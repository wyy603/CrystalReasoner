"""
Backend for EXPERIMENT §1.5: compute tables for thinking vs no-thinking comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup

from crysreas.experiment.assets.plot_helpers import wilson_ci_95

from .common import (
    ComparePaths,
    as_bool_series,
    count_atoms,
    ensure_thinking_dirs,
    parse_instruction_spacegroup_id,
)

DATA_1_5_SG_BAR = "thinking_exp_1_5_spacegroup_consistency_bar.parquet"
DATA_1_5_METRIC_CURVE = "thinking_exp_1_5_metrics_vs_atoms_curve.parquet"

TARGET_SG_SYMBOLS = ["P1", "C2/c", "Amm2", "I4m2", "P3", "P6_3/mmc", "F-43m"]
METRICS = ["structure_validity", "smact_validity", "composition_consistency"]


def _symbol_to_id(symbol: str) -> int:
    manual = {"I4m2": 119}
    if symbol in manual:
        return manual[symbol]
    return int(SpaceGroup(symbol).int_number)


def _build_base_frame(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "mp_id": df["mp_id"].astype(str).to_numpy(),
            "model": model_label,
            "sg_id_instruction": [parse_instruction_spacegroup_id(x) for x in df["prompt"].tolist()],
            "spacegroup_consistency": as_bool_series(df["spacegroup_consistency"]).astype(float),
            "n_atoms": [count_atoms(x) for x in df["simple_structure"].tolist()],
            "structure_validity": as_bool_series(df["structure_validity"]).astype(float),
            "smact_validity": as_bool_series(df["smact_validity"]).astype(float),
            "composition_consistency": as_bool_series(df["composition_consistency"]).astype(float),
        }
    )
    return out


def _load_existing_curve(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _load_existing_bar(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir)
    think_cols = [
        "mp_id",
        "prompt",
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "composition_consistency",
        "spacegroup_consistency",
    ]
    no_cols = [
        "mp_id",
        "prompt",
        "simple_structure",
        "structure_validity",
    ]
    p_bar = paths.data_dir / DATA_1_5_SG_BAR
    p_curve = paths.data_dir / DATA_1_5_METRIC_CURVE
    existing_bar = _load_existing_bar(p_bar)
    existing_curve = _load_existing_curve(p_curve)

    df_think = pd.read_parquet(paths.thinking_parquet, columns=think_cols)
    df_no = pd.read_parquet(paths.no_thinking_parquet, columns=no_cols)
    if not df_think["mp_id"].astype(str).equals(df_no["mp_id"].astype(str)):
        raise ValueError("mp_id sequence is not aligned between thinking and no-thinking parquet.")

    think = _build_base_frame(df_think, "thinking")
    no_think = _build_base_frame(
        df_no.assign(
            smact_validity=pd.NA,
            composition_consistency=pd.NA,
            spacegroup_consistency=pd.NA,
        ),
        "no_thinking",
    )
    merged = pd.concat([think, no_think], ignore_index=True)

    sg_id_map = {sym: _symbol_to_id(sym) for sym in TARGET_SG_SYMBOLS}
    sg_rows: list[dict[str, object]] = []
    for sym in TARGET_SG_SYMBOLS:
        sid = sg_id_map[sym]
        g_think = think[think["sg_id_instruction"] == sid]
        g_no = no_think[no_think["sg_id_instruction"] == sid]
        think_true = int(g_think["spacegroup_consistency"].fillna(0).astype(bool).sum())
        think_stats = wilson_ci_95(think_true, len(g_think))
        if existing_bar is not None:
            g_no_old = existing_bar[existing_bar["spacegroup_id"] == sid]
        else:
            g_no_old = pd.DataFrame()
        if len(g_no_old) > 0:
            no_n = int(g_no_old["n_no_thinking"].iloc[0])
            no_true = int(round(float(g_no_old["spacegroup_consistency_no_thinking"].iloc[0]) * no_n))
        else:
            no_n = int(len(g_no))
            no_true = int(g_no["spacegroup_consistency"].fillna(0).astype(bool).sum())
        no_stats = wilson_ci_95(no_true, no_n)
        sg_rows.append(
            {
                "spacegroup_symbol": sym,
                "spacegroup_id": sid,
                "n_thinking": int(len(g_think)),
                "n_no_thinking": no_stats["n_samples"],
                "spacegroup_consistency_thinking_true_count": think_stats["true_count"],
                "spacegroup_consistency_thinking": think_stats["ratio"],
                "spacegroup_consistency_thinking_ci_low": think_stats["ci_low"],
                "spacegroup_consistency_thinking_ci_high": think_stats["ci_high"],
                "spacegroup_consistency_no_thinking_true_count": no_stats["true_count"],
                "spacegroup_consistency_no_thinking": no_stats["ratio"],
                "spacegroup_consistency_no_thinking_ci_low": no_stats["ci_low"],
                "spacegroup_consistency_no_thinking_ci_high": no_stats["ci_high"],
            }
        )
    sg_bar = pd.DataFrame(sg_rows)

    metric_rows: list[dict[str, object]] = []
    for model, g_model in merged.groupby("model", observed=True):
        g_model = g_model[g_model["n_atoms"].notna()].copy()
        g_model["n_atoms"] = g_model["n_atoms"].astype(int)
        grouped = g_model.groupby("n_atoms", observed=True)
        for n_atoms, g_atoms in grouped:
            row: dict[str, object] = {
                "model": model,
                "n_atoms": int(n_atoms),
                "n_samples": int(len(g_atoms)),
            }
            for metric in METRICS:
                if model == "no_thinking" and metric != "structure_validity" and existing_curve is not None:
                    old_row = existing_curve[
                        (existing_curve["model"] == model) & (existing_curve["n_atoms"] == int(n_atoms))
                    ]
                    if len(old_row) > 0:
                        old_n = int(old_row["n_samples"].iloc[0])
                        old_ratio = float(old_row[metric].iloc[0])
                        old_true = int(round(old_ratio * old_n))
                        stats = wilson_ci_95(old_true, old_n)
                    else:
                        stats = wilson_ci_95(0, 0)
                else:
                    valid = g_atoms[metric].dropna().astype(bool)
                    stats = wilson_ci_95(int(valid.sum()), int(valid.size))
                row[metric] = stats["ratio"]
                row[f"{metric}_true_count"] = stats["true_count"]
                row[f"{metric}_ci_low"] = stats["ci_low"]
                row[f"{metric}_ci_high"] = stats["ci_high"]
            metric_rows.append(row)
    metric_curve = pd.DataFrame(metric_rows).sort_values(["model", "n_atoms"]).reset_index(drop=True)

    sg_bar.to_parquet(p_bar, index=False)
    metric_curve.to_parquet(p_curve, index=False)
    print(f"1.5 backend wrote {p_bar} and {p_curve}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.5 backend (data only).")
    parser.add_argument("--thinking", type=str, default="checkpoints_merged/thinking/conditional+thinking.parquet")
    parser.add_argument(
        "--no-thinking",
        type=str,
        default="checkpoints_merged/no_thinking/conditional+thinking.parquet",
    )
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
        )
    )


if __name__ == "__main__":
    main()
