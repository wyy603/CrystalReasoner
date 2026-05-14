"""
Backend for conditioning spacegroup comparison:
compute grouped bar data for two target checkpoints only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup
from tqdm import tqdm

from crysreas.experiment.thinking.common import (
    crystal_section,
    parse_instruction_spacegroup_id,
    parse_simple_structure,
    parse_space_group_id,
    split_thinking_and_tail,
    structure_spacegroup_id,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_BAR = "conditioning_spacegroup_bar_compare.parquet"

DEFAULT_RL_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "rl_thinking_mix"
    / "conditional+thinking.parquet"
)
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "thinking"
    / "conditional+thinking.parquet"
)

TARGET_SG_SYMBOLS = ["P1", "C2/c", "Amm2", "I4m2", "P3", "P6_3/mmc", "F-43m"]


def _symbol_to_id(symbol: str) -> int:
    manual = {"I4m2": 119}
    if symbol in manual:
        return manual[symbol]
    return int(SpaceGroup(symbol).int_number)


def _build_per_row(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for prompt, response, simple in tqdm(zip(
        df["prompt"].tolist(),
        df["responses"].tolist(),
        df["simple_structure"].tolist(),
        strict=True,
    ), total=len(df)):
        thinking, _ = split_thinking_and_tail(response)
        crystal = crystal_section(thinking)
        structure = parse_simple_structure(str(simple))

        sg_instruction = parse_instruction_spacegroup_id(prompt)
        sg_claimed = parse_space_group_id(crystal)
        sg_structure = structure_spacegroup_id(structure)
        rows.append(
            {
                "sg_id_instruction": sg_instruction,
                "sg_id_claimed": sg_claimed,
                "sg_id_structure": sg_structure,
                "sg_id_claimed_eq_structure": (
                    bool(sg_claimed == sg_structure)
                    if sg_claimed is not None and sg_structure is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    per_row = _build_per_row(df)
    per_row = per_row[per_row["sg_id_instruction"].notna()].copy()
    sg_id_map = {sym: _symbol_to_id(sym) for sym in TARGET_SG_SYMBOLS}
    target_ids = set(sg_id_map.values())
    sel = per_row[per_row["sg_id_instruction"].isin(target_ids)].copy()

    grp_rows: list[dict[str, object]] = []
    for sym in TARGET_SG_SYMBOLS:
        print("sym", sym)
        sid = sg_id_map[sym]
        g = sel[sel["sg_id_instruction"] == sid]
        c1 = g["sg_id_claimed_eq_structure"].dropna()
        grp_rows.append(
            {
                "model": model_name,
                "spacegroup_symbol": sym,
                "spacegroup_id": sid,
                "n_rows": int(len(g)),
                "ratio_claimed_eq_structure": float(c1.mean()) if len(c1) else float("nan"),
            }
        )
    return pd.DataFrame(grp_rows)


def run(rl_path: Path, baseline_path: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    print("???")

    df_rl = pd.read_parquet(rl_path, columns=["prompt", "responses", "simple_structure"])
    df_baseline = pd.read_parquet(baseline_path, columns=["prompt", "responses", "simple_structure"])

    print("???")

    out_df = pd.concat(
        [
            _aggregate(df_rl, "rl_thinking_mix"),
            _aggregate(df_baseline, "thinking"),
        ],
        ignore_index=True,
    )
    model_total = out_df.groupby("model", observed=True)["n_rows"].transform("sum")
    out_df["ratio_in_model"] = np.where(model_total > 0, out_df["n_rows"] / model_total, np.nan)
    out_path = data_dir / OUT_BAR
    out_df.to_parquet(out_path, index=False)
    print(f"conditioning backend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute conditioning spacegroup bar data for two target checkpoints."
    )
    parser.add_argument("--rl-path", type=str, default=str(DEFAULT_RL_PATH))
    parser.add_argument("--baseline-path", type=str, default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()
    run(
        rl_path=Path(args.rl_path),
        baseline_path=Path(args.baseline_path),
        data_dir=Path(args.data_dir),
    )


if __name__ == "__main__":
    main()
