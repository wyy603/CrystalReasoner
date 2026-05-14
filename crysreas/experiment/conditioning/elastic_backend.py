"""
Backend for conditioning elastic comparison:
compute grouped bar data for elastic_reward_all == 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import wilson_ci_95

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_BAR = "conditioning_elastic_reward_all_eq2_bar.parquet"

DEFAULT_RL_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "rl_thinking_mix"
    / "elastic+thinking.parquet"
)
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "elastic_thinking"
    / "elastic+thinking.parquet"
)

TARGET_REWARD = 2.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(x):
        return None
    return x


def _parse_range(value: Any) -> tuple[float, float] | None:
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        return None
    if len(seq) != 2:
        return None
    left = _to_float(seq[0])
    right = _to_float(seq[1])
    if left is None or right is None:
        return None
    return (left, right)


def _range_str(r: tuple[float, float]) -> str:
    left, right = r
    if np.isinf(right):
        right_str = "inf"
    else:
        right_str = f"{right:g}"
    return f"[{left:g}, {right_str}]"


def _build_pair_stats_frame(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["gt", "elastic_reward_all"]).copy()
    df["bulk_range"] = df["gt"].map(
        lambda x: _parse_range(x.get("bulk_modulus")) if isinstance(x, dict) else None
    )
    df["shear_range"] = df["gt"].map(
        lambda x: _parse_range(x.get("shear_modulus")) if isinstance(x, dict) else None
    )
    df = df[df["bulk_range"].notna() & df["shear_range"].notna()].copy()
    grouped = df.groupby(["bulk_range", "shear_range"], observed=True)
    out = grouped.size().rename("n_total").reset_index()
    reward_counts = (
        df[df["elastic_reward_all"] == TARGET_REWARD]
        .groupby(["bulk_range", "shear_range"], observed=True)
        .size()
        .rename("n_reward_2")
        .reset_index()
    )
    out = out.merge(reward_counts, on=["bulk_range", "shear_range"], how="left")
    out["n_reward_2"] = out["n_reward_2"].fillna(0).astype(int)
    out["ratio_in_category"] = np.where(out["n_total"] > 0, out["n_reward_2"] / out["n_total"], np.nan)
    out["model"] = model_name
    return out


def run(rl_path: Path, baseline_path: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    rl = _build_pair_stats_frame(rl_path, "rl_thinking_mix")
    baseline = _build_pair_stats_frame(baseline_path, "elastic_thinking")
    merged = pd.concat([rl, baseline], ignore_index=True)

    all_pairs = (
        merged.groupby(["bulk_range", "shear_range"], observed=True)["n_total"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    rows: list[dict[str, object]] = []
    for i, (bulk_r, shear_r) in enumerate(all_pairs, start=1):
        sample_name = f"B{_range_str(bulk_r)} | G{_range_str(shear_r)}"
        for model_name in ("rl_thinking_mix", "elastic_thinking"):
            g = merged[
                (merged["bulk_range"] == bulk_r)
                & (merged["shear_range"] == shear_r)
                & (merged["model"] == model_name)
            ]
            n_val = int(g["n_reward_2"].iloc[0]) if len(g) else 0
            rows.append(
                {
                    "sample_id": i,
                    "sample_label": sample_name,
                    "bulk_range": bulk_r,
                    "shear_range": shear_r,
                    "model": model_name,
                    "n_total": int(g["n_total"].iloc[0]) if len(g) else 0,
                    "n_reward_2": n_val,
                    "ratio_in_category": float(g["ratio_in_category"].iloc[0]) if len(g) else float("nan"),
                    "ratio_in_category_ci_low": wilson_ci_95(
                        n_val,
                        int(g["n_total"].iloc[0]) if len(g) else 0,
                    )["ci_low"],
                    "ratio_in_category_ci_high": wilson_ci_95(
                        n_val,
                        int(g["n_total"].iloc[0]) if len(g) else 0,
                    )["ci_high"],
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["sample_id", "model"]).reset_index(drop=True)
    out_path = data_dir / OUT_BAR
    out_df.to_parquet(out_path, index=False)
    print(f"conditioning elastic backend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute conditioning elastic bar data for two target checkpoints."
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
