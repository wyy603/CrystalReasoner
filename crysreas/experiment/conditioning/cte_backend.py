"""
Backend for conditioning CTE comparison:
compute grouped bar data for two target checkpoints only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import wilson_ci_95

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_BAR = "conditioning_cte_reward_all_eq1_bar.parquet"

DEFAULT_RL_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "rl_thinking_mix"
    / "cte+thinking.parquet"
)
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[3]
    / "checkpoints_merged"
    / "rl_cte_thinking"
    / "cte+thinking.parquet"
)

TARGET_REWARD = 1.0


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


def _parse_gt_cte_range(gt: Any) -> tuple[float, float] | None:
    if not isinstance(gt, dict):
        return None
    cte_range = gt.get("thermal_expansion_300k")
    if isinstance(cte_range, np.ndarray):
        seq = cte_range.tolist()
    elif isinstance(cte_range, (list, tuple)):
        seq = list(cte_range)
    else:
        return None
    if len(seq) != 2:
        return None
    left = _to_float(seq[0])
    right = _to_float(seq[1])
    if left is None or right is None:
        return None
    return (left, right)


def _range_sort_key(r: tuple[float, float]) -> tuple[float, float]:
    left, right = r
    left_key = -np.inf if np.isneginf(left) else left
    right_key = np.inf if np.isposinf(right) else right
    return left_key, right_key


def _range_label(r: tuple[float, float]) -> str:
    left, right = r
    left_str = "-inf" if np.isneginf(left) else f"{left:g}"
    right_str = "inf" if np.isposinf(right) else f"{right:g}"
    return f"[{left_str}, {right_str}]"


def _keep_non_negative_range(r: tuple[float, float]) -> bool:
    left, right = r
    if np.isnan(left) or np.isnan(right):
        return False
    # Remove ranges below zero; keep bins starting from zero.
    return left >= 0.0


def _build_range_stats(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["gt", "cte_reward_all"]).copy()
    df["cte_range"] = df["gt"].map(_parse_gt_cte_range)
    df = df[df["cte_range"].notna()].copy()
    df = df[df["cte_range"].map(_keep_non_negative_range)].copy()
    out = df.groupby("cte_range", observed=True).size().rename("n_total").reset_index()
    reward_counts = (
        df[df["cte_reward_all"] == TARGET_REWARD]
        .groupby("cte_range", observed=True)
        .size()
        .rename("n_reward_1")
        .reset_index()
    )
    out = out.merge(reward_counts, on="cte_range", how="left")
    out["n_reward_1"] = out["n_reward_1"].fillna(0).astype(int)
    out["ratio_in_category"] = np.where(out["n_total"] > 0, out["n_reward_1"] / out["n_total"], np.nan)
    out["model"] = model_name
    return out


def run(rl_path: Path, baseline_path: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rl = _build_range_stats(rl_path, "rl_thinking_mix")
    baseline = _build_range_stats(baseline_path, "rl_cte_thinking")
    merged = pd.concat([rl, baseline], ignore_index=True)

    reward_by_range = (
        merged.pivot_table(
            index="cte_range",
            columns="model",
            values="n_reward_1",
            aggfunc="first",
        )
        .fillna(0)
        .astype(int)
    )
    keep_ranges = reward_by_range[(reward_by_range > 0).all(axis=1)].index.tolist()
    all_ranges = sorted(keep_ranges, key=_range_sort_key)

    rows: list[dict[str, object]] = []
    for idx, cte_range in enumerate(all_ranges, start=1):
        label = _range_label(cte_range)
        for model_name in ("rl_thinking_mix", "rl_cte_thinking"):
            g = merged[(merged["model"] == model_name) & (merged["cte_range"] == cte_range)]
            n_val = int(g["n_reward_1"].iloc[0]) if len(g) else 0
            rows.append(
                {
                    "range_id": idx,
                    "range_label": label,
                    "cte_range": cte_range,
                    "model": model_name,
                    "n_total": int(g["n_total"].iloc[0]) if len(g) else 0,
                    "n_reward_1": n_val,
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
    out = pd.DataFrame(rows).sort_values(["range_id", "model"]).reset_index(drop=True)
    out_path = data_dir / OUT_BAR
    out.to_parquet(out_path, index=False)
    print(f"conditioning cte backend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute conditioning CTE reward_all==1 bar data for two target checkpoints."
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
