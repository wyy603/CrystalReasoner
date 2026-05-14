"""
Frontend for EXPERIMENT §1.3: read backend tables and render figures only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ComparePaths, ensure_thinking_dirs
from .part_1_3_backend import DATA_1_3_PER_ROW, DATA_1_3_PREDICTION_EXAMPLES, DATA_1_3_SG_BAR

PREDICTION_JSON = "prediction.json"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Increase all figure font sizes.
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def _quantile_binned_mean(x: np.ndarray, y: np.ndarray, *, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x.astype(float)[mask]
    y = y.astype(float)[mask]
    if len(x) < 10:
        return np.array([]), np.array([])
    n_bins = int(min(n_bins, max(8, len(x) // 20)))
    try:
        cats = pd.qcut(x, q=n_bins, duplicates="drop")
    except ValueError:
        uniq = len(np.unique(x))
        cats = pd.cut(x, bins=min(n_bins, max(3, uniq)))
    g = pd.DataFrame({"x": x, "y": y, "b": cats}).groupby("b", observed=True)
    xm = g["x"].mean()
    ym = g["y"].mean()
    order = np.argsort(xm.to_numpy())
    return xm.to_numpy()[order], ym.to_numpy()[order]


def _plot_atoms_vs_metric(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    m = (x >= 2) & (x <= 21) & np.isfinite(x) & np.isfinite(y)   
    x = x[m].astype(float)
    y = y[m].astype(float)
    if len(x) == 0:
        return
    xm, ym = _quantile_binned_mean(x, y)
    plt.figure(figsize=(10, 6), dpi=220)
    plt.scatter(x, y, s=6, alpha=0.1, color="#4c78a8", label="Samples")
    if len(xm) > 0:
        plt.plot(xm, ym, color="#d62728", linewidth=2.2, label="Mean Curve")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.ylim(0, 10)
    xmax = float(np.nanpercentile(x, 99.5))
    plt.xlim(0, max(xmax * 1.05, 1.0))
    plt.xticks(np.arange(2, 22, step=4))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _write_prediction_json(examples_df: pd.DataFrame, out_path: Path) -> None:
    examples: list[dict[str, object]] = []
    for row in examples_df.to_dict(orient="records"):
        examples.append(
            {
                "mp_id": row["mp_id"],
                "n_atoms": None if pd.isna(row["n_atoms"]) else int(row["n_atoms"]),
                "prompt": row["prompt"],
                "response": row["response"],
                "ground_truth_response": row["ground_truth_response"],
                "response_ground_truth_similarity": row["response_ground_truth_similarity"],
                "response_ground_truth_similarity_rank": row["response_ground_truth_similarity_rank"],
                "response_ground_truth_similarity_subset_size": row[
                    "response_ground_truth_similarity_subset_size"
                ],
                "selection_method": row.get("selection_method"),
                "selection_seed": None if pd.isna(row.get("selection_seed")) else int(row["selection_seed"]),
                "consistency_subset_size": None
                if pd.isna(row.get("consistency_subset_size"))
                else int(row["consistency_subset_size"]),
                "final_cif": row["final_cif"],
                "consistency_v2": json.loads(row["consistency_v2_json"]),
            }
        )
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(examples, handle, indent=2, ensure_ascii=False)


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir, figure_dir=paths.figure_dir)
    p_row = paths.data_dir / DATA_1_3_PER_ROW
    p_bar = paths.data_dir / DATA_1_3_SG_BAR
    p_examples = paths.data_dir / DATA_1_3_PREDICTION_EXAMPLES
    if not p_row.is_file() or not p_bar.is_file() or not p_examples.is_file():
        raise FileNotFoundError(f"Missing backend data under {paths.data_dir}; run part_1_3_backend first.")
    per_row = pd.read_parquet(p_row)
    grp_df = pd.read_parquet(p_bar)
    examples_df = pd.read_parquet(p_examples)

    x = np.arange(len(grp_df))
    w = 0.24
    plt.figure(figsize=(13, 6), dpi=220)
    plt.bar(x - w, grp_df["ratio_claimed_eq_structure"], width=w, label="claimed == structure")
    plt.bar(x, grp_df["ratio_instruction_eq_claimed"], width=w, label="instruction == claimed")
    plt.xticks(x, grp_df["spacegroup_symbol"])
    plt.ylim(0, 1.02)
    plt.ylabel("Ratio of Spacegroup Corrections")
    plt.xlabel("Spacegroup Instruction")
    plt.title("Spacegroup Consistency by Instruction Group")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figure_dir / "thinking_exp_1_3_spacegroup_bar.pdf")
    plt.close()

    n_atoms = per_row["n_atoms"].to_numpy(dtype=float)
    vol_pct = per_row["volume_rel_diff_pct"].to_numpy(dtype=float)
    _plot_atoms_vs_metric(
        n_atoms,
        vol_pct,
        xlabel="Number of Atoms",
        ylabel="Volume Difference Relative (%)",
        title="Volume Relative Error vs Number of Atoms",
        out_path=paths.figure_dir / "thinking_exp_1_3_volume_rel_pct_vs_n_atoms.pdf",
    )

    bond_rel = per_row["bond_median_min_rel_pct"].to_numpy(dtype=float)
    _plot_atoms_vs_metric(
        n_atoms,
        bond_rel,
        xlabel="Number of Atoms",
        ylabel="Bond Length Difference Relative (%)",
        title="Bond Length Relative Error vs Number of Atoms",
        out_path=paths.figure_dir / "thinking_exp_1_3_bond_rel_pct_vs_n_atoms.pdf",
    )

    prediction_path = paths.data_dir / PREDICTION_JSON
    _write_prediction_json(examples_df, prediction_path)

    print(f"1.3 frontend wrote figures under {paths.figure_dir} and {prediction_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.3 frontend (figures only).")
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
    parser.add_argument("--figure-dir", type=str, default=str(ComparePaths().figure_dir))
    args = parser.parse_args()
    run(
        ComparePaths(
            data_dir=Path(args.data_dir),
            figure_dir=Path(args.figure_dir),
        )
    )


if __name__ == "__main__":
    main()
