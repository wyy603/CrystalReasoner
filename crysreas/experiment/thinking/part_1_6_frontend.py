"""
Frontend for EXPERIMENT §1.6: read backend table and render trace-ablation figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import (
    add_panel_label,
    aggregate_mean_curve,
    apply_publication_style,
    plot_curve_with_ci,
)
from .common import ComparePaths, ensure_thinking_dirs
from .part_1_6_backend import DATA_1_6_CURVE, DATA_1_6_PER_ROW

apply_publication_style()


def load_curve(paths: ComparePaths) -> pd.DataFrame:
    src = paths.data_dir / DATA_1_6_CURVE
    if src.is_file():
        curve = pd.read_parquet(src)
    else:
        curve = pd.DataFrame()
    need_cols = {
        "condition",
        "n_atoms",
        "average_of_information",
        "average_of_information_ci_low",
        "average_of_information_ci_high",
    }
    if not need_cols.issubset(curve.columns):
        p_row = paths.data_dir / DATA_1_6_PER_ROW
        if not p_row.is_file():
            raise FileNotFoundError(f"Missing backend data {src} and {p_row}; run part_1_6_backend first.")
        per_row = pd.read_parquet(p_row)
        per_row = per_row[per_row["n_atoms"].notna()].copy()
        per_row["n_atoms"] = per_row["n_atoms"].astype(int)
        curve = pd.concat(
            [
                aggregate_mean_curve(
                    per_row.rename(columns={"average_of_information_with_trace": "average_of_information"}),
                    group_cols=["n_atoms"],
                    value_col="average_of_information",
                ).assign(condition="with_trace"),
                aggregate_mean_curve(
                    per_row.rename(columns={"average_of_information_without_trace": "average_of_information"}),
                    group_cols=["n_atoms"],
                    value_col="average_of_information",
                ).assign(condition="without_trace"),
            ],
            ignore_index=True,
        )
    curve = curve[curve["n_atoms"].between(1, 21)].copy()
    return curve


def draw(ax: plt.Axes, paths: ComparePaths, *, panel_label: str | None = None) -> None:
    curve = load_curve(paths)
    plot_curve_with_ci(
        ax,
        df=curve,
        group_col="condition",
        groups=("with_trace", "without_trace"),
        x_col="n_atoms",
        y_col="average_of_information",
        xlabel="Number of Atoms",
        ylabel="Average CIF Information",
        title="Average CIF Information vs Number of Atoms",
        xticks=np.arange(1, 22, step=4),
        legend_loc="best",
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(figure_dir=paths.figure_dir)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    draw(ax, paths)
    fig.tight_layout()
    out_pdf = paths.figure_dir / "thinking_exp_1_6_average_of_information_vs_atoms.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"1.6 frontend wrote {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.6 frontend (figures only).")
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
