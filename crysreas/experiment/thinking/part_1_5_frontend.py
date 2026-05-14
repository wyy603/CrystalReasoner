"""
Frontend for EXPERIMENT §1.5: read backend tables and render figures only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import (
    MODEL_STYLE,
    add_panel_label,
    apply_publication_style,
    plot_curve_with_ci,
    plot_grouped_bars,
)
from .common import ComparePaths, ensure_thinking_dirs
from .part_1_5_backend import DATA_1_5_METRIC_CURVE, DATA_1_5_SG_BAR, METRICS

apply_publication_style()

METRIC_NAME = {
    "structure_validity": "Structure Validity",
    "smact_validity": "Chemical Validity",
    "composition_consistency": "Composition Consistency",
}


def load_data(paths: ComparePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_bar = paths.data_dir / DATA_1_5_SG_BAR
    p_curve = paths.data_dir / DATA_1_5_METRIC_CURVE
    if not p_bar.is_file() or not p_curve.is_file():
        raise FileNotFoundError(f"Missing backend data under {paths.data_dir}; run part_1_5_backend first.")
    sg_bar = pd.read_parquet(p_bar)
    metric_curve = pd.read_parquet(p_curve)
    metric_curve = metric_curve[metric_curve["n_atoms"].between(1, 21)].copy()
    return sg_bar, metric_curve


def draw_spacegroup_bar(ax: plt.Axes, df: pd.DataFrame, *, panel_label: str | None = None) -> None:
    x = np.arange(len(df))
    w = 0.36
    plot_grouped_bars(
        ax,
        x=x,
        series=[
            {
                "x": x - w / 2,
                "y": df["spacegroup_consistency_thinking"].to_numpy(dtype=float),
                "ci_low": df["spacegroup_consistency_thinking_ci_low"].to_numpy(dtype=float),
                "ci_high": df["spacegroup_consistency_thinking_ci_high"].to_numpy(dtype=float),
                "width": w,
                "color": MODEL_STYLE["thinking"]["color"],
                "label": MODEL_STYLE["thinking"]["label"],
            },
            {
                "x": x + w / 2,
                "y": df["spacegroup_consistency_no_thinking"].to_numpy(dtype=float),
                "ci_low": df["spacegroup_consistency_no_thinking_ci_low"].to_numpy(dtype=float),
                "ci_high": df["spacegroup_consistency_no_thinking_ci_high"].to_numpy(dtype=float),
                "width": w,
                "color": MODEL_STYLE["no_thinking"]["color"],
                "label": MODEL_STYLE["no_thinking"]["label"],
            },
        ],
        xticks=x,
        xtick_labels=df["spacegroup_symbol"].tolist(),
        ylabel="Spacegroup Consistency",
        xlabel="Spacegroup in Instruction",
        title="Spacegroup Consistency\n by Spacegroup Type",
        ylim=(0.0, 1.02),
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def draw_metric_curve(ax: plt.Axes, df: pd.DataFrame, metric: str, *, panel_label: str | None = None) -> None:
    plot_curve_with_ci(
        ax,
        df=df,
        group_col="model",
        groups=("thinking", "no_thinking"),
        x_col="n_atoms",
        y_col=metric,
        xlabel="Number of Atoms",
        ylabel=METRIC_NAME[metric],
        title=f"{METRIC_NAME[metric]} \nvs Number of Atoms",
        xticks=np.arange(1, 22, step=4),
        ylim=(0.0, 1.02),
        legend_loc="lower right",
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(figure_dir=paths.figure_dir)
    sg_bar, metric_curve = load_data(paths)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    draw_spacegroup_bar(ax, sg_bar)
    fig.tight_layout()
    fig.savefig(paths.figure_dir / "thinking_exp_1_5_spacegroup_consistency_bar.pdf")
    plt.close(fig)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
        draw_metric_curve(ax, metric_curve, metric)
        fig.tight_layout()
        fig.savefig(paths.figure_dir / f"thinking_exp_1_5_{metric}_vs_atoms.pdf")
        plt.close(fig)
    print(f"1.5 frontend wrote figures under {paths.figure_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.5 frontend (figures only).")
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
