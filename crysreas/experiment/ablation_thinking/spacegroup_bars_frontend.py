"""
Frontend for ablation thinking: bar chart of spacegroup consistency ratio
by removed thinking section (part1–3), with a no-ablation baseline from
conditional+thinking parquet.
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
    plot_grouped_bars,
)
from .spacegroup_bars_backend import DATA_DIR, OUT_BAR

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

apply_publication_style()

SECTION_ORDER = ("part1", "part2", "part3")
SECTION_DISPLAY = {
    "part1": "Part 1",
    "part2": "Part 2",
    "part3": "Part 3",
}


def _ratio_for(df: pd.DataFrame, *, operation: str, section: str) -> float:
    m = df[(df["operation"] == operation) & (df["target_section"] == section)]
    if len(m) == 0:
        return float("nan")
    return float(m["ratio"].iloc[0])


def _value_for(df: pd.DataFrame, *, operation: str, section: str, col: str) -> float:
    m = df[(df["operation"] == operation) & (df["target_section"] == section)]
    if len(m) == 0:
        return float("nan")
    return float(m[col].iloc[0])


def load_data(*, data_dir: Path) -> pd.DataFrame:
    src = data_dir / OUT_BAR
    if not src.is_file():
        raise FileNotFoundError(f"Missing backend data {src}; run spacegroup_bars_backend first.")
    return pd.read_parquet(src)

def draw(ax: plt.Axes, *, data_dir: Path, panel_label: str | None = None) -> None:
    df = load_data(data_dir=data_dir)
    baseline_ratio = float(df["baseline_ratio"].iloc[0]) if "baseline_ratio" in df.columns and len(df) > 0 else float("nan")
    baseline_ci_low = float(df["baseline_ci_low"].iloc[0]) if "baseline_ci_low" in df.columns and len(df) > 0 else float("nan")
    baseline_ci_high = float(df["baseline_ci_high"].iloc[0]) if "baseline_ci_high" in df.columns and len(df) > 0 else float("nan")

    section_x = np.arange(1, len(SECTION_ORDER) + 1)  # x=1,2,3 for Part 1/2/3
    original_x = 0.0
    w = 0.34
    tick_positions = np.concatenate(([original_x], section_x))
    tick_labels = ["Original"] + [SECTION_DISPLAY[s] for s in SECTION_ORDER]
    series = []
    if not np.isnan(baseline_ratio):
        style = MODEL_STYLE["original"]
        series.append(
            {
                "x": np.array([original_x], dtype=float),
                "y": np.array([baseline_ratio], dtype=float),
                "ci_low": np.array([baseline_ci_low], dtype=float),
                "ci_high": np.array([baseline_ci_high], dtype=float),
                "width": w,
                "color": style["color"],
                "label": style["label"],
            }
        )
    style = MODEL_STYLE["remove"]
    series.append(
        {
            "x": section_x,
            "y": np.array([_ratio_for(df, operation="remove", section=sec) for sec in SECTION_ORDER], dtype=float),
            "ci_low": np.array(
                [_value_for(df, operation="remove", section=sec, col="ci_low") for sec in SECTION_ORDER],
                dtype=float,
            ),
            "ci_high": np.array(
                [_value_for(df, operation="remove", section=sec, col="ci_high") for sec in SECTION_ORDER],
                dtype=float,
            ),
            "width": w,
            "color": style["color"],
            "label": style["label"],
        }
    )
    plot_grouped_bars(
        ax,
        x=tick_positions,
        series=series,
        xticks=tick_positions,
        xtick_labels=tick_labels,
        ylabel="Spacegroup Consistency Ratio",
        xlabel="Ablated Section in Thinking Trace",
        title="Spacegroup Consistency \nafter Removing Thinking Sections",
        ylim=(0.6, 1.0),
    )
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(*, data_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    draw(ax, data_dir=data_dir)
    fig.tight_layout()
    out_pdf = figure_dir / "ablation_thinking_spacegroup_part_bars.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"ablation spacegroup bars frontend wrote {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ablation thinking spacegroup bar figure.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()
    run(data_dir=Path(args.data_dir), figure_dir=Path(args.figure_dir))


if __name__ == "__main__":
    main()
