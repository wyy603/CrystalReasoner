"""
Frontend for conditioning spacegroup comparison:
read backend table and render bar figure only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import add_panel_label, apply_publication_style, plot_grouped_bars
from .spacegroup_backend import DATA_DIR, OUT_BAR

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

# -------- Editable plotting config (frontend only) --------
X_AXIS_LABEL = "Spacegroup Instruction"
Y_AXIS_LABEL = "Ratio of Compliant Structures"
PLOT_TITLE = "Ratio of Compliant Structures \n by Spacegroup Instruction"
Y_COLUMN = "ratio_claimed_eq_structure"  # keep ratio metric by default
SHOW_CATEGORY_INDEX = False
CATEGORY_INDEX_PREFIX = "G"
CUSTOM_XTICK_LABELS: dict[str, str] = {}

apply_publication_style()


def load_data(data_dir: Path) -> pd.DataFrame:
    p_bar = data_dir / OUT_BAR
    if not p_bar.is_file():
        raise FileNotFoundError(f"Missing backend data file: {p_bar}")
    return pd.read_parquet(p_bar)


def draw(ax: plt.Axes, data_dir: Path, *, panel_label: str | None = None) -> None:
    df = load_data(data_dir)
    rl = df[df["model"] == "rl_thinking_mix"].set_index("spacegroup_symbol")[Y_COLUMN].copy()
    baseline = df[df["model"] == "spacegroup_thinking"].set_index("spacegroup_symbol")[Y_COLUMN].copy()
    rl_low = df[df["model"] == "rl_thinking_mix"].set_index("spacegroup_symbol")[f"{Y_COLUMN}_ci_low"].copy()
    rl_high = df[df["model"] == "rl_thinking_mix"].set_index("spacegroup_symbol")[f"{Y_COLUMN}_ci_high"].copy()
    base_low = df[df["model"] == "spacegroup_thinking"].set_index("spacegroup_symbol")[f"{Y_COLUMN}_ci_low"].copy()
    base_high = df[df["model"] == "spacegroup_thinking"].set_index("spacegroup_symbol")[f"{Y_COLUMN}_ci_high"].copy()
    symbols = df["spacegroup_symbol"].drop_duplicates().tolist()
    rl_vals = np.array([rl.get(sym, np.nan) for sym in symbols], dtype=float)
    base_vals = np.array([baseline.get(sym, np.nan) for sym in symbols], dtype=float)
    rl_ci_low = np.array([rl_low.get(sym, np.nan) for sym in symbols], dtype=float)
    rl_ci_high = np.array([rl_high.get(sym, np.nan) for sym in symbols], dtype=float)
    base_ci_low = np.array([base_low.get(sym, np.nan) for sym in symbols], dtype=float)
    base_ci_high = np.array([base_high.get(sym, np.nan) for sym in symbols], dtype=float)

    x = np.arange(len(symbols))
    w = 0.30
    tick_labels: list[str] = []
    for i, sym in enumerate(symbols, start=1):
        if sym in CUSTOM_XTICK_LABELS:
            tick_labels.append(CUSTOM_XTICK_LABELS[sym])
            continue
        if SHOW_CATEGORY_INDEX:
            tick_labels.append(f"{CATEGORY_INDEX_PREFIX}{i}: {sym}")
        else:
            tick_labels.append(sym)
    plot_grouped_bars(
        ax,
        x=x,
        series=[
            {
                "x": x - w / 2,
                "y": rl_vals,
                "ci_low": rl_ci_low,
                "ci_high": rl_ci_high,
                "width": w,
                "color": "#4c78a8",
                "label": "CrysReas",
            },
            {
                "x": x + w / 2,
                "y": base_vals,
                "ci_low": base_ci_low,
                "ci_high": base_ci_high,
                "width": w,
                "color": "#f58518",
                "label": "CrysReas-Spacegroup",
            },
        ],
        xticks=x,
        xtick_labels=tick_labels,
        ylabel=Y_AXIS_LABEL,
        xlabel=X_AXIS_LABEL,
        title=PLOT_TITLE,
        ylim=(0, 1.02) if Y_COLUMN.startswith("ratio_") else None,
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(data_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    draw(ax, data_dir)
    fig.tight_layout()
    out_path = figure_dir / "conditioning_spacegroup_bar_compare.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"conditioning frontend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conditioning spacegroup bar comparison figure.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()
    run(data_dir=Path(args.data_dir), figure_dir=Path(args.figure_dir))


if __name__ == "__main__":
    main()
