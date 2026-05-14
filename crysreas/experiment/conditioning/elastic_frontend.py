"""
Frontend for conditioning elastic comparison:
read backend table and render bar figure only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import add_panel_label, apply_publication_style, plot_grouped_bars
from .elastic_backend import DATA_DIR, OUT_BAR

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

# -------- Editable plotting config (frontend only) --------
X_AXIS_LABEL = r"Elastic Properties Range Requirement ($\text{GPa}$)"
Y_AXIS_LABEL = "Ratio of Compliant Structures"
PLOT_TITLE = "Ratio of Compliant Structures \n by Elastic Properties Range Requirement"
Y_COLUMN = "ratio_in_category"  # "ratio_in_category" or "n_reward_2"
SHOW_CATEGORY_INDEX = False
CATEGORY_INDEX_PREFIX = "S"
CUSTOM_XTICK_LABELS: dict[int, str] = {}

apply_publication_style()
plt.rcParams["xtick.labelsize"] = 18


def load_data(data_dir: Path) -> pd.DataFrame:
    p_bar = data_dir / OUT_BAR
    if not p_bar.is_file():
        raise FileNotFoundError(f"Missing backend data file: {p_bar}")
    return pd.read_parquet(p_bar)


def draw(ax: plt.Axes, data_dir: Path, *, panel_label: str | None = None) -> None:
    df = load_data(data_dir)
    order = (
        df[["sample_id", "sample_label"]]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
        .head(5)
    )
    x = np.arange(len(order))
    w = 0.30
    label_map = dict(zip(order["sample_id"].tolist(), order["sample_label"].tolist(), strict=True))

    rl = (
        df[df["model"] == "rl_thinking_mix"][["sample_id", Y_COLUMN]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(0)
    )
    rl_low = (
        df[df["model"] == "rl_thinking_mix"][["sample_id", f"{Y_COLUMN}_ci_low"]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(np.nan)
    )
    rl_high = (
        df[df["model"] == "rl_thinking_mix"][["sample_id", f"{Y_COLUMN}_ci_high"]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(np.nan)
    )
    baseline = (
        df[df["model"] == "elastic_thinking"][["sample_id", Y_COLUMN]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(0)
    )
    baseline_low = (
        df[df["model"] == "elastic_thinking"][["sample_id", f"{Y_COLUMN}_ci_low"]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(np.nan)
    )
    baseline_high = (
        df[df["model"] == "elastic_thinking"][["sample_id", f"{Y_COLUMN}_ci_high"]]
        .set_index("sample_id")
        .reindex(order["sample_id"])
        .fillna(np.nan)
    )
    tick_labels: list[str] = []
    for sid in order["sample_id"].tolist():
        if sid in CUSTOM_XTICK_LABELS:
            tick_labels.append(CUSTOM_XTICK_LABELS[sid])
            continue
        base = label_map[sid]
        if SHOW_CATEGORY_INDEX:
            tick_labels.append(f"{CATEGORY_INDEX_PREFIX}{sid}: {base}")
        else:
            tick_labels.append(base)
    plot_grouped_bars(
        ax,
        x=x,
        series=[
            {
                "x": x - w / 2,
                "y": rl[Y_COLUMN].to_numpy(dtype=float),
                "ci_low": rl_low[f"{Y_COLUMN}_ci_low"].to_numpy(dtype=float),
                "ci_high": rl_high[f"{Y_COLUMN}_ci_high"].to_numpy(dtype=float),
                "width": w,
                "color": "#4c78a8",
                "label": "CrysReas",
            },
            {
                "x": x + w / 2,
                "y": baseline[Y_COLUMN].to_numpy(dtype=float),
                "ci_low": baseline_low[f"{Y_COLUMN}_ci_low"].to_numpy(dtype=float),
                "ci_high": baseline_high[f"{Y_COLUMN}_ci_high"].to_numpy(dtype=float),
                "width": w,
                "color": "#f58518",
                "label": "CrysReas-ElasticProperties",
            },
        ],
        xticks=x,
        xtick_labels=tick_labels,
        ylabel=Y_AXIS_LABEL,
        xlabel=X_AXIS_LABEL,
        title=PLOT_TITLE,
        ylim=(0.3, 0.9) if Y_COLUMN.startswith("ratio_") else None,
        rotation=20,
        ha="right",
    )
    if Y_COLUMN.startswith("ratio_"):
        ax.set_yticks(np.arange(0.3, 0.91, 0.2))
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(data_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    draw(ax, data_dir)
    fig.tight_layout()
    out_path = figure_dir / "conditioning_elastic_reward_all_eq2_bar.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"conditioning elastic frontend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conditioning elastic reward==2 bar figure.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()
    run(data_dir=Path(args.data_dir), figure_dir=Path(args.figure_dir))


if __name__ == "__main__":
    main()
