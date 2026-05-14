"""
Frontend for conditioning CTE comparison:
read backend table and render bar figure only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import add_panel_label, apply_publication_style, plot_grouped_bars
from .cte_backend import DATA_DIR, OUT_BAR

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

# -------- Editable plotting config (frontend only) --------
X_AXIS_LABEL = r"Thermal Expansion Range Requirement ($10^{-6} \, \mathrm{K}^{-1}$)"
Y_AXIS_LABEL = "Ratio of Compliant Structures"
PLOT_TITLE = "Ratio of Compliant Structures \n by Thermal Expansion Range Requirement"
Y_COLUMN = "ratio_in_category"  # "ratio_in_category" or "n_reward_1"
SHOW_CATEGORY_INDEX = False
CATEGORY_INDEX_PREFIX = "R"
CUSTOM_XTICK_LABELS: dict[int, str] = {}

apply_publication_style()
plt.rcParams["xtick.labelsize"] = 18


def _format_sci_latex(v: float) -> str:
    if np.isposinf(v):
        return r"\infty"
    if np.isneginf(v):
        return r"-\infty"
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    coeff = v / (10 ** exp)
    coeff_str = f"{coeff:.1f}".rstrip("0").rstrip(".")
    return rf"{coeff_str} \times 10^{{{exp}}}"


def _range_to_latex(label: str) -> str:
    text = label.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return text
    body = text[1:-1]
    parts = [p.strip() for p in body.split(",")]
    if len(parts) != 2:
        return text

    def parse_part(p: str) -> float | None:
        if p == "inf":
            return np.inf
        if p == "-inf":
            return -np.inf
        try:
            return float(p)
        except ValueError:
            return None

    left = parse_part(parts[0])
    right = parse_part(parts[1])
    if left is None or right is None:
        return text
    return rf"$[{_format_sci_latex(left)}, {_format_sci_latex(right)}]$"


def load_data(data_dir: Path) -> pd.DataFrame:
    p_bar = data_dir / OUT_BAR
    if not p_bar.is_file():
        raise FileNotFoundError(f"Missing backend data file: {p_bar}")
    return pd.read_parquet(p_bar)


def draw(ax: plt.Axes, data_dir: Path, *, panel_label: str | None = None) -> None:
    df = load_data(data_dir)
    order = (
        df[["range_id", "range_label"]]
        .drop_duplicates()
        .sort_values("range_id")
        .reset_index(drop=True)
    )
    x = np.arange(len(order))
    w = 0.30

    rl = (
        df[df["model"] == "rl_thinking_mix"][["range_id", Y_COLUMN]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(0)
    )
    rl_low = (
        df[df["model"] == "rl_thinking_mix"][["range_id", f"{Y_COLUMN}_ci_low"]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(np.nan)
    )
    rl_high = (
        df[df["model"] == "rl_thinking_mix"][["range_id", f"{Y_COLUMN}_ci_high"]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(np.nan)
    )
    baseline = (
        df[df["model"] == "rl_cte_thinking"][["range_id", Y_COLUMN]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(0)
    )
    baseline_low = (
        df[df["model"] == "rl_cte_thinking"][["range_id", f"{Y_COLUMN}_ci_low"]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(np.nan)
    )
    baseline_high = (
        df[df["model"] == "rl_cte_thinking"][["range_id", f"{Y_COLUMN}_ci_high"]]
        .set_index("range_id")
        .reindex(order["range_id"])
        .fillna(np.nan)
    )
    tick_labels: list[str] = []
    for range_id, range_label in zip(order["range_id"].tolist(), order["range_label"].tolist(), strict=True):
        if range_id in CUSTOM_XTICK_LABELS:
            tick_labels.append(CUSTOM_XTICK_LABELS[range_id])
            continue
        base = _range_to_latex(range_label)
        if SHOW_CATEGORY_INDEX:
            tick_labels.append(f"{CATEGORY_INDEX_PREFIX}{range_id}: {base}")
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
                "label": "CrysReas-ThermalExpansion",
            },
        ],
        xticks=x,
        xtick_labels=tick_labels,
        ylabel=Y_AXIS_LABEL,
        xlabel=X_AXIS_LABEL,
        title=PLOT_TITLE,
        ylim=(0.1, 0.8) if Y_COLUMN.startswith("ratio_") else None,
        rotation=20,
        ha="right",
    )
    if Y_COLUMN.startswith("ratio_"):
        ax.set_yticks(np.arange(0.1, 0.81, 0.1))
    if panel_label is not None:
        add_panel_label(ax, panel_label)


def run(data_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
    draw(ax, data_dir)
    fig.tight_layout()
    out_path = figure_dir / "conditioning_cte_reward_all_eq1_bar.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"conditioning cte frontend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conditioning CTE reward==1 bar comparison figure.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()
    run(data_dir=Path(args.data_dir), figure_dir=Path(args.figure_dir))


if __name__ == "__main__":
    main()
