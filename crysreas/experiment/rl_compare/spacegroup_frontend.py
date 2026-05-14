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

from .spacegroup_backend import DATA_DIR, OUT_BAR

FIGURE_DIR = Path(__file__).resolve().parent / "figure"

# -------- Editable plotting config (frontend only) --------
X_AXIS_LABEL = "Spacegroup Instruction"
Y_AXIS_LABEL = "Ratio of Spacegroup Corrections"
PLOT_TITLE = "Spacegroup Consistency by Instruction Group"
Y_COLUMN = "ratio_claimed_eq_structure"  # keep ratio metric by default
SHOW_CATEGORY_INDEX = False
CATEGORY_INDEX_PREFIX = "G"
CUSTOM_XTICK_LABELS: dict[str, str] = {}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 22
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def run(data_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    p_bar = data_dir / OUT_BAR
    if not p_bar.is_file():
        raise FileNotFoundError(f"Missing backend data file: {p_bar}")
    df = pd.read_parquet(p_bar)

    rl = df[df["model"] == "rl_thinking_mix"].set_index("spacegroup_symbol")[Y_COLUMN].copy()
    baseline = df[df["model"] == "thinking"].set_index("spacegroup_symbol")[Y_COLUMN].copy()
    symbols = df["spacegroup_symbol"].drop_duplicates().tolist()
    rl_vals = np.array([rl.get(sym, np.nan) for sym in symbols], dtype=float)
    base_vals = np.array([baseline.get(sym, np.nan) for sym in symbols], dtype=float)

    x = np.arange(len(symbols))
    w = 0.35
    plt.figure(figsize=(13, 6), dpi=220)
    plt.bar(x - w / 2, base_vals, width=w, label="CrysReas-Thinking")
    plt.bar(x + w / 2, rl_vals, width=w, label="CrysReas")
    tick_labels: list[str] = []
    for i, sym in enumerate(symbols, start=1):
        if sym in CUSTOM_XTICK_LABELS:
            tick_labels.append(CUSTOM_XTICK_LABELS[sym])
            continue
        if SHOW_CATEGORY_INDEX:
            tick_labels.append(f"{CATEGORY_INDEX_PREFIX}{i}: {sym}")
        else:
            tick_labels.append(sym)
    plt.xticks(x, tick_labels)
    plt.ylabel(Y_AXIS_LABEL)
    plt.xlabel(X_AXIS_LABEL)
    plt.title(PLOT_TITLE)
    if Y_COLUMN.startswith("ratio_"):
        plt.ylim(0, 1.02)
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    out_path = figure_dir / "conditioning_spacegroup_bar_compare.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"conditioning frontend wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render conditioning spacegroup bar comparison figure.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()
    run(data_dir=Path(args.data_dir), figure_dir=Path(args.figure_dir))


if __name__ == "__main__":
    main()
