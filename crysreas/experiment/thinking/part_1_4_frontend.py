"""
Frontend for EXPERIMENT §1.4: read backend table and render CIF information figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ComparePaths, ensure_thinking_dirs
from .part_1_4_backend import DATA_1_4_CURVE

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]

# 2. 全局放大字体
plt.rcParams['font.size'] = 22          # 基础大小
plt.rcParams['axes.titlesize'] = 22     # 标题大小
plt.rcParams['axes.labelsize'] = 22     # 轴标签大小
plt.rcParams['xtick.labelsize'] = 22    # 坐标轴数字大小
plt.rcParams['legend.fontsize'] = 22    # 图例大小
plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗

MODEL_STYLE = {
    "thinking": {"label": "Thinking", "color": "#4c78a8", "marker": "o"},
    "no_thinking": {"label": "No Thinking", "color": "#f58518", "marker": "s"},
}


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(figure_dir=paths.figure_dir)
    src = paths.data_dir / DATA_1_4_CURVE
    if not src.is_file():
        raise FileNotFoundError(f"Missing backend data {src}; run part_1_4_backend first.")
    curve = pd.read_parquet(src)
    curve = curve[curve["n_atoms"].between(1, 21)].copy()

    plt.figure(figsize=(10, 6), dpi=220)
    for model_name in ("thinking", "no_thinking"):
        g = curve[curve["model"] == model_name].sort_values("n_atoms")
        if len(g) == 0:
            continue
        style = MODEL_STYLE[model_name]
        plt.plot(
            g["n_atoms"].to_numpy(),
            g["cif_info_avg"].to_numpy(),
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4.5,
            label=style["label"],
        )

    plt.xlabel("Number of Atoms")
    plt.ylabel("Average CIF Information (nats/token)")
    plt.title("CIF Information vs Number of Atoms")
    plt.xticks(np.arange(1, 22, step=4))
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(loc="best")
    plt.tight_layout()
    out_pdf = paths.figure_dir / "thinking_exp_1_4_cif_information_vs_atoms.pdf"
    plt.savefig(out_pdf)
    plt.close()
    print(f"1.4 frontend wrote {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.4 frontend (figures only).")
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
