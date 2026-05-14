"""
Build the combined E_hull asset from DFT_new graph sources.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crysreas.experiment.DFT_new.graph1 import draw as draw_graph1
from crysreas.experiment.DFT_new.graph1 import load_e_above_hull_values
from crysreas.experiment.DFT_new.graph2 import draw as draw_graph2
from crysreas.experiment.DFT_new.graph2 import load_e_above_hull_by_mp_id
from crysreas.experiment.assets.plot_helpers import apply_publication_style, ensure_dir

apply_publication_style()

OUTPUT_PATH = Path(__file__).resolve().parent / "e_hull.pdf"


def run(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(28, 7), dpi=400)
    draw_graph1(axes[0], load_e_above_hull_values(), panel_label="(a)")
    draw_graph2(axes[1:], load_e_above_hull_by_mp_id())
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved combined plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined E_hull asset.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()
    run(Path(args.output))


if __name__ == "__main__":
    main()
