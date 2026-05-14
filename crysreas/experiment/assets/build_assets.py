"""
Rebuild combined vector-PDF assets directly from source tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from crysreas.experiment.ablation_thinking.spacegroup_bars_frontend import (
    DATA_DIR as ABLATION_DATA_DIR,
    draw as draw_ablation_spacegroup,
)
from crysreas.experiment.assets.plot_helpers import add_panel_label, apply_publication_style, ensure_dir
from crysreas.experiment.conditioning.cte_frontend import DATA_DIR as CTE_DATA_DIR, draw as draw_cte
from crysreas.experiment.conditioning.elastic_frontend import DATA_DIR as ELASTIC_DATA_DIR, draw as draw_elastic
from crysreas.experiment.conditioning.spacegroup_frontend import (
    DATA_DIR as CONDITIONING_SG_DATA_DIR,
    draw as draw_conditioning_spacegroup,
)
from crysreas.experiment.thinking.common import ComparePaths
from crysreas.experiment.thinking.part_1_2_frontend import draw as draw_tokens_curve
from crysreas.experiment.thinking.part_1_5_frontend import (
    draw_metric_curve,
    draw_spacegroup_bar,
    load_data as load_metrics_data,
)

apply_publication_style()

ASSET_DIR = Path(__file__).resolve().parent


def _save_metrics(asset_dir: Path, thinking_paths: ComparePaths) -> None:
    sg_bar, metric_curve = load_metrics_data(thinking_paths)
    fig, axes = plt.subplots(1, 3, figsize=(28, 7), dpi=400)
    draw_metric_curve(axes[0], metric_curve, "structure_validity", panel_label="(a)")
    draw_metric_curve(axes[1], metric_curve, "composition_consistency", panel_label="(b)")
    draw_spacegroup_bar(axes[2], sg_bar, panel_label="(c)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(asset_dir / "metrics.pdf")
    plt.close(fig)


def _save_thinking_trace(asset_dir: Path, thinking_paths: ComparePaths) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=400)
    draw_tokens_curve(axes[0], thinking_paths)
    add_panel_label(axes[0], "(a)")
    draw_ablation_spacegroup(axes[1], data_dir=ABLATION_DATA_DIR, panel_label="(b)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(asset_dir / "thinking_trace.pdf")
    plt.close(fig)


def _save_conditioning(asset_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(30, 7), dpi=400)
    draw_conditioning_spacegroup(axes[0], CONDITIONING_SG_DATA_DIR, panel_label="(a)")
    draw_elastic(axes[1], ELASTIC_DATA_DIR, panel_label="(b)")
    draw_cte(axes[2], CTE_DATA_DIR, panel_label="(c)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.15)
    fig.savefig(asset_dir / "conditioning.pdf")
    plt.close(fig)


def run(asset_dir: Path, thinking_data_dir: Path, thinking_figure_dir: Path) -> None:
    ensure_dir(asset_dir)
    thinking_paths = ComparePaths(data_dir=thinking_data_dir, figure_dir=thinking_figure_dir)
    _save_metrics(asset_dir, thinking_paths)
    _save_thinking_trace(asset_dir, thinking_paths)
    _save_conditioning(asset_dir)
    print(f"asset builder wrote metrics.pdf, thinking_trace.pdf, conditioning.pdf under {asset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined experiment assets.")
    parser.add_argument("--asset-dir", type=str, default=str(ASSET_DIR))
    parser.add_argument("--thinking-data-dir", type=str, default=str(ComparePaths().data_dir))
    parser.add_argument("--thinking-figure-dir", type=str, default=str(ComparePaths().figure_dir))
    args = parser.parse_args()
    run(
        asset_dir=Path(args.asset_dir),
        thinking_data_dir=Path(args.thinking_data_dir),
        thinking_figure_dir=Path(args.thinking_figure_dir),
    )


if __name__ == "__main__":
    main()
