"""
Frontend for EXPERIMENT §1.2: read backend tables and render figures only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crysreas.experiment.assets.plot_helpers import (
    aggregate_mean_curve,
    apply_publication_style,
    plot_curve_with_ci,
)

from .common import ComparePaths, ensure_thinking_dirs
from .part_1_2_backend import DATA_1_2_PER_ROW

apply_publication_style()


def load_curve(paths: ComparePaths) -> pd.DataFrame:
    src = paths.data_dir / DATA_1_2_PER_ROW
    if not src.is_file():
        raise FileNotFoundError(f"Missing backend data {src}; run part_1_2_backend first.")
    per_row = pd.read_parquet(src)
    mask = per_row["n_atoms"].notna() & per_row["thinking_tokens"].notna()
    mask &= per_row["n_atoms"].between(1, 21)
    per_row = per_row.loc[mask].copy()
    per_row["n_atoms"] = per_row["n_atoms"].astype(int)
    curve = aggregate_mean_curve(per_row, group_cols=["n_atoms"], value_col="thinking_tokens")
    curve["condition"] = "thinking"
    return curve.sort_values("n_atoms").reset_index(drop=True)


def draw(ax: plt.Axes, paths: ComparePaths) -> None:
    curve = load_curve(paths)
    plot_curve_with_ci(
        ax,
        df=curve,
        group_col="condition",
        groups=["thinking"],
        x_col="n_atoms",
        y_col="thinking_tokens",
        xlabel="Number of Atoms in Structures",
        ylabel="Number of Tokens in Thinking Trace",
        title="Thinking Trace Token Count vs Atom Count",
        xticks=np.arange(1, 22, step=4),
        legend_loc="upper left",
    )


def run(paths: ComparePaths) -> None:
    ensure_thinking_dirs(figure_dir=paths.figure_dir)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=220)
    draw(ax, paths)
    fig.tight_layout()
    out_pdf = paths.figure_dir / "thinking_exp_1_2_tokens_vs_atoms_curve.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"1.2 frontend wrote {out_pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.2 frontend (figures only).")
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
