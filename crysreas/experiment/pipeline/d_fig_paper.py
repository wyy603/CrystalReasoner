"""Manuscript-sized version of the complexity figure, four variants.

Same data and same panels as `the experiment data directory, figures/fig_d_complexity.pdf`, drawn at a
width close to the one it is printed at, so the tick labels survive the scaling
the LaTeX \\includegraphics does.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pickle  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import d_report as R  # noqa: E402

# Figure 2 of the paper compares the four variants of Table 2, in the same order;
# the prior works are in Table 1 and do not need a second appearance.
TAGS = ["base", "thinking", "rl", "full"]
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = ROOT / "papers" / "CrystalReasoner_paper" / "assets" / "complexity.pdf"


def main() -> None:
    R.apply_publication_style()
    plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 15, "axes.labelsize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 14})
    data = {t: R.load(t) for t in TAGS}
    gt_atoms = pickle.load(open(DATA_DIR / "gt_atoms.pkl", "rb"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=220)

    ax = axes[0]
    for t in TAGS:
        d = data[t]
        key = d.mp_id.map(gt_atoms).to_numpy(dtype=float)
        pts = R.binned(d, "structure_validity", key, R.ATOM_BINS)
        ax.errorbar(range(len(R.ATOM_LAB)), [100 * p[0] for p in pts],
                    yerr=[100 * p[1] for p in pts], marker="o", lw=2, capsize=2,
                    color=R.COLOR[t], label=R.LABEL[t])
    ax.set_xticks(range(len(R.ATOM_LAB)), R.ATOM_LAB)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Structural Validity (%)")
    ax.grid(alpha=.25, ls="--")
    ax.set_title("(a)", loc="left", fontsize=17, fontweight="bold")

    ax = axes[1]
    # crystal systems are categories with no order to interpolate along, so bars
    labs = [s[2] for s in R.SYS]
    x = np.arange(len(labs))
    w = 0.8 / len(TAGS)
    for k, t in enumerate(TAGS):
        d = data[t]
        pts = R.binned(d, "sgc", d.req_spg.to_numpy(dtype=float),
                       [(a, b) for a, b, _ in R.SYS])
        off = (k - (len(TAGS) - 1) / 2) * w
        y = [100 * q[0] for q in pts]
        ax.bar(x + off, y, width=w, color=R.COLOR[t], label=R.LABEL[t], zorder=2)
        ax.errorbar(x + off, y, yerr=[100 * q[1] for q in pts], fmt="none",
                    ecolor="0.25", lw=0.8, capsize=1.2, zorder=3)
    ax.set_ylim(0, 92)
    ax.set_xticks(x, labs, rotation=25, ha="right")
    ax.set_xlabel("Crystal System")
    ax.set_ylabel("Space Group Consistency (%)")
    ax.grid(axis="y", alpha=.25, ls="--")
    ax.set_axisbelow(True)
    ax.set_title("(b)", loc="left", fontsize=17, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    h, lab = ax.get_legend_handles_labels()
    # bbox_inches="tight" below grows the saved canvas to hold the legend, so it
    # can sit under the panels without eating into them
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.16), columnspacing=1.4, handlelength=1.8)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
