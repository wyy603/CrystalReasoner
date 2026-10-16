"""Thinking-trace segment shuffle ablation: tables + figure."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(ROOT))
from crysreas.experiment.assets.plot_helpers import (  # noqa: E402
    apply_publication_style, wilson_ci_95)
from crysreas.experiment.make_graph import compute  # noqa: E402

FIG = DATA_DIR / "figures"
TAB = DATA_DIR / "tables"
apply_publication_style()
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 19, "axes.titlesize": 21, "axes.labelsize": 20,
                     "xtick.labelsize": 15, "ytick.labelsize": 16, "legend.fontsize": 15})


def panel(ax, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=24, fontweight="bold", clip_on=False)

SEG = {"baseline": "Baseline\n(intact)",
       "shuffle_part1": "part 1\nsymmetry",
       "shuffle_part2": "part 2\ncoordination",
       "shuffle_part3": "part 3\nvalidity",
       "shuffle_part4": "part 4\nproperties",
       "shuffle_all": "all four"}
METRICS = [("structure_validity", "Structural Validity"),
           ("composition_consistency", "Composition Consistency"),
           ("spacegroup_consistency", "Space Group Consistency")]
COLORS = ["#54a24b", "#4c78a8", "#f58518", "#e45756", "#b279a2", "#9d755d"]


def main() -> None:
    src = DATA_DIR / "shuffle/shuffle_ablation.parquet"
    if not src.exists():
        print(f"missing {src}"); return
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(src)
    rows = []
    for v in SEG:
        s = d[d.variant == v]
        if not len(s):
            continue
        r = dict(variant=v, label=SEG[v].replace("\n", " "), n=len(s),
                 tok_orig=float(s.original_section_token_count.mean()),
                 tok_new=float(s.replacement_section_token_count.mean()))
        for m, _ in METRICS:
            r[m] = compute(s, m)
            r[m + "_v"] = float(r[m].split("±")[0].strip().rstrip("%"))
        rows.append(r)
    t = pd.DataFrame(rows)
    base = t[t.variant == "baseline"].iloc[0]
    for m, _ in METRICS:
        t[m + "_drop"] = base[m + "_v"] - t[m + "_v"]
    t.to_parquet(TAB / "t1_shuffle.parquet", index=False)
    print("\n=== Thinking-trace segment shuffle (drop vs baseline, pp) ===")
    print(t[["label", "n", "tok_orig", "tok_new"] +
            [m for m, _ in METRICS] +
            [m + "_drop" for m, _ in METRICS]].to_string(
                index=False, float_format=lambda v: f"{v:.2f}"))

    fig, axes = plt.subplots(1, 3, figsize=(30, 7), dpi=220)
    for ax, (m, title) in zip(axes, METRICS):
        y = t[m + "_v"].to_numpy()
        ci = [wilson_ci_95(v/100*n, n) for v, n in zip(y, t["n"])]
        x = np.arange(len(t))
        ax.bar(x, y, width=.62, color=COLORS[:len(t)], zorder=2)
        ax.errorbar(x, y, yerr=[[v-c["ci_low"]*100 for v, c in zip(y, ci)],
                                [c["ci_high"]*100-v for v, c in zip(y, ci)]],
                    fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=3)
        ax.axhline(base[m + "_v"], color="#54a24b", ls="--", lw=2, zorder=1)
        ax.set_xticks(x, [SEG[v] for v in t["variant"]], rotation=0)
        ax.set_ylabel(f"{title} (%)"); ax.set_title(f"{title}")
        ax.grid(axis="y", alpha=.25, ls="--")
        ax.tick_params(axis="x", labelsize=15)
    fig.suptitle("Shuffling one trace segment across examples "
                 "(format and length preserved, information destroyed)",
                 fontsize=26, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(FIG / "fig_t1_shuffle.pdf"); fig.savefig(FIG / "fig_t1_shuffle.png", dpi=100); plt.close(fig)
    print(f"wrote {FIG/'fig_t1_shuffle.pdf'}")


if __name__ == "__main__":
    main()
