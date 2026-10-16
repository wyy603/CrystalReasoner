"""S.U.N. under both novelty criteria, and under both sampling protocols.

Protocols
  out     1024 prompts x 10 samples   (the rerun of the paper's design)
  denovo  8141 prompts x  2 samples   (every distinct test formula)

Novelty criteria
  novel_mp     absent from MP+Alexandria (the paper's definition)
  novel_train  absent from the SFT training split

Uniqueness is set-size sensitive (the paper's own appendix says so), so the two
protocols are NOT directly comparable on uniqueness; they are reported side by
side to show exactly that.
"""
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

SUN = DATA_DIR / "sun"
FIG = DATA_DIR / "figures"
TAB = DATA_DIR / "tables"
apply_publication_style()
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 19, "axes.titlesize": 21, "axes.labelsize": 20,
                     "xtick.labelsize": 15, "ytick.labelsize": 16, "legend.fontsize": 15})


def panel(ax, label: str) -> None:
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=24, fontweight="bold", clip_on=False)
ORDER = ["base", "thinking", "rl", "full", "validity_only", "energy_only",
         "crystaltextllm", "plaid_wyckoff"]
LABEL = {"base": "Base", "thinking": "Thinking", "rl": "RL", "full": "CrysReas",
         "validity_only": "Validity Only", "energy_only": "Energy Only",
         "crystaltextllm": "CrystalTextLLM", "plaid_wyckoff": "PLAID++ Wyckoff"}
BLUE, ORANGE = "#4c78a8", "#f58518"
COLS = ["is_stable", "is_unique", "is_novel_mp", "is_novel_train", "sun_mp", "sun_train"]


def table(proto: str) -> pd.DataFrame:
    rows = []
    for tag in ORDER:
        p = SUN / f"{proto}.main.{tag}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        r = dict(protocol=proto, model=LABEL[tag], tag=tag, n=len(d))
        for c in COLS:
            r[c] = compute(d, c)
            r[c + "_v"] = float(r[c].split("±")[0].strip().rstrip("%"))
        r["n_sun_mp"] = int(d.sun_mp.sum())
        r["n_sun_train"] = int(d.sun_train.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def fig(dn: pd.DataFrame, ex: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 6.5), dpi=220)
    ax = axes[0]
    x = np.arange(len(dn)); w = .38
    for off, col, c, lab in ((-w/2, "sun_mp_v", BLUE, "S.U.N. (novel vs MP+Alexandria)"),
                             (w/2, "sun_train_v", ORANGE, "S.U.N. (novel vs training split)")):
        y = dn[col].to_numpy()
        ci = [wilson_ci_95(v/100*n, n) for v, n in zip(y, dn["n"])]
        ax.bar(x+off, y, width=w, color=c, label=lab, zorder=2)
        ax.errorbar(x+off, y,
                    yerr=[[v-c_["ci_low"]*100 for v, c_ in zip(y, ci)],
                          [c_["ci_high"]*100-v for v, c_ in zip(y, ci)]],
                    fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=3)
    ax.set_xticks(x, [LABEL[k] for k in dn["tag"]], rotation=28, ha="right")
    ax.set_ylabel("S.U.N. (%)"); ax.set_yscale("log")
    ax.set_title("De-novo protocol (8141 x 2)")
    ax.grid(axis="y", alpha=.25, ls="--"); ax.legend(loc="upper right", fontsize=16)
    panel(ax, "(a)")

    ax = axes[1]
    both = ex.merge(dn, on="tag", suffixes=("_ex", "_dn"))
    x = np.arange(len(both)); w = .38
    ax.bar(x-w/2, both.is_unique_v_ex, width=w, color=BLUE,
           label="1024 x 10 (paper design)", zorder=2)
    ax.bar(x+w/2, both.is_unique_v_dn, width=w, color=ORANGE,
           label="8141 x 2 (de novo)", zorder=2)
    ax.set_xticks(x, [LABEL[k] for k in both["tag"]], rotation=28, ha="right")
    ax.set_ylabel("Uniqueness (%)")
    ax.set_title("Uniqueness depends on samples/prompt")
    ax.grid(axis="y", alpha=.25, ls="--"); ax.legend(loc="upper left", fontsize=16)
    panel(ax, "(b)")
    fig.tight_layout(); fig.savefig(FIG / "fig_t1_sun.pdf"); fig.savefig(FIG / "fig_t1_sun.png", dpi=110); plt.close(fig)
    print(f"wrote {FIG/'fig_t1_sun.pdf'}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    ex, dn = table("out"), table("denovo")
    show = ["model", "n", "is_stable", "is_unique", "is_novel_mp", "is_novel_train",
            "sun_mp", "n_sun_mp", "sun_train", "n_sun_train"]
    for name, t in (("1024 x 10 (paper design)", ex), ("8141 x 2 (de novo)", dn)):
        if len(t):
            print(f"\n=== S.U.N. -- {name} ===")
            print(t[show].to_string(index=False))
    out = pd.concat([ex, dn], ignore_index=True)
    out.to_parquet(TAB / "t1_sun.parquet", index=False)
    if len(dn) and len(ex):
        fig(dn, ex)


if __name__ == "__main__":
    main()
