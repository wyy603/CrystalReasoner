"""Manuscript figure 2, redrawn on the 1,024 x 10-seed protocol.

Same panels and same styling as `pipeline/d_fig_paper.py`, but the data come from
`data/out/` (the 1,024 prompts of assets/MP/split_generation.json, one sample
per seed) instead of the 8,141 x 2 de-novo run, and the error bars are standard
errors across the 10 seeds, matching the tables.
"""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_report as R  # noqa: E402

TAGS = ["base", "thinking", "rl", "full"]
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = ROOT / "papers" / "CrystalReasoner_paper" / "assets" / "complexity.pdf"


def load_out(tag: str) -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / f"t0/out.main.{tag}.parquet")
    b = lambda s: s.astype("boolean").fillna(False).to_numpy(dtype=bool)
    valid = b(d.structure_validity) & b(d.smact_validity)
    d = d.assign(sgc=np.where((d.sgc_supergroup == True).to_numpy() & valid, True, False),
                 seed=d.groupby("mp_id").cumcount().to_numpy())
    return d


def binned_seed(d, col, key, bins):
    """(mean over seeds, se across seeds, n distinct prompts) per bin."""
    out = []
    for lo, hi in bins:
        sub = d[(key >= lo) & (key <= hi)]
        if not len(sub):
            out.append((np.nan, np.nan, 0)); continue
        v = (sub[col] == True).astype(float) if sub[col].dtype != float else sub[col].astype(float)
        per = v.groupby(sub["seed"]).mean().to_numpy()
        out.append((per.mean(), per.std(ddof=1) / np.sqrt(len(per)), sub.mp_id.nunique()))
    return out


def main() -> None:
    R.apply_publication_style()
    plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 15,
                         "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "legend.fontsize": 14})
    data = {t: load_out(t) for t in TAGS}
    gt_atoms = pickle.load(open(DATA_DIR / "gt_atoms.pkl", "rb"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=220)

    ax = axes[0]
    for t in TAGS:
        d = data[t]
        key = d.mp_id.map(gt_atoms).to_numpy(dtype=float)
        pts = binned_seed(d, "structure_validity", key, R.ATOM_BINS)
        ax.errorbar(range(len(R.ATOM_LAB)), [100 * p[0] for p in pts],
                    yerr=[100 * p[1] for p in pts], marker="o", lw=2, capsize=2,
                    color=R.COLOR[t], label=R.LABEL[t])
        if t == "full":
            print("atom bins (prompts):", [(l, p[2]) for l, p in zip(R.ATOM_LAB, pts)])
    ax.set_xticks(range(len(R.ATOM_LAB)), R.ATOM_LAB)
    ax.set_ylabel("Structural Validity (%)")
    ax.grid(alpha=.25, ls="--")
    ax.set_title("(a)", loc="left", fontsize=17, fontweight="bold")
    ax.set_title("Number of Atoms", loc="center", fontsize=16, fontweight="bold")

    ax = axes[1]
    labs = [s[2] for s in R.SYS]; x = np.arange(len(labs)); w = 0.8 / len(TAGS)
    for k, t in enumerate(TAGS):
        d = data[t]
        pts = binned_seed(d, "sgc", d.req_spg.to_numpy(dtype=float),
                          [(a, b) for a, b, _ in R.SYS])
        off = (k - (len(TAGS) - 1) / 2) * w
        ax.bar(x + off, [100 * q[0] for q in pts], width=w, color=R.COLOR[t],
               label=R.LABEL[t], zorder=2)
        ax.errorbar(x + off, [100 * q[0] for q in pts], yerr=[100 * q[1] for q in pts],
                    fmt="none", ecolor="0.25", lw=0.8, capsize=1.2, zorder=3)
        if t == "full":
            print("crystal systems (prompts):", [(l, q[2]) for l, q in zip(labs, pts)])
    ax.set_ylim(0, 92)
    ax.set_xticks(x, labs, rotation=25, ha="right")
    ax.set_ylabel("Space Group Consistency (%)")
    ax.grid(axis="y", alpha=.25, ls="--"); ax.set_axisbelow(True)
    ax.set_title("(b)", loc="left", fontsize=17, fontweight="bold")
    ax.set_title("Crystal System", loc="center", fontsize=16, fontweight="bold")

    fig.tight_layout(w_pad=2.5)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="center left", ncol=1, frameon=False,
               bbox_to_anchor=(1.005, 0.5), labelspacing=0.85, handlelength=1.8)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print("wrote", OUT)
    d = data["full"]
    one = d.drop_duplicates("mp_id")
    print("distinct requested space groups:", one.req_spg.nunique())


if __name__ == "__main__":
    main()
