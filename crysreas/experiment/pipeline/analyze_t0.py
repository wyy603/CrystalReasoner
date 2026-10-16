"""Tables + figures for T0-a (space group), T0-b (precision), T0-e (trace length).

Aggregation uses the repo's own `crysreas.experiment.make_graph.compute`, so the
means and error bars are produced exactly as the paper's tables were.
Figure style comes from `crysreas.experiment.assets.plot_helpers`.
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
    apply_publication_style, wilson_ci_95,
)
from crysreas.experiment.make_graph import compute  # noqa: E402

T0 = DATA_DIR / "t0"
FIG = DATA_DIR / "figures"
TAB = DATA_DIR / "tables"
apply_publication_style()
# The repo style is sized for figures shrunk into the paper; these report figures
# are viewed at 1x, so tame the largest sizes and keep everything else identical.
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"],  # Arial absent on this host
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
BLUE, ORANGE, GREEN = "#4c78a8", "#f58518", "#54a24b"


def pct(s: str | float) -> float:
    if isinstance(s, str):
        return float(s.split("±")[0].strip().rstrip("%"))
    return float(s)


def gate(d: pd.DataFrame) -> pd.DataFrame:
    """Apply the repo's own gate: `spacegroup_consistency` is None unless the
    structure passed structural AND chemical validity. Without this the strict
    column does not reproduce the paper's number (Base: 43.4% ungated vs the
    published 39.0% gated), so strict and supergroup must both be gated for the
    comparison to be apples-to-apples."""
    ok = (d["structure_validity"].astype("boolean").fillna(False)
          & d["smact_validity"].astype("boolean").fillna(False))
    for c in ("sgc_strict", "sgc_supergroup"):
        d[c + "_g"] = d[c].where(ok, other=None)
    return d


def load(tag: str) -> pd.DataFrame | None:
    p = T0 / f"out.main.{tag}.parquet"
    return gate(pd.read_parquet(p)) if p.exists() else None


# ---------------------------------------------------------------- T0-a
def t0a() -> pd.DataFrame:
    rows = []
    for tag in ORDER:
        d = load(tag)
        if d is None:
            continue
        strict, sup = compute(d, "sgc_strict_g"), compute(d, "sgc_supergroup_g")
        rows.append(dict(model=LABEL[tag], tag=tag, n=len(d),
                         strict=strict, supergroup=sup,
                         strict_v=pct(strict), sup_v=pct(sup),
                         delta=pct(sup) - pct(strict)))
    t = pd.DataFrame(rows)
    t.to_parquet(TAB / "t0a_spacegroup.parquet", index=False)
    print("\n=== T0-a: space-group consistency, strict vs supergroup-aware ===")
    print(t[["model", "n", "strict", "supergroup", "delta"]].to_string(index=False))
    return t


def t0a_per_group() -> pd.DataFrame:
    rows = []
    for p in sorted(T0.glob("out.sg.*.parquet")):
        _, _, grp, who = p.stem.split(".", 3)
        d = gate(pd.read_parquet(p))
        rows.append(dict(group=grp.replace("_", "/"), who=who, n=len(d),
                         strict_v=pct(compute(d, "sgc_strict_g")),
                         sup_v=pct(compute(d, "sgc_supergroup_g"))))
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).sort_values(["group", "who"])
    t.to_parquet(TAB / "t0a_spacegroup_per_group.parquet", index=False)
    print("\n=== T0-a: per requested space group ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return t


def fig_t0a(t: pd.DataFrame, pg: pd.DataFrame) -> None:
    ncol = 2 if len(pg) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(11 * ncol, 6.5), dpi=220)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    x = np.arange(len(t)); w = 0.38
    for off, col, c, lab in ((-w/2, "strict_v", BLUE, "Strict (paper metric)"),
                             (w/2, "sup_v", ORANGE, "Supergroup-aware")):
        y = t[col].to_numpy()
        ci = [wilson_ci_95(v/100*n, n) for v, n in zip(y, t["n"])]
        ax.bar(x+off, y, width=w, color=c, label=lab, zorder=2)
        ax.errorbar(x+off, y, yerr=[[v-c_["ci_low"]*100 for v, c_ in zip(y, ci)],
                                    [c_["ci_high"]*100-v for v, c_ in zip(y, ci)]],
                    fmt="none", ecolor="black", capsize=3, lw=1.2, zorder=3)
    ax.set_xticks(x, [LABEL[k] for k in t["tag"]], rotation=28, ha="right")
    ax.set_ylabel("Space Group Consistency (%)")
    ax.set_title("Exact match vs. supergroup-aware")
    ax.grid(axis="y", alpha=.25, ls="--"); ax.legend(loc="upper right")
    panel(ax, "(a)")

    if len(pg):
        ax = axes[1]
        piv = pg[pg.who == "full"].set_index("group")[["strict_v", "sup_v"]]
        x = np.arange(len(piv)); w = 0.38
        ax.bar(x-w/2, piv.strict_v, width=w, color=BLUE, label="Strict", zorder=2)
        ax.bar(x+w/2, piv.sup_v, width=w, color=ORANGE, label="Supergroup-aware", zorder=2)
        ax.set_xticks(x, piv.index.tolist(), rotation=28, ha="right")
        ax.set_ylabel("Space Group Consistency (%)")
        ax.set_title("CrysReas, by requested space group")
        ax.grid(axis="y", alpha=.25, ls="--"); ax.legend(loc="upper left")
        panel(ax, "(b)")
    fig.tight_layout(); fig.savefig(FIG / "fig_t0a_spacegroup.pdf"); fig.savefig(FIG / "fig_t0a_spacegroup.png", dpi=110); plt.close(fig)
    print(f"wrote {FIG/'fig_t0a_spacegroup.pdf'}")


# ---------------------------------------------------------------- T0-b
def t0b() -> pd.DataFrame:
    rows = []
    for tag in ORDER:
        d = load(tag)
        if d is None:
            continue
        m = d[d.relax_rmsd.notna()]
        r = dict(model=LABEL[tag], tag=tag, n_relaxed=len(m),
                 rmsd_mean=float(m.relax_rmsd.mean()),
                 rmsd_median=float(m.relax_rmsd.median()),
                 maxdisp_median=float(m.relax_maxdisp.median()),
                 dvol_median=float(m.dvol_frac.median()),
                 dabc_median=float(m.dabc_frac.median()))
        for k in ("1e2", "1e3", "1e4"):
            r[f"sym_{k}"] = pct(compute(d, f"sym_survives_{k}"))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_parquet(TAB / "t0b_precision.parquet", index=False)
    print("\n=== T0-b: precision (relaxation displacement + symmetry survival) ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return t


def t0b_paired() -> pd.DataFrame:
    """Same comparison restricted to prompts every model relaxed successfully.

    Unpaired, the RMSD column is selection-biased: Base relaxes 75.4% of rows and
    CrysReas 85.8%, so the two averages are taken over different, differently hard
    structures. Pairing removes that."""
    ds = {tag: load(tag) for tag in ORDER}
    ds = {k: v for k, v in ds.items() if v is not None}
    if len(ds) < 2:
        return pd.DataFrame()
    ok = None
    for d in ds.values():
        m = d.relax_rmsd.notna().to_numpy()
        ok = m if ok is None else (ok & m)
    rows = []
    for tag, d in ds.items():
        sub = d[ok]
        r = dict(model=LABEL[tag], tag=tag, n_common=int(ok.sum()),
                 rmsd_median=float(sub.relax_rmsd.median()),
                 dvol_median=float(sub.dvol_frac.median()))
        for k in ("1e2", "1e3", "1e4"):
            r[f"sym_{k}"] = pct(compute(sub, f"sym_survives_{k}"))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_parquet(TAB / "t0b_precision_paired.parquet", index=False)
    print("\n=== T0-b paired: only prompts every model relaxed ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return t


def fig_t0b(t: pd.DataFrame) -> None:
    """Both panels use the PAIRED subset (prompts every model relaxed), so the
    models are compared on identical structures."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 6.5), dpi=220)
    ax = axes[0]
    x = np.arange(len(t))
    ax.bar(x, t.rmsd_median, width=.6, color=BLUE, zorder=2)
    ax.set_xticks(x, [LABEL[k] for k in t["tag"]], rotation=28, ha="right")
    ax.set_ylabel("Median RMSD to relaxed (Å)")
    ax.set_title(f"Relaxation displacement (paired, n={int(t.n_common.iloc[0])})")
    ax.grid(axis="y", alpha=.25, ls="--")
    panel(ax, "(a)")

    ax = axes[1]
    sym = [0.1, 0.01, 0.001, 0.0001]
    for tag, c in zip(("base", "rl", "thinking", "full", "crystaltextllm"),
                      (ORANGE, GREEN, BLUE, "#e45756", "#b279a2")):
        r = t[t.tag == tag]
        if not len(r):
            continue
        y = [100.0] + [float(r[f"sym_{k}"].iloc[0]) for k in ("1e2", "1e3", "1e4")]
        ax.plot(range(len(sym)), y, marker="o", color=c, lw=2.5, label=LABEL[tag])
    ax.set_xticks(range(len(sym)), [f"{s:g}" for s in sym])
    ax.set_xlabel("symprec (Å)"); ax.set_ylabel("Symmetry retained (%)")
    ax.set_title("Symmetry vs. tolerance")
    ax.grid(alpha=.25, ls="--"); ax.legend(loc="lower left")
    panel(ax, "(b)")
    fig.tight_layout(); fig.savefig(FIG / "fig_t0b_precision.pdf"); fig.savefig(FIG / "fig_t0b_precision.png", dpi=110); plt.close(fig)
    print(f"wrote {FIG/'fig_t0b_precision.pdf'}")


# ---------------------------------------------------------------- T0-e
def fig_t0e() -> None:
    sl = T0 / "t0e_slopes.parquet"
    if not sl.exists():
        return
    slopes = pd.read_parquet(sl)
    gt = pd.read_parquet(T0 / "t0e_trace_length_gt.parquet")
    gen = pd.read_parquet(T0 / "t0e_trace_length_per_row.parquet")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=220)

    def curve(df, xcol, ycol, color, label):
        d = df[df[xcol].between(1, 21)]
        g = d.groupby(d[xcol].astype(int))[ycol].agg(["mean", "sem", "count"])
        g = g[g["count"] >= 5]
        ax.errorbar(g.index, g["mean"], yerr=1.96*g["sem"].fillna(0), marker="o",
                    color=color, lw=2.5, capsize=3, label=label)

    s_gt = slopes[slopes.series.str.startswith("SFT")].iloc[0]
    curve(gt, "gt_atoms", "gt_tokens", GREEN,
          f"SFT corpus (template)  slope={s_gt.slope:.1f}")
    for tag, c in (("thinking", BLUE), ("full", "#e45756")):
        g = gen[gen.model == tag]
        if not len(g):
            continue
        s = slopes[(slopes.series.str.startswith(tag)) & (slopes.x == "gt_atoms")]
        lab = f"{LABEL[tag]} (generated)  slope={s.slope.iloc[0]:.1f}" if len(s) else LABEL[tag]
        curve(g, "gt_atoms", "tokens", c, lab)
    ax.set_xlabel("Number of Atoms (ground truth)")
    ax.set_ylabel("Thinking Trace Tokens")
    ax.set_title("Trace length: model vs. its template")
    ax.grid(alpha=.25, ls="--"); ax.legend(loc="upper left", fontsize=17)
    fig.tight_layout(); fig.savefig(FIG / "fig_t0e_tracelen.pdf"); fig.savefig(FIG / "fig_t0e_tracelen.png", dpi=110); plt.close(fig)
    print(f"wrote {FIG/'fig_t0e_tracelen.pdf'}")
    print("\n=== T0-e: slopes (tokens per atom) ===")
    print(slopes.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    t = t0a(); pg = t0a_per_group(); fig_t0a(t, pg)
    t0b(); tp = t0b_paired(); fig_t0b(tp)
    fig_t0e()


if __name__ == "__main__":
    main()
