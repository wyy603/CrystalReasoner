"""Tables and figures for report.md.

One protocol for everything: the 8,141 x 2 de-novo generations
(`data/denovo/main.<tag>.parquet`), their per-row novelty flags
(`data/sun/denovo.main.<tag>.parquet`) and the structure pass that adds the
subgroup-aware space-group columns and the relaxation displacements
(`data/t0/denovo.main.<tag>.parquet`).

Two definitions are fixed here:

  S.U.N.    stable (E_hull < 0.1 eV/atom) AND unique AND absent from the SFT
            training split.
  S.S.U.N.  the same three factors AND space-group compliant.

Space-group compliance means the generated structure carries at least the
requested symmetry: the detected group equals the requested one, or the
requested one is a proper subgroup of it. The same criterion is used wherever
space-group consistency is reported. It is gated on structural and chemical
validity exactly as the repo metric is, so the two readings are comparable.

Error bars are the standard error of the mean over the 8,141 per-prompt means.
The repo's own estimator keeps only within-prompt variance, which is close to
zero at two samples per prompt and would understate the uncertainty here.
"""
from __future__ import annotations

import pickle
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

from crysreas.experiment.assets.plot_helpers import apply_publication_style  # noqa: E402

FIG = DATA_DIR / "figures"
TAB = DATA_DIR / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

TAGS = ["base", "thinking", "rl", "full", "validity_only", "energy_only",
        "crystaltextllm", "plaid_wyckoff"]
LABEL = {"base": "CrysReas-Base", "thinking": "CrysReas-Thinking", "rl": "CrysReas-RL",
         "full": "CrysReas", "validity_only": "Validity Only",
         "energy_only": "Energy Only", "crystaltextllm": "CrystalTextLLM",
         "plaid_wyckoff": "PLAID++ Wyckoff"}
TEXLAB = {"base": r"\texttt{\method-Base}", "thinking": r"\texttt{\method-Thinking}",
          "rl": r"\texttt{\method-RL}", "full": r"\method",
          "validity_only": "Validity Only", "energy_only": "Energy Only",
          "crystaltextllm": "CrystalTextLLM", "plaid_wyckoff": "PLAID++ Wyckoff"}
COLOR = {"base": "#4c78a8", "thinking": "#f58518", "rl": "#54a24b", "full": "#e45756",
         "crystaltextllm": "#b279a2", "plaid_wyckoff": "#9c755f",
         "validity_only": "#72b7b2", "energy_only": "#eeca3b"}


# --------------------------------------------------------------------- loading
def load(tag: str) -> pd.DataFrame:
    """Per-row frame with every flag the report needs."""
    t0 = pd.read_parquet(DATA_DIR / f"t0/denovo.main.{tag}.parquet")
    sun = pd.read_parquet(DATA_DIR / f"sun/denovo.main.{tag}.parquet")
    assert (t0.mp_id.values == sun.mp_id.values).all(), tag

    d = t0.copy()
    for c in ("is_stable", "is_unique", "is_novel_train", "is_novel_mp"):
        d[c] = sun[c].astype("boolean").fillna(False)

    valid = (d.structure_validity.astype("boolean").fillna(False)
             & d.smact_validity.astype("boolean").fillna(False))
    # same gate as crysreas/metric_process/basic.py::ensure_spacegroup_consistency
    d["sgc"] = d.sgc_supergroup.where(valid, other=None)
    d["sgc_strict"] = d.sgc_strict.where(valid, other=None)

    ok = lambda c: d[c] == True  # noqa: E731
    d["sun"] = d.is_stable & d.is_unique & d.is_novel_train
    d["ssun"] = d.sun & ok("sgc")
    d["ssun_strict"] = d.sun & ok("sgc_strict")
    d["sun_mp"] = d.is_stable & d.is_unique & d.is_novel_mp
    return d


# ----------------------------------------------------------------- aggregation
def agg(d: pd.DataFrame, col: str, mode: str = "bool") -> tuple[float, float, int]:
    """Mean over per-prompt means, with the standard error across prompts."""
    v = d[col].astype(float) if mode == "float" else (d[col] == True).astype(float)
    g = pd.DataFrame({"g": d.mp_id, "v": v}).groupby("g")["v"].mean().dropna()
    return float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), len(g)


def cell(d: pd.DataFrame, col: str, mode: str = "bool") -> str:
    m, se, _ = agg(d, col, mode)
    if mode == "float":
        return f"{m:.3f} ± {se:.3f}"
    return f"{100*m:.2f}% ± {100*se:.2f}%"


METRICS = [
    ("Structural Validity", "structure_validity", "bool", "up"),
    ("Chemical Validity", "smact_validity", "bool", "up"),
    ("Composition Consistency", "composition_consistency", "bool", "up"),
    ("Space Group Consistency", "sgc", "bool", "up"),
    ("Stability", "is_stable", "bool", "up"),
    ("Uniqueness", "is_unique", "bool", "up"),
    ("Energy (eV/atom)", "energy_above_hull", "float", "down"),
    ("S.U.N.", "sun", "bool", "up"),
    ("S.S.U.N.", "ssun", "bool", "up"),
]


def table(data: dict[str, pd.DataFrame], tags: list[str]) -> pd.DataFrame:
    rows = []
    for name, col, mode, direction in METRICS:
        r = {"metric": name, "mode": mode, "direction": direction}
        for t in tags:
            m, se, _ = agg(data[t], col, mode)
            r[t] = m
            r[t + "_se"] = se
            r[t + "_str"] = cell(data[t], col, mode)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- rendering
def md_table(t: pd.DataFrame, tags: list[str]) -> str:
    head = "| Metric | " + " | ".join(LABEL[x] for x in tags) + " |"
    rule = "|---" * (len(tags) + 1) + "|"
    out = [head, rule]
    for _, r in t.iterrows():
        vals = [r[x] for x in tags]
        best = min(vals) if r["direction"] == "down" else max(vals)
        arrow = "↓" if r["direction"] == "down" else "↑"
        cells = []
        for x in tags:
            s = r[x + "_str"]
            cells.append(f"**{s}**" if r[x] == best else s)
        out.append(f"| {r['metric']} ({arrow}) | " + " | ".join(cells) + " |")
    return "\n".join(out)


def tex_table(t: pd.DataFrame, tags: list[str], caption: str, label: str) -> str:
    cols = "@{}l" + "c" * len(tags) + "@{}"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\small", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{" + cols + "}", r"\toprule",
             r"\textbf{Metric} & "
             + " & ".join(r"\textbf{" + TEXLAB[x] + "}" for x in tags) + r" \\",
             r"\midrule"]
    for _, r in t.iterrows():
        vals = [r[x] for x in tags]
        best = min(vals) if r["direction"] == "down" else max(vals)
        arrow = r"($\uparrow$)" if r["direction"] == "up" else r"($\downarrow$)"
        cells = []
        for x in tags:
            if r["mode"] == "float":
                v, se = f"{r[x]:.3f}", f"{r[x+'_se']:.3f}"
                body = (r"\textbf{" + v + "}" if r[x] == best else v) + r" $\pm$ " + se
            else:
                v, se = f"{100*r[x]:.2f}\\%", f"{100*r[x+'_se']:.2f}\\%"
                body = (r"\textbf{" + v + "}" if r[x] == best else v) + r" $\pm$ " + se
            cells.append(body)
        lines.append(f"{r['metric']} {arrow} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------- shuffle ablation
SHUF_ORDER = ["baseline", "shuffle_part1", "shuffle_part2", "shuffle_part3",
              "shuffle_part4", "shuffle_all"]
SHUF_LABEL = {"baseline": "Full trace (baseline)",
              "shuffle_part1": "Part 1 shuffled (symmetry, formula)",
              "shuffle_part2": "Part 2 shuffled (coordination environment)",
              "shuffle_part3": "Part 3 shuffled (validity statement)",
              "shuffle_part4": "Part 4 shuffled (predicted properties)",
              "shuffle_all": "All four parts shuffled"}
SHUF_METRICS = [("Structural Validity", "structure_validity"),
                ("Chemical Validity", "smact_validity"),
                ("Composition Consistency", "composition_consistency"),
                ("Space Group Consistency", "sgc")]


def shuffle_table() -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / "shuffle/shuffle_sgc.parquet")
    d["sgc"] = d["sgc_sub"]
    rows = []
    base = {}
    for v in SHUF_ORDER:
        sub = d[d.variant == v]
        r = {"variant": v, "label": SHUF_LABEL[v], "n": len(sub)}
        for name, col in SHUF_METRICS:
            m, se, _ = agg(sub, col)
            r[col], r[col + "_se"] = m, se
            r[col + "_str"] = f"{100*m:.2f}% ± {100*se:.2f}%"
            if v == "baseline":
                base[col] = m
            r[col + "_delta"] = 100 * (m - base[col])
        rows.append(r)
    return pd.DataFrame(rows)


def shuffle_md(t: pd.DataFrame) -> str:
    head = "| Condition | " + " | ".join(n for n, _ in SHUF_METRICS) + " |"
    out = [head, "|---" * (len(SHUF_METRICS) + 1) + "|"]
    for _, r in t.iterrows():
        cells = []
        for _, c in SHUF_METRICS:
            s = r[c + "_str"]
            if r.variant != "baseline":
                s += f" ({r[c+'_delta']:+.2f})"
            cells.append(s)
        out.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def shuffle_tex(t: pd.DataFrame, caption: str, label: str) -> str:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\small", r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{@{}l" + "c" * len(SHUF_METRICS) + "@{}}", r"\toprule",
             r"\textbf{Condition} & "
             + " & ".join(r"\textbf{" + n + r"} ($\uparrow$)" for n, _ in SHUF_METRICS)
             + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        cells = []
        for _, c in SHUF_METRICS:
            s = f"{100*r[c]:.2f}\\% $\\pm$ {100*r[c+'_se']:.2f}\\%"
            if r.variant != "baseline":
                s += f" ({r[c+'_delta']:+.2f})"
            cells.append(s)
        lines.append(r["label"] + " & " + " & ".join(cells) + r" \\")
        if r.variant == "baseline":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


# ----------------------------------------------------------------- figures
ATOM_BINS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 99)]
ATOM_LAB = ["1-4", "5-8", "9-12", "13-16", "17+"]
SYS = [(1, 2, "Triclinic"), (3, 15, "Monoclinic"), (16, 74, "Orthorhombic"),
       (75, 142, "Tetragonal"), (143, 167, "Trigonal"), (168, 194, "Hexagonal"),
       (195, 230, "Cubic")]


def binned(d: pd.DataFrame, col: str, key: np.ndarray, bins) -> list[tuple[float, float, int]]:
    out = []
    for lo, hi in bins:
        m = (key >= lo) & (key <= hi)
        sub = d[m]
        out.append(agg(sub, col) if len(sub) else (np.nan, np.nan, 0))
    return out


def fig_complexity(data: dict[str, pd.DataFrame], tags: list[str]) -> pd.DataFrame:
    gt_atoms = pickle.load(open(DATA_DIR / "gt_atoms.pkl", "rb"))
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.6), dpi=220)
    rows = []

    ax = axes[0]
    for t in tags:
        d = data[t]
        key = d.mp_id.map(gt_atoms).to_numpy(dtype=float)
        pts = binned(d, "structure_validity", key, ATOM_BINS)
        y = [100 * p[0] for p in pts]
        e = [100 * p[1] for p in pts]
        ax.errorbar(range(len(ATOM_LAB)), y, yerr=e, marker="o", lw=2.5, capsize=3,
                    color=COLOR[t], label=LABEL[t])
        for lab, p in zip(ATOM_LAB, pts):
            rows.append(dict(panel="atoms", model=LABEL[t], bin=lab,
                             value=100 * p[0], se=100 * p[1], n_prompts=p[2]))
    ax.set_xticks(range(len(ATOM_LAB)), ATOM_LAB)
    ax.set_xlabel("Number of Atoms in the Requested Material")
    ax.set_ylabel("Structural Validity (%)")
    ax.grid(alpha=.25, ls="--")
    ax.set_title("(a)", loc="left", fontsize=22, fontweight="bold")

    ax = axes[1]
    # crystal systems are categories with no order to interpolate along, so bars
    labs = [s[2] for s in SYS]
    xs = np.arange(len(labs))
    w = 0.8 / len(tags)
    for k, t in enumerate(tags):
        d = data[t]
        key = d.req_spg.to_numpy(dtype=float)
        pts = binned(d, "sgc", key, [(a, b) for a, b, _ in SYS])
        y = [100 * q[0] for q in pts]
        e = [100 * q[1] for q in pts]
        off = (k - (len(tags) - 1) / 2) * w
        ax.bar(xs + off, y, width=w, color=COLOR[t], label=LABEL[t], zorder=2)
        ax.errorbar(xs + off, y, yerr=e, fmt="none", ecolor="0.25", lw=1.0,
                    capsize=1.5, zorder=3)
        for lab, q in zip(labs, pts):
            rows.append(dict(panel="spacegroup", model=LABEL[t], bin=lab,
                             value=100 * q[0], se=100 * q[1], n_prompts=q[2]))
    ax.set_ylim(0, 92)
    ax.set_xticks(xs, labs, rotation=22, ha="right")
    ax.set_ylabel("Space Group Consistency (%)")
    ax.set_xlabel("Crystal System of the Requested Space Group")
    ax.grid(axis="y", alpha=.25, ls="--")
    ax.set_axisbelow(True)
    ax.set_title("(b)", loc="left", fontsize=22, fontweight="bold")

    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_d_complexity.pdf")
    fig.savefig(FIG / "fig_d_complexity.png", dpi=110)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_parquet(TAB / "report_d_complexity.parquet", index=False)
    print(f"wrote {FIG/'fig_d_complexity.pdf'}")
    return t


def fig_precision(data: dict[str, pd.DataFrame], tags: list[str]) -> pd.DataFrame:
    """Paired over prompts every model relaxed, so the models are compared on
    the same materials; relaxation success itself differs by model."""
    common = None
    for t in tags:
        m = data[t].relax_rmsd.notna().groupby(data[t].mp_id).all()
        common = m if common is None else (common & m)
    keep = set(common[common].index)

    rows = []
    for t in tags:
        d = data[t]
        sub = d[d.mp_id.isin(keep) & d.relax_rmsd.notna()]
        r = dict(model=LABEL[t], tag=t, n_prompts=len(keep), n_rows=len(sub),
                 rmsd_median=float(sub.relax_rmsd.median()),
                 rmsd_mean=float(sub.relax_rmsd.mean()),
                 dvol_median=float(sub.dvol_frac.median()))
        for k in ("1e2", "1e3", "1e4"):
            m, se, _ = agg(d[d.mp_id.isin(keep)], f"sym_survives_{k}")
            r[f"sym_{k}"], r[f"sym_{k}_se"] = 100 * m, 100 * se
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_parquet(TAB / "report_d_precision.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(19, 6.6), dpi=220)
    ax = axes[0]
    x = np.arange(len(t))
    ax.bar(x, t.rmsd_median * 1000, width=.6,
           color=[COLOR[k] for k in t.tag], zorder=2)
    ax.set_xticks(x, [LABEL[k] for k in t.tag], rotation=25, ha="right")
    ax.set_ylabel("Median displacement (mÅ)")
    ax.grid(axis="y", alpha=.25, ls="--")
    ax.set_title("(a)", loc="left", fontsize=22, fontweight="bold")

    ax = axes[1]
    sym = [0.1, 0.01, 0.001, 0.0001]
    for tag in t.tag:
        r = t[t.tag == tag].iloc[0]
        y = [100.0] + [float(r[f"sym_{k}"]) for k in ("1e2", "1e3", "1e4")]
        ax.plot(range(len(sym)), y, marker="o", lw=2.5, color=COLOR[tag], label=LABEL[tag])
    ax.set_xticks(range(len(sym)), [f"{s:g}" for s in sym])
    ax.set_xlabel("Symmetry tolerance symprec (Å)")
    ax.set_ylabel("Detected space group retained (%)")
    ax.grid(alpha=.25, ls="--")
    ax.set_title("(b)", loc="left", fontsize=22, fontweight="bold")

    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, fontsize=16)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig_d_precision.pdf")
    fig.savefig(FIG / "fig_d_precision.png", dpi=110)
    plt.close(fig)
    print(f"wrote {FIG/'fig_d_precision.pdf'}")
    return t


# ---------------------------------------------------------------------- main
def main() -> None:
    apply_publication_style()
    plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 16})

    data = {t: load(t) for t in TAGS}
    for t in TAGS:
        d = data[t]
        print(f"{LABEL[t]:20s} n={len(d)} sun={int(d.sun.sum()):5d} ssun={int(d.ssun.sum()):5d} "
              f"sun_mp={int(d.sun_mp.sum()):4d} ssun_strict={int(d.ssun_strict.sum()):5d}")

    priors = ["full", "crystaltextllm", "plaid_wyckoff"]
    variants = ["base", "thinking", "rl", "full"]
    rewards = ["validity_only", "energy_only", "full"]

    t_pri = table(data, priors)
    t_var = table(data, variants)
    t_rew = table(data, rewards)
    t_pri.to_parquet(TAB / "report_d_priors.parquet", index=False)
    t_var.to_parquet(TAB / "report_d_variants.parquet", index=False)
    t_rew.to_parquet(TAB / "report_d_rewards.parquet", index=False)
    t_shuf = shuffle_table()
    t_shuf.to_parquet(TAB / "report_d_shuffle.parquet", index=False)

    # extra columns the text quotes but the tables do not carry
    extra = []
    for t in TAGS:
        d = data[t]
        extra.append(dict(model=LABEL[t], tag=t,
                          sgc_strict=cell(d, "sgc_strict"), sgc_sub=cell(d, "sgc"),
                          sun_mp=cell(d, "sun_mp"), sun=cell(d, "sun"),
                          ssun=cell(d, "ssun"), ssun_strict=cell(d, "ssun_strict"),
                          n_sun=int(d.sun.sum()), n_ssun=int(d.ssun.sum()),
                          n_sun_mp=int(d.sun_mp.sum()),
                          novel_train=cell(d, "is_novel_train"),
                          novel_mp=cell(d, "is_novel_mp"),
                          relax_rate=cell(d, "relax_rmsd")))
        extra[-1]["relax_rate"] = f"{100*d.relax_rmsd.notna().mean():.2f}%"
        extra[-1]["sgc_p1_only"] = cell(d[d.req_spg == 1], "sgc")
        extra[-1]["sgc_no_p1"] = cell(d[d.req_spg != 1], "sgc")
        extra[-1]["sgc_strict_no_p1"] = cell(d[d.req_spg != 1], "sgc_strict")
    ex = pd.DataFrame(extra)
    ex.to_parquet(TAB / "report_d_extra.parquet", index=False)
    cols = [("model", "Model"), ("sgc_strict", "SGC strict"), ("sgc_sub", "SGC subgroup"),
            ("sgc_p1_only", "SGC subgroup, P1 requests only"),
            ("sun", "S.U.N."), ("n_sun", "count"), ("ssun", "S.S.U.N."), ("n_ssun", "count"),
            ("sun_mp", "S.U.N. vs MP+Alexandria"), ("n_sun_mp", "count"),
            ("relax_rate", "relaxed")]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    lines = [head, "|---" * len(cols) + "|"]
    for _, r in ex.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c, _ in cols) + " |")
    (TAB / "report_d_extra.md").write_text("\n".join(lines) + "\n")

    cx = fig_complexity(data, ["base", "thinking", "rl", "full",
                               "crystaltextllm", "plaid_wyckoff"])
    pr = fig_precision(data, ["base", "thinking", "rl", "full",
                              "crystaltextllm", "plaid_wyckoff"])

    cap_pri = ("Comparison against two prior LLM-based works, re-implemented in the "
               r"\method pipeline with the same dataset, base model and hyperparameters. "
               r"All three models are evaluated on the same $16{,}282$ generated "
               r"structures ($8{,}141$ held-out compositions $\times$ $2$ samples). "
               r"Here \textbf{PLAID++ Wyckoff} means that only the representation of "
               r"PLAID++ is used, without its DPO stage. Space group consistency counts "
               r"a structure when it carries at least the requested symmetry; S.U.N. and "
               r"S.S.U.N. are defined in Section~\ref{sec:eval1}.")
    cap_var = (r"Performance comparison of model variants: \texttt{\method-Base} (SFT "
               r"baseline), \texttt{\method-Thinking} (SFT with thinking traces), "
               r"\texttt{\method-RL} (RL on the base model), and the full \method. "
               r"Thinking traces and RL each improve almost every metric, and the two "
               r"are complementary. All four columns come from one evaluation of "
               r"$16{,}282$ generated structures through a single metric pipeline.")
    cap_rew = ("Ablation of reward designs: Validity Only (structural and chemical "
               r"validity, $\alpha_{\text{validity}}=1, \alpha_{\text{stability}}=0$), "
               r"Energy Only ($\alpha_{\text{validity}}=0, \alpha_{\text{stability}}=10$) "
               r"and Mixed Reward ($\alpha_{\text{validity}}=1, "
               r"\alpha_{\text{stability}}=10$, the full \method). Each configuration "
               "leads on the metric it optimizes, and the mixed reward is the only one "
               "that stays competitive on validity and on stability alike.")
    cap_shuf = ("Ablation on the four segments of the thinking trace. Each segment is "
                "replaced by the same-named segment of another randomly drawn sample, so "
                "format, style and length distribution stay intact and only the "
                "information the segment carries about this material is destroyed. "
                "Replacing a segment leaves the mean token count of that condition "
                "unchanged. Numbers in "
                "parentheses are the change from the full trace. Source rows are "
                r"generations that were correct on all four metrics, which is why the "
                r"baseline sits above the numbers of Table~\ref{tab:compare_priors}; "
                "read the relative drops.")

    md = ["## Table: comparison against two prior LLM-based works", "",
          md_table(t_pri, priors), "",
          "## Table: comparison against three alternatives", "",
          md_table(t_var, variants), "",
          "## Table: ablation study on reward components", "",
          md_table(t_rew, rewards), "",
          "## Table: thinking-trace segment ablation", "",
          shuffle_md(t_shuf), ""]
    (TAB / "report_d_tables.md").write_text("\n".join(md))
    for key, body in (("priors", md_table(t_pri, priors)),
                      ("variants", md_table(t_var, variants)),
                      ("rewards", md_table(t_rew, rewards)),
                      ("shuffle", shuffle_md(t_shuf))):
        (TAB / f"report_d_{key}.md").write_text(body + "\n")

    tex = [tex_table(t_pri, priors, cap_pri, "tab:compare_priors"), "",
           tex_table(t_var, variants, cap_var, "tab:overall_performance"), "",
           tex_table(t_rew, rewards, cap_rew, "tab:reward_comparison"), "",
           shuffle_tex(t_shuf, cap_shuf, "tab:trace_shuffle"), ""]
    (TAB / "report_d_tables.tex").write_text("\n".join(tex))
    for key, body in (("priors", tex_table(t_pri, priors, cap_pri, "tab:compare_priors")),
                      ("variants", tex_table(t_var, variants, cap_var, "tab:overall_performance")),
                      ("rewards", tex_table(t_rew, rewards, cap_rew, "tab:reward_comparison")),
                      ("shuffle", shuffle_tex(t_shuf, cap_shuf, "tab:trace_shuffle"))):
        (TAB / f"report_d_{key}.tex").write_text(body + "\n")

    print((TAB / "report_d_tables.md").read_text())
    print(pd.DataFrame(extra).to_string(index=False))
    print(cx.to_string(index=False))
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
