"""Paper-format LaTeX for the four tables that go into the manuscript.

Same numbers as `pipeline/d_report.py`, formatted the way the manuscript wants
them: the variant table keeps its metric-category column, and every float
carries `\\color{red}` because a colour command does not cross a float
boundary, so `\\rev{}` around the float would have no effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import d_report as R  # noqa: E402
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

OUT = DATA_DIR / "tables" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

DATA = {t: R.load(t) for t in R.TAGS}
ROWS = {name: (col, mode, direction) for name, col, mode, direction in R.METRICS}


def fmt(tag: str, col: str, mode: str, best: bool) -> str:
    m, se, _ = R.agg(DATA[tag], col, mode)
    if mode == "float":
        v, s = f"{m:.3f}", f"{se:.3f}"
    else:
        v, s = f"{100*m:.2f}\\%", f"{100*se:.2f}\\%"
    return (r"\textbf{" + v + "}" if best else v) + r" $\pm$ " + s


def best_tag(tags: list[str], col: str, mode: str, direction: str) -> str:
    vals = {t: R.agg(DATA[t], col, mode)[0] for t in tags}
    return (min if direction == "down" else max)(vals, key=vals.get)


def arrow(direction: str) -> str:
    return r"($\uparrow$)" if direction == "up" else r"($\downarrow$)"


def flat(tags: list[str], headers: list[str], caption: str, label: str) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\color{red}",
         r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l" + "c" * len(tags) + "@{}}", r"\toprule",
         r"\textbf{Metric} & " + " & ".join(r"\textbf{" + h + "}" for h in headers) + r" \\",
         r"\midrule"]
    for name, col, mode, direction in R.METRICS:
        b = best_tag(tags, col, mode, direction)
        L.append(f"{name} {arrow(direction)} & "
                 + " & ".join(fmt(t, col, mode, t == b) for t in tags) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L) + "\n"


CATEGORIES = [
    ("Validity", [("Structural", "Structural Validity"), ("Chemical", "Chemical Validity")]),
    ("Instruction Following", [("Composition", "Composition Consistency"),
                               ("Space Group", "Space Group Consistency")]),
    ("Stability", [("Stability", "Stability"), ("Energy", "Energy (eV/atom)")]),
    ("Discovery", [("Uniqueness", "Uniqueness"), ("S.U.N.", "S.U.N."),
                   ("S.S.U.N.", "S.S.U.N.")]),
]


def categorized(tags: list[str], headers: list[str], caption: str, label: str) -> str:
    L = [r"\begin{table}[t]", r"\centering", r"\color{red}",
         r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}ll" + "c" * len(tags) + "@{}}", r"\toprule",
         r"\textbf{Metric Category} & \textbf{Metric Name} & "
         + " & ".join(r"\textbf{" + h + "}" for h in headers) + r" \\", r"\midrule"]
    for k, (cat, items) in enumerate(CATEGORIES):
        if k:
            L += [r"\addlinespace", r"\midrule"]
        for j, (short, name) in enumerate(items):
            col, mode, direction = ROWS[name]
            b = best_tag(tags, col, mode, direction)
            lead = (r"\multirow{" + str(len(items)) + r"}{*}{" + cat + "}") if j == 0 else ""
            L.append(f"{lead} & " + r"\textbf{" + short + "} " + arrow(direction) + " & "
                     + " & ".join(fmt(t, col, mode, t == b) for t in tags) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L) + "\n"


SHUF_SHORT = {"baseline": "Full trace",
              "shuffle_part1": "Part 1 swapped: symmetry, formula",
              "shuffle_part2": "Part 2 swapped: coordination",
              "shuffle_part3": "Part 3 swapped: validity check",
              "shuffle_part4": "Part 4 swapped: properties",
              "shuffle_all": "All four parts swapped"}
SHUF_HEAD = ["Structural", "Chemical", "Composition", "Space Group"]


def shuffle(caption: str, label: str) -> str:
    """Compact enough to sit at full text width without adjustbox shrinking it."""
    t = R.shuffle_table()
    L = [r"\begin{table}[t]", r"\centering", r"\color{red}",
         r"\caption{" + caption + "}", r"\label{" + label + "}", r"\small",
         r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{@{}l" + "c" * len(R.SHUF_METRICS) + "@{}}", r"\toprule",
         r"\textbf{Thinking trace fed to the model} & "
         + " & ".join(r"\textbf{" + h + r"} ($\uparrow$)" for h in SHUF_HEAD) + r" \\",
         r"\midrule"]
    for _, r in t.iterrows():
        cells = []
        for _, c in R.SHUF_METRICS:
            s_ = f"{100*r[c]:.2f} $\\pm$ {100*r[c+'_se']:.2f}"
            if r.variant != "baseline":
                s_ += f" ({r[c+'_delta']:+.1f})"
            cells.append(s_)
        L.append(SHUF_SHORT[r["variant"]] + " & " + " & ".join(cells) + r" \\")
        if r.variant == "baseline":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L) + "\n"


CAP_PRI = (r"\rev{Comparison against two prior LLM-based works, re-implemented in the "
           r"\method pipeline with the same dataset, base model and hyperparameters. "
           r"Here \textbf{PLAID++ Wyckoff} means that only the representation of PLAID++ "
           r"is used, without its DPO stage. All three models are evaluated on the same "
           r"$16{,}282$ generated structures, and the metrics are the ones defined in "
           r"Section~\ref{sec:eval1}. \method wins every column but uniqueness, where "
           r"CrystalTextLLM produces the more scattered set without turning that spread "
           r"into stable structures.}")
CAP_VAR = (r"\rev{Performance comparison of model variants: \texttt{\method-Base} (SFT "
           r"baseline), \texttt{\method-Thinking} (SFT with thinking traces), "
           r"\texttt{\method-RL} (RL on the base model), and the full \method. Thinking "
           r"traces and RL each improve almost every metric and the full model leads on "
           r"all of them. All four columns come from one evaluation of $16{,}282$ "
           r"generated structures through a single metric pipeline.}")
CAP_REW = (r"\rev{Ablation of reward designs: Validity Only "
           r"($\alpha_{\text{validity}}=1, \alpha_{\text{stability}}=0$), Energy Only "
           r"($\alpha_{\text{validity}}=0, \alpha_{\text{stability}}=10$) and Mixed "
           r"Reward ($\alpha_{\text{validity}}=1, \alpha_{\text{stability}}=10$, the full "
           r"\method). Each configuration leads on the metric it optimizes. The two "
           r"configurations that carry the energy term are close everywhere, so the "
           r"energy term is what moves stability and the validity terms add a small "
           r"amount on top.}")
CAP_SHUF = (r"\rev{Ablation on the four segments of the thinking trace. Each segment is "
            r"replaced by the same-named segment of another randomly drawn sample, so "
            r"format, style and length distribution stay intact and only the information "
            r"the segment carries about this material is destroyed. Replacing a segment "
            r"leaves the mean token count of that condition unchanged, so no drop here "
            r"is a length effect. Numbers in parentheses are the change from the full "
            r"trace, in points. All four columns are percentages: structural validity, "
            r"chemical validity, composition consistency, and space group consistency. "
            r"The source rows were selected to be correct on all four metrics, "
            r"which places the baseline above the numbers of "
            r"Table~\ref{tab:overall_performance}; the drops are what this table "
            r"measures.}")


def main() -> None:
    (OUT / "tab_compare_priors.tex").write_text(flat(
        ["full", "crystaltextllm", "plaid_wyckoff"],
        [r"\method", "CrystalTextLLM", "PLAID++ Wyckoff"], CAP_PRI, "tab:compare_priors"))
    (OUT / "tab_overall_performance.tex").write_text(categorized(
        ["base", "thinking", "rl", "full"],
        [r"\method-Base", r"\method-Thinking", r"\method-RL", r"\method"],
        CAP_VAR, "tab:overall_performance"))
    (OUT / "tab_reward_comparison.tex").write_text(flat(
        ["validity_only", "energy_only", "full"],
        ["Validity Only", "Energy Only", "Mixed Reward"], CAP_REW, "tab:reward_comparison"))
    (OUT / "tab_trace_shuffle.tex").write_text(shuffle(CAP_SHUF, "tab:trace_shuffle"))
    for f in sorted(OUT.glob("*.tex")):
        print("wrote", f)


if __name__ == "__main__":
    main()
