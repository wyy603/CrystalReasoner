"""Aggregation for every number the paper reports.

This module is the only place results are produced. The metric pipeline
(``crysreas.metric_process``) writes one row per generated structure; this
module turns those rows into the rates, error bars and tables that appear in
the manuscript, so that a table can never disagree with the columns it came
from.

Protocol
--------
An evaluation draws ``n_samples`` structures for each held-out composition.
Row ``j`` of every prompt is labelled seed ``j`` (see
``crysreas.metric_process.basic.ensure_seed``), so each seed holds exactly one
structure per composition and is a complete evaluation of the split on its own.
Every rate below is the mean over seeds of the per-seed rate, and every error
bar is the standard error across seeds. That estimator reports run-to-run
variation. It deliberately does *not* include the variation that comes from
choosing which compositions to evaluate on, so these error bars are not
comparable with a per-composition clustered standard error.

Thresholds and judgements come from the config dict documented in
``crysreas.metric_process.config``; nothing here hard-codes them.

Usage
-----
    from crysreas.experiment.evaluate import load_run, summarize, compare

    runs = {"CrysReas": load_run("path/to/full.parquet")}
    print(compare(runs).to_string())

or from the command line::

    python -m crysreas.experiment.evaluate run_a.parquet run_b.parquet --latex
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from crysreas.metric_process.config import merge_metric_process_config
from crysreas.metric_process.io import load_metrics_dataframe


@dataclass(frozen=True)
class Metric:
    """One reported row: display label, source column, and how to read it."""

    label: str
    column: str
    mode: str = "bool"          # "bool" or "float"
    higher_is_better: bool = True


#: The nine rows of the manuscript tables, in table order.
METRICS: tuple[Metric, ...] = (
    Metric("Structural Validity", "structure_validity"),
    Metric("Chemical Validity", "smact_validity"),
    Metric("Composition Consistency", "composition_consistency"),
    Metric("Space Group Consistency", "spacegroup_consistency"),
    Metric("Stability", "is_stable"),
    Metric("Uniqueness", "is_unique"),
    Metric("Energy (eV/atom)", "energy_above_hull", "float", False),
    Metric("S.U.N.", "stable_unique_novel"),
    Metric("S.S.U.N.", "spacegroup_stable_unique_novel"),
)


def load_run(path: str | Path, config: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Read one evaluated run, ensuring it carries a seed label."""
    df = load_metrics_dataframe(Path(path))
    cfg = merge_metric_process_config(None, dict(config or {}))
    column = cfg["seed_column"]
    if column not in df.columns:
        group = cfg["group_column"]
        if group not in df.columns:
            raise KeyError(
                f"{path} has neither a {column!r} nor a {group!r} column; "
                "seeds cannot be recovered"
            )
        df[column] = df.groupby(group).cumcount()
    return df


def _values(df: pd.DataFrame, metric: Metric) -> np.ndarray:
    """Per-row numeric values, with unscored rows charged to the denominator.

    A row whose relaxation or parse failed holds ``None``. For a boolean rate
    that counts as a failure, which keeps the denominator equal to the number of
    structures drawn.
    """
    if metric.column not in df.columns:
        return np.full(len(df), np.nan)
    if metric.mode == "float":
        return pd.to_numeric(df[metric.column], errors="coerce").to_numpy(dtype=float)
    return (df[metric.column] == True).to_numpy(dtype=float)  # noqa: E712


def per_seed(
    df: pd.DataFrame,
    metric: Metric,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """The metric evaluated separately on each seed."""
    cfg = merge_metric_process_config(None, dict(config or {}))
    seeds = df[cfg["seed_column"]].to_numpy()
    values = _values(df, metric)
    out = []
    for seed in np.unique(seeds):
        rows = values[seeds == seed]
        out.append(np.nanmean(rows) if metric.mode == "float" else rows.mean())
    return np.asarray(out, dtype=float)


def missing_columns(
    df: pd.DataFrame, metrics: Iterable[Metric] = METRICS
) -> list[str]:
    """Metrics whose source column the table does not carry.

    A missing column yields ``nan`` throughout, which is easy to miss in a
    printed table, so callers should surface this before reporting anything.
    """
    return [m.label for m in metrics if m.column not in df.columns]


def summarize(
    df: pd.DataFrame,
    metrics: Iterable[Metric] = METRICS,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Mean over seeds, standard error across seeds, and the seed count."""
    rows = []
    for metric in metrics:
        values = per_seed(df, metric, config)
        n = len(values)
        scale = 1.0 if metric.mode == "float" else 100.0
        se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append(
            {
                "metric": metric.label,
                "mean": scale * float(values.mean()),
                "se": scale * se,
                "n_seeds": n,
                "n_structures": len(df),
                "higher_is_better": metric.higher_is_better,
                "mode": metric.mode,
            }
        )
    return pd.DataFrame(rows)


def _format(mean: float, se: float, mode: str, bold: bool) -> str:
    digits = 3 if mode == "float" else 2
    unit = "" if mode == "float" else "%"
    body = f"{mean:.{digits}f}{unit}"
    if bold:
        body = f"**{body}**"
    return f"{body} ± {se:.{digits}f}{unit}"


def compare(
    runs: Mapping[str, pd.DataFrame],
    metrics: Iterable[Metric] = METRICS,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """One manuscript table: metrics down the rows, runs across the columns.

    The best entry in each row is marked. Ties are broken by column order, so a
    mark is not by itself evidence of a difference; compare the gap with the
    error bars before reading anything into it.
    """
    metrics = tuple(metrics)
    summaries = {name: summarize(df, metrics, config) for name, df in runs.items()}
    table = []
    for i, metric in enumerate(metrics):
        means = {name: s.loc[i, "mean"] for name, s in summaries.items()}
        best = (max if metric.higher_is_better else min)(means.values())
        row = {"Metric": metric.label + (" (↑)" if metric.higher_is_better else " (↓)")}
        for name, s in summaries.items():
            row[name] = _format(
                s.loc[i, "mean"], s.loc[i, "se"], metric.mode, means[name] == best
            )
        table.append(row)
    return pd.DataFrame(table)


def to_latex(
    runs: Mapping[str, pd.DataFrame],
    metrics: Iterable[Metric] = METRICS,
    config: Mapping[str, Any] | None = None,
    caption: str = "",
    label: str = "tab:results",
) -> str:
    """The same table as a LaTeX ``tabular``, ready to paste into the paper."""
    metrics = tuple(metrics)
    summaries = {name: summarize(df, metrics, config) for name, df in runs.items()}
    names = list(runs)
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(names) + r"@{}}",
        r"\toprule",
        r"\textbf{Metric} & "
        + " & ".join(rf"\textbf{{{n}}}" for n in names)
        + r" \\",
        r"\midrule",
    ]
    for i, metric in enumerate(metrics):
        means = {n: summaries[n].loc[i, "mean"] for n in names}
        best = (max if metric.higher_is_better else min)(means.values())
        arrow = r"($\uparrow$)" if metric.higher_is_better else r"($\downarrow$)"
        cells = []
        for n in names:
            mean, se = summaries[n].loc[i, "mean"], summaries[n].loc[i, "se"]
            digits = 3 if metric.mode == "float" else 2
            unit = "" if metric.mode == "float" else r"\%"
            body = f"{mean:.{digits}f}{unit}"
            if means[n] == best:
                body = rf"\textbf{{{body}}}"
            cells.append(rf"{body} $\pm$ {se:.{digits}f}{unit}")
        lines.append(f"{metric.label} {arrow} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    if caption:
        return (
            "\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{table}"
        )
    return body


def separation(
    a: pd.DataFrame, b: pd.DataFrame, metric: Metric,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Gap between two runs in units of the standard error of the difference.

    Reported alongside any claim that one run beats another, because the seed
    counts here are small enough that a visible gap is often not a separation.
    """
    va, vb = per_seed(a, metric, config), per_seed(b, metric, config)
    sea = va.std(ddof=1) / np.sqrt(len(va))
    seb = vb.std(ddof=1) / np.sqrt(len(vb))
    denom = float(np.hypot(sea, seb))
    if denom == 0:
        return float("nan")
    return float(abs(va.mean() - vb.mean()) / denom)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="evaluated parquet files")
    parser.add_argument("--names", nargs="*", help="column labels, defaults to stems")
    parser.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    args = parser.parse_args(argv)

    names = args.names or [Path(p).stem for p in args.runs]
    if len(names) != len(args.runs):
        parser.error("--names must have one entry per run")
    runs = {n: load_run(p) for n, p in zip(names, args.runs)}

    first = next(iter(runs.values()))
    n_seeds = int(first["seed"].nunique())
    print(
        f"{len(first)} structures, {first['mp_id'].nunique()} compositions, "
        f"{n_seeds} seeds; error bars are standard errors across seeds"
    )
    for name, df in runs.items():
        gaps = missing_columns(df)
        if gaps:
            print(
                f"WARNING  {name} has no column for: {', '.join(gaps)}. "
                f"Those rows report nan. Score the table first:\n"
                f"         python -m crysreas.metric_process --path <parquet> "
                f"--metrics-name spacegroup_stable_unique_novel"
            )
    print()
    print(to_latex(runs) if args.latex else compare(runs).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
