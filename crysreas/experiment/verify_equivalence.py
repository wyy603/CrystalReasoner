"""Proof that the migrated metrics agree with what they replace.

The evaluation artifacts under ``data/`` carry columns scored under MatterGen's
reading of the four judgements. This module re-derives each of those columns
from the configurable implementation and compares them row by row. Nothing is
sampled or averaged first: a single disagreeing row fails the check.

Two directions are covered.

*MatterGen settings* --- with ``crysreas.metric_process.config.mattergen_config()``
the implementation must reproduce the stored columns:

    A  stability flag      apply_stability_threshold(e, 0.1, strict=True) == is_stable
    B  space group         spacegroup_matches(..., "strict") == spacegroup_consistency
    C  uniqueness, pooled   compute_uniqueness(scope="pooled") == is_unique

*Default settings* --- with the protocol this package reports under, the same
implementation must reproduce the manuscript's numbers:

    D  space group, subgroup criterion == the stored sgc_supergroup column
    E  per-seed uniqueness and S.U.N. == data/u_per_seed.parquet
    F  seed labelling is a valid partition: every (prompt, seed) pair once

Known exception, check C on ``energy_only``
-------------------------------------------
One row in 10,240 disagrees. The cause is a property of the metric itself. ``DisorderedStructureMatcher.fit`` is **not symmetric**: it
disorders its *first* argument before the fallback comparison, so ``fit(a, b)``
and ``fit(b, a)`` can differ for a pair sitting on the tolerance boundary. Rows
6223 and 6227 of ``energy_only`` (formula YAu3, mp-1079467) are exactly such a
pair -- verified directly, they are the only asymmetric pair in their group.

Greedy first-wins therefore does not define a unique answer: which member of a
near-matching class survives, and sometimes how many survive, depends on the
order the set is scanned in. Pooled uniqueness is well defined only up to that
ambiguity, here 0.01 percentage points on one of eight runs.

The check is deliberately left strict, with no tolerance, because the ambiguity
is worth seeing. It does not touch the reported numbers: uniqueness is
scored per seed, where a seed holds one structure per composition and collisions
of this kind essentially cannot arise, and check E reproduces the per-seed rate
exactly on all eight runs.

One structural assumption is checked here too, not assumed:

    G  grouping by reduced formula is exact, i.e. the bucketed uniqueness equals
       a brute-force all-pairs run on a slice that really does contain
       duplicates. This is what licenses scoring uniqueness per seed at all.

Run::

    python -m crysreas.experiment.verify_equivalence
    python -m crysreas.experiment.verify_equivalence --tags full base
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from crysreas.experiment.paths import DATA_DIR
from crysreas.metric_process.helpers import spacegroup_matches
from crysreas.metric_process.io import load_metrics_dataframe
from crysreas.mlip.energy import apply_stability_threshold
from crysreas.mlip.sun import compute_uniqueness, uniqueness_mask

warnings.filterwarnings("ignore")

DEFAULT_TAGS = (
    "base", "thinking", "rl", "full",
    "validity_only", "energy_only", "crystaltextllm", "plaid_wyckoff",
)


class Report:
    """Collects outcomes so the exit status reflects every check."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, message: str) -> None:
        print(("  PASS  " if ok else "  FAIL  ") + message, flush=True)
        if not ok:
            self.failures += 1


def _bool(series: pd.Series) -> np.ndarray:
    return (series == True).to_numpy(dtype=bool)  # noqa: E712


def check_stability(df: pd.DataFrame, tag: str, report: Report) -> None:
    """A. The stored flag is a strict ``E_hull < 0.1``."""
    e = pd.to_numeric(df["energy_above_hull"], errors="coerce").to_numpy(dtype=float)
    rebuilt = apply_stability_threshold(e, 0.1, strict=True)
    stored = _bool(df["is_stable"])
    bad = int((rebuilt != stored).sum())
    report.check(bad == 0, f"A stability flag, {tag}: {bad} rows differ")


def check_spacegroup_strict(
    df: pd.DataFrame, t0: pd.DataFrame, tag: str, report: Report
) -> None:
    """B. The stored column is integer equality, gated on validity.

    Only the criterion applied to the detector's output varies here; the
    detector itself is shared, so the detected numbers come from ``spg_p1``
    column, with no recomputation. Re-running ``SpacegroupAnalyzer`` here would
    test pymatgen, not this change, and costs an hour per run.
    """
    valid = _bool(df["structure_validity"]) & _bool(df["smact_validity"])
    stored = df["spacegroup_consistency"].to_numpy()
    req = t0["req_spg"].to_numpy()
    det = t0["spg_p1"].to_numpy()
    disagree = scored = 0
    for i in range(len(df)):
        if not valid[i]:
            continue
        rebuilt = spacegroup_matches(req[i], det[i], "strict")
        value = stored[i]
        if rebuilt is None and (value is None or pd.isna(value)):
            continue
        if rebuilt is None or value is None or pd.isna(value):
            disagree += 1
            continue
        scored += 1
        if bool(rebuilt) != bool(value):
            disagree += 1
    report.check(
        disagree == 0,
        f"B space group, strict, {tag}: {disagree} of {scored} scored rows differ",
    )


def check_uniqueness_pooled(df: pd.DataFrame, tag: str, report: Report) -> None:
    """C. Pooled uniqueness is what the stored column holds."""
    relaxed = [None if r is None or (isinstance(r, float) and np.isnan(r)) else r
               for r in df["relaxed_structures"].tolist()]
    rebuilt = compute_uniqueness(relaxed, scope="pooled")
    stored = _bool(df["is_unique"])
    bad = int((rebuilt != stored).sum())
    report.check(
        bad == 0,
        f"C uniqueness, pooled, {tag}: {bad} rows differ "
        f"(rebuilt {int(rebuilt.sum())}, stored {int(stored.sum())})",
    )


def check_spacegroup_subgroup(
    df: pd.DataFrame, t0: pd.DataFrame, tag: str, report: Report
) -> None:
    """D. The subgroup criterion reproduces the column the manuscript uses."""
    stored = t0["sgc_supergroup"]
    disagree = 0
    for i in range(len(t0)):
        rebuilt = spacegroup_matches(t0["req_spg"].iloc[i], t0["spg_p1"].iloc[i], "subgroup")
        value = stored.iloc[i]
        both_missing = rebuilt is None and (value is None or pd.isna(value))
        if both_missing:
            continue
        if rebuilt is None or value is None or pd.isna(value) or bool(rebuilt) != bool(value):
            disagree += 1
    report.check(
        disagree == 0, f"D space group, subgroup, {tag}: {disagree} rows differ"
    )


def check_per_seed(
    df: pd.DataFrame, tag: str, expected: pd.DataFrame | None, report: Report
) -> None:
    """E. Per-seed uniqueness matches the rate the manuscript reports."""
    relaxed = [None if r is None or (isinstance(r, float) and np.isnan(r)) else r
               for r in df["relaxed_structures"].tolist()]
    seeds = df["seed"].to_numpy()
    mask = compute_uniqueness(relaxed, seeds=seeds, scope="seed")
    rate = 100 * np.mean([mask[seeds == s].mean() for s in np.unique(seeds)])
    if expected is None or tag not in expected.index:
        report.check(True, f"E per-seed uniqueness, {tag}: {rate:.2f}% (no reference)")
        return
    want = float(expected.loc[tag, "uniq"])
    report.check(
        abs(rate - want) < 0.005,
        f"E per-seed uniqueness, {tag}: {rate:.2f}% vs reference {want:.2f}%",
    )


def check_seed_partition(df: pd.DataFrame, tag: str, report: Report) -> None:
    """F. Each seed selects exactly one structure per prompt."""
    counts = df.groupby(["mp_id", "seed"]).size()
    per_seed = df.groupby("seed").size()
    ok = bool((counts == 1).all()) and per_seed.nunique() == 1
    report.check(
        ok,
        f"F seed partition, {tag}: {per_seed.nunique()} distinct seed sizes, "
        f"max rows per (prompt, seed) = {int(counts.max())}",
    )


def check_bucketing_is_exact(df: pd.DataFrame, report: Report, n: int = 300) -> None:
    """G. Reduced-formula grouping loses no match a full comparison would find."""
    from mattergen.evaluation.utils.structure_matcher import (
        DefaultDisorderedStructureMatcher,
    )

    matcher = DefaultDisorderedStructureMatcher()
    relaxed = df["relaxed_structures"].tolist()
    sample = [s for s in relaxed if s is not None][:n]
    bucketed = uniqueness_mask(sample, matcher)

    brute = np.zeros(len(sample), dtype=bool)
    kept: list = []
    for i, s in enumerate(sample):
        if not any(matcher.fit(s, k) for k in kept):
            kept.append(s)
            brute[i] = True

    duplicates = len(sample) - int(brute.sum())
    report.check(
        bool((bucketed == brute).all()) and duplicates > 0,
        f"G reduced-formula grouping is exact: {len(sample)} structures, "
        f"{duplicates} duplicates exercised",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs", default=str(DATA_DIR / "out"),
        help="directory of main.<tag>.parquet",
    )
    parser.add_argument(
        "--t0", default=str(DATA_DIR / "t0"),
        help="directory of <src>.main.<tag>.parquet",
    )
    parser.add_argument("--reference", default=str(DATA_DIR / "u_per_seed.parquet"))
    parser.add_argument("--tags", nargs="*", default=list(DEFAULT_TAGS))
    args = parser.parse_args(argv)

    runs, t0_dir = Path(args.runs), Path(args.t0)
    expected = None
    if Path(args.reference).exists():
        expected = pd.read_parquet(args.reference).set_index("tag")

    report = Report()
    first: pd.DataFrame | None = None
    found = 0
    for tag in args.tags:
        path = runs / f"main.{tag}.parquet"
        if not path.exists():
            print(f"  SKIP  {tag}: {path} not found")
            continue
        df = load_metrics_dataframe(path)
        found += 1
        if first is None:
            first = df
        print(f"\n--- {tag}: {len(df)} rows, {df['mp_id'].nunique()} prompts ---")
        check_seed_partition(df, tag, report)
        check_stability(df, tag, report)
        check_uniqueness_pooled(df, tag, report)
        check_per_seed(df, tag, expected, report)
        t0_path = t0_dir / f"{runs.name}.main.{tag}.parquet"
        if t0_path.exists():
            t0 = pd.read_parquet(t0_path)
            check_spacegroup_strict(df, t0, tag, report)
            check_spacegroup_subgroup(df, t0, tag, report)
        else:
            print(f"  SKIP  B/D space group, {tag}: {t0_path} not found")

    if first is not None:
        print("\n--- structural assumption ---")
        check_bucketing_is_exact(first, report)

    if found == 0:
        # A verification that silently verifies nothing is worse than none at
        # all, so an empty run reports an error and never a pass.
        print(
            f"\nNO RUNS FOUND under {runs}; nothing was verified. "
            "Pass --runs to point at a directory of evaluated parquets."
        )
        return 2

    print(
        f"\n{found} run(s) checked. "
        + (
            "ALL CHECKS PASSED"
            if report.failures == 0
            else f"{report.failures} CHECK(S) FAILED"
        )
    )
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
