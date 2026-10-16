"""Stable / unique / novel from MatterGen evaluator (``GetSUN``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from crysreas.mlip.models import get_reference


def coerce_bool_array(values: Any, n: int, name: str) -> np.ndarray:
    """Convert a nullable/object metric column to a conservative bool mask."""

    arr = np.asarray(list(values) if not np.isscalar(values) else [values], dtype=object)
    if len(arr) != n:
        raise ValueError(f"{name} length mismatch: expected {n}, got {len(arr)}")

    out = np.full((n,), False, dtype=np.bool_)
    for i, value in enumerate(arr):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)):
            out[i] = bool(value)
            continue
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "1", "yes", "y"}:
                out[i] = True
            elif token in {"false", "0", "no", "n", "none", "nan", ""}:
                out[i] = False
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            continue
        out[i] = bool(value)
    return out


def compute_stable_unique_novel_batch(
    structures: list[Any],
    relaxed_structures: list[Any],
    energies: list[Any],
    is_stable: Any,
    reference: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (is_novel, is_unique, stable_unique_novel) length-n arrays.
    ``stable_unique_novel`` matches metrics.GetSUN: unique & novel & is_stable.
    """
    from mattergen.evaluation.metrics.energy import FracNovelUniqueStableStructures
    from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
    from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher

    reference = reference or get_reference()
    n = len(structures)
    is_stable = coerce_bool_array(is_stable, n, "is_stable")
    is_novel = np.full((n,), False, dtype=np.bool_)
    is_unique = np.full((n,), False, dtype=np.bool_)

    success_indices = [i for i, s in enumerate(relaxed_structures) if s is not None]
    rs = [relaxed_structures[i] for i in success_indices]
    es = [energies[i] for i in success_indices]

    if len(rs) > 0:
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=rs,
            energies=es,
            original_structures=structures,
            reference=reference,
            structure_matcher=DefaultDisorderedStructureMatcher(),
        )
        metric: FracNovelUniqueStableStructures = evaluator._get_metric(
            FracNovelUniqueStableStructures
        )
        is_novel[success_indices] = metric.structure_capability.is_novel
        is_unique[success_indices] = metric.structure_capability.is_unique

    sun = is_unique & is_novel & is_stable
    return is_novel, is_unique, sun


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------
# MatterGen's uniqueness is a property of a *set*: structures are grouped by
# reduced formula and, within a group, the first member of each matching class
# is kept and the rest are dropped. Its matcher returns False immediately when
# two ordered structures have different reduced formulas
# (``mattergen.evaluation.utils.structure_matcher.DisorderedStructureMatcher.fit``),
# so grouping by reduced formula is not an approximation: it is exactly an
# all-pairs comparison over the scored set, computed in O(sum n_i^2) instead of
# O(n^2). ``crysreas.experiment.verify_equivalence`` checks that claim directly.
#
# What changes with the protocol is *which* set is scored. Under
# ``uniqueness_scope="pooled"`` the whole table is one set; because every prompt
# pins one held-out composition, every comparison then falls between samples of
# a single composition and the metric rewards a model for answering the same
# composition differently each time. Under ``uniqueness_scope="seed"`` each seed
# is scored on its own; a seed holds exactly one structure per composition, so a
# duplicate can only arise between two *different* requested compositions, i.e.
# when the model returned the same structure whatever it was asked for.


def _reduced_formula(structure: Any) -> str | None:
    try:
        return structure.composition.reduced_formula
    except Exception:
        return None


def uniqueness_mask(
    structures: list[Any],
    matcher: Any | None = None,
) -> np.ndarray:
    """Greedy first-wins uniqueness over one scored set.

    ``structures[i] is None`` counts as not unique, matching the pipeline's
    treatment of a failed relaxation: the row stays in the denominator.
    """
    from mattergen.evaluation.utils.dataset_matcher import get_unique
    from mattergen.evaluation.utils.structure_matcher import (
        DefaultDisorderedStructureMatcher,
    )

    matcher = matcher or DefaultDisorderedStructureMatcher()
    n = len(structures)
    out = np.full((n,), False, dtype=np.bool_)

    groups: dict[str, list[int]] = {}
    for i, s in enumerate(structures):
        if s is None:
            continue
        key = _reduced_formula(s)
        if key is None:
            continue
        groups.setdefault(key, []).append(i)

    for members in groups.values():
        kept = get_unique(matcher, [structures[i] for i in members])
        for k in kept:
            out[members[k]] = True
    return out


def compute_uniqueness(
    structures: list[Any],
    seeds: Any | None = None,
    scope: str = "seed",
    matcher: Any | None = None,
) -> np.ndarray:
    """Uniqueness under the configured scope.

    ``scope="seed"`` scores each seed separately and requires ``seeds``;
    ``scope="pooled"`` scores the whole input as one set.
    """
    if scope == "pooled":
        return uniqueness_mask(structures, matcher)
    if scope != "seed":
        raise ValueError(f"unknown uniqueness_scope: {scope!r}")
    if seeds is None:
        raise ValueError("uniqueness_scope='seed' requires a seed column")

    from mattergen.evaluation.utils.structure_matcher import (
        DefaultDisorderedStructureMatcher,
    )

    matcher = matcher or DefaultDisorderedStructureMatcher()
    seed_arr = np.asarray(seeds)
    out = np.full((len(structures),), False, dtype=np.bool_)
    for seed in np.unique(seed_arr):
        rows = np.flatnonzero(seed_arr == seed)
        sub = uniqueness_mask([structures[i] for i in rows], matcher)
        out[rows[sub]] = True
    return out


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def build_training_reference(split_path: Any = None, shelve_path: Any = None) -> dict:
    """Reduced formula -> list of training structures, for novelty-vs-train.

    Every evaluation prompt fixes a *known* Materials Project composition, so
    "absent from MP+Alexandria" is equivalent to "failed to produce the known
    phase" and the metric penalises accuracy. Measured against the training
    split instead, novelty asks the answerable question: did the model produce
    something it was not trained on?
    """
    import json
    import shelve

    from crysreas import Config

    split_path = split_path or (Config.DATA_PATH / "split_cdvae.json")
    shelve_path = shelve_path or (Config.DATA_PATH / "MP_shelve")
    ids = json.load(open(split_path))["train"]

    index: dict[str, list] = {}
    with shelve.open(str(shelve_path), flag="r") as db:
        for key in ids:
            try:
                structure = db[key]["structure"]
            except Exception:
                continue
            formula = _reduced_formula(structure)
            if formula is not None:
                index.setdefault(formula, []).append(structure)
    return index


def novelty_vs_training(
    structures: list[Any],
    reference: dict,
    matcher: Any | None = None,
) -> np.ndarray:
    """True when no training structure of the same reduced formula matches."""
    from mattergen.evaluation.utils.structure_matcher import (
        DefaultDisorderedStructureMatcher,
    )

    matcher = matcher or DefaultDisorderedStructureMatcher()
    out = np.full((len(structures),), False, dtype=np.bool_)
    for i, s in enumerate(structures):
        if s is None:
            continue
        candidates = reference.get(_reduced_formula(s) or "", ())
        out[i] = not any(matcher.fit(s, ref) for ref in candidates)
    return out
