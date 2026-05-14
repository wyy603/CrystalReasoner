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
