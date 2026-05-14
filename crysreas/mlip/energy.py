"""
Energy-above-hull from relaxed structures (MatterGen MetricsEvaluator).

Migrated from ``crysreas.metric.metrics.GetEnergy.calculate_batch``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import Structure
import ray
from crysreas.mlip.models import get_reference

@ray.remote(num_gpus=0.5)
def compute_energy_above_hull_batch(
    structures: list[Any],
    relaxed_structures: list[Any],
    energies: list[Any],
    reference: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row energy above hull and stability flag (``is_stable``: E_hull < 0.1 eV/atom).

    Rows with ``relaxed_structures[i] is None`` stay NaN / False.
    """
    from mattergen.evaluation.metrics.energy import AvgEnergyAboveHullPerAtom
    from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
    from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher

    reference = reference or get_reference()
    n = len(structures)
    energy_above_hull = np.full((n,), float("nan"))
    is_stable = np.full((n,), False, dtype=np.bool_)

    success_indices = [i for i, s in enumerate(relaxed_structures) if s is not None]
    rs = [relaxed_structures[i] for i in success_indices]
    es = [energies[i] for i in success_indices]

    if len(rs) > 0:
        matcher = DefaultDisorderedStructureMatcher()
        evaluator = MetricsEvaluator.from_structures_and_energies(
            structures=rs,
            energies=es,
            original_structures=structures,
            reference=reference,
            structure_matcher=matcher,
        )
        try:
            metric: AvgEnergyAboveHullPerAtom = evaluator._get_metric(AvgEnergyAboveHullPerAtom)
            energy_above_hull[success_indices] = metric.energy_capability.energy_above_hull
            is_stable[success_indices] = energy_above_hull[success_indices] < 0.1
        except Exception:
            for j, i in enumerate(success_indices):
                try:
                    ev_i = MetricsEvaluator.from_structures_and_energies(
                        structures=[rs[j]],
                        energies=[es[j]],
                        original_structures=[structures[i]],
                        reference=reference,
                        structure_matcher=matcher,
                    )
                    m_i: AvgEnergyAboveHullPerAtom = ev_i._get_metric(AvgEnergyAboveHullPerAtom)
                    hull = np.asarray(
                        m_i.energy_capability.energy_above_hull, dtype=float
                    ).reshape(-1)
                    energy_above_hull[i] = float(hull[0])
                    is_stable[i] = energy_above_hull[i] < 0.1
                except Exception:
                    energy_above_hull[i] = None
                    is_stable[i] = False

    return energy_above_hull, is_stable
