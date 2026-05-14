"""Scalar rewards derived from energy_above_hull (from ``crysreas.metric.metrics``)."""

from __future__ import annotations

import math


def energy_reward_scalar(energy: float | None) -> float:
    if energy is None or (isinstance(energy, float) and math.isnan(energy)):
        return 0.0
    if energy <= 1.0:
        return 1.0 - 0.5 * energy
    return 0.5 / energy


def smooth_energy_reward_scalar(energy: float | None, composition_consistent: bool) -> float:
    if not composition_consistent:
        return -1.0
    if energy is None or (isinstance(energy, float) and math.isnan(energy)):
        return -1.0
    if energy < 0.036:
        return 1.0 - (energy / 0.036) * 0.1
    if energy < 0.1:
        return 0.7 - ((energy - 0.036) / (0.1 - 0.036)) * 0.3
    return 0.4 * math.exp(-2.0 * (energy - 0.1))
