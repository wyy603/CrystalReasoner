"""Per-process lazy singletons for MatterGen / MatterSim evaluation (see metric.metrics)."""

from __future__ import annotations

from typing import Any

_reference: Any = None
_potential: Any = None


def get_reference():
    """ReferenceMP2020Correction for hull / novelty metrics (singleton per process)."""
    global _reference
    if _reference is None:
        from mattergen.evaluation.reference.presets import ReferenceMP2020Correction

        _reference = ReferenceMP2020Correction()
    return _reference


def get_potential(device: str | None = None, load_path: str | None = None):
    """MatterSim Potential for relaxation (singleton per process)."""
    global _potential
    if _potential is None:
        from mattergen.common.utils.globals import get_device as mattergen_get_device
        from mattersim.forcefield.potential import Potential

        dev = device if device is not None else str(mattergen_get_device())
        _potential = Potential.from_checkpoint(
            device=dev, load_path=load_path, load_training_state=False
        )
    return _potential


def reset_singletons_for_tests() -> None:
    """Clear cached model/reference (pytest only)."""
    global _reference, _potential
    _reference = None
    _potential = None
