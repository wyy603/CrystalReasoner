"""
MLIP-backed computation (MatterSim relaxation, hull energy, etc.).

Shared lazy singletons live in ``crysreas.mlip.models`` (one load per process).
Migrated from ``crysreas.metric.metrics`` mattergen globals — keep behavior aligned.
"""

from crysreas.mlip.models import get_potential, get_reference

__all__ = ["get_potential", "get_reference"]
