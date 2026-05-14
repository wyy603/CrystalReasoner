"""Module ``crysreas.metric_process.config`` keeps ``set_config`` / ``get_config`` for tests.

The metric pipeline must not read that global state: callers pass ``config`` explicitly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from pymatgen.core import Lattice, Structure

from crysreas.metric_process import merge_metric_process_config, run_metrics
from crysreas.metric_process.config import get_config, set_config


def test_poisoned_global_does_not_affect_run_metrics_explicit_config() -> None:
    """``run_metrics(..., config=...)`` uses the passed dict, not ``get_config()``."""
    lattice = Lattice.cubic(5.43)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    elem = {
        "material_id": "mp-999",
        "structure": structure,
        "bulk_modulus": 100.0,
        "shear_modulus": 50.0,
    }
    mock_db = MagicMock()
    mock_db.__getitem__ = lambda self, k: elem

    class MockShelve:
        def __enter__(self):
            return mock_db

        def __exit__(self, *args):
            return False

    prev = get_config().copy()
    try:
        set_config({"prompt_type": ["conditional", "thinking"]})
        cfg = merge_metric_process_config(None, {"prompt_type": ["elastic"]})
        with patch("crysreas.metric_process.basic.shelve.open", return_value=MockShelve()):
            df = pd.DataFrame({"mp_id": ["mp-999"]})
            out = run_metrics(df, ["gt"], config=cfg)
        gt0 = out["gt"].iloc[0]
        assert "bulk_modulus" in gt0
    finally:
        set_config(prev)


def test_get_config_set_config_merge_semantics_unchanged() -> None:
    """``set_config`` still merges into module copy; ``get_config`` returns a snapshot."""
    prev = get_config().copy()
    try:
        set_config({"prompt_type": ["a", "b"]})
        g = get_config()
        assert g["prompt_type"] == ["a", "b"]
        set_config({"prompt_type": ["c"]})
        assert get_config()["prompt_type"] == ["c"]
    finally:
        set_config(prev)
