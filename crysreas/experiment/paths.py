"""Canonical locations for experiment code and data.

Everything the experiments read or write lives under this package, so a script
never has to guess where the repository root is or reach outside it. Import
these constants instead of rebuilding paths from ``__file__``: the depth of a
module inside the package is not a stable thing to depend on.
"""

from __future__ import annotations

from pathlib import Path

#: ``crysreas/experiment``
EXPERIMENT_DIR = Path(__file__).resolve().parent

#: Evaluation artifacts: generated structures, relaxations, per-row metric
#: columns, aggregated tables and figures. Versioned through Git LFS, see the
#: repository ``.gitattributes``.
DATA_DIR = EXPERIMENT_DIR / "data"

#: Repository root, for the few places that need the training assets.
ROOT = EXPERIMENT_DIR.parent.parent

FIGURES_DIR = DATA_DIR / "figures"
TABLES_DIR = DATA_DIR / "tables"
