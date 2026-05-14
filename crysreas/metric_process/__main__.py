"""Allow ``python -m crysreas.metric_process`` (CLI parity with ``crysreas.metric_process.cli``)."""

from __future__ import annotations

import tyro

from crysreas.metric_process.cli import main

if __name__ == "__main__":
    tyro.cli(main)
