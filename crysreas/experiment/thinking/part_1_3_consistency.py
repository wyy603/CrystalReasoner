"""
Run EXPERIMENT.md §1.3 (consistency): backend (data) then frontend (figures).

Dependencies:
- part_1_3_backend, part_1_3_frontend
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ComparePaths, THINKING_DATA_DIR, THINKING_FIGURE_DIR
from .part_1_3_backend import run as run_backend
from .part_1_3_frontend import run as run_frontend


def run(paths: ComparePaths) -> None:
    run_backend(paths)
    run_frontend(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.3 (backend + frontend).")
    parser.add_argument("--thinking", type=str, default=str(ComparePaths().thinking_parquet))
    parser.add_argument("--no-thinking", type=str, default=str(ComparePaths().no_thinking_parquet))
    parser.add_argument("--data-dir", type=str, default=str(THINKING_DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(THINKING_FIGURE_DIR))
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
            figure_dir=Path(args.figure_dir),
        ),
    )


if __name__ == "__main__":
    main()
