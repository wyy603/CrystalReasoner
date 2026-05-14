"""
Run EXPERIMENT §1.4: backend (data) then frontend (figure).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ComparePaths, THINKING_DATA_DIR, THINKING_FIGURE_DIR
from .part_1_4_backend import run as run_backend
from .part_1_4_frontend import run as run_frontend


def run(paths: ComparePaths, *, batch_size: int = 2, max_rows: int | None = None) -> None:
    run_backend(paths, batch_size=batch_size, max_rows=max_rows)
    run_frontend(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.4 (backend + frontend).")
    parser.add_argument("--thinking", type=str, default="checkpoints_merged/thinking/conditional+thinking.parquet")
    parser.add_argument(
        "--no-thinking",
        type=str,
        default="checkpoints_merged/no_thinking/conditional+thinking.parquet",
    )
    parser.add_argument("--data-dir", type=str, default=str(THINKING_DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(THINKING_FIGURE_DIR))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
            figure_dir=Path(args.figure_dir),
        ),
        batch_size=int(args.batch_size),
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
