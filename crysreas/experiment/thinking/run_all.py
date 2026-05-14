"""
Run all modular parts for EXPERIMENT.md §1.2–§1.6.

Dependencies:
- part_1_2_entropy.py
- part_1_3_consistency.py
- part_1_4_recovery.py
- part_1_5_compare_models.py
- part_1_6_compare_information.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ComparePaths, THINKING_DATA_DIR, THINKING_FIGURE_DIR
from .part_1_2_entropy import run as run_1_2
from .part_1_3_consistency import run as run_1_3
from .part_1_4_recovery import run as run_1_4
from .part_1_5_compare_models import run as run_1_5
from .part_1_6_compare_information import run as run_1_6


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all parts: 1.2, 1.3, 1.4, 1.5, 1.6")
    parser.add_argument("--thinking", type=str, default=str(ComparePaths().thinking_parquet))
    parser.add_argument("--no-thinking", type=str, default=str(ComparePaths().no_thinking_parquet))
    parser.add_argument("--data-dir", type=str, default=str(THINKING_DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(THINKING_FIGURE_DIR))
    parser.add_argument("--batch-size", type=int, default=2, help="batch size for part 1.4 scoring")
    parser.add_argument("--max-rows", type=int, default=None, help="optional row cap for part 1.4 debugging")
    args = parser.parse_args()

    paths = ComparePaths(
        thinking_parquet=Path(args.thinking),
        no_thinking_parquet=Path(args.no_thinking),
        data_dir=Path(args.data_dir),
        figure_dir=Path(args.figure_dir),
    )
    run_1_2(paths)
    run_1_3(paths)
    run_1_4(paths, batch_size=args.batch_size, max_rows=args.max_rows)
    run_1_5(paths)
    run_1_6(paths, batch_size=args.batch_size, max_rows=args.max_rows)
    print("All done: 1.2 + 1.3 + 1.4 + 1.5 + 1.6")


if __name__ == "__main__":
    main()
