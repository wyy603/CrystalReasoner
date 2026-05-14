"""
Run EXPERIMENT.md §1.2 (entropy): backend (data) then frontend (figures).

Dependencies:
- part_1_2_backend, part_1_2_frontend
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import ComparePaths, DEFAULT_TOKENIZER_CHECKPOINT, THINKING_DATA_DIR, THINKING_FIGURE_DIR
from .part_1_2_backend import run as run_backend
from .part_1_2_frontend import run as run_frontend


def run(paths: ComparePaths, *, tokenizer_checkpoint: Path | None = None) -> None:
    run_backend(paths, tokenizer_checkpoint=tokenizer_checkpoint)
    run_frontend(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.2 (backend + frontend).")
    parser.add_argument("--thinking", type=str, default=str(ComparePaths().thinking_parquet))
    parser.add_argument("--no-thinking", type=str, default=str(ComparePaths().no_thinking_parquet))
    parser.add_argument("--data-dir", type=str, default=str(THINKING_DATA_DIR))
    parser.add_argument("--figure-dir", type=str, default=str(THINKING_FIGURE_DIR))
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default=str(DEFAULT_TOKENIZER_CHECKPOINT),
        help="HF checkpoint directory containing tokenizer files.",
    )
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
            figure_dir=Path(args.figure_dir),
        ),
        tokenizer_checkpoint=Path(args.tokenizer_checkpoint),
    )


if __name__ == "__main__":
    main()
