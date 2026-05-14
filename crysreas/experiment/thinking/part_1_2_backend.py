"""
Backend for EXPERIMENT §1.2: compute per-row metrics and write tables only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    ComparePaths,
    DEFAULT_TOKENIZER_CHECKPOINT,
    count_atoms,
    count_thinking_tokens,
    ensure_thinking_dirs,
    load_frames,
    load_thinking_tokenizer,
    split_thinking_and_tail,
)

DATA_1_2_PER_ROW = "thinking_exp_1_2_per_row.parquet"


def run(paths: ComparePaths, *, tokenizer_checkpoint: Path | None = None) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir)
    df_think, _ = load_frames(paths)

    tok = load_thinking_tokenizer(tokenizer_checkpoint)
    thinking, _ = zip(*[split_thinking_and_tail(x) for x in df_think["responses"].tolist()], strict=True)
    n_tokens = np.array([count_thinking_tokens(tok, t) for t in thinking], dtype=float)
    n_atoms = np.array([count_atoms(s) for s in df_think["simple_structure"].tolist()], dtype=float)

    per_row = pd.DataFrame(
        {
            "mp_id": df_think["mp_id"].astype(str).to_numpy(),
            "n_atoms": n_atoms,
            "thinking_tokens": n_tokens,
        }
    )
    out = paths.data_dir / DATA_1_2_PER_ROW
    per_row.to_parquet(out, index=False)
    print(f"1.2 backend wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.2 backend (data only).")
    parser.add_argument("--thinking", type=str, default=str(ComparePaths().thinking_parquet))
    parser.add_argument("--no-thinking", type=str, default=str(ComparePaths().no_thinking_parquet))
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
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
        ),
        tokenizer_checkpoint=Path(args.tokenizer_checkpoint),
    )


if __name__ == "__main__":
    main()
