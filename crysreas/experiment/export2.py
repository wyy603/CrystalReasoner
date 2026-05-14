from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _EXPERIMENT_DIR


@dataclass
class FileSpec:
    input_path: Path
    output_path: Path
    target_rows: int


@dataclass
class Args:
    files: tuple[FileSpec, ...] = (
        FileSpec(
            input_path=Path("sample_1024.parquet"),
            output_path=_OUTPUT_DIR / "sample_512.csv",
            target_rows=512,
        ),
        FileSpec(
            input_path=Path("sample_512_elastic.parquet"),
            output_path=_OUTPUT_DIR / "sample_256_elastic.csv",
            target_rows=256,
        ),
    )
    converged_col: str = "converged"
    mp_id_col: str = "mp_id"
    model_name_col: str = "model_name"


def _ordered_unique(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(values.astype(str).tolist()))


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.fillna(0).astype(int).astype(bool)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )


def _valid_mp_ids(df: pd.DataFrame, mp_id_col: str, converged_col: str, model_name_col: str) -> set[str]:
    if mp_id_col not in df.columns:
        raise ValueError(f"Missing required column: {mp_id_col}")
    if converged_col not in df.columns:
        raise ValueError(
            f"Missing required column: {converged_col}. "
            f"Please run run_metric and save back to parquet first."
        )
    if model_name_col not in df.columns:
        raise ValueError(f"Missing required column: {model_name_col}")

    expected_models = df[model_name_col].astype(str).nunique()
    g = df.groupby(df[mp_id_col].astype(str), sort=False)
    has_all_models = g[model_name_col].nunique() == expected_models
    all_converged_true = g[converged_col].apply(lambda s: _to_bool_series(s).all())
    mask = has_all_models & all_converged_true
    return set(mask[mask].index.tolist())


def main(args: Args) -> None:
    if len(args.files) < 1:
        raise ValueError("files must contain at least one FileSpec.")

    for spec in args.files:
        df = pd.read_parquet(spec.input_path)

        before_rows = len(df)
        before_ids = df[args.mp_id_col].astype(str).nunique()

        valid_ids = _valid_mp_ids(
            df,
            mp_id_col=args.mp_id_col,
            converged_col=args.converged_col,
            model_name_col=args.model_name_col,
        )

        ordered_mp_ids = _ordered_unique(df[args.mp_id_col])
        kept_mp_ids: list[str] = []
        kept_rows = 0

        g = df.groupby(df[args.mp_id_col].astype(str), sort=False)
        for mp_id in ordered_mp_ids:
            if mp_id not in valid_ids:
                continue
            try:
                n_rows = len(g.get_group(str(mp_id)))
            except KeyError:
                continue
            if kept_rows + n_rows > spec.target_rows:
                break
            kept_mp_ids.append(str(mp_id))
            kept_rows += n_rows

        filtered = df[df[args.mp_id_col].astype(str).isin(set(kept_mp_ids))].copy()
        after_rows = len(filtered)
        after_ids = filtered[args.mp_id_col].astype(str).nunique()

        if after_rows != kept_rows:
            raise RuntimeError(
                f"Internal error: expected kept_rows={kept_rows}, got after_rows={after_rows} for {spec.input_path}"
            )

        output_path = spec.output_path
        if not output_path.is_absolute():
            output_path = _OUTPUT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(output_path, index=False)
        print(
            f"{spec.input_path} -> {output_path} | kept_mp_ids={after_ids}/{before_ids} | kept_rows={after_rows}/{before_rows}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
