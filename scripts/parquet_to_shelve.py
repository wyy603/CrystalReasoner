from __future__ import annotations

import argparse
import json
import re
import shelve
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from crysreas.utils.crystal import SimpleCrystal

try:
    from emmet.core.mpid import MPID
except Exception:  # pragma: no cover - optional compatibility fallback
    MPID = None


DEFAULT_SOURCE = Path("assets/MP/MP_shelve.parquet")
DEFAULT_OUTPUT = Path("assets/MP/MP_shelve_recreated")
JSON_STRING_COLUMNS = frozenset(("bulk_modulus", "shear_modulus"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate a Materials Project shelve database from Parquet."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows; useful for dry-runs.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write the shelve database. By default this script dry-runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to an existing shelve path when --write is set.",
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def normalize_mp_id(value: Any) -> str:
    text = str(value)
    match = re.fullmatch(r"MPID\(([^)]+)\)", text)
    if match:
        return match.group(1)
    return text


def restore_material_id(value: Any) -> Any:
    mp_id = normalize_mp_id(value)
    if MPID is None:
        return mp_id
    try:
        return MPID(mp_id)
    except Exception:
        return mp_id


def restore_json_string(value: Any) -> Any:
    if is_missing(value):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return float(value)
        except ValueError:
            return value


def restore_scalar(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def restore_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "material_id" not in row:
        raise KeyError("Parquet row is missing material_id")
    if "structure" not in row:
        raise KeyError("Parquet row is missing structure")

    mp_id = normalize_mp_id(row["material_id"])
    restored: dict[str, Any] = {}
    for key, value in row.items():
        if key == "structure":
            restored[key] = SimpleCrystal.from_simple_no_sym(str(value)).structure
        elif key == "material_id":
            restored[key] = restore_material_id(value)
        elif key in JSON_STRING_COLUMNS:
            restored[key] = restore_json_string(value)
        else:
            restored[key] = restore_scalar(value)
    return mp_id, restored


def output_exists(output: Path) -> bool:
    return any(output.parent.glob(output.name + "*"))


def iter_rows(source: Path, batch_size: int):
    parquet_file = pq.ParquetFile(source)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for row in df.to_dict(orient="records"):
            yield row


def convert_parquet_to_shelve(
    source: Path,
    output: Path,
    batch_size: int,
    limit: int | None,
    write: bool,
    overwrite: bool,
) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if write and output_exists(output) and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    row_iter = iter_rows(source, batch_size)
    progress = tqdm(total=limit, desc="Restoring rows")
    examples: list[tuple[str, dict[str, Any]]] = []
    processed = 0

    db = shelve.open(str(output), flag="n") if write else None
    try:
        for row in row_iter:
            key, record = restore_record(row)
            if db is not None:
                db[key] = record
            if len(examples) < 3:
                examples.append((key, record))
            processed += 1
            progress.update(1)
            if limit is not None and processed >= limit:
                break
    finally:
        progress.close()
        if db is not None:
            db.close()

    mode = "Wrote" if write else "Dry-run restored"
    print(f"{mode} {processed} rows from {source}.")
    if not write:
        print(f"No shelve files were created. Target path would be: {output}")
    for key, record in examples:
        structure = record["structure"]
        print(
            f"example key={key} material_id={record['material_id']} "
            f"structure_type={type(structure).__name__} sites={len(structure)}"
        )


def main() -> None:
    args = parse_args()
    convert_parquet_to_shelve(
        source=args.source,
        output=args.output,
        batch_size=args.batch_size,
        limit=args.limit,
        write=args.write,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
