from __future__ import annotations

import argparse
import json
import shelve
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from crysreas.utils.crystal import SimpleCrystal


DROP_COLUMNS = frozenset(("condensed_structure", "decomposes_to"))
JSON_STRING_COLUMNS = frozenset(("bulk_modulus", "shear_modulus"))
DEFAULT_SOURCE = Path("assets/MP/MP_shelve")
DEFAULT_OUTPUT = Path("assets/MP/MP_shelve.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Materials Project shelve database to Parquet."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def parquet_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [parquet_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        safe_dict = {str(key): parquet_safe(item) for key, item in value.items()}
        return json.dumps(safe_dict, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        safe_list = [parquet_safe(item) for item in value]
        return json.dumps(safe_list, ensure_ascii=False)
    return str(value)


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in record.items():
        if key in DROP_COLUMNS:
            continue
        if key == "structure":
            converted[key] = SimpleCrystal(value).to_simple_no_sym()
        elif key in JSON_STRING_COLUMNS:
            safe_value = parquet_safe(value)
            converted[key] = None if is_missing(safe_value) else str(safe_value)
        else:
            converted[key] = parquet_safe(value)
    return converted


def write_batch(
    rows: list[dict[str, Any]],
    columns: list[str],
    writer: pq.ParquetWriter | None,
    output: Path,
) -> pq.ParquetWriter:
    df = pd.DataFrame.from_records(rows, columns=columns)
    df = df.astype(object)
    for column in df.columns:
        df[column] = df[column].map(lambda value: None if is_missing(value) else value)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output, table.schema)
    else:
        table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def collect_columns(source: Path) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    with shelve.open(str(source), flag="r") as db:
        for key in tqdm(db.keys(), desc="Collecting columns"):
            for column in db[key].keys():
                if column in DROP_COLUMNS or column in seen:
                    continue
                seen.add(column)
                columns.append(column)
    return columns


def convert_shelve(source: Path, output: Path, batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()

    columns = collect_columns(source)
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    try:
        with shelve.open(str(source), flag="r") as db:
            for key in tqdm(db.keys(), total=len(db), desc="Converting rows"):
                rows.append(convert_record(db[key]))
                if len(rows) >= batch_size:
                    writer = write_batch(rows, columns, writer, temp_output)
                    rows.clear()
            if rows:
                writer = write_batch(rows, columns, writer, temp_output)
    finally:
        if writer is not None:
            writer.close()

    temp_output.replace(output)


def main() -> None:
    args = parse_args()
    convert_shelve(args.source, args.output, args.batch_size)


if __name__ == "__main__":
    main()
