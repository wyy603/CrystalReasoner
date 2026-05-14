"""Verify that every mp_id in split_cte.json has thermal_expansion_300k in MP_shelve."""

import json
import numbers
import shelve
import sys
from pathlib import Path

from tqdm import tqdm

from crysreas import Config


def _is_numeric_scalar(x: object) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def main() -> int:
    json_path = Config.DATA_PATH / "split_cte.json"
    db_path = Config.DATA_PATH / "MP_shelve"

    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return 1

    db_files_exist = any(Path(str(db_path) + ext).exists() for ext in [".db", ".dat", ".dir"])
    if not db_files_exist:
        print(f"Error: Database MP_shelve not found at: {db_path}")
        return 1

    with open(json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    keys_to_check = (
        split_data.get("train", [])
        + split_data.get("val", [])
        + split_data.get("test", [])
    )

    if not keys_to_check:
        print("Warning: No mp_ids in split_cte.json.")
        return 1

    total = len(keys_to_check)
    ok = 0
    missing_db = 0
    bad = 0
    examples: list[str] = []

    with shelve.open(str(db_path), flag="r") as db:
        for key in tqdm(keys_to_check, desc="check_cte"):
            if key not in db:
                missing_db += 1
                if len(examples) < 10:
                    examples.append(f"{key}: missing from database")
                continue

            entry = db[key]
            if not isinstance(entry, dict):
                bad += 1
                if len(examples) < 10:
                    examples.append(f"{key}: entry is not a dict")
                continue

            val = entry.get("thermal_expansion_300k")
            if val is None or not _is_numeric_scalar(val):
                bad += 1
                if len(examples) < 10:
                    examples.append(
                        f"{key}: missing or invalid thermal_expansion_300k ({type(val).__name__})"
                    )
                continue

            ok += 1

    print("\n--- check_cte (thermal_expansion_300k) ---")
    print(f"Total mp_ids in split_cte.json: {total}")
    print(f"OK: {ok}")
    print(f"Missing in DB: {missing_db}")
    print(f"Missing or invalid thermal_expansion_300k: {bad}")
    print("-------------------------------------------\n")

    if examples:
        print("Examples:")
        for line in examples:
            print(f"  - {line}")

    if ok == total:
        print("All split_cte entries have thermal_expansion_300k in MP_shelve.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
