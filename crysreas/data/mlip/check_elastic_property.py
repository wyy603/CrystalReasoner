
import json
import shelve
from pathlib import Path
from tqdm import tqdm
from crysreas import Config

def main():
    """
    Checks if the keys in split_elastic.json have been correctly updated in the MP_shelve database.

    An entry is considered "updated" if it's a dictionary containing 'bulk_modulus' and 
    'shear_modulus' as float values.
    """
    # Define paths relative to the script's location in assets/MP/
    script_dir = Config.DATA_PATH
    json_path = script_dir / "split_elastic.json"
    db_path = script_dir / "MP_shelve"

    # --- 1. Validate that necessary files exist ---
    if not json_path.exists():
        print(f"Error: JSON config file 'split_elastic.json' not found at: {json_path}")
        return

    # Shelve creates multiple files, we check for the common .db extension
    # Note: Depending on the system, shelve might create .bak, .dat, .dir
    db_files_exist = any(Path(str(db_path) + ext).exists() for ext in ['.db', '.dat', '.dir'])
    if not db_files_exist:
        print(f"Error: Database file 'MP_shelve' not found at: {db_path}")
        print("Please run pkl_to_shelve.py first to generate the database.")
        return

    # --- 2. Load all keys from the JSON split file ---
    print(f"Loading keys from {json_path}...")
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    keys_to_check = (
        split_data.get('train', []) + 
        split_data.get('val', []) + 
        split_data.get('test', [])
    )

    if not keys_to_check:
        print("Warning: No keys found in 'split_elastic.json'.")
        return

    total_keys = len(keys_to_check)
    print(f"Found {total_keys} total keys to check.")

    # --- 3. Open the database and check each key ---
    updated_count = 0
    missing_count = 0
    malformed_count = 0
    malformed_details = []

    print(f"Opening database {db_path} and starting validation...")
    with shelve.open(str(db_path), flag='r') as db:
        for key in tqdm(keys_to_check, desc="Validating database entries"):
            if key not in db:
                missing_count += 1
                if len(malformed_details) < 10: # Log first few examples
                    malformed_details.append(f"{key}: Missing from database")
                continue

            entry = db[key]
            
            # Check conditions for "updated"
            is_dict = isinstance(entry, dict)
            has_bulk = 'bulk_modulus' in entry if is_dict else False
            has_shear = 'shear_modulus' in entry if is_dict else False
            is_bulk_float = isinstance(entry.get('bulk_modulus'), float) if has_bulk else False
            is_shear_float = isinstance(entry.get('shear_modulus'), float) if has_shear else False

            if is_dict and has_bulk and has_shear and is_bulk_float and is_shear_float:
                updated_count += 1
            else:
                malformed_count += 1
                if len(malformed_details) < 10:
                    reason = []
                    if not is_dict: reason.append(f"Not a dict (type is {type(entry).__name__})")
                    else:
                        if not has_bulk: reason.append("Missing 'bulk_modulus'")
                        elif not is_bulk_float: reason.append("'bulk_modulus' is not a float")
                        if not has_shear: reason.append("Missing 'shear_modulus'")
                        elif not is_shear_float: reason.append("'shear_modulus' is not a float")
                    malformed_details.append(f"{key}: Malformed ({', '.join(reason)})")

    # --- 4. Print the final report ---
    print("\n--- Validation Report ---")
    print(f"Total Keys from JSON: {total_keys}")
    print(f"✅ Correctly Updated Entries: {updated_count}")
    print(f"❌ Missing Entries in DB: {missing_count}")
    print(f"⚠️ Malformed or Outdated Entries: {malformed_count}")
    print("------------------------\n")

    if malformed_count > 0 or missing_count > 0:
        print("Issues found. Here are some examples of malformed/missing keys:")
        for detail in malformed_details:
            print(f"- {detail}")
    else:
        print("🎉 All keys from split_elastic.json have been correctly updated in the database.")

if __name__ == "__main__":
    main()
