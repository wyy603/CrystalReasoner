import json
import random
from pathlib import Path

import pandas as pd

from crysreas import Config
from crysreas.utils.crystal import SimpleCrystal


def read_split(path):
    if isinstance(path, str) and not path.startswith(str(Config.DATA_PATH)):
        path = Config.DATA_PATH / path
    path = Path(path)
    with open(path, "r") as f:
        split = json.load(f)
    return split


def _resolve_data_path(path) -> Path:
    """Relative paths are resolved under Config.DATA_PATH."""
    p = Path(path)
    if not p.is_absolute() and not str(p).startswith(str(Config.DATA_PATH)):
        p = Config.DATA_PATH / p
    return p


def mask_few_atom_sites(df: pd.DataFrame, max_atoms: int) -> pd.Series:
    """Row-aligned boolean mask: True when simple_structure has fewer than max_atoms sites."""
    out = []
    for _, row in df.iterrows():
        try:
            n = SimpleCrystal.from_simple_no_sym(row["simple_structure"]).structure.num_sites
        except Exception:
            n = float("inf")
        out.append(n < max_atoms)
    return pd.Series(out, index=df.index)


def gen_split(valid_mask, new_split_path, from_split: dict, df: pd.DataFrame):
    """
    valid_mask: boolean Series aligned with df rows (same index as df).
    New split = each key in from_split, keeping mp_ids that appear in df rows selected by valid_mask.
    """
    new_split_path = _resolve_data_path(new_split_path)
    if new_split_path.exists():
        with open(new_split_path, "r") as f:
            return json.load(f)

    valid_mp_ids = set(df.loc[valid_mask, "mp_id"].astype(str))
    split = {}
    for key in ["train", "test", "val"]:
        split[key] = [mp_id for mp_id in from_split.get(key, []) if str(mp_id) in valid_mp_ids]
    with open(new_split_path, "w") as f:
        json.dump(split, f, indent=4)
    print(new_split_path, len(split["train"]), len(split["test"]), len(split["val"]))
    return split


def sub_split(split, path, train_num=8000, test_num=512):
    split = split.copy()
    path = _resolve_data_path(path)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)

    if len(split["train"]) < train_num:
        if train_num > 0:
            print(f"Warning: Only {len(split['train'])} valid train samples remaining, less than {train_num}. Using all valid samples.")
        train = split["train"]
    else:
        train = random.sample(split["train"], train_num)

    if len(split["test"]) < test_num:
        if test_num > 0:
            print(f"Warning: Only {len(split['test'])} valid test samples remaining, less than {test_num}. Using all valid samples.")
        test = split["test"]
    else:
        test = random.sample(split["test"], test_num)

    val = split["val"]

    new_split = {
        "train": train,
        "test": test,
        "val": val,
    }
    with open(path, "w") as f:
        json.dump(new_split, f, indent=4)
    print(path, len(new_split["train"]), len(new_split["test"]), len(new_split["val"]))
    return new_split


if __name__ == "__main__":
    # These json are generated from other python code
    split_cdvae_metric = pd.read_parquet(Config.DATA_PATH / "split_cdvae_metric.parquet")
    split_cdvae = read_split("split_cdvae.json")
    split_small_atoms_metric = pd.read_parquet(Config.DATA_PATH / "split_small_atoms_metric.parquet")

    random.seed(42)
    # Generate full split (structure validity & smact_validity)
    mask_valid = (split_cdvae_metric["structure_validity"] == True) & (split_cdvae_metric["smact_validity"] == True)
    split_full = gen_split(mask_valid, "split_full.json", from_split=split_cdvae, df=split_cdvae_metric)
    # split_generation = sub_split(split_cdvae, "split_generation.json", train_num=0, test_num=1024)

    # Generate full split for elastic reward (bulk_modulus available)
    split_elastic_full = gen_split(
        split_cdvae_metric["elastic_properties"].notna(),
        "split_elastic_full.json",
        from_split=split_cdvae,
        df=split_cdvae_metric,
    )

    # Generate 
    split = sub_split(split_full, "split.json", train_num=8000, test_num=512)
    split_elastic = sub_split(split_elastic_full, "split_elastic.json", train_num=4000, test_num=256)
    split_generation_elastic = sub_split(split_elastic_full, "split_generation_elastic.json", train_num=0, test_num=512)

    random.seed(42)
    mask_small = mask_few_atom_sites(split_cdvae_metric, max_atoms=10)
    split_small_atoms = gen_split(
        mask_valid & mask_small,
        "split_small_atoms.json",
        from_split=split_cdvae,
        df=split_cdvae_metric,
    )

    random.seed(42)
    mask_cte = (
        split_small_atoms_metric["cte"].apply(lambda x: isinstance(x, dict))
        & (split_small_atoms_metric["structure_validity"] == True)
        & (split_small_atoms_metric["smact_validity"] == True)
    )
    split_cte_full = gen_split(
        mask_cte,
        "split_cte_full.json",
        from_split=split_full,
        df=split_small_atoms_metric,
    )
    split_cte = sub_split(split_cte_full, "split_cte.json", train_num=4000, test_num=256)
    # Generation split for CTE tasks: directly sampled from split_cte.json (test=256).
    sub_split(split_cte, "split_generation_cte.json", train_num=0, test_num=256)
