"""Build a stratified per-space-group generation split for the 7 conditioning targets."""
import json, pickle, sys, collections
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "scripts")
from parquet_to_shelve import restore_record
from crysreas.metric_process.helpers import return_gt_dict
from pymatgen.symmetry.groups import SpaceGroup

TARGETS = ["P1", "C2/c", "Amm2", "I4m2", "P3", "P6_3/mmc", "F-43m"]
MANUAL = {"I4m2": 119}
SG_IDS = {s: (MANUAL[s] if s in MANUAL else int(SpaceGroup(s).int_number)) for s in TARGETS}
WANT = set(SG_IDS.values())
N_PROMPTS = 63          # x16 samples = 1008 structures per space group

held_out = set(json.load(open("assets/MP/split_cdvae.json"))["test"])
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
cache = Path(str(DATA_DIR / "sg_scan.pkl"))
if cache.exists():
    by_sg = pickle.load(open(cache, "rb"))
else:
    by_sg = collections.defaultdict(list)
    f = pq.ParquetFile("assets/MP/MP_shelve.parquet")
    seen = 0
    for batch in f.iter_batches(batch_size=4000):
        for row in batch.to_pandas().to_dict(orient="records"):
            k, r = restore_record(row)
            if k not in held_out:
                continue
            seen += 1
            try:
                spg = return_gt_dict(r, "conditional")["spg"]
            except Exception:
                continue
            if spg in WANT:
                by_sg[spg].append(k)
    print("scanned held-out materials:", seen, file=sys.stderr)
    pickle.dump(dict(by_sg), open(cache, "wb"))

out = {}
for sym, num in SG_IDS.items():
    keys = sorted(by_sg.get(num, []))
    sel = keys[:N_PROMPTS]
    out[sym] = sel
    print(f"{sym:10s} (No. {num:3d}): {len(keys):5d} available -> using {len(sel)} prompts "
          f"= {len(sel) * 16} structures")

Path(str(DATA_DIR / "splits")).mkdir(parents=True, exist_ok=True)
for sym, keys in out.items():
    safe = sym.replace("/", "_").replace("-", "m")
    json.dump({"train": [], "test": keys, "val": []},
              open(DATA_DIR / "splits" / f"split_sg_{safe}.json", "w"))
print("wrote", len(out), "per-space-group splits to", DATA_DIR / "splits")
