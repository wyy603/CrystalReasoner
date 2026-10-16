"""Scan the generation test split for prompts that specify only a subset of properties."""
import json, pickle, re, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from parquet_to_shelve import restore_record
from crysreas.data.prompt_generator import get_info

keys = set(json.load(open("assets/MP/split_generation.json"))["test"])
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
cache = Path(str(DATA_DIR / "test_records.pkl"))
if cache.exists():
    recs = pickle.load(open(cache, "rb"))
else:
    recs = {}
    f = pq.ParquetFile("assets/MP/MP_shelve.parquet")
    for batch in f.iter_batches(batch_size=4000):
        for row in batch.to_pandas().to_dict(orient="records"):
            k, r = restore_record(row)
            if k in keys:
                recs[k] = r
        if len(recs) == len(keys):
            break
    pickle.dump(recs, open(cache, "wb"))
print("records:", len(recs), file=sys.stderr)

SENT = re.compile(r"The ([a-z ]+?) is ([^.]*)\.")
rows = []
for k, r in recs.items():
    q = get_info(r, "conditional+thinking", seed=k[3:])["question"]
    body = q.split("Below is a description of a bulk material.")[1].split("Generate a description")[0]
    fields = [m.group(1).strip() for m in SENT.finditer(body)]
    struct = r.get("structure")
    n = len(struct) if struct is not None else -1
    rows.append((len(fields), n, k, fields, q))

rows.sort()
for nf, n, k, fields, q in rows:
    if nf <= 2 and 2 <= n <= 12:
        print(f"{k}\tnfields={nf}\tnatoms={n}\t{fields}")
        print(f"   {q.strip()}")
