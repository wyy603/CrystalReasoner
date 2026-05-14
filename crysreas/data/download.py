import os
import json
import tyro
import shelve
import pickle
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
from mp_api.client import MPRester
from robocrys import StructureCondenser, StructureDescriber
from crysreas import Config

@dataclass
class Args:
    """Arguments for downloading material data from Materials Project."""
    db_path: str = str(Config.DATA_PATH / "MP_shelve")
    mp_ids: List[str] = None
    api_key: str = Config.API_KEY['MP']
    download_type: str = "robocrys" # "summary" or "robocrys"
    fields: List[str] = field(default_factory=list)
    add_new: bool = False
    chunk_size: int = 16384

def main(args: Args):
    """Main entry point for downloading material data."""
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Default fields if not provided
    if not args.fields:
        if args.download_type == "summary":
            args.fields = ['material_id', 'structure', 
                'energy_above_hull', 'formation_energy_per_atom', 'is_stable',
                'decomposes_to', 'band_gap', 'is_metal', 'efermi',
                'bulk_modulus', 'shear_modulus', 'homogeneous_poisson',
                "formula_pretty"
            ]
        else: # robocrys
            args.fields = ["material_id", "description", "condensed_structure"]

    download(args.db_path, args.mp_ids, args.api_key, args.download_type, args.fields, args.add_new, args.chunk_size)

def download(db_path: str, mp_ids: List[str], api_key: str, download_type: str, fields: List[str], add_new: bool = False, chunk_size: int = 16384):
    """Download material data from MP and save to a shelve database."""
    db = shelve.open(db_path)
    mpr = MPRester(api_key)
    
    if download_type == "summary":
        docs = mpr.materials.summary.search(
            material_ids=mp_ids,
            chunk_size=chunk_size,
            fields=fields
        )
    elif download_type == "robocrys":
        docs = mpr.materials.robocrys.search_docs(
            material_ids=mp_ids,
            chunk_size=chunk_size,
            fields=fields
        )

    download_count = 0
    error_ids = []
    for doc in tqdm(docs):
        try:
            mp_id = doc.material_id
            elem = {key: doc[key] for key in fields}
            
            if mp_id in db:
                dbelem = db[mp_id]
                dbelem.update(elem)
                db[mp_id] = dbelem
            elif add_new:
                db[mp_id] = elem
            
            if download_count % 100 == 0:
                db.sync()

            download_count += 1
        except:
            error_ids.append(mp_id)

    print(f"\nSuccessfully downloaded {download_count} materials.")
    if error_ids:
        print(f"Failed to collect {len(error_ids)} materials. Log saved to log.json")
    with open("log.json", "w", encoding="utf-8") as f:
        json.dump(error_ids, f, indent=4)
    db.close()

if __name__ == "__main__":
    tyro.cli(main)