import pandas as pd
import os
from crysreas import Config
import shelve
from tqdm import tqdm

def insert_elastic_properties():
    df: pd.DataFrame = pd.read_parquet(Config.DATA_PATH / "split_cdvae_metric.parquet")

    with shelve.open(os.path.join(Config.DATA_PATH, "MP_shelve"), "w") as db:
        count = 0
        for elem in df.itertuples():
            if isinstance(elem.elastic_properties, dict):
                print(db[elem.mp_id]["bulk_modulus"], elem.elastic_properties["bulk_modulus"])
                temp = db[elem.mp_id]
                temp["bulk_modulus"] = elem.elastic_properties["bulk_modulus"]
                temp["shear_modulus"] = elem.elastic_properties["shear_modulus"]
                db[elem.mp_id] = temp
            else:
                count += 1
        print(count, len(df), count / len(df))

def insert_thermal_conductivity():
    df: pd.DataFrame = pd.read_parquet(Config.DATA_PATH / 'split_small_atoms_metric.parquet')

    with shelve.open(os.path.join(Config.DATA_PATH, "MP_shelve"), "w") as db:
        count = 0
        for elem in tqdm(list(df.itertuples())):
            if isinstance(elem.cte, dict):
                idx = elem.cte["temperatures"].tolist().index(300)
                thermal_expansion_300k = elem.cte["thermal_expansion"][idx]
                #print(thermal_expansion_300k)
                temp = db[elem.mp_id]
                temp["thermal_expansion_300k"] = thermal_expansion_300k
                db[elem.mp_id] = temp
            else:
                count += 1
        print(count, len(df), count / len(df))

if __name__ == "__main__":
    #insert_elastic_properties()
    insert_thermal_conductivity()
