"""Load evaluation tables (parity with ``crysreas.metric.run_metric``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from crysreas import Config
from crysreas.metric_process.helpers import df_deserialize


def _add_seed_column(df: pd.DataFrame, group: str = "mp_id", column: str = "seed") -> None:
    """Label each sample of a prompt with the seed it came from.

    ``explode`` above preserves the order in which generation drew the
    ``n_samples`` completions of a prompt, so the position of a row inside its
    prompt identifies one draw. Taking one row per prompt at a fixed position
    therefore yields one complete evaluation of the split. See
    ``crysreas.metric_process.basic.ensure_seed`` for what the label does and
    does not mean.
    """
    if column in df.columns or group not in df.columns:
        return
    df[column] = df.groupby(group).cumcount()


def load_metrics_dataframe(path: Path | str) -> pd.DataFrame:
    """Read pickle / parquet / csv and apply the same preprocessing as ``run_metric``."""
    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "responses" in df.columns:
            first_val = df.iloc[0]["responses"]
            if isinstance(first_val, (list, np.ndarray)) and len(first_val) > 1:
                if "extra_args" in df.columns:
                    df = df.explode(["responses", "extra_args"], ignore_index=True)
                else:
                    df = df.explode("responses", ignore_index=True)
            elif isinstance(first_val, (list, np.ndarray)) and len(first_val) == 1:
                df["responses"] = df["responses"].apply(
                    lambda x: x[0] if len(x) > 0 else None
                )
                if "extra_args" in df.columns:
                    df["extra_args"] = df["extra_args"].apply(
                        lambda x: x[0] if len(x) > 0 else None
                    )
        _add_seed_column(df)
        df_deserialize(df)
        return df
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df_deserialize(df)
        return df
    if path.suffix == ".json":
        with open(path, "r") as f:
            split = json.load(f)
        keys = split["test"] + split["train"] + split["val"]
        import shelve

        data_list = []
        with shelve.open(str(Config.DATA_PATH / "MP_shelve"), flag="r") as db:
            for key in tqdm(keys):
                data_list.append(
                    {"simple_structure": db[key]["structure"], "mp_id": key}
                )
        return pd.DataFrame(data_list)
    raise ValueError(f"Unsupported file type: {path.suffix}")
