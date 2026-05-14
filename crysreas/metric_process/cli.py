"""CLI for ``metric_process`` (spawn + optional multiprocessing)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import warnings
from pathlib import Path

import tyro

from crysreas.metric_process import MetricProcess, merge_metric_process_config
from crysreas.metric_process.helpers import df_serialize
from crysreas.metric_process.io import load_metrics_dataframe
from crysreas.metric_process.registry import run_metrics


def main(
    path: str,
    metrics_name: list[str] | None = None,
    output_path: str | None = None,
    prompt_type: str = "conditional+thinking",
    forced: bool = False,
    num_workers: int = 16,
    level: str | None = None,
) -> None:
    """
    Compute registered metrics on a table and optionally save (serialized) results.

    With ``num_workers > 1``, chunks are processed via Ray. Prefer ``num_workers=1``
    for heavy MLIP metrics (they always run in-process).
    """
    if level:
        key = level.strip().upper()
        if key not in logging._nameToLevel:
            raise ValueError(
                f"Invalid log level {level!r}; expected one of: "
                f"{', '.join(sorted(logging._nameToLevel.keys()))}"
            )
        lvl = logging._nameToLevel[key]
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            force=True,
        )
        logging.getLogger().setLevel(lvl)
        # Survives ``ray.init()`` lowering levels; ``basic._batched_remote_results_with_debug_tqdm`` reads this.
        if key == "DEBUG":
            os.environ["AI4SCI_METRIC_PROGRESS_DEBUG"] = "1"

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set (e.g. re-entry)

    warnings.filterwarnings("ignore", category=Warning, module=r"mattergen(\.|$)")
    warnings.filterwarnings("ignore", category=Warning, module=r"uncertainties(\.|$)")
    warnings.filterwarnings("ignore", category=Warning, module=r"pymatgen(\.|$)")

    path_p = Path(path)
    if output_path is None:
        output_path = str(path_p)
    save_path = Path(output_path)

    df = load_metrics_dataframe(path_p)
    print(df.iloc[0])


    print("prompt_type", prompt_type)

    mp_config = merge_metric_process_config(
        None, {"prompt_type": list(prompt_type.split("+"))}
    )
    names = list(metrics_name or [])
    if not names:
        raise ValueError(
            "metrics_name is empty: pass at least one metric (e.g. --metrics-name cte). "
            "Legacy ``run_metric`` with no metrics computed nothing."
        )

    # ``forced`` semantics live only in CLI: if requested, remove target columns
    # from the input table so the corresponding ``ensure_*`` logic will
    # recompute them (because it only checks for column existence).
    if forced:
        to_drop = [n for n in names if n in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)

    processed = df
    if num_workers <= 1:
        processed = run_metrics(
            processed.copy(), names, config=mp_config, forced=forced, log=True
        )
    else:
        with MetricProcess(mp_config) as mp:
            processed = mp.process(processed.copy(), names, forced=forced, log=True)

    processed_copy = processed.copy()
    df_serialize(processed_copy)
    if save_path.suffix == ".parquet":
        processed_copy.to_parquet(save_path)
    elif save_path.suffix == ".pkl":
        processed_copy.to_pickle(save_path)
    elif save_path.suffix == ".csv":
        processed_copy.to_csv(save_path)
    else:
        raise ValueError(f"Unsupported output suffix: {save_path.suffix}")

    print("Wrote:", save_path)


if __name__ == "__main__":
    tyro.cli(main)
