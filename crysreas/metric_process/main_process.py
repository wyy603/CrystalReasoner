"""Multiprocess DataFrame pipeline: same metric API, row chunks merged by index (Ray)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import ray

from .basic import _batched_remote_results_with_debug_tqdm
from .config import merge_metric_process_config
from .registry import resolve_metric_closure_sorted, run_metrics, metric_is_heavy


def _ensure_ray() -> None:
    if not ray.is_initialized():
        print("ray.init main_process")
        ray.init(ignore_reinit_error=True)


@ray.remote
def _ray_worker_run_metrics_list1(
    payload: list[
        tuple[pd.DataFrame, tuple[str, ...], bool, bool, dict[str, Any]]
    ],
) -> list[pd.DataFrame]:
    """Top-level for pickling; must import basic so METRICS is populated in worker."""
    chunk, names, log, forced, mp_config = payload[0]
    from crysreas.metric_process import basic as _basic  # noqa: F401

    return [
        run_metrics(
            chunk, list(names), config=mp_config, log=log, forced=forced
        )
    ]


def _split_df(df: pd.DataFrame, chunk_size: int = 64) -> list[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if len(df) == 0:
        return [df.copy()]
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


class MetricProcess:
    """
    Parallel metric runner using Ray.
    Metrics are run in topological order and always dispatched through Ray workers.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._closed = False
        self._config = merge_metric_process_config(None, config)
        from crysreas.metric_process import basic as _basic  # noqa: F401

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

    def __enter__(self) -> MetricProcess:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def process(
        self,
        df: pd.DataFrame,
        metric_names: Sequence[str],
        *,
        log: bool = False,
        forced: bool = False,
    ) -> pd.DataFrame:
        """Return a new DataFrame with requested metrics computed."""
        if self._closed:
            raise RuntimeError("MetricProcess is closed")

        names_sorted = resolve_metric_closure_sorted(list(metric_names))
        out = df.copy()

        for name in names_sorted:
            if metric_is_heavy(name):
                out = run_metrics(
                    out, [name], config=self._config, log=log, forced=forced
                )
                continue

            chunks = _split_df(out, 64)
            _ensure_ray()
            payloads = [
                (ch, (name,), log, forced, self._config)
                for ch in chunks
            ]
            parts: list[pd.DataFrame] = _batched_remote_results_with_debug_tqdm(
                remote_method=_ray_worker_run_metrics_list1,
                items=payloads,
                metric_name=f"metric_process:{name}",
                batch_size=1,
            )
            out = pd.concat(parts, axis=0).sort_index()

        return out

    def __call__(
        self,
        df: pd.DataFrame,
        metric_names: Sequence[str],
        *,
        log: bool = False,
        forced: bool = False,
    ) -> pd.DataFrame:
        return self.process(df, metric_names, log=log, forced=forced)
