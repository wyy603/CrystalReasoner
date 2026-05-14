"""String registry for metric functions: DataFrame in, DataFrame out."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import merge_metric_process_config


@dataclass(frozen=True)
class MetricType:
    """Per-metric scheduling metadata (topological layer + optional tags)."""

    topo: int
    tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


METRICS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]] = {}
METRIC_TOPO: dict[str, MetricType] = {}


def register(
    name: str | None = None,
    *,
    topo: int = 0,
    tags: tuple[str, ...] = (),
    dependencies: Sequence[str] = (),
) -> Callable:
    """Register a metric name, topological order, and tags in one step."""

    def decorator(fn: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]):
        key = name or getattr(fn, "__name__", "unknown")
        if key in METRICS:
            raise ValueError(f"Duplicate metric registration: {key}")
        METRICS[key] = fn
        METRIC_TOPO[key] = MetricType(
            topo=topo,
            tags=tags,
            dependencies=tuple(dependencies),
        )
        return fn

    return decorator


def get_metric(name: str) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame] | None:
    return METRICS.get(name)


def get_metric_tags(name: str) -> tuple[str, ...]:
    mt = METRIC_TOPO.get(name)
    return mt.tags if mt is not None else ()


def get_metric_dependencies(name: str) -> tuple[str, ...]:
    mt = METRIC_TOPO.get(name)
    return mt.dependencies if mt is not None else ()


def sort_metrics_by_topo(names: Sequence[str]) -> list[str]:
    """Stable sort by ``MetricType.topo`` then name. Unknown names sort last."""

    def key(n: str) -> tuple[int, str]:
        mt = METRIC_TOPO.get(n)
        return (mt.topo if mt is not None else 10_000, n)

    return sorted(names, key=key)


def resolve_metric_closure_sorted(metric_names: Sequence[str]) -> list[str]:
    """Expand requested metrics to include dependencies, then sort by topo."""
    closure: set[str] = set()
    visiting: set[str] = set()

    def dfs(name: str) -> None:
        if name in closure:
            return
        if name in visiting:
            raise ValueError(f"Cyclic metric dependency detected at {name!r}")
        if name not in METRICS:
            raise KeyError(f"Unknown metric: {name!r}. Known: {list_metrics()}")
        visiting.add(name)
        for dep in get_metric_dependencies(name):
            dfs(dep)
        visiting.remove(name)
        closure.add(name)

    for metric_name in metric_names:
        dfs(metric_name)
    return sort_metrics_by_topo(list(closure))


def metric_is_heavy(name: str) -> bool:
    """True if requested metric is tagged heavy (MLIP / GPU path)."""
    tags = get_metric_tags(name)
    return "heavy" in tags


def list_metrics() -> list[str]:
    return sorted(METRICS.keys())


def run_metrics(
    df: pd.DataFrame,
    names: Sequence[str],
    *,
    config: dict[str, Any] | None = None,
    log: bool = False,
    forced: bool = False,
) -> pd.DataFrame:
    """Run metrics in order; each function mutates and returns the DataFrame."""
    # NOTE: ``forced`` currently only takes effect in ``metric_process.cli`` by
    # deleting target columns from the input DataFrame.
    cfg = merge_metric_process_config(None, config)
    out = df
    expanded_names = resolve_metric_closure_sorted(names)
    for n in expanded_names:
        fn = METRICS.get(n)
        if fn is None:
            raise KeyError(f"Unknown metric: {n!r}. Known: {list_metrics()}")
        if log:
            print(f"[crysreas/metric_process/registry.py] Running metric {n}.")
        out = fn(out, cfg)
        if log:
            print(f"[crysreas/metric_process/registry.py] Finished metric {n}.")
    return out
