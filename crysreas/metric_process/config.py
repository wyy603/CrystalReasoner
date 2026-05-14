"""Runtime config for metric_process (e.g. prompt_type for GT buckets).

Module-level ``set_config`` / ``get_config`` operate on a private process-global
dict for tests and legacy introspection only. The metric pipeline does **not**
read this global state; callers pass an explicit ``config`` dict into
``run_metrics``, ``MetricProcess``, and ``ensure_*``.
"""

from typing import Any

DEFAULT_METRIC_PROCESS_CONFIG: dict[str, Any] = {
    "prompt_type": ["conditional", "thinking"],
}

_CONFIG: dict[str, Any] = {**DEFAULT_METRIC_PROCESS_CONFIG}


def merge_metric_process_config(
    base: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge config dicts like ``{**base_with_defaults, **updates}`` (no globals)."""
    b = {**DEFAULT_METRIC_PROCESS_CONFIG, **(base or {})}
    return {**b, **(updates or {})}


def set_config(config: dict) -> None:
    global _CONFIG
    _CONFIG = {**_CONFIG, **config}


def get_config() -> dict:
    return dict(_CONFIG)
