"""
Functional metric pipeline: DataFrame in/out, string registry, optional multiprocessing.
"""

from . import basic as _basic  # noqa: F401 - register metrics on import
from .config import DEFAULT_METRIC_PROCESS_CONFIG, merge_metric_process_config
from .main_process import MetricProcess
from .registry import (
    METRIC_TOPO,
    MetricType,
    get_metric,
    get_metric_tags,
    list_metrics,
    register,
    run_metrics,
)

__all__ = [
    "METRIC_TOPO",
    "MetricType",
    "MetricProcess",
    "DEFAULT_METRIC_PROCESS_CONFIG",
    "merge_metric_process_config",
    "get_metric",
    "get_metric_tags",
    "list_metrics",
    "register",
    "run_metrics",
]
