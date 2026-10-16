"""Runtime config for metric_process (e.g. prompt_type for GT buckets).

Module-level ``set_config`` / ``get_config`` operate on a private process-global
dict for tests and introspection only. The metric pipeline does **not** read
this global state; callers pass an explicit ``config`` dict into
``run_metrics``, ``MetricProcess``, and ``ensure_*``.

Evaluation protocol
-------------------
The keys below fix the four judgements that define S.U.N. and S.S.U.N. Each one
also accepts the setting MatterGen uses, collected in ``mattergen_config()``, so
a table scored under either reading can be produced from the same code path.

``stability_threshold``
    A structure counts as stable when ``E_hull <= stability_threshold``
    (eV/atom). ``0.0`` means on or below the hull, which is what the
    ``Stability`` column and the stability factor of S.U.N. require. Set
    ``stability_threshold=0.1`` with ``stability_strict=True`` for the
    MatterGen form, a strict ``E_hull < 0.1``.

``metastability_threshold``
    The wider window used by S.S.U.N. only, ``E_hull <= 0.1`` eV/atom.

``spacegroup_criterion``
    ``"subgroup"`` counts a structure when it carries *at least* the requested
    symmetry: the requested group equals the detected one or is a proper
    subgroup of it. ``"strict"`` requires integer equality.

``novelty_reference``
    ``"train"`` measures novelty against the SFT training split, which asks
    whether the model produced something it was not trained on.
    ``"mp_alexandria"`` measures it against MatterGen's combined Materials
    Project and Alexandria reference set.

``uniqueness_scope``
    ``"seed"`` deduplicates inside one seed, where a duplicate can only arise
    between two different requested compositions. ``"pooled"`` deduplicates over
    the whole evaluated table.
"""

from typing import Any

DEFAULT_METRIC_PROCESS_CONFIG: dict[str, Any] = {
    "prompt_type": ["conditional", "thinking"],
    # --- evaluation protocol -------------------------------------------------
    "stability_threshold": 0.0,
    "stability_strict": False,
    "metastability_threshold": 0.1,
    "spacegroup_criterion": "subgroup",
    "spacegroup_symprec": 0.1,
    "spacegroup_angle_tolerance": 5,
    "novelty_reference": "train",
    "uniqueness_scope": "seed",
    "seed_column": "seed",
    "group_column": "mp_id",
}

#: MatterGen's reading of the same four judgements.
MATTERGEN_METRIC_CONFIG: dict[str, Any] = {
    "stability_threshold": 0.1,
    "stability_strict": True,
    "spacegroup_criterion": "strict",
    "novelty_reference": "mp_alexandria",
    "uniqueness_scope": "pooled",
}

_CONFIG: dict[str, Any] = {**DEFAULT_METRIC_PROCESS_CONFIG}


def merge_metric_process_config(
    base: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge config dicts like ``{**base_with_defaults, **updates}`` (no globals)."""
    b = {**DEFAULT_METRIC_PROCESS_CONFIG, **(base or {})}
    return {**b, **(updates or {})}


def mattergen_config(updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """MatterGen's settings for the four judgements.

    Pooled set-level uniqueness, novelty against MP+Alexandria, integer-equality
    space groups and a strict ``E_hull < 0.1`` stability window. This is the
    combination the stored evaluation artifacts were scored under, so it is also
    what ``crysreas.experiment.verify_equivalence`` compares against.
    """
    return merge_metric_process_config(MATTERGEN_METRIC_CONFIG, updates)


def set_config(config: dict) -> None:
    global _CONFIG
    _CONFIG = {**_CONFIG, **config}


def get_config() -> dict:
    return dict(_CONFIG)
