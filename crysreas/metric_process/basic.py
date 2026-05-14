"""Registered metrics: CPU ``ensure_*`` helpers + MLIP-backed ensures (single module)."""

from __future__ import annotations

import io
import logging
import math
import os
import shelve
import sys
from typing import Any

import numpy as np
import pandas as pd
import ray
from pymatgen.core import Structure
from tqdm import tqdm

from crysreas import Config
from crysreas.utils.basic_eval import chemical_symbols, smact_validity
from crysreas.utils.crystal import SimpleCrystal

from .helpers import (
    parse_crystaltext_structure_from_response,
    parse_plaid_wyckoff_structure_from_response,
    parse_simple_structure_from_response,
    return_gt_dict,
    spacegroup_number_safe,
    structure_validity,
)
from .registry import get_metric_dependencies, register, run_metrics

logger = logging.getLogger(__name__)

_HEAVY = ("heavy",)


def _is_fail(x: Any) -> bool:
    return x is False or x is None or (isinstance(x, float) and np.isnan(x))


class _TqdmToLogger(io.StringIO):
    """File-like stream that redirects tqdm output to logger.debug."""

    def __init__(self, log_func: Any) -> None:
        super().__init__()
        self._log_func = log_func

    def write(self, buf: str) -> int:
        msg = buf.strip()
        if msg:
            self._log_func(msg)
        return len(buf)

    def flush(self) -> None:
        return None


def _get_tqdm_file() -> _TqdmToLogger:
    return _TqdmToLogger(lambda msg: logger.debug(msg))


def _want_metric_batch_progress_bar() -> bool:
    """True when user asked for verbose batch progress (survives Ray resetting log levels)."""
    if os.environ.get("AI4SCI_METRIC_PROGRESS_DEBUG") == "1":
        return True
    return logger.isEnabledFor(logging.DEBUG)


def _batched_remote_results_with_debug_tqdm(
    remote_method: Any,
    items: list[Any],
    metric_name: str,
    batch_size: int = 16,
) -> list[Any]:
    """Run batched Ray tasks; tqdm on DEBUG, else one print line per finished batch."""
    if not items:
        return []

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    refs = [remote_method.remote(batch) for batch in batches]

    show_pbar = _want_metric_batch_progress_bar()
    pbar = None
    if show_pbar:
        # Prefer stderr tqdm when forced via env: Ray often lowers logger levels after ray.init().
        tqdm_file: io.TextIOBase | _TqdmToLogger = (
            sys.stderr
            if os.environ.get("AI4SCI_METRIC_PROGRESS_DEBUG") == "1"
            else _get_tqdm_file()
        )
        pbar = tqdm(
            total=len(refs),
            desc=f"calculating {metric_name} batches",
            unit="batch",
            dynamic_ncols=True,
            file=tqdm_file,
        )

    results_by_batch: list[list[Any] | None] = [None] * len(refs)
    pending: dict[Any, int] = {ref: i for i, ref in enumerate(refs)}

    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        done_ref = done_refs[0]
        batch_idx = pending.pop(done_ref)
        results_by_batch[batch_idx] = ray.get(done_ref)
        done = len(refs) - len(pending)
        if pbar is not None:
            pbar.set_postfix_str(f"{metric_name}: done {done}/{len(refs)}")
            pbar.update(1)
        else:
            print(
                f"[crysreas/metric_process/basic.py] {metric_name}: batch {done}/{len(refs)}",
                flush=True,
            )

    if pbar is not None:
        pbar.close()

    return [elem for batch in results_by_batch if batch is not None for elem in batch]


def _ensure_registered_dependencies(
    df: pd.DataFrame, metric_name: str, config: dict[str, Any]
) -> None:
    """Backwards-compatible dependency fulfillment for direct ensure_* calls."""
    deps = [
        dep for dep in get_metric_dependencies(metric_name) if dep not in df.columns
    ]
    if deps:
        run_metrics(df, deps, config=config)


# --- ensure_* (column cache semantics) -------------------------------------------------


def ensure_simple_structure(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "simple_structure" in df.columns:
        return
    if "responses" not in df.columns:
        raise ValueError("ensure_simple_structure requires column 'responses'")
    prompt_type = config.get("prompt_type") or ["conditional"]
    if prompt_type[0] in ("crystaltextllm_generation", "crystaltextllm_8_generation"):
        parser = parse_crystaltext_structure_from_response
    elif prompt_type[0] in ("plaid_wyckoff_generation", "plaid_wyckoff_8_generation"):
        parser = parse_plaid_wyckoff_structure_from_response
    else:
        parser = parse_simple_structure_from_response
    series = []
    for _, row in df.iterrows():
        series.append(parser(row["responses"]))
    df["simple_structure"] = series


def ensure_gt(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "gt" in df.columns:
        return
    if "mp_id" not in df.columns:
        raise ValueError("ensure_gt requires column 'mp_id'")
    pt0 = config["prompt_type"][0] if config.get("prompt_type") else "conditional"
    path = Config.DATA_PATH / "MP_shelve"
    gts: list[Any] = []
    with shelve.open(str(path)) as db:
        for mp_id in df["mp_id"].tolist():
            elem = db[mp_id]
            gts.append(return_gt_dict(elem, pt0))
    df["gt"] = np.array(gts, dtype=object)


def ensure_structure_validity(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "structure_validity" in df.columns:
        return
    _ensure_registered_dependencies(df, "structure_validity", config)
    vals = []
    for _, row in df.iterrows():
        s = row["simple_structure"]
        if _is_fail(s):
            vals.append(None)
        else:
            vals.append(structure_validity(s))
    df["structure_validity"] = vals


def ensure_smact_validity(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "smact_validity" in df.columns:
        return
    _ensure_registered_dependencies(df, "smact_validity", config)
    vals = []
    for _, row in df.iterrows():
        structure = row["simple_structure"]
        if _is_fail(structure):
            vals.append(None)
            continue
        try:
            element_amount_dict = structure.composition.get_el_amt_dict()
            elem = list(element_amount_dict.keys())
            elem = [chemical_symbols.index(str(x)) for x in elem]
            comp_float = list(element_amount_dict.values())
            comp = np.array([int(round(x)) for x in comp_float], dtype=int)
            comp = comp / np.gcd.reduce(comp)
            comp = np.array(comp, dtype=int).tolist()
            vals.append(smact_validity(elem, comp))
        except Exception as e:
            print(f"smact_validity row failed: {e}")
            logger.debug("smact_validity row failed: %s", e)
            vals.append(None)
    df["smact_validity"] = vals


def ensure_composition_consistency(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "composition_consistency" in df.columns:
        return
    _ensure_registered_dependencies(df, "composition_consistency", config)
    vals = []
    for _, row in df.iterrows():
        gt = row["gt"]
        s = row["simple_structure"]
        if _is_fail(gt) or _is_fail(s):
            vals.append(None)
            continue
        vals.append(str(s.composition.reduced_composition) == gt["comp"])
    df["composition_consistency"] = vals


def ensure_spacegroup_consistency(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "spacegroup_consistency" in df.columns:
        return
    _ensure_registered_dependencies(df, "spacegroup_consistency", config)
    vals = []
    for _, row in df.iterrows():
        gt = row["gt"]
        s = row["simple_structure"]
        sv = row["structure_validity"]
        sm = row["smact_validity"]
        if _is_fail(gt) or _is_fail(s) or _is_fail(sv) or _is_fail(sm):
            vals.append(None)
            continue
        if not sv or not sm:
            vals.append(None)
            continue
        spg = spacegroup_number_safe(s)
        if spg is None:
            vals.append(None)
        else:
            vals.append(spg == gt["spg"])
    df["spacegroup_consistency"] = vals


def ensure_fit_format(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "fit_format" in df.columns:
        return
    if "responses" not in df.columns:
        raise ValueError("ensure_fit_format requires columns 'responses'")

    out = []
    for _, row in df.iterrows():
        response_str = row["responses"]

        if response_str is None or (
            isinstance(response_str, float) and pd.isna(response_str)
        ):
            out.append(0)
            continue

        response_str = str(response_str)

        start_pattern = "<CIF>"
        end_pattern = "</CIF>"
        im_end_pattern = "<|im_end|>"
        endoftext_pattern = "<|endoftext|>"

        start_idx = response_str.find(start_pattern)
        end_idx = response_str.find(end_pattern)
        im_end_idx = response_str.find(im_end_pattern)
        endoftext_idx = response_str.find(endoftext_pattern)

        if start_idx == -1 or end_idx == -1 or endoftext_idx == -1 or im_end_idx == -1:
            out.append(0)
            continue
        if not (start_idx < end_idx < im_end_idx < endoftext_idx):
            out.append(0)
            continue
        distance = im_end_idx - (end_idx + len(end_pattern))
        if distance > 4:
            out.append(0)
            continue
        distance = endoftext_idx - (im_end_idx + len(im_end_pattern))
        if distance > 4:
            out.append(0)
            continue

        out.append(1)

    df["fit_format"] = out


def ensure_small_atoms(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """True when ``num_sites < 10`` for the parsed simple structure (see ``SimpleCrystal``)."""
    if "small_atoms" in df.columns:
        return
    _ensure_registered_dependencies(df, "small_atoms", config)
    vals: list[bool | None] = []
    for _, row in df.iterrows():
        s = row["simple_structure"]
        if _is_fail(s):
            vals.append(None)
            continue
        try:
            n = s.num_sites
        except Exception as e:
            logger.debug("small_atoms row failed: %s", e)
            vals.append(None)
            continue
        vals.append(n < 10)
    df["small_atoms"] = vals


def ensure_relaxed_structures(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "relaxed_structures" in df.columns:
        return
    _ensure_registered_dependencies(df, "relaxed_structures", config)

    from crysreas.mlip.relax import relax_structures_batch

    index = df[
        (df["structure_validity"] == True) & (df["smact_validity"] == True)
    ].index
    structures: list[Any] = df.loc[index]["simple_structure"].tolist()
    relaxed, energies, converged = ray.get(relax_structures_batch.remote(structures))
    df.loc[index, "relaxed_structures"] = pd.Series(
        relaxed, index=index, dtype="object"
    )
    df.loc[index, "energies"] = energies


def ensure_relaxed_structures_300(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "relaxed_structures_300" in df.columns:
        return
    _ensure_registered_dependencies(df, "relaxed_structures_300", config)

    from crysreas.mlip.relax import relax_structures_batch

    index = df[
        (df["structure_validity"] == True) & (df["smact_validity"] == True)
    ].index
    structures: list[Any] = df.loc[index]["simple_structure"].tolist()
    relaxed, energies, converged = ray.get(
        relax_structures_batch.remote(structures, steps=300)
    )
    df.loc[index, "relaxed_structures_300"] = pd.Series(
        relaxed, index=index, dtype="object"
    )
    df.loc[index, "converged"] = converged
    df.loc[index, "cif"] = [x.to(fmt="cif") if x is not None else None for x in relaxed]
    df["converged"] = converged
    df["cif"] = [x.to(fmt="cif") if x is not None else None for x in relaxed]


def ensure_energy_above_hull(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "energy_above_hull" in df.columns:
        return
    _ensure_registered_dependencies(df, "energy_above_hull", config)

    from crysreas.mlip.energy import compute_energy_above_hull_batch

    index = df[df["relaxed_structures"].notna()].index
    structures = df.loc[index]["simple_structure"].tolist()
    relaxed_structures = df.loc[index]["relaxed_structures"].tolist()
    energies = df.loc[index]["energies"].tolist()

    e_hull, is_stable = ray.get(
        compute_energy_above_hull_batch.remote(structures, relaxed_structures, energies)
    )
    df.loc[index, "energy_above_hull"] = e_hull
    df.loc[index, "is_stable"] = is_stable


def ensure_stable_unique_novel(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "stable_unique_novel" in df.columns:
        return
    _ensure_registered_dependencies(df, "stable_unique_novel", config)

    from crysreas.mlip.sun import coerce_bool_array, compute_stable_unique_novel_batch

    index = df[df["relaxed_structures"].notna()].index
    structures = df.loc[index]["simple_structure"].tolist()
    relaxed_structures = df.loc[index]["relaxed_structures"].tolist()
    energies = df.loc[index]["energies"].tolist()
    is_stable = coerce_bool_array(df.loc[index]["is_stable"], len(index), "is_stable")

    is_novel, is_unique, sun = compute_stable_unique_novel_batch(
        structures, relaxed_structures, energies, is_stable
    )
    df.loc[index, "is_novel"] = is_novel
    df.loc[index, "is_unique"] = is_unique
    df.loc[index, "stable_unique_novel"] = sun


def ensure_energy_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "energy_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "energy_reward", config)

    from crysreas.mlip.rewards import energy_reward_scalar

    rewards = []
    for _, row in df.iterrows():
        e = row["energy_above_hull"]
        ev = (
            float(e)
            if e is not None and not (isinstance(e, float) and math.isnan(e))
            else None
        )
        rewards.append(energy_reward_scalar(ev))
    df["energy_reward"] = rewards


def ensure_smooth_energy_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "smooth_energy_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "smooth_energy_reward", config)

    from crysreas.mlip.rewards import smooth_energy_reward_scalar

    rewards = []
    for _, row in df.iterrows():
        e = row["energy_above_hull"]
        comp = row["composition_consistency"]
        ev = (
            float(e)
            if e is not None and not (isinstance(e, float) and math.isnan(e))
            else None
        )
        rewards.append(smooth_energy_reward_scalar(ev, bool(comp)))
    df["smooth_energy_reward"] = rewards


def ensure_fmax(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "fmax" in df.columns:
        return
    _ensure_registered_dependencies(df, "fmax", config)

    from crysreas.mlip.elastic_api import fmax_numpy

    index = df[df["relaxed_structures"].notna()].index
    structures = df.loc[index]["relaxed_structures"].tolist()
    df.loc[index, "fmax"] = fmax_numpy(structures)


def ensure_elastic_properties(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "elastic_properties" in df.columns:
        return
    _ensure_registered_dependencies(df, "elastic_properties", config)

    from crysreas.mlip.elastic_api import elastic_properties_list

    index = df[df["relaxed_structures"].notna()].index
    structures = df.loc[index]["relaxed_structures"].tolist()
    # Keep debug on for now to diagnose unexpected negative moduli cases.
    debug = False
    if debug:
        elastic_props, elastic_debug = elastic_properties_list(
            structures, relaxed=False, debug=True
        )
        df.loc[index, "elastic_properties"] = elastic_props
        df.loc[index, "elastic_debug"] = elastic_debug
    else:
        df.loc[index, "elastic_properties"] = elastic_properties_list(
            structures, relaxed=False, debug=False
        )


def ensure_elastic_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "elastic_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "elastic_reward", config)

    from crysreas.mlip.elastic_api import elastic_properties_list
    from crysreas.mlip.inseg import inseg

    index = df[df["relaxed_structures"].notna()].index
    structures = df.loc[index]["relaxed_structures"].tolist()
    energies = df.loc[index]["energies"].tolist()

    n = len(structures)
    rewards = [0] * n
    elastic_properties_16: list[Any] = [None] * n

    valid_indices = [
        i
        for i, (s, e) in enumerate(zip(structures, energies))
        if s is not None and not math.isnan(e)
    ]
    if not valid_indices:
        df.loc[index, "elastic_properties_16"] = elastic_properties_16
        df.loc[index, "elastic_reward"] = rewards
        return

    top_indices = sorted(valid_indices, key=lambda i: energies[i])[:16]
    top_structures = [structures[i] for i in top_indices]

    top_results = elastic_properties_list(top_structures, relaxed=True)

    for i, res in zip(top_indices, top_results):
        elastic_properties_16[i] = res

    valid_reward_indices = [
        i for i in top_indices if elastic_properties_16[i] is not None
    ]
    if valid_reward_indices:
        bulk_modulus = [
            elastic_properties_16[i]["bulk_modulus"] for i in valid_reward_indices
        ]
        shear_modulus = [
            elastic_properties_16[i]["shear_modulus"] for i in valid_reward_indices
        ]
        bulk_modulus_gt = [
            df["gt"].iloc[i]["bulk_modulus"] for i in valid_reward_indices
        ]
        shear_modulus_gt = [
            df["gt"].iloc[i]["shear_modulus"] for i in valid_reward_indices
        ]

        bulk_scores = inseg(bulk_modulus, bulk_modulus_gt)
        shear_scores = inseg(shear_modulus, shear_modulus_gt)
        for i, b, s in zip(valid_reward_indices, bulk_scores, shear_scores):
            rewards[i] = b + s

    df.loc[index, "elastic_properties_16"] = elastic_properties_16
    df.loc[index, "elastic_reward"] = rewards


def ensure_elastic_reward_all(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "elastic_reward_all" in df.columns:
        return
    _ensure_registered_dependencies(df, "elastic_reward_all", config)

    from crysreas.mlip.inseg import inseg

    index = df[df["relaxed_structures"].notna()].index
    all_elastic_properties = (
        df.loc[index]["elastic_properties"].tolist()
        if "elastic_properties" in df.columns
        else [None] * len(index)
    )
    gts = df.loc[index]["gt"].tolist()

    rewards: list[int] = []
    for i, properties in enumerate(all_elastic_properties):
        if properties is None:
            rewards.append(0)
            continue

        bulk_modulus = (
            properties.get("bulk_modulus") if isinstance(properties, dict) else None
        )
        shear_modulus = (
            properties.get("shear_modulus") if isinstance(properties, dict) else None
        )

        gt = gts[i]
        bulk_modulus_gt = gt.get("bulk_modulus") if isinstance(gt, dict) else None
        shear_modulus_gt = gt.get("shear_modulus") if isinstance(gt, dict) else None

        if bulk_modulus is None or shear_modulus is None:
            rewards.append(0)
            continue

        bulk_score = inseg([bulk_modulus], [bulk_modulus_gt])[0]
        shear_score = inseg([shear_modulus], [shear_modulus_gt])[0]
        rewards.append(bulk_score + shear_score)

    df.loc[index, "elastic_reward_all"] = rewards


def _range_reward_quadratic(x: float | None, l: float | None, r: float | None) -> float:
    """
    Quadratic + Exponential Reward, from -1 to 1
    Guide x towards (l+r)/2.
    """
    if x is None or l is None or r is None or r <= l:
        return 0.0

    mid = 0.5 * (l + r)
    denom = r - l
    z = (x - mid) / denom

    y = 1.0 - 2.0 * (z * z)

    if isinstance(x, np.ndarray):
        return np.where(y >= 0, y, np.exp(y) - 1.0)
    return y if y >= 0 else math.exp(y) - 1.0


def ensure_new_elastic_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "new_elastic_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "new_elastic_reward", config)

    rewards = [0.0] * len(df)
    index = df[df["elastic_properties"].notna()].index
    props_list = df.loc[index, "elastic_properties"].tolist()
    gts = df.loc[index, "gt"].tolist()
    row_positions = df.index.get_indexer(index)

    for row_pos, props, gt in zip(row_positions.tolist(), props_list, gts):
        if row_pos < 0:
            continue
        if not isinstance(props, dict) or not isinstance(gt, dict):
            continue
        bulk_x = props.get("bulk_modulus")
        shear_x = props.get("shear_modulus")
        bulk_range = gt.get("bulk_modulus")
        shear_range = gt.get("shear_modulus")
        if bulk_range is None or shear_range is None:
            continue
        if not (
            isinstance(bulk_range, (tuple, list))
            and len(bulk_range) == 2
            and isinstance(shear_range, (tuple, list))
            and len(shear_range) == 2
        ):
            continue

        bl, br = float(bulk_range[0]), float(bulk_range[1])
        sl, sr = float(shear_range[0]), float(shear_range[1])
        bulk_r = _range_reward_quadratic(
            float(bulk_x) if bulk_x is not None else None, bl, br
        )
        shear_r = _range_reward_quadratic(
            float(shear_x) if shear_x is not None else None, sl, sr
        )
        rewards[row_pos] = bulk_r + shear_r

    df["new_elastic_reward"] = rewards


def ensure_sn_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "sn_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "sn_reward", config)

    from crysreas.mlip.rewards import energy_reward_scalar

    rewards: list[float] = []
    for _, row in df.iterrows():
        e = row.get("energy_above_hull")
        ev = (
            float(e)
            if e is not None and not (isinstance(e, float) and math.isnan(e))
            else None
        )
        base = float(energy_reward_scalar(ev))
        novel = bool(row.get("is_novel")) if row.get("is_novel") is not None else False
        rewards.append(base + (1.0 if novel else 0.0))
    df["sn_reward"] = rewards


def ensure_cte(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "cte" in df.columns:
        return
    _ensure_registered_dependencies(df, "cte", config)

    from crysreas.mlip.cte import calculate_qha

    index = df[
        (df["structure_validity"] == True)
        & (df["smact_validity"] == True)
        & df["simple_structure"].notna()
        & (df["small_atoms"] == True)
    ].index
    structures = df.loc[index]["simple_structure"].tolist()
    if not structures:
        df["cte"] = None
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    results = _batched_remote_results_with_debug_tqdm(
        remote_method=calculate_qha,
        items=structures,
        metric_name="cte",
        batch_size=16,
    )
    dumped = [r.model_dump() if r is not None else None for r in results]
    df["cte"] = None
    df.loc[index, "cte"] = dumped

    thermal_expansion_300k: list[float | None] = []
    for elem in dumped:
        if elem is None:
            thermal_expansion_300k.append(None)
            continue
        temps = elem["temperatures"]
        tlist = temps.tolist() if hasattr(temps, "tolist") else list(temps)
        ti = tlist.index(300)
        thermal_expansion_300k.append(elem["thermal_expansion"][ti])

    df["thermal_expansion_300k"] = None
    df.loc[index, "thermal_expansion_300k"] = thermal_expansion_300k


def ensure_cte_reward(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "cte_reward" in df.columns:
        return
    _ensure_registered_dependencies(df, "cte_reward", config)

    from crysreas.mlip.cte import calculate_qha

    index = df[
        (df["structure_validity"] == True)
        & (df["smact_validity"] == True)
        & (df["small_atoms"] == True)
        & df["simple_structure"].notna()
    ].index.tolist()
    if not index:
        df["cte_reward"] = 0
        return

    n = len(index)
    rewards = [0.0] * n

    energies = (
        df.loc[index, "energies"].tolist()
        if "energies" in df.columns
        else [math.nan] * n
    )
    structures = df.loc[index, "simple_structure"].tolist()

    valid_indices: list[int] = []
    for ii, st in enumerate(structures):
        gt = df.loc[index[ii], "gt"]
        gt_val = gt.get("thermal_expansion_300k") if isinstance(gt, dict) else None
        if gt_val is None:
            continue
        e = energies[ii]
        if e is None or (isinstance(e, float) and math.isnan(e)):
            continue
        if st is None:
            continue
        valid_indices.append(ii)

    if valid_indices:
        top_indices = sorted(valid_indices, key=lambda i: energies[i])[:16]
        top_structures = [structures[i] for i in top_indices]

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        qha_results = _batched_remote_results_with_debug_tqdm(
            remote_method=calculate_qha,
            items=top_structures,
            metric_name="cte_reward",
            batch_size=16,
        )

        scores: list[float] = []
        reward_pos: list[int] = []
        for i, res in zip(top_indices, qha_results):
            if res is None:
                continue
            dumped = res.model_dump() if hasattr(res, "model_dump") else res
            if dumped is None:
                continue
            temps = dumped["temperatures"]
            tlist = temps.tolist() if hasattr(temps, "tolist") else list(temps)
            if 300 not in tlist:
                continue
            ti = tlist.index(300)
            pred = dumped["thermal_expansion"][ti]
            if pred is None or (isinstance(pred, float) and math.isnan(pred)):
                continue
            gt = df.loc[index[i], "gt"]
            gt_val = gt.get("thermal_expansion_300k") if isinstance(gt, dict) else None
            if gt_val is None:
                continue
            if not (isinstance(gt_val, (tuple, list)) and len(gt_val) == 2):
                continue

            l, r = gt_val
            score = _range_reward_quadratic(
                float(pred),
                float(l) if l is not None else None,
                float(r) if r is not None else None,
            )
            scores.append(score)
            reward_pos.append(i)

        for i, score in zip(reward_pos, scores):
            rewards[i] = score

    df.loc[index, "cte_reward"] = rewards


def ensure_cte_reward_all(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if "cte_reward_all" in df.columns:
        return
    _ensure_registered_dependencies(df, "cte_reward_all", config)

    m = (
        (df["structure_validity"] == True)
        & (df["smact_validity"] == True)
        & df["simple_structure"].notna()
        & (df["small_atoms"] == True)
    )
    if not m.any():
        df["cte_reward_all"] = 0
        return

    out: list[int] = []
    for pred, gt in zip(df.loc[m, "thermal_expansion_300k"], df.loc[m, "gt"]):
        rng = gt.get("thermal_expansion_300k") if isinstance(gt, dict) else None
        if pred is None or (isinstance(pred, float) and math.isnan(pred)):
            out.append(0)
            continue
        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            out.append(0)
            continue
        lo, hi = rng[0], rng[1]
        if lo is None or hi is None:
            out.append(0)
            continue
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            out.append(0)
            continue
        pv = float(pred)
        out.append(1 if lo <= pv <= hi else 0)

    df.loc[m, "cte_reward_all"] = out


# --- registered metric wrappers --------------------------------------------------------


@register("simple_structure", topo=0, dependencies=())
def metric_simple_structure(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_simple_structure(df, config)
    return df


@register("gt", topo=0, dependencies=())
def metric_gt(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_gt(df, config)
    return df


@register("structure_validity", topo=1, dependencies=("simple_structure",))
def metric_structure_validity(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_structure_validity(df, config)
    return df


@register("smact_validity", topo=1, dependencies=("simple_structure",))
def metric_smact_validity(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_smact_validity(df, config)
    return df


@register("small_atoms", topo=1, dependencies=("simple_structure",))
def metric_small_atoms(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_small_atoms(df, config)
    return df


@register("composition_consistency", topo=2, dependencies=("gt", "simple_structure"))
def metric_composition_consistency(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_composition_consistency(df, config)
    return df


@register(
    "spacegroup_consistency",
    topo=3,
    dependencies=("gt", "simple_structure", "structure_validity", "smact_validity"),
)
def metric_spacegroup_consistency(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_spacegroup_consistency(df, config)
    return df


@register("fit_format", topo=0, dependencies=())
def metric_fit_format(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_fit_format(df, config)
    return df


@register(
    "relaxed_structures",
    topo=4,
    tags=_HEAVY,
    dependencies=("simple_structure", "structure_validity", "smact_validity"),
)
def metric_relaxed_structures(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_relaxed_structures(df, config)
    return df


@register(
    "relaxed_structures_300",
    topo=4,
    tags=_HEAVY,
    dependencies=("simple_structure", "structure_validity", "smact_validity"),
)
def metric_relaxed_structures_300(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_relaxed_structures_300(df, config)
    return df


@register(
    "energy_above_hull", topo=5, tags=_HEAVY, dependencies=("relaxed_structures",)
)
def metric_energy_above_hull(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_energy_above_hull(df, config)
    return df


@register(
    "stable_unique_novel", topo=6, tags=_HEAVY, dependencies=("energy_above_hull",)
)
def metric_stable_unique_novel(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_stable_unique_novel(df, config)
    return df


@register(
    "energy_reward",
    topo=7,
    tags=_HEAVY,
    dependencies=("energy_above_hull", "composition_consistency"),
)
def metric_energy_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_energy_reward(df, config)
    return df


@register(
    "smooth_energy_reward",
    topo=7,
    tags=_HEAVY,
    dependencies=("energy_above_hull", "composition_consistency"),
)
def metric_smooth_energy_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_smooth_energy_reward(df, config)
    return df


@register(
    "sn_reward",
    topo=8,
    tags=_HEAVY,
    dependencies=("energy_above_hull", "stable_unique_novel"),
)
def metric_sn_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_sn_reward(df, config)
    return df


@register("fmax", topo=5, tags=_HEAVY, dependencies=("relaxed_structures",))
def metric_fmax(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_fmax(df, config)
    return df


@register(
    "elastic_properties", topo=5, tags=_HEAVY, dependencies=("relaxed_structures",)
)
def metric_elastic_properties(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_elastic_properties(df, config)
    return df


@register(
    "elastic_reward",
    topo=8,
    tags=_HEAVY,
    dependencies=(
        "relaxed_structures",
        "composition_consistency",
        "energy_above_hull",
        "gt",
    ),
)
def metric_elastic_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_elastic_reward(df, config)
    return df


@register(
    "elastic_reward_all",
    topo=9,
    tags=_HEAVY,
    dependencies=("composition_consistency", "elastic_properties", "gt"),
)
def metric_elastic_reward_all(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_elastic_reward_all(df, config)
    return df


@register(
    "new_elastic_reward",
    topo=9,
    tags=_HEAVY,
    dependencies=("elastic_properties", "gt"),
)
def metric_new_elastic_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_new_elastic_reward(df, config)
    return df


@register(
    "cte",
    topo=4,
    tags=_HEAVY,
    dependencies=(
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "small_atoms",
    ),
)
def metric_cte(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_cte(df, config)
    return df


@register(
    "cte_reward",
    topo=10,
    tags=_HEAVY,
    dependencies=(
        "gt",
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "small_atoms",
        "relaxed_structures",
    ),
)
def metric_cte_reward(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_cte_reward(df, config)
    return df


@register(
    "cte_reward_all",
    topo=10,
    tags=_HEAVY,
    dependencies=("cte", "gt"),
)
def metric_cte_reward_all(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    ensure_cte_reward_all(df, config)
    return df
