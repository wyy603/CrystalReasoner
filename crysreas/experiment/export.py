"""
Export interleaved evaluation tables from merged checkpoint Parquet files.

Pipeline (see ``energy_export``):

1. Load one Parquet per model; each row is an ``mp_id`` with MLIP score, structure, etc.
2. Keep only ``mp_id``s that appear in **every** model (intersection).
3. Build ``score_df``: one row per common ``mp_id``, plus ``rl_gain`` (weighted score diff).
4. **First pick (buffer)**: choose which ``mp_id``s will be sent to MLIP relaxation.
   This is the expensive step: cost is ``len(buffer_ids) * number_of_models`` relax jobs,
   because each model has its own generated structure for the same ``mp_id``.
5. Relax (300 steps); drop ``mp_id``s where any model fails to converge / produce CIF.
6. **Second pick (final)**: from surviving ``mp_id``s, choose the subset written to CSV.
7. Write interleaved rows: for each chosen ``mp_id``, one row per model.

For stratified exports (e.g. 32 ``mp_id``s), use a **small buffer** by oversampling per
stratum before relax---do **not** relax all common ``mp_id``s unless you accept ~N×models cost.

CLI::

    python -m crysreas.experiment.export                     # default: elastic512
    python -m crysreas.experiment.export elastic_stratified32
"""

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray

from crysreas.mlip.relax import relax_structures_batch
from crysreas.utils.crystal import SimpleCrystal

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _EXPERIMENT_DIR

_MODULUS_KEYS = ("bulk_modulus", "shear_modulus")


def _slice_modulus_dict(obj):
    """Keep only bulk_modulus and shear_modulus; whole field None stays None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: obj.get(k) for k in _MODULUS_KEYS}
    try:
        if pd.api.types.is_scalar(obj) and pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    getter = getattr(obj, "get", None)
    if callable(getter):
        return {k: getter(k) for k in _MODULUS_KEYS}
    try:
        return {k: obj[k] for k in _MODULUS_KEYS}
    except Exception:
        return {k: None for k in _MODULUS_KEYS}


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _cell_for_csv(val):
    """Serialize nested values for CSV; UTF-8 text stays plain (no JSON wrap for str)."""
    if val is None:
        return ""
    try:
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return ""
    except (ValueError, TypeError):
        pass
    if isinstance(val, (dict, list, tuple)):
        return json.dumps(val, ensure_ascii=False, default=_json_default)
    if isinstance(val, np.ndarray):
        return json.dumps(val.tolist(), ensure_ascii=False)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def _dataframe_for_utf8_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_cell_for_csv)
    return out


def _parse_simple_structure(s):
    if s is None:
        return None
    try:
        return SimpleCrystal.from_simple_no_sym(s).structure
    except Exception:
        return None


def _relaxed_exports(relaxed_structure):
    """Convert relaxed pymatgen Structure to simple text and CIF."""
    if relaxed_structure is None:
        return None, None
    try:
        relaxed_crystal = SimpleCrystal.from_sym_structure(relaxed_structure)
        return relaxed_crystal.to_simple_no_sym(), relaxed_crystal.to_cif()
    except Exception:
        return None, None


@dataclass(frozen=True)
class BufferPickArgs:
    """
    Extra knobs passed into the first-stage (buffer) picker.

    The default picker uses ``A`` and ``B`` only. Custom pickers (e.g. stratified buffer)
    may ignore them; they still receive this object for a uniform call signature.
    """

    A: int
    B: int
    random_state: int | None


@dataclass(frozen=True)
class FinalPickArgs:
    """Context for the second-stage picker: full per-model frames, score column name, and legacy C/D for defaults."""

    dfs: dict[str, pd.DataFrame]
    key: str
    model_order: list[str]
    C: int
    D: int
    random_state: int | None


PickBufferIds = Callable[[pd.DataFrame, BufferPickArgs], list[Any]]
PickFinalMpIds = Callable[[pd.DataFrame, FinalPickArgs], list[Any]]


def pick_buffer_all_mp_ids(score_df: pd.DataFrame, _args: BufferPickArgs) -> list[Any]:
    """
    Relax **every** common ``mp_id`` in ``score_df``.

    Use only when you truly need full coverage (e.g. debugging). Cost is
    ``len(score_df) * num_models`` relaxations. For stratified small exports,
    prefer ``make_stratified_buffer_picker`` instead.
    """
    return score_df["mp_id"].tolist()


def default_pick_buffer_ids(score_df: pd.DataFrame, args: BufferPickArgs) -> list[Any]:
    """Prefer high ``rl_gain`` (first ``A`` rows), then random exploration (up to ``B`` more)."""
    buffer_top = score_df.head(args.A)["mp_id"].tolist()
    remain_df = score_df.iloc[len(buffer_top) :]
    buffer_rand_n = min(args.B, len(remain_df))
    buffer_rand = (
        remain_df.sample(buffer_rand_n, random_state=args.random_state)["mp_id"].tolist()
        if buffer_rand_n > 0
        else []
    )
    return buffer_top + buffer_rand


def default_pick_final_mp_ids(final_pool: pd.DataFrame, args: FinalPickArgs) -> list[Any]:
    """Second pass: top ``C`` by ``rl_gain``, then ``D`` random from the tail."""
    if len(final_pool) < args.C + args.D:
        raise ValueError("Not enough rows in final_pool for C + D.")
    head = final_pool.head(args.C)["mp_id"].tolist()
    tail_sample = final_pool.iloc[args.C :].sample(args.D, random_state=args.random_state)["mp_id"].tolist()
    return head + tail_sample


def make_stratified_buffer_picker(
    *,
    model_1: str,
    model_2: str,
    score_column_key: str,
    strata: Sequence[tuple[str, int, Callable[[pd.DataFrame], pd.Series]]],
    oversample_factor: float = 2.5,
    random_state: int | None = 42,
) -> PickBufferIds:
    """
    First-stage picker: stratified **oversample** before relaxation.

    ``score_df`` (from ``energy_export``) has one column per model named
    ``f"{score_column_key}_{model_name}"``. We rename model_1 / model_2 columns to
    ``score_m1`` and ``score_m2`` so the same predicates as ``make_stratified_final_picker``
    can be reused.

    For each stratum tuple ``(label, n_final, pred)``, ``n_final`` is how many ``mp_id``s
    you plan to take **after** relax (second stage). Here we pre-sample roughly
    ``ceil(n_final * oversample_factor)`` (capped by pool size) so that enough survive
    MLIP failures. If a pool is smaller than that target, we take the whole pool.

    Returned buffer is the union across strata (strata are disjoint, so no dedup needed).
    """

    def pick(score_df: pd.DataFrame, args: BufferPickArgs) -> list[Any]:
        col1 = f"{score_column_key}_{model_1}"
        col2 = f"{score_column_key}_{model_2}"
        if col1 not in score_df.columns or col2 not in score_df.columns:
            raise KeyError(f"Expected columns {col1!r} and {col2!r} on score_df.")
        m = score_df[["mp_id", col1, col2]].copy()
        m = m.rename(columns={col1: "score_m1", col2: "score_m2"})
        m["score_m1"] = pd.to_numeric(m["score_m1"], errors="coerce")
        m["score_m2"] = pd.to_numeric(m["score_m2"], errors="coerce")
        if m["score_m1"].isna().any() or m["score_m2"].isna().any():
            raise ValueError("Stratified buffer pick: non-numeric scores in score_df.")

        chosen: list[Any] = []
        rs0 = 0 if random_state is None else int(random_state)
        for i, (label, n_final, pred) in enumerate(strata):
            mask = pred(m)
            pool_idx = m.index[mask]
            pool_mp = m.loc[pool_idx, "mp_id"]
            n_pool = len(pool_mp)
            if n_pool < n_final:
                raise ValueError(
                    f"Stratum {label!r}: need at least {n_final} mp_ids before relax, pool has {n_pool}."
                )
            target = min(n_pool, max(n_final, int(math.ceil(n_final * oversample_factor))))
            rs_i = None if random_state is None else (rs0 + 100 + i)
            take = pool_mp.sample(n=target, random_state=rs_i).tolist()
            chosen.extend(take)
        # Disjoint strata → usually already unique; keep order and drop accidental duplicates.
        out: list[Any] = []
        seen: set[Any] = set()
        for x in chosen:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return pick


def make_stratified_final_picker(
    *,
    model_1: str,
    model_2: str,
    strata: Sequence[tuple[str, int, Callable[[pd.DataFrame], pd.Series]]],
    random_state: int | None = 42,
) -> PickFinalMpIds:
    """
    Second-stage picker: after relaxation, draw a fixed count from each score-defined bucket.

    ``final_pool`` only lists ``mp_id``s that survived relax for **all** models. We merge
    MLIP scores from ``args.dfs`` so predicates see ``score_m1`` / ``score_m2`` (same layout
    as ``make_stratified_buffer_picker``).

    Each stratum is ``(label, n_take, predicate)``: sample exactly ``n_take`` distinct ``mp_id``s
    from rows where ``predicate(merged_df)`` is True. Strata should be disjoint.
    """

    def pick(final_pool: pd.DataFrame, args: FinalPickArgs) -> list[Any]:
        key_col = args.key
        d1 = args.dfs[model_1][["mp_id", key_col]].rename(columns={key_col: "score_m1"})
        d2 = args.dfs[model_2][["mp_id", key_col]].rename(columns={key_col: "score_m2"})
        m = final_pool[["mp_id"]].drop_duplicates().merge(d1, on="mp_id").merge(d2, on="mp_id")
        m["score_m1"] = pd.to_numeric(m["score_m1"], errors="coerce")
        m["score_m2"] = pd.to_numeric(m["score_m2"], errors="coerce")
        if m["score_m1"].isna().any() or m["score_m2"].isna().any():
            raise ValueError("Stratified pick: missing numeric scores for some mp_id in final_pool.")
        picked: list[Any] = []
        rs0 = 0 if random_state is None else int(random_state)
        for i, (label, n, pred) in enumerate(strata):
            mask = pred(m)
            pool = m.loc[mask, "mp_id"]
            if len(pool) < n:
                raise ValueError(f"Stratum {label!r}: need {n} mp_ids, got {len(pool)} after relax.")
            take = pool.sample(n=n, random_state=rs0 + i).tolist()
            picked.extend(take)
        return picked

    return pick


def energy_export(
    files,
    weights,
    key: str,
    output_name: str,
    other_keys: list[str],
    A: int = 150,
    B: int = 150,
    C: int = 80,
    D: int = 48,
    random_state: int | None = 42,
    pick_buffer_ids: PickBufferIds | None = None,
    pick_final_mp_ids: PickFinalMpIds | None = None,
    min_final_pool_size: int | None = None,
):
    """
    Two-stage sampling (override with ``pick_buffer_ids`` / ``pick_final_mp_ids``):

    * **Buffer** (cheap to choose, expensive to run): which ``mp_id``s get relaxed.
    * **Final** (cheap): which relaxed ``mp_id``s appear in the export.

    ``min_final_pool_size``: minimum rows in ``final_pool`` after relax. If unset, uses ``C + D``
    (the default second-stage counts). Set to ``32`` when your final picker returns 32 ids.

    **Why relax count is not ``32``:** each model may output a different structure for the same
    ``mp_id``. The export interleaves both, so we must relax **each model's** structure for every
    buffered ``mp_id``. MLIP work ~= ``len(buffer_ids) * len(files)`` structure relaxations.
    """
    # --- Step 1: load per-model Parquet (scores + structures + optional extra columns) ---
    print("Step 1: Loading and Filtering Invalid Energies...")
    dfs = {}
    for name, path in files.items():
        cols = ["mp_id", key, "simple_structure"] + other_keys
        df = pd.read_parquet(path, columns=cols)
        df[key] = pd.to_numeric(df[key], errors="coerce")
        df = df.groupby("mp_id").head(1).copy()
        total_len = len(df)
        df = df.dropna(subset=[key, "simple_structure"]).copy()
        dfs[name] = df
        print(f"  - {name}: {len(df)} / {total_len} valid records found.")

    # Only compare ``mp_id``s present in every model file (apples-to-apples).
    common_ids = sorted(set.intersection(*(set(df["mp_id"]) for df in dfs.values())))
    print(f"\nStep 2: Found {len(common_ids)} common mp_ids with valid MLIP energies.")

    # --- Step 3: wide table ``score_df`` --- one row per common ``mp_id`` ---
    # Columns: mp_id, <key>_<model> for each model, then rl_gain = sum(weight * score).
    # Example (two models): rl_gain = w_el * score_elastic + w_rl * score_rl; with (-1,+1) it is elastic - rl.
    print("Step 3: Calculating RL improvement scores (Gain)...")
    score_df = pd.DataFrame({"mp_id": list(common_ids)})
    for name, df in dfs.items():
        temp = df[df["mp_id"].isin(common_ids)][["mp_id", key]]
        temp = temp.rename(columns={key: f"{key}_{name}"})
        score_df = score_df.merge(temp, on="mp_id")

    score_df["rl_gain"] = sum(weights[name] * score_df[f"{key}_{name}"] for name in weights)
    score_df = score_df.sort_values("rl_gain", ascending=False)

    print(len(score_df))

    # --- First pick: buffer_ids (subset of mp_id to relax) ---
    buffer_fn = pick_buffer_ids or default_pick_buffer_ids
    buffer_ids = buffer_fn(
        score_df,
        BufferPickArgs(A=A, B=B, random_state=random_state),
    )
    n_models = len(files)
    n_unique_buffer = len(set(buffer_ids))
    if len(buffer_ids) != n_unique_buffer:
        print(f"  Note: buffer list has {len(buffer_ids)} entries but {n_unique_buffer} unique mp_ids.")
    print(
        f"\nStep 4: MLIP relaxation — {n_unique_buffer} distinct mp_ids × {n_models} models "
        f"= {n_unique_buffer * n_models} structure relaxations (300 steps each)."
    )

    # --- Step 4: run relaxation **per model** (structures differ per model for same mp_id) ---
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    valid_relaxed_pool = {name: {} for name in files.keys()}
    for name, df in dfs.items():
        subset = df[df["mp_id"].isin(buffer_ids)]
        ids = subset["mp_id"].tolist()
        structures = [_parse_simple_structure(s) for s in subset["simple_structure"].tolist()]
        relaxed, _energies, converged = ray.get(relax_structures_batch.remote(structures, steps=300))
        for mpid, r_struct, ok in zip(ids, relaxed, converged):
            if ok is True and r_struct is not None:
                relaxed_simple, relaxed_cif = _relaxed_exports(r_struct)
                if relaxed_simple is not None and relaxed_cif is not None:
                    valid_relaxed_pool[name][mpid] = {
                        "relaxed_structure": relaxed_simple,
                        "cif": relaxed_cif,
                    }

    # Keep mp_ids that succeeded for **every** model (intersection of per-model success sets).
    strictly_valid_ids = sorted(set.intersection(*(set(v.keys()) for v in valid_relaxed_pool.values())))

    print(f"\nStep 5: {len(strictly_valid_ids)} IDs passed both score and converged-relax validation.")

    # Survivors joined back to ``score_df`` for rl_gain ordering / stratified predicates.
    final_pool = score_df[score_df["mp_id"].isin(strictly_valid_ids)].sort_values("rl_gain", ascending=False)
    need_final = min_final_pool_size if min_final_pool_size is not None else (C + D)
    if len(final_pool) < need_final:
        raise ValueError(
            f"Not enough valid samples after relax (need {need_final}, got {len(final_pool)}). "
            "Try increasing the buffer size or adjusting picks."
        )
    # --- Second pick: which survivors become CSV rows (default: head C + sample D from tail) ---
    final_fn = pick_final_mp_ids or default_pick_final_mp_ids
    final_selected_ids = final_fn(
        final_pool,
        FinalPickArgs(
            dfs=dfs,
            key=key,
            model_order=list(files.keys()),
            C=C,
            D=D,
            random_state=random_state,
        ),
    )

    # --- Step 6: expand each mp_id to len(files) rows (interleaved by model_order) ---
    print("\nStep 6: Packaging final CSV with interleaved storage...")
    final_rows = []
    model_order = list(files.keys())
    for mpid in final_selected_ids:
        # One block per mp_id: rows appear model_a, model_b, ... so spreadsheets align side-by-side.
        current_gain = final_pool[final_pool["mp_id"] == mpid]["rl_gain"].values[0]
        for name in model_order:
            energy_val = dfs[name][dfs[name]["mp_id"] == mpid][key].values[0]
            current_dict = {
                "mp_id": mpid,
                "model_name": name,
                "rl_gain_score": current_gain,
                "mlip_score": energy_val,
            }
            if "simple_structure" in dfs[name].columns:
                current_dict["simple_structure"] = dfs[name][dfs[name]["mp_id"] == mpid][
                    "simple_structure"
                ].values[0]
                current_dict["relaxed_structure"] = valid_relaxed_pool[name][mpid]["relaxed_structure"]
                current_dict["cif"] = valid_relaxed_pool[name][mpid]["cif"]
            for other_key in other_keys:
                val = dfs[name][dfs[name]["mp_id"] == mpid][other_key].values[0]
                if other_key in ("gt", "elastic_properties"):
                    val = _slice_modulus_dict(val)
                current_dict[other_key] = val
            final_rows.append(current_dict)

    output_df = pd.DataFrame(final_rows)
    out_path = Path(output_name)
    if not out_path.is_absolute():
        out_path = _OUTPUT_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(output_name).lower().endswith(".csv"):
        csv_df = _dataframe_for_utf8_csv(output_df)
        csv_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )
    else:
        output_df.to_parquet(out_path, index=False)

    print("\n--- MISSION SUCCESS ---")
    print(f"File: {out_path}")
    n_rows = len(output_df)
    n_mp = int(output_df["mp_id"].nunique())
    n_models = len(files)
    print(
        f"CSV rows: {n_rows}  (= {n_mp} distinct mp_id × {n_models} models). "
        "Each row is one structure for that (mp_id, model): columns simple_structure (input), "
        "relaxed_structure + cif (after 300-step MLIP relax)."
    )
    print("mp_id order: from second-stage picker (default: top rl_gain, then random from remainder)")
    print(f"Format: Interleaved ({n_models} models per mp_id)")
    print("Validation: All rows are from 300-step relaxed+converged structures, with valid CIF strings.")
    if str(output_name).lower().endswith(".csv"):
        print(
            "CSV: simple_structure = original; relaxed_structure = converged relaxed simple format; "
            "cif = relaxed_structure as pymatgen CIF text."
        )


def _legacy_cli_example_four_models_energy_above_hull():
    """Previously used ``__main__`` snippet; kept for reference (not called)."""
    # np.random.seed(42)
    # random.seed(42)
    # energy_export(
    #     files = {
    #         "With_Thinking": "checkpoints_merged/thinking/conditional+thinking.parquet",
    #         "RL_With_Thinking": "checkpoints_merged/rl_thinking_mix/conditional+thinking.parquet",
    #         "No_Thinking": "checkpoints_merged/no_thinking/conditional+thinking.parquet",
    #         "RL_No_Thinking": "checkpoints_merged/rl_no_thinking_mix/conditional+thinking.parquet",
    #     },
    #     weights = {
    #         "With_Thinking": 1,
    #         "RL_With_Thinking": -1,
    #         "No_Thinking": 1,
    #         "RL_No_Thinking": -1,
    #     },
    #     key = 'energy_above_hull',
    #     output_name = 'sample_1024.parquet',
    #     other_keys = []
    # )


def _verify_elastic_csv_readback(out_path: str) -> None:
    reread = pd.read_csv(out_path, encoding="utf-8-sig")
    json.loads(reread.loc[0, "gt"])
    json.loads(reread.loc[0, "elastic_properties"])
    cif0 = reread.loc[0, "cif"]
    assert isinstance(cif0, str) and (
        "data_" in cif0 or "_symmetry" in cif0 or "_cell_length" in cif0
    )
    assert "simple_structure" in reread.columns and "cif" in reread.columns
    print(
        f"CSV read-back OK: {len(reread)} rows, encoding=utf-8-sig, "
        "simple_structure preserved, cif is pymatgen CIF"
    )


def _main():
    """
    Default export: ``C=80`` + ``D=48`` → **128** distinct ``mp_id``s → **256** CSV rows (2 models).

    Output filename still says ``sample_512_elastic.csv`` for historical reasons; row count follows ``C+D``.
    """
    np.random.seed(42)
    random.seed(42)
    out_path = str(_OUTPUT_DIR / "sample_512_elastic.csv")
    energy_export(
        files={
            "rl_thinking": "checkpoints_merged/rl_thinking_mix/elastic+thinking.parquet",
            "elastic_reward": "checkpoints_merged/rl_elastic_thinking_new/elastic+thinking.parquet",
        },
        weights={
            "rl_thinking": -1,
            "elastic_reward": 1,
        },
        key="elastic_reward_all",
        output_name=out_path,
        other_keys=["gt", "elastic_properties"],
    )
    if out_path.lower().endswith(".csv"):
        _verify_elastic_csv_readback(out_path)


def _main_stratified_elastic_32():
    """
    Export 32 ``mp_id``s (64 CSV rows: 2 models × 32).

    Buffer stage uses stratified **oversample** so we do not relax ~all common ids.
    Tune ``oversample_factor`` upward if Step 5 shrinks the pool and the final stratified
    pick raises "need n mp_ids, got ...".
    """
    np.random.seed(42)
    random.seed(42)
    rs = 42
    rl_name, elastic_name = "rl_thinking", "elastic_reward"
    score_key = "elastic_reward_all"
    # Same predicates for buffer (pre-relax) and final (post-relax); score_m1=RL, score_m2=elastic.
    strata = (
        ("model2_wins", 12, lambda m: (m["score_m2"] - m["score_m1"]) > 0),
        ("model1_wins", 8, lambda m: (m["score_m2"] - m["score_m1"]) < 0),
        ("tie_high_both_2", 6, lambda m: (m["score_m1"] == 2) & (m["score_m2"] == 2)),
        ("tie_zero_both_0", 6, lambda m: (m["score_m1"] == 0) & (m["score_m2"] == 0)),
    )
    pick_buffer = make_stratified_buffer_picker(
        model_1=rl_name,
        model_2=elastic_name,
        score_column_key=score_key,
        strata=strata,
        oversample_factor=2.5,
        random_state=rs,
    )
    pick_final = make_stratified_final_picker(
        model_1=rl_name,
        model_2=elastic_name,
        random_state=rs,
        strata=strata,
    )
    out_path = str(_OUTPUT_DIR / "sample_32_elastic_stratified.csv")
    energy_export(
        files={
            rl_name: "checkpoints_merged/rl_thinking_mix/elastic+thinking.parquet",
            elastic_name: "checkpoints_merged/rl_elastic_thinking_new/elastic+thinking.parquet",
        },
        weights={rl_name: -1, elastic_name: 1},
        key=score_key,
        output_name=out_path,
        other_keys=["gt", "elastic_properties"],
        random_state=rs,
        pick_buffer_ids=pick_buffer,
        pick_final_mp_ids=pick_final,
        min_final_pool_size=32,
        C=0,
        D=0,
    )
    if out_path.lower().endswith(".csv"):
        _verify_elastic_csv_readback(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Export elastic-task CSV from merged Parquets. "
            "Default preset does NOT run stratified sampling; pass elastic_stratified32 for that."
        )
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="elastic_stratified32",
        choices=("elastic512", "elastic_stratified32"),
        help=(
            "elastic512: sample_512_elastic.csv — 128 mp_ids × 2 models = 256 rows (default energy_export C/D). "
            "elastic_stratified32: sample_32_elastic_stratified.csv — 32 mp_ids × 2 = 64 rows."
        ),
    )
    args = parser.parse_args()
    if args.preset == "elastic512":
        _main()
    else:
        _main_stratified_elastic_32()
