from __future__ import annotations

import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tyro


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _latest_global_step(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("global_step_*"):
        match = re.search(r"global_step_(\d+)$", path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No global_step_* checkpoint found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list):
        if len(prompt) == 1 and isinstance(prompt[0], dict) and "content" in prompt[0]:
            return str(prompt[0]["content"])
        if all(isinstance(item, dict) and "content" in item for item in prompt):
            return "\n".join(str(item["content"]) for item in prompt)
        return "\n".join(str(item) for item in prompt)
    if isinstance(prompt, dict):
        return str(prompt.get("content", prompt))
    return str(prompt)


def _response_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _energy_bucket(row: pd.Series, include_invalid_rejected: bool) -> str | None:
    try:
        e_hull = float(row.get("energy_above_hull"))
    except (TypeError, ValueError):
        e_hull = float("nan")
    if math.isfinite(e_hull):
        if e_hull <= 0:
            return "stable"
        if e_hull <= 0.08:
            return "metastable"
        return "unstable"
    return "unstable" if include_invalid_rejected else None


def _sample_rows(rows: list[int], count: int, rng: random.Random) -> list[int]:
    if not rows or count <= 0:
        return []
    if len(rows) >= count:
        return rng.sample(rows, count)
    return [rng.choice(rows) for _ in range(count)]


def _append_pair(
    out: list[dict[str, Any]],
    df: pd.DataFrame,
    chosen_idx: int,
    rejected_idx: int,
    pair_type: str,
    chosen_bucket: str,
    rejected_bucket: str,
) -> None:
    chosen_row = df.loc[chosen_idx]
    rejected_row = df.loc[rejected_idx]
    prompt = _prompt_to_text(chosen_row["prompt"])
    chosen = _response_text(chosen_row["responses"])
    rejected = _response_text(rejected_row["responses"])
    if chosen is None or rejected is None:
        return
    out.append(
        {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "pair_type": pair_type,
            "chosen_bucket": chosen_bucket,
            "rejected_bucket": rejected_bucket,
            "chosen_mp_id": chosen_row.get("mp_id"),
            "rejected_mp_id": rejected_row.get("mp_id"),
        }
    )


def build_plaid_dpo_pairs(
    scored_path: str | Path,
    output_path: str | Path,
    seed: int = 1,
    include_invalid_rejected: bool = True,
) -> int:
    """Build PLaID++-style same-prompt preference pairs from a scored generation parquet."""
    scored_path = Path(scored_path)
    output_path = Path(output_path)
    df = pd.read_parquet(scored_path)
    if "responses" not in df.columns or "prompt" not in df.columns:
        raise ValueError("Scored parquet must contain 'prompt' and 'responses' columns.")
    if "energy_above_hull" not in df.columns:
        raise ValueError("Scored parquet must contain 'energy_above_hull'. Run metric_process first.")

    df = df.reset_index(drop=True)
    df["_prompt_text"] = df["prompt"].apply(_prompt_to_text)
    if "mp_id" in df.columns:
        group_keys = ["mp_id", "_prompt_text"]
    else:
        group_keys = ["_prompt_text"]

    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for _, group in df.groupby(group_keys, dropna=False, sort=False):
        stable: list[int] = []
        metastable: list[int] = []
        unstable: list[int] = []
        for idx, row in group.iterrows():
            bucket = _energy_bucket(row, include_invalid_rejected=include_invalid_rejected)
            if bucket == "stable":
                stable.append(idx)
            elif bucket == "metastable":
                metastable.append(idx)
            elif bucket == "unstable":
                unstable.append(idx)

        for idx in stable:
            for rejected_idx in _sample_rows(metastable, 1, rng):
                _append_pair(pairs, df, idx, rejected_idx, "stable_vs_metastable", "stable", "metastable")
            for rejected_idx in _sample_rows(unstable, 2, rng):
                _append_pair(pairs, df, idx, rejected_idx, "stable_vs_unstable", "stable", "unstable")

        for idx in metastable:
            for rejected_idx in _sample_rows(unstable, 2, rng):
                _append_pair(pairs, df, idx, rejected_idx, "metastable_vs_unstable", "metastable", "unstable")

        if "is_novel" in group.columns:
            stable_novel = [idx for idx in stable if bool(df.loc[idx].get("is_novel"))]
            stable_not_novel = [idx for idx in stable if not bool(df.loc[idx].get("is_novel"))]
            for idx in stable_novel:
                for rejected_idx in _sample_rows(stable_not_novel, 1, rng):
                    _append_pair(pairs, df, idx, rejected_idx, "stable_novel_vs_stable_not_novel", "stable_novel", "stable_not_novel")

        if "spacegroup_consistency" in group.columns:
            acceptable = stable + metastable
            sg_match = [idx for idx in acceptable if bool(df.loc[idx].get("spacegroup_consistency"))]
            sg_mismatch = [idx for idx in acceptable if not bool(df.loc[idx].get("spacegroup_consistency"))]
            for idx in sg_match:
                for rejected_idx in _sample_rows(sg_mismatch, 1, rng):
                    _append_pair(pairs, df, idx, rejected_idx, "matching_sg_vs_non_matching_sg", "acceptable_sg_match", "acceptable_sg_mismatch")

    if not pairs:
        raise ValueError(
            "No DPO pairs were created. Increase data.n_samples, check metric columns, or inspect generated stability rates."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_parquet(output_path)
    print(f"Wrote {len(pairs)} DPO pairs to {output_path}", flush=True)
    return len(pairs)


@dataclass
class IterativeDPOConfig:
    base_model: str = "checkpoints_merged/plaid_wyckoff"
    work_dir: str = "checkpoints_merged/plaid_wyckoff_dpo_iterations"
    rounds: int = 1
    generation_gpus: int = 1
    dpo_gpus: int = 1
    n_samples: int = 16
    generation_batch_size: int = 128
    split_path: str = "assets/MP/split_generation.json"
    split_type: str = "test"
    db_path: str = "assets/MP/MP_shelve"
    generation_prompt_type: str = "plaid_wyckoff_generation+no_thinking"
    dpo_prompt_type: str = "plaid_wyckoff_train+no_thinking"
    metric_names: list[str] = field(
        default_factory=lambda: ["energy_above_hull", "stable_unique_novel", "spacegroup_consistency"]
    )
    metric_workers: int = 1
    seed: int = 1
    include_invalid_rejected: bool = True
    skip_generation: bool = False
    skip_metric: bool = False
    skip_train: bool = False


def run_iterative_dpo(config: IterativeDPOConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    work_dir = Path(config.work_dir)
    current_model = Path(config.base_model)
    if config.rounds < 1:
        raise ValueError("rounds must be >= 1")

    for round_idx in range(1, config.rounds + 1):
        round_dir = work_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        generated_path = round_dir / "generations.parquet"
        scored_path = round_dir / "scored.parquet"
        pairs_path = round_dir / "dpo_pairs_train.parquet"
        checkpoint_dir = round_dir / "checkpoints"
        merged_model = round_dir / "model"

        if not config.skip_generation:
            _run(
                [
                    sys.executable,
                    "-m",
                    "crysreas.trainer.main_generation",
                    f"model.path={current_model}",
                    f"trainer.n_gpus_per_node={config.generation_gpus}",
                    f"data.n_samples={config.n_samples}",
                    f"data.batch_size={config.generation_batch_size}",
                    f"data.output_path={generated_path}",
                    f"data.custom_data.prompt_type={config.generation_prompt_type}",
                    f"data.custom_data.split_path={config.split_path}",
                    f"data.custom_data.split_type={config.split_type}",
                    f"data.custom_data.db_path={config.db_path}",
                ],
                cwd=project_root,
            )

        if not config.skip_metric:
            _run(
                [
                    sys.executable,
                    "-m",
                    "crysreas.metric_process",
                    "--path",
                    str(generated_path),
                    "--output-path",
                    str(scored_path),
                    "--prompt-type",
                    config.generation_prompt_type,
                    "--num-workers",
                    str(config.metric_workers),
                    "--forced",
                    "--metrics-name",
                    *config.metric_names,
                ],
                cwd=project_root,
            )

        build_plaid_dpo_pairs(
            scored_path=scored_path,
            output_path=pairs_path,
            seed=config.seed + round_idx,
            include_invalid_rejected=config.include_invalid_rejected,
        )

        if config.skip_train:
            continue

        _run(
            [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={config.dpo_gpus}",
                "-m",
                "crysreas.trainer.fsdp_dpo_trainer",
                f"model.partial_pretrain={current_model}",
                f"model.reference_pretrain={current_model}",
                f"data.train_files={pairs_path}",
                f"data.val_files={pairs_path}",
                f"data.custom_data.prompt_type={config.dpo_prompt_type}",
                f"trainer.default_local_dir={checkpoint_dir}",
                f"trainer.experiment_name=dpo_plaid_wyckoff_round_{round_idx}",
                "trainer.resume_mode=disable",
            ],
            cwd=project_root,
        )

        latest_ckpt = _latest_global_step(checkpoint_dir)
        _run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(latest_ckpt),
                "--target_dir",
                str(merged_model),
            ],
            cwd=project_root,
        )
        current_model = merged_model
        print(f"Round {round_idx} complete. Next model: {current_model}", flush=True)


def main(config: IterativeDPOConfig) -> None:
    run_iterative_dpo(config)


if __name__ == "__main__":
    tyro.cli(main)
