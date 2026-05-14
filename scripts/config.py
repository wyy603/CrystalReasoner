"""
Experiment registry for ``scripts/run.py``: one key -> one local job spec.

Keys mirror former ``task_*.sh`` / ``run_*.sh`` names; see ``EXPERIMENTS`` docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ExperimentSpec:
    """Single experiment definition."""

    kind: Literal["ppo", "sft", "dpo", "metric", "merge", "generate"]
    description: str = ""

    # --- PPO (``run_pipeline.py ppo``) ---
    gpu: int = 1
    ppo_args: str = ""

    # --- SFT ---
    sft_gpu: int = 1
    sft_local_dir: str = "checkpoints/thinking"
    sft_args: str = ""

    # --- Offline DPO ---
    dpo_gpu: int = 1
    dpo_local_dir: str = "checkpoints/dpo_plaid_wyckoff"
    dpo_args: str = ""

    # --- Metric (``python -m crysreas.metric_process``) ---
    metric_argv: list[str] = field(default_factory=list)
    # If True, ``extras`` after the experiment name are passed through as metric_process CLI args.
    metric_argv_from_cli: bool = False

    # --- Merge (``python -m verl.model_merger merge``) ---
    merge_path: str = ""
    merge_output_path: str = ""

    # --- Generate (``python -m crysreas.trainer.main_generation``) ---
    generate_gpu: int = 2
    generate_args: str = ""


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "rl_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_thinking.sh — energy + thinking mix.",
        ppo_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "~custom_reward_function.weights.smooth_energy_reward "
            "+custom_reward_function.weights.energy_reward=10 "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=rl_thinking_mix"
        ),
    ),
    "rl_thinking_novel": ExperimentSpec(
        kind="ppo",
        description="rl_thinking mix, but use sn_reward (energy_reward_scalar + novel bonus).",
        ppo_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "~custom_reward_function.weights.smooth_energy_reward "
            "+custom_reward_function.weights.sn_reward=10 "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=rl_thinking_novel"
        ),
    ),
    "rl_thinking_only_energy": ExperimentSpec(
        kind="ppo",
        description="thinking only energy",
        ppo_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "~custom_reward_function.weights "
            "+custom_reward_function.weights.energy_reward=10 "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=thinking_only_energy"
        ),
    ),
    "rl_thinking_only_validity": ExperimentSpec(
        kind="ppo",
        description="thinking only validity",
        ppo_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "~custom_reward_function.weights.smooth_energy_reward "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=thinking_only_energy"
        ),
    ),
    "rl_spacegroup_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_spacegroup_thinking.sh.",
        ppo_args=(
            "data.custom_data.prompt_type=spacegroup+thinking "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "~custom_reward_function.weights.smooth_energy_reward "
            "+custom_reward_function.weights.spacegroup_consistency=1 "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=spacegroup_thinking"
        ),
    ),
    "rl_elastic_no_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_elastic_no_thinking.sh.",
        ppo_args=(
            "data.custom_data.prompt_type=elastic+no_thinking "
            "data.custom_data.split_path=assets/MP/split_elastic.json "
            "~custom_reward_function.weights +custom_reward_function.weights.elastic_reward=1 "
            "actor_rollout_ref.model.path=checkpoints_merged/rl_no_thinking "
            "trainer.experiment_name=elastic_no_thinking"
        ),
    ),
    "rl_cte_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_cte_thinking.sh.",
        ppo_args=(
            "data.custom_data.prompt_type=cte+thinking "
            "data.custom_data.split_path=assets/MP/split_cte.json "
            "~custom_reward_function.weights +custom_reward_function.weights.cte_reward=5 "
            "+custom_reward_function.weights.energy_reward=1 "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=cte_thinking"
        ),
    ),
    "rl_elastic_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_elastic_thinking.sh.",
        ppo_args=(
            "data.custom_data.prompt_type=elastic+thinking "
            "data.custom_data.split_path=assets/MP/split_elastic.json "
            "~custom_reward_function.weights +custom_reward_function.weights.elastic_reward_all=1 "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.experiment_name=elastic_thinking2"
        ),
    ),
    "rl_elastic_thinking_new": ExperimentSpec(
        kind="ppo",
        description="elastic+thinking with 5*new_elastic_reward + energy_reward (2 epochs).",
        ppo_args=(
            "data.custom_data.prompt_type=elastic+thinking "
            "data.custom_data.split_path=assets/MP/split_elastic.json "
            "~custom_reward_function.weights "
            "+custom_reward_function.weights.new_elastic_reward=5 "
            "+custom_reward_function.weights.energy_reward=1 "
            "actor_rollout_ref.model.path=checkpoints_merged/thinking "
            "trainer.total_epochs=2 "
            "trainer.experiment_name=rl_elastic_thinking_new2"
        ),
    ),
    "rl_no_thinking": ExperimentSpec(
        kind="ppo",
        description="Former task_no_thinking.sh.",
        ppo_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "~custom_reward_function.weights.smooth_energy_reward "
            "+custom_reward_function.weights.energy_reward=10 "
            "data.custom_data.split_path=assets/MP/split_rl.json "
            "actor_rollout_ref.model.path=checkpoints_merged/no_thinking "
            "trainer.experiment_name=rl_no_thinking"
        ),
    ),
    "sft_no_thinking": ExperimentSpec(
        kind="sft",
        description="Former task_sft.sh.",
        sft_local_dir="checkpoints/no_thinking",
        sft_args=(
            "data.custom_data.prompt_type=conditional+no_thinking "
            "trainer.experiment_name=no_thinking "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "sft_thinking": ExperimentSpec(
        kind="sft",
        description="Former task_sft.sh.",
        sft_local_dir="checkpoints/thinking",
        sft_args=(
            "data.custom_data.prompt_type=conditional+thinking "
            "trainer.experiment_name=thinking "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "sft_crystaltextllm": ExperimentSpec(
        kind="sft",
        description="crysreas CrystalTextLLM reimplementation with generation+infill training mix.",
        sft_local_dir="checkpoints/crystaltextllm",
        sft_args=(
            "data.custom_data.prompt_type=crystaltextllm_train+no_thinking "
            "trainer.experiment_name=sft_crystaltextllm "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "sft_crystaltextllm_8": ExperimentSpec(
        kind="sft",
        description="crysreas CrystalTextLLM reimplementation with 8-digit structure precision.",
        sft_local_dir="checkpoints/crystaltextllm_8",
        sft_args=(
            "data.custom_data.prompt_type=crystaltextllm_8_train+no_thinking "
            "trainer.experiment_name=sft_crystaltextllm_8 "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "sft_plaid_wyckoff": ExperimentSpec(
        kind="sft",
        description="PLaID++ Wyckoff-style SFT with generation+infill training mix.",
        sft_local_dir="checkpoints/plaid_wyckoff",
        sft_args=(
            "data.custom_data.prompt_type=plaid_wyckoff_train+no_thinking "
            "trainer.experiment_name=sft_plaid_wyckoff "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "dpo_plaid_wyckoff": ExperimentSpec(
        kind="dpo",
        description="Offline DPO for the PLaID++ Wyckoff model.",
        dpo_local_dir="checkpoints/dpo_plaid_wyckoff",
        dpo_args=(
            "model.partial_pretrain=checkpoints_merged/plaid_wyckoff "
            "model.reference_pretrain=checkpoints_merged/plaid_wyckoff "
            "data.custom_data.prompt_type=plaid_wyckoff_train+no_thinking "
            "trainer.experiment_name=dpo_plaid_wyckoff"
        ),
    ),
    "sft_plaid_wyckoff_8": ExperimentSpec(
        kind="sft",
        description="PLaID++ Wyckoff-style SFT with 8-digit structure precision.",
        sft_local_dir="checkpoints/plaid_wyckoff_8",
        sft_args=(
            "data.custom_data.prompt_type=plaid_wyckoff_8_train+no_thinking "
            "trainer.experiment_name=sft_plaid_wyckoff_8 "
            "data.custom_data.split_path=assets/MP/split_cdvae.json"
        ),
    ),
    "run_metric": ExperimentSpec(
        kind="metric",
        description="Custom metric_process CLI: pass flags after run_metric (e.g. --path ... --metrics-name ...).",
        metric_argv_from_cli=True,
    ),
    "metric_qha": ExperimentSpec(
        kind="metric",
        description="Former run_qha.sh — precompute QHA / CTE-related columns (cte metric).",
        metric_argv=[
            "--path",
            "assets/MP/split_rl.json",
            "--output-path",
            "assets/MP/split_rl_qha.pkl",
            "--metrics-name",
            "cte",
        ],
    ),
    "metric_elastic": ExperimentSpec(
        kind="metric",
        description="Former run_elastic.sh — elastic_reward on split_rl.json.",
        metric_argv=[
            "--path",
            "assets/MP/split_rl.json",
            "--output-path",
            "assets/MP/split_rl_elastic.pkl",
            "--metrics-name",
            "elastic_reward",
        ],
    ),
    "merge": ExperimentSpec(
        kind="merge",
        description="Merge FSDP checkpoint to HuggingFace format (accept --path and --output_path).",
    ),
    "generate": ExperimentSpec(
        kind="generate",
        description="Run main_generation.py (default generation split), pass model dir as first arg.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=conditional+thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
    "generate_rl": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation.json for RL models.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=conditional+thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
    "test_generate": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation.json for RL models. (with log_prob)",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=conditional+thinking "
            "data.custom_data.split_path=assets/MP/split_generation_one.json "
            "rollout.calculate_log_probs=true "
        ),
    ),
    "generate_rl_log_prob": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation.json for RL models. (with log_prob)",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=conditional+thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json "
            "rollout.calculate_log_probs=true "
        ),
    ),
    "generate_rl_log_prob2": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation.json for RL models. (with log_prob)",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=conditional+thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json "
            "rollout.calculate_log_probs=true "
        ),
    ),
    "generate_elastic": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation_elastic.json.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=elastic+thinking "
            "data.custom_data.split_path=assets/MP/split_generation_elastic.json"
        ),
    ),
    "generate_cte": ExperimentSpec(
        kind="generate",
        description="Generate on split_generation_cte.json (cte+thinking).",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=cte+thinking "
            "data.custom_data.split_path=assets/MP/split_generation_cte.json"
        ),
    ),
    "generate_qha": ExperimentSpec(
        kind="generate",
        description="Alias of generate_cte for backward compatibility.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=cte+thinking "
            "data.custom_data.split_path=assets/MP/split_generation_cte.json"
        ),
    ),
    "generate_crystaltextllm": ExperimentSpec(
        kind="generate",
        description="Generate CrystalTextLLM-style structures on split_generation.json.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=crystaltextllm_generation+no_thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
    "generate_crystaltextllm_8": ExperimentSpec(
        kind="generate",
        description="Generate CrystalTextLLM-style structures with 8-digit structure precision.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=crystaltextllm_8_generation+no_thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
    "generate_plaid_wyckoff": ExperimentSpec(
        kind="generate",
        description="Generate PLaID++ Wyckoff-style structures on split_generation.json.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=plaid_wyckoff_generation+no_thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
    "generate_plaid_wyckoff_8": ExperimentSpec(
        kind="generate",
        description="Generate PLaID++ Wyckoff-style structures with 8-digit structure precision.",
        generate_args=(
            "trainer.n_gpus_per_node=1 "
            "data.n_samples=16 "
            "data.custom_data.prompt_type=plaid_wyckoff_8_generation+no_thinking "
            "data.custom_data.split_path=assets/MP/split_generation.json"
        ),
    ),
}
