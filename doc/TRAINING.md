# Training

This document describes the live training stack in this checkout. User-facing
commands go through `scripts/run.py`, while the real training logic lives under
`crysreas/trainer/`.

## Entrypoints

| Task | User command | Internal entrypoint |
| --- | --- | --- |
| SFT | `scripts/run.py sft_*` | `crysreas.trainer.fsdp_sft_trainer` |
| RL/GRPO | `scripts/run.py rl_*` | `crysreas.trainer.main_ppo` |
| Offline DPO | `scripts/run.py dpo_plaid_wyckoff` | `crysreas.trainer.fsdp_dpo_trainer` |
| Iterative DPO loop | direct module use | `crysreas.trainer.main_dpo` |
| Generation | `scripts/run.py generate*` | `crysreas.trainer.main_generation` |
| Merge | `scripts/run.py merge` | `verl.model_merger merge` |

Experiment names and overrides are defined in `scripts/config.py`.

## SFT Defaults

SFT defaults live in `crysreas/trainer/config/sft_trainer.yaml`.

| Field | Value |
| --- | --- |
| base model | `MegaScience/Qwen2.5-3B-MegaScience` |
| training split | `assets/MP/split_cdvae.json` |
| dataset class | `crysreas/trainer/crystal_dataset.py::CrystalDataset` |
| global batch size | `32` |
| micro batch per GPU | `1` |
| max length | `4096` |
| epochs | `2` |
| optimizer | Adam, lr `1e-4`, betas `(0.9, 0.95)`, weight decay `0.01` |
| LR schedule | cosine, warmup ratio `0.1` |
| gradient clipping | `1.0` |
| dtype | bf16 |
| strategy | FSDP2 |
| gradient checkpointing | enabled |
| LoRA | disabled by default (`lora_rank=0`) |

Registered SFT jobs:

| Job | prompt type | output dir |
| --- | --- | --- |
| `sft_no_thinking` | `conditional+no_thinking` | `checkpoints/no_thinking` |
| `sft_thinking` | `conditional+thinking` | `checkpoints/thinking` |
| `sft_crystaltextllm` | `crystaltextllm_train+no_thinking` | `checkpoints/crystaltextllm` |
| `sft_crystaltextllm_8` | `crystaltextllm_8_train+no_thinking` | `checkpoints/crystaltextllm_8` |
| `sft_plaid_wyckoff` | `plaid_wyckoff_train+no_thinking` | `checkpoints/plaid_wyckoff` |
| `sft_plaid_wyckoff_8` | `plaid_wyckoff_8_train+no_thinking` | `checkpoints/plaid_wyckoff_8` |

Example:

```bash
python scripts/run.py sft_thinking
```

## SFT Dataset Behavior

`CrystalDataset` reads:

- `config.custom_data.split_path`,
- `config.custom_data.db_path`,
- `config.custom_data.prompt_type`.

For each `mp_id`, it loads the Materials Project entry from `assets/MP/MP_shelve`
and calls `crysreas.data.prompt_generator.get_info`.

For prior-work training prompt families, the dataset mixes:

- generation examples with probability `0.66`,
- infill examples with probability `0.34`.

This applies to CrystalTextLLM-style and PLaID++ Wyckoff-style train families.

## RL/GRPO Defaults

RL defaults live in `crysreas/trainer/config/ppo_trainer.yaml` and its component
YAML files. The code uses the Verl PPO trainer stack with
`algorithm.adv_estimator=grpo`.

| Field | Value |
| --- | --- |
| dataset class | `crysreas/trainer/crystal_dataset_rl.py::CrystalDatasetRL` |
| default split | `assets/MP/split_cte.json` |
| default prompt type | `cte+no_thinking` |
| train batch size | `64` |
| validation batch size | `64` |
| max prompt length | `256` |
| max response length | `4096` |
| rollout backend | vLLM |
| rollout samples per prompt | `8` |
| rollout temperature | `1.0` |
| rollout top-p | `1.0` |
| actor lr | `1e-5` |
| PPO mini batch | `32` |
| micro batch per GPU | `1` |
| clip ratio | `0.2` |
| entropy coeff | `0` |
| epochs | `1` |

GRPO settings:

- `gamma=0.98`
- `lam=0.9`
- `norm_adv_by_std_in_grpo=True`
- `use_kl_in_reward=False`
- adaptive KL controller with coefficient `0.001`, target `0.05`, horizon `10000`

Default reward configuration:

- necessary: `fit_format=1`
- weighted: `simple_structure=0.1`, `structure_validity=0.3`,
  `smact_validity=0.3`, `composition_consistency=0.3`,
  `smooth_energy_reward=1`

## Registered RL Jobs

| Job | Start model | prompt type | split | main reward override |
| --- | --- | --- | --- | --- |
| `rl_thinking` | `checkpoints_merged/thinking` | `conditional+thinking` | `split_rl.json` | `energy_reward=10` |
| `rl_thinking_novel` | `checkpoints_merged/thinking` | `conditional+thinking` | `split_rl.json` | `sn_reward=10` |
| `rl_thinking_only_energy` | `checkpoints_merged/thinking` | `conditional+thinking` | `split_rl.json` | only `energy_reward=10` |
| `rl_thinking_only_validity` | `checkpoints_merged/thinking` | `conditional+thinking` | `split_rl.json` | removes smooth energy reward |
| `rl_no_thinking` | `checkpoints_merged/no_thinking` | `conditional+thinking` | `split_rl.json` | `energy_reward=10` |
| `rl_spacegroup_thinking` | `checkpoints_merged/thinking` | `spacegroup+thinking` | `split_rl.json` | `spacegroup_consistency=1` |
| `rl_elastic_thinking_new` | `checkpoints_merged/thinking` | `elastic+thinking` | `split_elastic.json` | `new_elastic_reward=5`, `energy_reward=1` |
| `rl_cte_thinking` | `checkpoints_merged/thinking` | `cte+thinking` | `split_cte.json` | `cte_reward=5`, `energy_reward=1` |

Example:

```bash
python scripts/run.py rl_thinking_novel
```

## Generation

Generation defaults live in `crysreas/trainer/config/generation.yaml`.
`scripts/config.py` overrides the default to use one GPU and 16 generations per
prompt for the registered generation jobs.

| Job | prompt type | split |
| --- | --- | --- |
| `generate` / `generate_rl` | `conditional+thinking` | `split_generation.json` |
| `generate_elastic` | `elastic+thinking` | `split_generation_elastic.json` |
| `generate_cte` / `generate_qha` | `cte+thinking` | `split_generation_cte.json` |
| `generate_crystaltextllm` | `crystaltextllm_generation+no_thinking` | `split_generation.json` |
| `generate_plaid_wyckoff` | `plaid_wyckoff_generation+no_thinking` | `split_generation.json` |

Example:

```bash
python scripts/run.py generate checkpoints_merged/thinking
```

## Merge

FSDP checkpoints must be merged before normal generation:

```bash
python scripts/run.py merge \
  --path checkpoints/thinking/global_step_1514 \
  --output-path checkpoints_merged/thinking
```
