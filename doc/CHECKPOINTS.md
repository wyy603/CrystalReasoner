# Checkpoints

This document maps checkpoint folders to the model names used in the paper and records the standard merge and generation flow.

## Checkpoints

| Folder name in `checkpoints_merged` | Model Name | Notes |
| --- | --- | --- |
| `no_thinking` | [🤗 Qwen2.5-3B-CrysReas-Base](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-Base) | SFT baseline without thinking traces. |
| `thinking` | [🤗 Qwen2.5-3B-CrysReas-Thinking](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-Thinking) | SFT baseline with thinking traces. |
| `rl_no_thinking` | [🤗 Qwen2.5-3B-CrysReas-RL](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-RL) | RL from the no-thinking baseline. |
| `rl_thinking_mix` | [🤗 Qwen2.5-3B-CrysReas](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas) | Main RL model from the thinking baseline. |
| `thinking_only_validity` | [🤗 Qwen2.5-3B-CrysReas-NoEnergyTerm](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-NoEnergyTerm) | Validity-focused ablation (no energy term). |
| `thinking_only_energy` | [🤗 Qwen2.5-3B-CrysReas-NoValidityTerm](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-NoValidityTerm) | Energy-focused ablation (no validity term). |
| `spacegroup_thinking` | [🤗 Qwen2.5-3B-CrysReas-SpaceGroup](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-SpaceGroup) | Space-group property specialist. |
| `rl_elastic_thinking_new` | [🤗 Qwen2.5-3B-CrysReas-Elastic](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-Elastic) | Elastic-property specialist. |
| `rl_cte_thinking` | [🤗 Qwen2.5-3B-CrysReas-ThermalExpansion](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-ThermalExpansion) | Thermal-expansion specialist. |
| `crystaltextllm` | [🤗 Qwen2.5-3B-CrysReas-CrystalTextLLM](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-CrystalTextLLM) | Prior-work text format reimplementation. |
| `plaid_wyckoff` | [🤗 Qwen2.5-3B-CrysReas-PLaIDWyckoff](https://huggingface.co/CrystalReasoner/Qwen2.5-3B-CrysReas-PLaIDWyckoff) | Prior-work Wyckoff representation reimplementation. |

## Local Checkpoint Layout

Training writes sharded checkpoints under `checkpoints/`. Generation normally
uses merged Hugging Face style checkpoints under `checkpoints_merged/`.

Typical layout:

```text
checkpoints/
  thinking/global_step_1514/
  crystal_rl/rl_thinking_mix/global_step_*/
checkpoints_merged/
  thinking/
  rl_thinking_mix/
```

## Merge Command

Use the unified runner:

```bash
python scripts/run.py merge \
  --path checkpoints/thinking/global_step_1514 \
  --output-path checkpoints_merged/thinking
```

For an RL actor checkpoint, the input path is usually the `actor` subdirectory:

```bash
python scripts/run.py merge \
  --path checkpoints/crystal_rl/rl_thinking_mix/global_step_125/actor \
  --output-path checkpoints_merged/rl_thinking_mix
```

## Generation Command

```bash
python scripts/run.py generate checkpoints_merged/thinking
```

Specialized generation jobs:

```bash
python scripts/run.py generate_elastic checkpoints_merged/rl_elastic_thinking_new
python scripts/run.py generate_cte checkpoints_merged/rl_cte_thinking
python scripts/run.py generate_crystaltextllm checkpoints_merged/crystaltextllm
python scripts/run.py generate_plaid_wyckoff checkpoints_merged/plaid_wyckoff
```

## Metric Command

```bash
python scripts/run.py run_metric \
  --path checkpoints_merged/thinking/conditional+thinking.parquet \
  --metrics-name composition_consistency spacegroup_consistency stable_unique_novel \
  --prompt-type conditional+thinking \
  --forced
```

## Notes

- Keep large generated parquet files under `checkpoints_merged/<model>/`.
- Use the folder names above when building paper tables and figures.
- If a checkpoint is sharded or FSDP-formatted, merge it before generation.
