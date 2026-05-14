# Checkpoints

This document maps checkpoint folders to the model names used in the paper and records the standard merge and generation flow.

## Public Checkpoint Folder

Precomputed experiment checkpoints and outputs are available from NYU Box:

```text
https://nyu.app.box.com/folder/361279226287?s=8ufevmo5jlhj4gfwjgzgowo56mftj87g
```

## Checkpoints

| Folder name | Model name in the paper | Notes |
| --- | --- | --- |
| `no_thinking` | CrysReas-Base | SFT baseline without thinking traces. |
| `thinking` | CrysReas-Thinking | SFT baseline with thinking traces. |
| `rl_no_thinking` | CrysReas-RL | RL from the no-thinking baseline. |
| `rl_thinking_mix` | CrysReas | Main RL model from the thinking baseline. |
| `thinking_only_validity` | RL with no energy term | Validity-focused ablation. |
| `thinking_only_energy` | RL with no validity term | Energy-focused ablation. |
| `spacegroup_thinking` | CrysReas-space-group | Space-group property specialist. |
| `rl_elastic_thinking_new` | CrysReas-ElasticProperties | Elastic-property specialist. |
| `rl_cte_thinking` | CrysReas-ThermalExpansion | Thermal-expansion specialist. |
| `crystaltextllm` | CrystalTextLLM-style baseline | Prior-work text format reimplementation. |
| `plaid_wyckoff` | PLaID++ Wyckoff-style baseline | Prior-work Wyckoff representation reimplementation. |

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
