# Dataset Preparation

This project stores Materials Project data and derived split files under
`assets/MP/`. The training and evaluation code expects the shelve database,
split JSON files, and selected metric-precomputed parquet files to be available
there.

## Main Data Assets

| Path | Role |
| --- | --- |
| `assets/MP/MP_shelve` | Shelve database keyed by Materials Project id. Used by SFT, RL, generation, and metrics. |
| `assets/MP/split_cdvae.json` | Upstream CDVAE-style MP-20 split used for SFT and prior-work baselines. |
| `assets/MP/split_rl.json` | Stability/RL training subset. |
| `assets/MP/split_elastic.json` | Elastic-property RL subset. |
| `assets/MP/split_cte.json` | Thermal-expansion RL subset. |
| `assets/MP/split_generation*.json` | Evaluation prompt sets for generation. |

## Build the Materials Project Shelve

```bash
python -m crysreas.data.download --args.download_type summary --args.add_new
python -m crysreas.data.download --args.download_type robocrys
```

The shelve entries are later read by:

- `crysreas/trainer/crystal_dataset.py`
- `crysreas/trainer/crystal_dataset_rl.py`
- `crysreas/trainer/main_generation.py`
- `crysreas.metric_process` ground-truth metrics

## Logic of Split Files

The split files are already in the github repo. But if you want to reproduce them, please run the following scripts.

```bash
python -m crysreas.metric_process \
  --path assets/MP/split_cdvae.json \
  --output-path assets/MP/split_cdvae_metric.parquet \
  --metrics-name relaxed_structures fmax elastic_properties \
  --num-workers 1 \
  --forced

python -m crysreas.metric_process \
  --path assets/MP/split_small_atoms.json \
  --output-path assets/MP/split_small_atoms_metric.parquet \
  --metrics-name cte \
  --num-workers 1 \
  --forced

python -m crysreas.data.generate_split
```

Current local split sizes are:

| Split | train | test | val |
| --- | ---: | ---: | ---: |
| `split_cdvae.json` | 24231 | 8141 | 8095 |
| `split_full.json` | 21834 | 7389 | 7285 |
| `split_rl.json` | 8000 | 512 | 7285 |
| `split_elastic.json` | 4000 | 256 | 7281 |
| `split_cte.json` | 4000 | 256 | 3409 |
| `split_generation.json` | 0 | 1024 | 8095 |
| `split_generation_elastic.json` | 0 | 512 | 7281 |
| `split_generation_cte.json` | 0 | 256 | 3409 |
| `split_small_atoms.json` | 10243 | 3554 | 3467 |

## Insert MLIP Properties

```bash
python -m crysreas.data.mlip.insert_properties
```

This step stores additional MLIP-derived fields back into the project data
assets for property-conditioned prompts and rewards.

## Troubleshooting

If Materials Project schema imports fail, refresh the API packages:

```bash
python -m pip install -U mp-api emmet-core
```