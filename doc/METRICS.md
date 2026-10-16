# Metrics

`crysreas.metric_process` is the preferred evaluation pipeline. It takes a
`pandas.DataFrame`, computes named metrics, and writes metric-annotated parquet,
pickle, or CSV files. It is used both for offline evaluation and for reward
components that share the same parsing and MLIP machinery.

## CLI

```bash
python -m crysreas.metric_process \
  --path checkpoints_merged/thinking/conditional+thinking.parquet \
  --output-path checkpoints_merged/thinking/conditional+thinking_metric.parquet \
  --metrics-name simple_structure structure_validity smact_validity composition_consistency spacegroup_consistency stable_unique_novel \
  --prompt-type conditional+thinking \
  --num-workers 1 \
  --forced
```

Important flags:

- `--path`: input JSON, parquet, pickle, or CSV.
- `--output-path`: output file. If omitted, the input file is overwritten.
- `--metrics-name`: one or more registered metric names.
- `--prompt-type`: prompt family, split by `+`, used for ground-truth fields.
- `--num-workers`: Ray parallelism for lightweight metrics.
- `--forced`: recompute target columns even though they already exist.
- `--level=debug`: enables more verbose metric progress logs.

The same CLI is available through the unified runner:

```bash
python scripts/run.py run_metric \
  --path checkpoints_merged/thinking/conditional+thinking.parquet \
  --metrics-name composition_consistency spacegroup_consistency stable_unique_novel \
  --prompt-type conditional+thinking \
  --forced
```

## Python API

```python
import pandas as pd
from crysreas.metric_process import MetricProcess, merge_metric_process_config, run_metrics

df = pd.read_parquet("samples.parquet")
config = merge_metric_process_config(None, {"prompt_type": ["conditional", "thinking"]})

out = run_metrics(
    df.copy(),
    ["simple_structure", "structure_validity", "composition_consistency"],
    config=config,
    forced=True,
    log=True,
)

with MetricProcess(config) as mp:
    out = mp.process(df.copy(), ["spacegroup_consistency"], forced=True, log=True)
```

Importing `crysreas.metric_process` registers the built-in metrics from
`crysreas/metric_process/basic.py`.

## Scheduling Model

Metrics are registered in `crysreas/metric_process/registry.py` with a metric
function, topological order, optional dependencies, and optional tags such as
`heavy`.

`run_metrics` expands dependencies and runs metrics in topological order.
`MetricProcess` uses Ray for lightweight metrics and keeps heavy MLIP metrics in
the current process.

## Core Metrics

| Metric | Main columns | Notes |
| --- | --- | --- |
| `simple_structure` | `simple_structure` | Parses generated text into `SimpleCrystal` / pymatgen structure objects. |
| `gt` | `gt` | Loads ground truth from `assets/MP/MP_shelve`. |
| `structure_validity` | `structure_validity` | Geometric validity checks. |
| `smact_validity` | `smact_validity` | Chemical validity through SMACT. |
| `composition_consistency` | `composition_consistency` | Reduced composition match against ground truth. |
| `spacegroup_consistency` | `spacegroup_consistency` | Does the structure carry the requested symmetry? Gated on both validity checks. The test follows `spacegroup_criterion`: `subgroup` accepts a detected group of which the request is a proper subgroup, `strict` demands integer equality. |
| `fit_format` | `fit_format` | Output-format check used by reward calculation. |
| `small_atoms` | `small_atoms` | Flags structures below the atom-count cutoff. |
| `seed` | `seed` | Index of a sample within its prompt, read as the seed it came from. Each seed value selects one structure per prompt, so it picks out one complete evaluation of the split. |

## MLIP and Stability Metrics

| Metric | Main columns | Notes |
| --- | --- | --- |
| `relaxed_structures` | `relaxed_structures`, `energies` | MatterSim relaxation path. |
| `relaxed_structures_300` | `relaxed_structures`, `energies` | Same path with a 300-step budget. |
| `energy_above_hull` | `energy_above_hull`, `is_stable` | MatterGen hull evaluation. |
| `is_stable` | `is_stable` | `E_hull <= stability_threshold`. Derived from the hull column every time it is requested and never cached, so the flag always matches the distances beside it. A missing hull distance counts as unstable. |
| `is_metastable` | `is_metastable` | `E_hull <= metastability_threshold`, the wider window S.S.U.N. uses. Derived, never cached. |
| `stable_unique_novel` | `is_novel`, `is_unique`, `stable_unique_novel` | S.U.N.: stable, unique and novel. Uniqueness follows `uniqueness_scope` and novelty follows `novelty_reference`. |
| `spacegroup_stable_unique_novel` | `spacegroup_stable_unique_novel` | S.S.U.N.: metastable, unique, novel and space-group consistent. Sits on the wider energy window, so it exceeds S.U.N. despite carrying one more factor; the two are not comparable to each other. |
| `fmax` | `fmax` | Maximum force norm from relaxed structures. |
| `elastic_properties` | `elastic_properties` | Elastic tensor and modulus calculation. |
| `cte` | `cte` | Thermal expansion / QHA path. |

Four of the columns above depend on settings in
`crysreas.metric_process.config` instead of on the row alone: the two stability
thresholds, `spacegroup_criterion`, `novelty_reference` and `uniqueness_scope`.
Passing a different config produces a different column under the same name, so
record the settings alongside any number taken from these metrics.
`mattergen_config()` collects MatterGen's setting for all four.

## Reward Metrics

| Metric | Role |
| --- | --- |
| `energy_reward` | Scalar reward from hull energy and composition consistency. |
| `smooth_energy_reward` | Smooth energy reward variant used by default RL config. |
| `elastic_reward` | Elastic-property reward using selected candidates. |
| `elastic_reward_all` | Elastic-property reward across all valid rows. |
| `cte_reward` | Thermal-expansion conditioned reward. |
| `cte_reward_all` | Thermal-expansion reward for full-row evaluation. |
| `new_elastic_reward` | Range-constraint elastic reward used by the current elastic specialist. |
| `sn_reward` | Stability-and-novelty reward combining the hull term with the novelty flag. |

## Prompt-Type Configuration

`prompt_type` is split on `+`. The first part controls which ground-truth fields
are added by the `gt` metric.

Examples:

- `conditional+thinking`: composition and space-group targets.
- `spacegroup+thinking`: explicit space-group conditioning.
- `elastic+thinking`: adds bucketed elastic targets.
- `cte+thinking`: adds thermal-expansion target ranges.
- `crystaltextllm_generation+no_thinking`: CrystalTextLLM-style parser path.
- `plaid_wyckoff_generation+no_thinking`: PLaID++ Wyckoff-style parser path.

## Serialization

Parquet cannot store pymatgen `Structure` objects directly. Before saving,
`metric_process` calls `df_serialize`, which converts structure columns to the
project's simple crystal text representation and converts array-like dict values
to serializable lists.

The corresponding helper `df_deserialize` restores known structure columns when
metric inputs are loaded.

## Adding a Metric

1. Implement a function that mutates and returns a DataFrame.
2. Register it with `@register("metric_name", topo=..., dependencies=(...))`.
3. Put shared column creation in an `ensure_*` helper if other metrics reuse it.
4. Add a focused test under `tests/test_metric_process/`.

Minimal pattern:

```python
from crysreas.metric_process.registry import register

@register("my_metric", topo=50, dependencies=("simple_structure",))
def metric_my_metric(df, config):
    df["my_metric"] = ...
    return df
```

## Tests

Metric tests live under `tests/test_metric_process/`. For parser or dependency
changes, run the metric tests together with any prompt-family tests that route
into the same parser.
