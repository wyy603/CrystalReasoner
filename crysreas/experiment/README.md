# Evaluation protocol and results

Every number in the paper is produced here, from columns written by
`crysreas.metric_process`. Nothing in this directory recomputes a judgement:
if a table and a column disagree, the table is wrong.

```
generation            metric_process                 experiment
-----------           --------------                 ----------
n_samples per    ->   one row per structure,     ->  rates, error bars,
held-out prompt       one column per judgement       tables, figures
```

## The four judgements

A structure is scored on four questions. Each one has a knob in
`crysreas.metric_process.config`, and `mattergen_config()` collects MatterGen's
setting for all four.

| Judgement | Config key | Default | MatterGen |
|---|---|---|---|
| Is it stable? | `stability_threshold` | `E_hull <= 0` | `E_hull < 0.1` |
| …metastable (S.S.U.N. only)? | `metastability_threshold` | `E_hull <= 0.1` | — |
| Does it have the requested symmetry? | `spacegroup_criterion` | `subgroup` | `strict` |
| Is it novel? | `novelty_reference` | `train` | `mp_alexandria` |
| Is it unique? | `uniqueness_scope` | `seed` | `pooled` |

### Stability

Two thresholds, both from PLaID++. The `Stability` column and the stability
factor of **S.U.N.** use `E_hull <= 0`, i.e. on or below the hull. **S.S.U.N.**
uses the wider `E_hull <= 0.1` metastability window. The two rates therefore sit
on different energy scales and are not meant to be read against each other:
S.S.U.N. is the larger of the two despite carrying one more factor.

`is_stable` and `is_metastable` are **derived, not cached**: both are
recomputed from `energy_above_hull` and the config every time they are
requested. Caching them would let a flag written under an older threshold
survive a config change and silently contradict the hull distances sitting next
to it. Recomputing is a vectorised comparison and costs nothing. `NaN` hull
distances — a failed relaxation — count as unstable, which keeps the denominator
equal to the number of structures drawn.

### Space group

`subgroup` counts a structure when it carries *at least* the requested
symmetry: the detected group equals the requested one, or the requested one is a
proper subgroup of it. The motivating case is P3, where the overwhelming
majority of structures generated for a P3 request are detected as P3m1, a
supergroup of P3, and score zero under integer equality.

The reading is one-sided — it can only raise a score — so it is an alternative,
not a correction, and it rewards over-symmetrisation. Report it as such.

### Novelty

Every evaluation prompt fixes a *known* Materials Project composition. Against
MP+Alexandria, "novel" therefore means "failed to produce the known phase", and
the metric penalises accuracy. Against the SFT training split it asks an
answerable question: did the model produce something it was not trained on?

Carry the limit with the number: train and test ids are disjoint, and only a
minority of test materials share a reduced formula with any training entry, so
novelty-vs-train can only fail on that minority.

### Uniqueness

Uniqueness is a property of a **set**, so the only thing that matters is which
set is scored. The matcher refuses any pair with different reduced formulas, so:

- **pooled** — one set for the whole table. Every prompt pins one composition,
  so every comparison falls between samples of a *single* composition and the
  metric rewards a model for answering the same composition differently each
  time. That is the opposite of what a composition-conditioned task asks.
- **seed** — one set per seed. A seed holds exactly one structure per
  composition, so a duplicate can only arise between two *different* requested
  compositions, i.e. when the model returned the same structure whatever it was
  asked for. That is the failure uniqueness should detect.

Uniqueness is sensitive to set size, so only compare runs scored over sets of
equal size.

## Seeds and error bars

Generation draws `n_samples` completions per prompt in one call, and the loader
explodes them in order, so the samples of a prompt are exchangeable. Row `j` of
every prompt is labelled seed `j` (`ensure_seed`). Each seed then holds exactly
one structure per composition and is a complete evaluation of the split.

Every rate is the mean over seeds of the per-seed rate; every error bar is the
standard error across seeds. Two consequences are worth stating wherever the
numbers appear:

1. These error bars report **run-to-run** variation. They do **not** include the
   variation from choosing which compositions to evaluate on, so they are not
   comparable with a per-composition clustered standard error.
2. Seed labels are **not aligned across models**. Seed 3 of one run is not the
   same random seed as seed 3 of another. Never pair seeds between runs.

With a small number of seeds a visible gap is often not a separation. Use
`evaluate.separation(a, b, metric)` and quote the result before claiming a win.

## Producing the results

```python
from crysreas.experiment.evaluate import load_run, compare, to_latex, separation
from crysreas.experiment.evaluate import METRICS

runs = {
    "CrysReas":       load_run("path/to/full.parquet"),
    "CrystalTextLLM": load_run("path/to/crystaltextllm.parquet"),
}
print(compare(runs).to_string(index=False))
print(to_latex(runs, caption="...", label="tab:compare_priors"))
print(separation(runs["CrysReas"], runs["CrystalTextLLM"], METRICS[-1]))
```

From the command line:

```bash
python -m crysreas.experiment.evaluate out/main.full.parquet out/main.base.parquet --latex
```

To score a table under MatterGen's reading of the same metrics:

```python
from crysreas.metric_process.config import mattergen_config
from crysreas.metric_process.registry import run_metrics

run_metrics(df, ["stable_unique_novel"], config=mattergen_config())
```

## Equivalence with what this replaced

```bash
python -m crysreas.experiment.verify_equivalence
```

Seven checks, each one row-by-row and never on an average:

| | Check |
|---|---|
| A | the `E_hull < 0.1` window reproduces the stored `is_stable` |
| B | `strict` criterion reproduces the stored `spacegroup_consistency` |
| C | `pooled` scope reproduces the stored `is_unique` |
| D | `subgroup` criterion reproduces the column behind the manuscript tables |
| E | per-seed uniqueness reproduces the reported rate |
| F | seed labels partition each prompt exactly once |
| G | reduced-formula grouping is exact — bucketed uniqueness equals a brute-force all-pairs run on a slice containing real duplicates |

A–C run under `mattergen_config()` and must reproduce the stored columns bit
for bit. D–E run under the default protocol and must reproduce the manuscript's
numbers. F–G
check assumptions the rest of the protocol rests on, none of them assumed.

Current status: **48 pass, 1 fails** across the eight runs.

The failure is check C on `energy_only`, one row in 10,240. The cause lies in
the metric itself.
`DisorderedStructureMatcher.fit` disorders its *first* argument before the
fallback comparison, so it is not symmetric: `fit(a, b)` and `fit(b, a)` can
differ for a pair on the tolerance boundary. Rows 6223 and 6227 of `energy_only`
are such a pair, and they are the only asymmetric pair in their group. Greedy
first-wins therefore does not define a unique answer — the survivor set depends
on scan order — so **pooled uniqueness is well defined only up to 0.01
percentage points here**.

The check is left strict, with no tolerance, because the ambiguity is worth
seeing. It does not reach the reported numbers: uniqueness is scored per
seed, where a seed holds one structure per composition and collisions of this
kind essentially cannot arise, and check E reproduces the per-seed rate exactly
on all eight runs.

## What is not verified here

- The upstream artifacts. Relaxation, `energy_above_hull`, and the detected
  space groups are taken as given; a defect there would affect the old numbers
  equally.
- Seed exchangeability. It follows from how generation draws samples, not from
  anything the data can show.
- MLIP accuracy. `E_hull <= 0` is a prediction, not a DFT result, and a
  threshold at zero is far more sensitive to potential error than one at 0.1.
  Support it with the DFT recomputation, never with the MLIP alone.
