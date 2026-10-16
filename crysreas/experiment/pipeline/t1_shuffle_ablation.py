"""Tier-1: thinking-trace segment SHUFFLE ablation.

Replaces the draft's Figure 3(b). That figure *removes* a trace segment at
inference from a model trained with every segment present, so each drop is
confounded with distribution shift, and its conclusion ("earlier segments
matter more") is confounded by construction: removing an earlier segment
perturbs more of the text that follows it.

Here segment k is instead SHUFFLED ACROSS EXAMPLES -- swapped for the
same-named segment of a different, randomly chosen sample. Format, register
and length distribution stay in-distribution; only the *information* becomes
wrong for this structure. The drop relative to `baseline` therefore measures
how much the CIF that follows actually depends on segment k.

Conditions: baseline (prefix untouched), shuffle_part1..4, shuffle_all.
Segments (as parsed by the original ablation code):
  part1  "First, consider space groups and atom numbers"  -> symmetry
  part2  "Second, consider band gaps"                     -> local coordination
  part3  "Third, consider structure validity"             -> validity sentence
  part4  "### Stability" + Electronic/Elastic Properties  -> functional properties
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tyro
from transformers import AutoTokenizer
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

sys.path.insert(0, str(ROOT))

from crysreas.experiment.ablation_thinking.run_ablation import (  # noqa: E402
    extract_section_spans,
    prompt_to_text,
    split_response,
)
from crysreas.metric_process.config import merge_metric_process_config  # noqa: E402
from crysreas.metric_process.helpers import df_serialize  # noqa: E402
from crysreas.metric_process.registry import run_metrics  # noqa: E402
from crysreas.trainer.generate_parquet import generate_from_dataframe  # noqa: E402

PARTS = ("part1", "part2", "part3", "part4")


@dataclass
class Args:
    parquet_path: Path = DATA_DIR / "out/main.thinking.parquet"
    model_path: Path = ROOT / "checkpoints_merged/thinking"
    output_path: Path = DATA_DIR / "shuffle/shuffle_ablation.parquet"
    num_source_rows: int = 256
    num_attempts_per_condition: int = 4
    sample_seed: int = 42
    generation_seed: int = 42
    batch_size: int = 64
    n_gpus_per_node: int = 1


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation with no fixed point, so no row ever donates to itself."""
    if n < 2:
        raise ValueError("need >= 2 rows to shuffle across examples")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def select_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Rows whose own generation was fully correct, so `baseline` is a clean ceiling
    and any drop is attributable to the corrupted segment."""
    need = ["structure_validity", "smact_validity",
            "composition_consistency", "spacegroup_consistency"]
    ok = df.copy()
    for c in need:
        ok = ok[ok[c].apply(lambda x: bool(x) is True)]
    parsed = []
    seen: set[str] = set()
    for idx, row in ok.iterrows():
        mp = str(row["mp_id"])
        if mp in seen:            # one row per structure, so a donor is always
            continue              # a *different* material, not another sample of it
        seen.add(mp)
        try:
            prefix, _ = split_response(str(row["responses"]))
            spans = extract_section_spans(prefix)
        except Exception:
            continue
        parsed.append((idx, prefix, spans))
        del mp
        if len(parsed) >= n * 3:            # small surplus, then subsample
            break
    if len(parsed) < n:
        raise ValueError(f"only {len(parsed)} usable rows, need {n}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(parsed), size=n, replace=False)
    recs = []
    for j in pick:
        idx, prefix, spans = parsed[j]
        recs.append(dict(src_index=int(idx), prefix=prefix, spans=spans,
                         mp_id=str(df.loc[idx, "mp_id"]),
                         prompt=prompt_to_text(df.loc[idx, "prompt"])))
    return pd.DataFrame(recs)


def seg(prefix: str, spans: dict, name: str) -> str:
    s = spans[name]
    return prefix[s.start:s.end]


def build_prefix(prefix: str, spans: dict, repl: dict[str, str]) -> str:
    """Rebuild the trace, substituting the named segments. Spans are replaced
    back-to-front so earlier offsets stay valid."""
    out = prefix
    for name in sorted(repl, key=lambda k: spans[k].start, reverse=True):
        s = spans[name]
        out = out[:s.start] + repl[name] + out[s.end:]
    return out + "\n## CIF File"


def main(args: Args) -> None:
    df = pd.read_parquet(args.parquet_path)
    rows = select_rows(df, args.num_source_rows, args.sample_seed)
    n = len(rows)
    rng = np.random.default_rng(args.sample_seed + 1)
    donor = {p: derangement(n, rng) for p in PARTS}

    tok = AutoTokenizer.from_pretrained(str(args.model_path))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def ntok(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=False))

    tasks: list[dict[str, Any]] = []
    for i in range(n):
        r = rows.iloc[i]
        prefix, spans = r["prefix"], r["spans"]
        conds: dict[str, dict[str, str]] = {"baseline": {}}
        for p in PARTS:
            d = int(donor[p][i])
            conds[f"shuffle_{p}"] = {p: seg(rows.iloc[d]["prefix"], rows.iloc[d]["spans"], p)}
        conds["shuffle_all"] = {
            p: seg(rows.iloc[int(donor[p][i])]["prefix"], rows.iloc[int(donor[p][i])]["spans"], p)
            for p in PARTS}
        for cond, repl in conds.items():
            orig_tok = sum(ntok(seg(prefix, spans, p)) for p in repl) if repl else 0
            new_tok = sum(ntok(v) for v in repl.values()) if repl else 0
            tasks.append(dict(
                mp_id=r["mp_id"], src_index=r["src_index"], variant=cond,
                target_section=("none" if cond == "baseline"
                                else ("all" if cond == "shuffle_all" else cond.split("_")[1])),
                original_prompt=r["prompt"],
                assistant_prefix=build_prefix(prefix, spans, repl),
                original_section_token_count=orig_tok,
                replacement_section_token_count=new_tok,
            ))

    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": t["original_prompt"]},
         {"role": "assistant", "content": t["assistant_prefix"]}],
        tokenize=False, add_generation_prompt=False, continue_final_message=True)
        for t in tasks]

    print(f"[shuffle] {n} source rows x {len(tasks)//n} conditions = {len(tasks)} prompts, "
          f"{args.num_attempts_per_condition} attempts each "
          f"-> {len(tasks)*args.num_attempts_per_condition} generations", flush=True)

    gen = generate_from_dataframe(
        pd.DataFrame({"prompt": prompts}),
        model_path=str(args.model_path), prompt_key="prompt", response_key="responses",
        batch_size=args.batch_size, num_samples=args.num_attempts_per_condition,
        seed=args.generation_seed, prompt_length=4096,
        n_gpus_per_node=args.n_gpus_per_node, device="cuda",
        preformatted_prompt=True, squeeze_single_sample=False,
    )

    recs = []
    for task, ptext, conts in zip(tasks, prompts, gen["responses"].tolist(), strict=True):
        for k, cont in enumerate(conts):
            row = {kk: vv for kk, vv in task.items() if kk != "assistant_prefix"}
            row.update(attempt_idx=k, prompt=ptext,
                       responses=task["assistant_prefix"] + str(cont))
            recs.append(row)
    res = pd.DataFrame(recs)

    cfg = merge_metric_process_config(None, {"prompt_type": ["conditional", "thinking"]})
    res = run_metrics(res.copy(), ["gt", "simple_structure", "structure_validity",
                                   "smact_validity", "composition_consistency",
                                   "spacegroup_consistency"],
                      config=cfg, log=True, forced=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out = res.copy()
    df_serialize(out)
    out.to_parquet(args.output_path, index=False)
    print(f"[shuffle] wrote {args.output_path} ({len(out)} rows)")


if __name__ == "__main__":
    tyro.cli(main)
