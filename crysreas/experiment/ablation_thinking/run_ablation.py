from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tyro
from transformers import AutoTokenizer

from crysreas.metric_process.config import merge_metric_process_config
from crysreas.metric_process.helpers import df_serialize
from crysreas.metric_process.registry import run_metrics
from crysreas.trainer.generate_parquet import generate_from_dataframe


@dataclass
class Args:
    parquet_path: Path = Path("checkpoints_merged/thinking/conditional+thinking.parquet")
    model_path: Path = Path("checkpoints_merged/thinking")
    redundant_path: Path = Path("crysreas/experiment/ablation_thinking/redundant.txt")
    output_path: Path = Path("crysreas/experiment/ablation_thinking/ablation_results.parquet")
    csv_output_path: Path = Path("crysreas/experiment/ablation_thinking/ablation_results.csv")
    mp_id: str = "mp-1068807"
    row_rank_within_mpid: int = 2
    num_source_rows: int = 64
    sample_seed: int = 42
    generation_seed: int = 42
    batch_size: int = 64
    num_attempts_per_condition: int = 16
    trust_remote_code: bool = False
    nnodes: int = 1
    n_gpus_per_node: int = 1
    device: str = "cuda"


@dataclass(frozen=True)
class SectionSpan:
    name: str
    start: int
    end: int


def prompt_to_text(prompt_obj: Any) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        return str(prompt_obj.get("content", ""))
    if hasattr(prompt_obj, "tolist"):
        prompt_obj = prompt_obj.tolist()
    if isinstance(prompt_obj, (list, tuple)):
        parts: list[str] = []
        for item in prompt_obj:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(prompt_obj)


def choose_target_row(df: pd.DataFrame, mp_id: str, row_rank_within_mpid: int) -> pd.Series:
    sub = df[df["mp_id"].astype(str) == mp_id].reset_index(drop=False)
    if len(sub) < row_rank_within_mpid:
        raise ValueError(
            f"mp_id={mp_id!r} has only {len(sub)} rows, cannot select rank {row_rank_within_mpid}"
        )
    row = sub.iloc[row_rank_within_mpid - 1]
    required_true = [
        "structure_validity",
        "smact_validity",
        "composition_consistency",
        "spacegroup_consistency",
    ]
    if not all(bool(row[key]) for key in required_true):
        raise ValueError(
            f"Selected row does not have all required validity flags true: "
            f"{json.dumps({k: row[k] for k in required_true}, ensure_ascii=False)}"
        )
    return row


def select_source_rows(
    df: pd.DataFrame,
    *,
    mp_id: str,
    row_rank_within_mpid: int,
    num_source_rows: int,
    sample_seed: int,
) -> pd.DataFrame:
    if num_source_rows < 1:
        raise ValueError("num_source_rows must be >= 1")
    target_row = choose_target_row(df, mp_id=mp_id, row_rank_within_mpid=row_rank_within_mpid)
    target_index = int(target_row["index"]) if "index" in target_row else int(target_row.name)

    if "spacegroup_consistency" not in df.columns:
        raise ValueError("Input parquet must contain 'spacegroup_consistency'")
    valid = df[df["spacegroup_consistency"].apply(lambda x: bool(x) is True)].reset_index(drop=False)
    if target_index not in set(valid["index"].tolist()):
        raise ValueError("Target row is not in spacegroup_consistency=True subset")
    if len(valid) < num_source_rows:
        raise ValueError(
            f"Only {len(valid)} rows satisfy spacegroup_consistency=True; "
            f"cannot sample {num_source_rows}"
        )

    others = valid[valid["index"] != target_index]
    need = num_source_rows - 1
    sampled_others = others.sample(n=need, random_state=sample_seed) if need > 0 else others.iloc[:0]
    selected = pd.concat([valid[valid["index"] == target_index], sampled_others], ignore_index=True)
    return selected


def split_response(response: str) -> tuple[str, str]:
    marker = "## CIF File"
    idx = response.find(marker)
    if idx == -1:
        raise ValueError("Could not find '## CIF File' marker in response")
    return response[:idx], response[idx:]


def extract_section_spans(prefix: str) -> dict[str, SectionSpan]:
    def heading_match(text: str, title: str) -> re.Match[str]:
        match = re.search(rf"(?m)^###\s+{re.escape(title)}\s*$", text)
        if not match:
            raise ValueError(f"Could not locate heading: ### {title}")
        return match

    def section_span_by_title(text: str, title: str) -> tuple[int, int]:
        start_match = heading_match(text, title)
        start = start_match.start()
        next_heading = re.search(r"(?m)^##+\s+.+$", text[start_match.end() :])
        if next_heading:
            end = start_match.end() + next_heading.start()
        else:
            end = len(text)
        return start, end

    def section_span_by_any_title(text: str, titles: tuple[str, ...]) -> tuple[int, int]:
        last_error: Exception | None = None
        for title in titles:
            try:
                return section_span_by_title(text, title)
            except ValueError as e:
                last_error = e
        raise ValueError(f"Could not locate any heading in {titles!r}") from last_error

    crystal_heading = heading_match(prefix, "Crystal Structure")
    _, crystal_section_end = section_span_by_title(prefix, "Crystal Structure")
    crystal_text = prefix[crystal_heading.end() : crystal_section_end]
    crystal_base = crystal_heading.end()

    first_match = re.search(
        r"First, consider space groups and atom numbers\.(.*?)(?=Second, consider band gaps\.)",
        crystal_text,
        flags=re.DOTALL,
    )
    second_match = re.search(
        r"Second, consider band gaps\.(.*?)(?=Third, consider structure validity\.)",
        crystal_text,
        flags=re.DOTALL,
    )
    third_match = re.search(
        r"Third, consider structure validity\.(.*?)\Z",
        crystal_text,
        flags=re.DOTALL,
    )

    if not all([first_match, second_match, third_match]):
        raise ValueError("Could not isolate Crystal Structure part1/part2/part3")

    def full_span(name: str, match_obj: re.Match[str], base: int) -> SectionSpan:
        return SectionSpan(
            name=name,
            start=base + match_obj.start(),
            end=base + match_obj.end(),
        )

    stability_start, stability_end = section_span_by_title(prefix, "Stability")
    electronic_start, electronic_end = section_span_by_any_title(
        prefix, ("Electronic Properties", "Elastic Properties")
    )
    part4_start = min(stability_start, electronic_start)
    part4_end = max(stability_end, electronic_end)

    return {
        "part1": full_span("part1", first_match, crystal_base),
        "part2": full_span("part2", second_match, crystal_base),
        "part3": full_span("part3", third_match, crystal_base),
        "part4": SectionSpan(name="part4", start=part4_start, end=part4_end),
    }


def build_token_matched_replacement(tokenizer: Any, source_text: str, target_token_count: int) -> str:
    if target_token_count <= 0:
        return ""
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    if not source_ids:
        raise ValueError("redundant.txt tokenized to an empty sequence")
    repeated_ids = (source_ids * ((target_token_count + len(source_ids) - 1) // len(source_ids)))[:target_token_count]
    replacement = tokenizer.decode(repeated_ids, clean_up_tokenization_spaces=False)
    actual = len(tokenizer.encode(replacement, add_special_tokens=False))
    if actual != target_token_count:
        raise ValueError(f"Replacement token mismatch: expected {target_token_count}, got {actual}")
    return replacement


def apply_variant(
    prefix: str,
    spans: dict[str, SectionSpan],
    section_name: str,
    operation: str,
    tokenizer: Any,
    redundant_text: str,
) -> tuple[str, int, int]:
    span = spans[section_name]
    original_text = prefix[span.start:span.end]
    original_token_count = len(tokenizer.encode(original_text, add_special_tokens=False))
    if operation == "remove":
        replacement = ""
    elif operation == "replace":
        replacement = build_token_matched_replacement(tokenizer, redundant_text, original_token_count)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    new_prefix = prefix[:span.start] + replacement + prefix[span.end:] + "\n## CIF File"
    replacement_token_count = len(tokenizer.encode(replacement, add_special_tokens=False))
    return new_prefix, original_token_count, replacement_token_count


def main(args: Args) -> None:
    df = pd.read_parquet(args.parquet_path)
    source_rows = select_source_rows(
        df,
        mp_id=args.mp_id,
        row_rank_within_mpid=args.row_rank_within_mpid,
        num_source_rows=args.num_source_rows,
        sample_seed=args.sample_seed,
    )

    redundant_text = args.redundant_path.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tasks: list[dict[str, Any]] = []
    for source_order, row in enumerate(source_rows.itertuples(index=False), start=0):
        original_prompt = prompt_to_text(getattr(row, "prompt"))
        original_response = str(getattr(row, "responses"))
        prefix_before_cif, cif_tail = split_response(original_response)
        spans = extract_section_spans(prefix_before_cif)
        mp_id = str(getattr(row, "mp_id"))
        source_index = int(getattr(row, "index"))

        for operation in ("remove", "replace"):
            for section_name in ("part1", "part2", "part3", "part4"):
                new_prefix, original_token_count, replacement_token_count = apply_variant(
                    prefix_before_cif,
                    spans,
                    section_name,
                    operation,
                    tokenizer,
                    redundant_text,
                )
                tasks.append(
                    {
                        "mp_id": mp_id,
                        "source_order": source_order,
                        "source_index": source_index,
                        "variant": f"{operation}_{section_name}",
                        "operation": operation,
                        "target_section": section_name,
                        "original_prompt": original_prompt,
                        "assistant_prefix": new_prefix,
                        "original_response": original_response,
                        "original_prefix_before_cif": prefix_before_cif,
                        "original_cif_tail": cif_tail,
                        "original_section_token_count": original_token_count,
                        "replacement_section_token_count": replacement_token_count,
                    }
                )

    preformatted_prompts: list[str] = []
    for task in tasks:
        messages = [
            {"role": "user", "content": task["original_prompt"]},
            {"role": "assistant", "content": task["assistant_prefix"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        preformatted_prompts.append(prompt_text)

    gen_input_df = pd.DataFrame({"prompt": preformatted_prompts})
    print("gen_input_df", gen_input_df)
    gen_output_df = generate_from_dataframe(
        gen_input_df,
        model_path=str(args.model_path),
        prompt_key="prompt",
        response_key="responses",
        batch_size=args.batch_size,
        num_samples=args.num_attempts_per_condition,
        seed=args.generation_seed,
        prompt_length=2048,
        trust_remote_code=args.trust_remote_code,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        device=args.device,
        preformatted_prompt=True,
        squeeze_single_sample=False,
    )

    records: list[dict[str, Any]] = []
    for task, prompt_text, continuations in zip(
        tasks,
        preformatted_prompts,
        gen_output_df["responses"].tolist(),
        strict=True,
    ):
        assistant_prefix = task["assistant_prefix"]
        for attempt_idx, continuation in enumerate(continuations):
            row = dict(task)
            row.pop("assistant_prefix")
            row["attempt_idx"] = attempt_idx
            row["prompt"] = prompt_text
            row["responses"] = assistant_prefix + str(continuation)
            records.append(row)

    result_df = pd.DataFrame(records)
    metric_names = [
        "gt",
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "composition_consistency",
        "spacegroup_consistency",
    ]
    metric_cfg = merge_metric_process_config(None, {"prompt_type": ["conditional", "thinking"]})
    result_df = run_metrics(result_df.copy(), metric_names, config=metric_cfg, log=True, forced=True)
    result_df = result_df[
        [
            "mp_id",
            "variant",
            "operation",
            "target_section",
            "attempt_idx",
            "source_order",
            "source_index",
            "original_prompt",
            "prompt",
            "responses",
            "simple_structure",
            "structure_validity",
            "smact_validity",
            "composition_consistency",
            "spacegroup_consistency",
            "original_section_token_count",
            "replacement_section_token_count",
            "original_response",
        ]
    ]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_df = result_df.copy()
    df_serialize(parquet_df)
    parquet_df.to_parquet(args.output_path, index=False)
    csv_df = result_df.copy()
    csv_df["simple_structure"] = csv_df["simple_structure"].apply(lambda x: None if x is None else str(x))
    csv_df.to_csv(args.csv_output_path, index=False)
    print(f"Wrote parquet to {args.output_path}")
    print(f"Wrote csv to {args.csv_output_path}")


if __name__ == "__main__":
    tyro.cli(main)
