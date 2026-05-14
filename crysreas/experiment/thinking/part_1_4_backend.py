"""
Backend for EXPERIMENT §1.4: compute CIF information curves vs atom counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import CIF_MARKERS, ComparePaths, count_atoms, ensure_thinking_dirs

DATA_1_4_PER_ROW = "thinking_exp_1_4_cif_information_per_row.parquet"
DATA_1_4_CURVE = "thinking_exp_1_4_cif_information_vs_atoms_curve.parquet"


def _locate_cif_span(text: Any) -> tuple[int, int] | None:
    if not isinstance(text, str) or not text:
        return None
    start = -1
    for marker in CIF_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            start = idx
            break
    if start == -1:
        return None

    end_tag = "</CIF>"
    end_pos = text.find(end_tag, start)
    if end_pos != -1:
        end = end_pos + len(end_tag)
    else:
        end = len(text)
    if end <= start:
        return None
    return start, end


def _load_model_and_tokenizer(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        low_cpu_mem_usage=True,
        torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _score_cif_information(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    spans: list[tuple[int, int]],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    info_sum_all = np.full(len(texts), np.nan, dtype=float)
    token_count_all = np.zeros(len(texts), dtype=int)

    for i in range(0, len(texts), batch_size):
        b_texts = texts[i : i + batch_size]
        b_spans = spans[i : i + batch_size]
        enc = tokenizer(
            b_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for bi, (start, end) in enumerate(b_spans):
            off = offsets[bi]
            tok_start = off[:, 0]
            tok_end = off[:, 1]
            overlap = (tok_end > start) & (tok_start < end)
            valid = attention_mask[bi].bool()
            target_mask[bi] = overlap.to(target_mask.device) & valid

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            labels = input_ids[:, 1:]
            nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            cif_mask = target_mask[:, 1:] & attention_mask[:, 1:].bool()
            info_sum = (nll * cif_mask).sum(dim=1)
            n_tok = cif_mask.sum(dim=1)

        j0 = i
        j1 = i + len(b_texts)
        info_sum_all[j0:j1] = info_sum.detach().cpu().numpy()
        token_count_all[j0:j1] = n_tok.detach().cpu().numpy().astype(int)

    return info_sum_all, token_count_all


def _build_per_row(
    df: pd.DataFrame,
    model_name: str,
    model_dir: Path,
    device: torch.device,
    *,
    batch_size: int,
) -> pd.DataFrame:
    model, tokenizer = _load_model_and_tokenizer(model_dir, device)
    keep_idx: list[int] = []
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for idx, response in enumerate(df["responses"].tolist()):
        span = _locate_cif_span(response)
        if span is None:
            continue
        keep_idx.append(idx)
        texts.append(str(response))
        spans.append(span)

    if not keep_idx:
        return pd.DataFrame(
            columns=[
                "mp_id",
                "model",
                "n_atoms",
                "cif_info_sum",
                "cif_token_count",
                "cif_info_avg",
            ]
        )

    info_sum, token_count = _score_cif_information(
        model,
        tokenizer,
        texts,
        spans,
        device,
        batch_size=batch_size,
    )
    info_avg = np.divide(
        info_sum,
        token_count,
        out=np.full_like(info_sum, np.nan, dtype=float),
        where=token_count > 0,
    )
    return pd.DataFrame(
        {
            "mp_id": df.iloc[keep_idx]["mp_id"].astype(str).to_numpy(),
            "model": model_name,
            "n_atoms": [count_atoms(x) for x in df.iloc[keep_idx]["simple_structure"].tolist()],
            "cif_info_sum": info_sum.astype(float),
            "cif_token_count": token_count.astype(int),
            "cif_info_avg": info_avg.astype(float),
        }
    )


def run(paths: ComparePaths, *, batch_size: int = 2, max_rows: int | None = None) -> None:
    ensure_thinking_dirs(data_dir=paths.data_dir)
    cols = ["mp_id", "responses", "simple_structure"]
    think_df = pd.read_parquet(paths.thinking_parquet, columns=cols)
    no_df = pd.read_parquet(paths.no_thinking_parquet, columns=cols)
    if max_rows is not None:
        think_df = think_df.iloc[:max_rows].copy()
        no_df = no_df.iloc[:max_rows].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    think_rows = _build_per_row(
        think_df,
        "thinking",
        paths.thinking_parquet.parent,
        device,
        batch_size=batch_size,
    )
    no_rows = _build_per_row(
        no_df,
        "no_thinking",
        paths.no_thinking_parquet.parent,
        device,
        batch_size=batch_size,
    )
    per_row = pd.concat([think_rows, no_rows], ignore_index=True)
    per_row = per_row[per_row["n_atoms"].notna() & per_row["cif_info_avg"].notna()].copy()
    per_row["n_atoms"] = per_row["n_atoms"].astype(int)

    curve = (
        per_row.groupby(["model", "n_atoms"], as_index=False)
        .agg(
            n_samples=("mp_id", "size"),
            cif_info_avg=("cif_info_avg", "mean"),
        )
        .sort_values(["model", "n_atoms"])
        .reset_index(drop=True)
    )

    p_row = paths.data_dir / DATA_1_4_PER_ROW
    p_curve = paths.data_dir / DATA_1_4_CURVE
    per_row.to_parquet(p_row, index=False)
    curve.to_parquet(p_curve, index=False)
    print(f"1.4 backend wrote {p_row} and {p_curve}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXPERIMENT §1.4 backend (data only).")
    parser.add_argument("--thinking", type=str, default="checkpoints_merged/thinking/conditional+thinking.parquet")
    parser.add_argument(
        "--no-thinking",
        type=str,
        default="checkpoints_merged/no_thinking/conditional+thinking.parquet",
    )
    parser.add_argument("--data-dir", type=str, default=str(ComparePaths().data_dir))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    run(
        ComparePaths(
            thinking_parquet=Path(args.thinking),
            no_thinking_parquet=Path(args.no_thinking),
            data_dir=Path(args.data_dir),
        ),
        batch_size=int(args.batch_size),
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
