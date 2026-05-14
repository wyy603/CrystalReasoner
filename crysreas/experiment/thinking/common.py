"""
Shared helpers for thinking-trace comparison experiments.

Dependencies:
- pandas, numpy
- pymatgen
- crysreas.utils.crystal.SimpleCrystal
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crysreas.utils.crystal import SimpleCrystal

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
# Packaged artifact layout: backend tables under data/, figures under figure/.
THINKING_EXPERIMENT_ROOT = Path(__file__).resolve().parents[0]
THINKING_DATA_DIR = THINKING_EXPERIMENT_ROOT / "data"
THINKING_FIGURE_DIR = THINKING_EXPERIMENT_ROOT / "figure"

DEFAULT_THINKING = EXPERIMENT_DIR.parent.parent / "checkpoints_merged/thinking/conditional+thinking.parquet"
DEFAULT_NO_THINKING = EXPERIMENT_DIR.parent.parent / "checkpoints_merged/no_thinking/conditional+thinking.parquet"

CIF_MARKERS = ("## CIF File", "<CIF>", "CIF File:")

# HF checkpoint with tokenizer files (model weights may be incomplete; tokenizer load is enough).
DEFAULT_TOKENIZER_CHECKPOINT = (
    EXPERIMENT_DIR.parent.parent / "checkpoints_merged" / "20260202" / "global_step_1626"
)


def load_thinking_tokenizer(checkpoint_dir: Path | str | None = None) -> Any:
    """Load HuggingFace tokenizer from a model checkpoint directory."""
    from transformers import AutoTokenizer

    p = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_TOKENIZER_CHECKPOINT
    return AutoTokenizer.from_pretrained(str(p))


def count_thinking_tokens(tokenizer: Any, thinking_text: str) -> int:
    """Token count for thinking-only text (no BOS/EOS or chat special tokens)."""
    if not thinking_text:
        return 0
    return len(tokenizer.encode(str(thinking_text), add_special_tokens=False))


def ensure_thinking_dirs(*, data_dir: Path | None = None, figure_dir: Path | None = None) -> None:
    """Create thinking experiment data/figure directories when needed."""
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ComparePaths:
    thinking_parquet: Path = DEFAULT_THINKING
    no_thinking_parquet: Path = DEFAULT_NO_THINKING
    out_dir: Path = EXPERIMENT_DIR
    data_dir: Path = THINKING_DATA_DIR
    figure_dir: Path = THINKING_FIGURE_DIR


def split_thinking_and_tail(response: Any) -> tuple[str, str]:
    """Split response into thinking text and CIF tail."""
    if not isinstance(response, str):
        return "", ""
    for marker in CIF_MARKERS:
        idx = response.find(marker)
        if idx != -1:
            return response[:idx].strip(), response[idx:].strip()
    return response.strip(), ""


def crystal_section(thinking: str) -> str:
    """Extract the crystal structure section used for consistency parsing."""
    m = re.search(r"###\s*Crystal Structure\s*(.*)", thinking, flags=re.S | re.I)
    if m:
        sec = m.group(1)
        for stop in ("### Stability", "## Stability", "### Electronic", "## Electronic Properties"):
            i = sec.find(stop)
            if i != -1:
                sec = sec[:i]
        return sec
    for stop in ("### Stability", "## Stability"):
        i = thinking.find(stop)
        if i != -1:
            return thinking[:i]
    return thinking


def parse_space_group_id(text: str) -> int | None:
    m = re.search(r"\(id\s*(\d+)\)", text, flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_volume(text: str) -> float | None:
    m = re.search(r"volume\s*([\d.]+)", text, flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_bond_lengths_angstrom(text: str) -> list[float]:
    vals: list[float] = []
    for a, b in re.findall(r"(\d+\.\d+)\s*[-–]?\s*(\d+\.\d+)\s*Å", text):
        try:
            vals.append(float(a))
            vals.append(float(b))
        except ValueError:
            continue
    for x in re.findall(r"(\d+\.\d+)\s*Å", text):
        try:
            v = float(x)
        except ValueError:
            continue
        if v not in vals:
            vals.append(v)
    return [v for v in vals if 0.45 <= v <= 5.0]


def parse_prompt_text(prompt_obj: Any) -> str:
    """Convert prompt field (array/list/dict/string) to plain text."""
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        return str(prompt_obj.get("content", ""))
    if isinstance(prompt_obj, (list, tuple, np.ndarray)):
        parts: list[str] = []
        for item in prompt_obj:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join([x for x in parts if x])
    return str(prompt_obj)


def parse_instruction_spacegroup_id(prompt_obj: Any) -> int | None:
    text = parse_prompt_text(prompt_obj)
    m = re.search(r"spacegroup\s*number\s*is\s*(\d+)", text, flags=re.I)
    if not m:
        m = re.search(r"space\s*group\s*number\s*is\s*(\d+)", text, flags=re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_simple_structure(simple_structure: str) -> Structure | None:
    try:
        return SimpleCrystal.from_simple_no_sym(str(simple_structure)).structure
    except Exception:
        return None


def count_atoms(simple_structure: str) -> int | None:
    st = parse_simple_structure(simple_structure)
    return int(len(st)) if st is not None else None


def structure_spacegroup_id(structure: Structure | None) -> int | None:
    if structure is None:
        return None
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return int(sga.get_space_group_number())
    except Exception:
        return None


def structure_volume(structure: Structure | None) -> float | None:
    if structure is None:
        return None
    try:
        return float(structure.volume)
    except Exception:
        return None


def structure_bond_pool(structure: Structure | None, max_d: float = 5.0) -> list[float]:
    if structure is None:
        return []
    dists: list[float] = []
    n = len(structure)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = float(structure.get_distance(i, j))
            except Exception:
                continue
            if d < max_d:
                dists.append(d)
    return dists


def median_min_abs_error(claimed: list[float], actual: list[float]) -> float | None:
    if not claimed or not actual:
        return None
    errs: list[float] = []
    for c in claimed:
        errs.append(min(abs(c - d) for d in actual))
    return float(np.median(errs))


def median_min_rel_pct_error(claimed: list[float], actual: list[float]) -> float | None:
    """Median over claimed bond lengths of min_d |c-d| / d expressed as percent (0–100 scale)."""
    if not claimed or not actual:
        return None
    rels: list[float] = []
    for c in claimed:
        best_abs: float | None = None
        best_d: float | None = None
        for d in actual:
            if d <= 0:
                continue
            ad = abs(float(c) - float(d))
            if best_abs is None or ad < best_abs:
                best_abs = ad
                best_d = float(d)
        if best_abs is not None and best_d is not None and best_d > 0:
            rels.append(100.0 * best_abs / best_d)
    if not rels:
        return None
    return float(np.median(rels))


def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes", "y"])


def load_frames(paths: ComparePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "mp_id",
        "prompt",
        "responses",
        "simple_structure",
        "structure_validity",
        "smact_validity",
        "composition_consistency",
        "is_stable",
        "is_novel",
        "is_unique",
        "stable_unique_novel",
        "spacegroup_consistency",
    ]
    df_think = pd.read_parquet(paths.thinking_parquet, columns=cols)
    df_no = pd.read_parquet(paths.no_thinking_parquet, columns=cols)
    if not df_think["mp_id"].equals(df_no["mp_id"]):
        raise ValueError("mp_id sequence is not aligned between thinking and no-thinking parquet.")
    return df_think, df_no

