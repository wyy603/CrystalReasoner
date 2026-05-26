#!/usr/bin/env python3
"""Upload CrystalReasoner checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


BASE_MODEL = "Qwen/Qwen2.5-3B"
ORG = "CrystalReasoner"
CHECKPOINT_ROOT = Path("checkpoints_merged")

README_PATH = Path("README.md")

# Stable Hub suffixes for models listed in README.md. The README table remains
# the source of truth for which checkpoint folders are uploaded.
VARIANT_NAMES = {
    "no_thinking": "Base",
    "thinking": "Thinking",
    "rl_no_thinking": "RL",
    "rl_thinking_mix": "",
    "thinking_only_validity": "NoEnergyTerm",
    "thinking_only_energy": "NoValidityTerm",
    "spacegroup_thinking": "SpaceGroup",
    "rl_elastic_thinking_new": "ElasticProperties",
    "rl_cte_thinking": "ThermalExpansion",
    "plaid_wyckoff": "PLaIDWyckoff",
    "crystaltextllm": "CrystalTextLLM",
}


README_TEMPLATE = """---
base_model: Qwen/Qwen2.5-3B
library_name: transformers
pipeline_tag: text-generation
tags:
- crystal-generation
- materials-science
- qwen2.5
- crystalreasoner
---

# CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation

CrystalReasoner (CrysReas) is an end-to-end LLM framework for generating crystal structures from natural language instructions. It uses supervised fine-tuning (SFT) to teach crystal-structure generation, thinking traces to introduce crystallographic and physical priors before coordinates, and reinforcement learning (RL) with verifiable rewards to improve validity, stability, and property conditioning.

# Qwen2.5-3B-CrysReas-[variant name]

## Quick Start

You can use this model directly with the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

model_id = "CrystalReasoner/Qwen2.5-3B-CrysReas-[variant name]"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

messages = [
    {"role": "user", "content": "Below is a description of a bulk material. The chemical formula is NaCl. The bulk_modulus is about 100 GPa. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(generated_text)
```

If you want the generated structure in pymatgen Structure format, please use this script after the previous generation:

```python
def get_structure(generated_text: str):
    import re
    from pymatgen.core import Lattice, Structure

    cif_match = re.search(r'<CIF>(.*?)</CIF>', generated_text, re.DOTALL)
    if cif_match:
        generated_text = cif_match.group(1)

    lines = [line.strip() for line in generated_text.strip().split('\\n') if line.strip()]
    if lines and not re.match(r'^[-+0-9.eE\\s]+$', lines[0]):
        lines = lines[1:]

    lengths = list(map(float, lines[0].split()))
    angles = list(map(float, lines[1].split()))
    lattice = Lattice.from_parameters(*lengths, *angles)

    species = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        species.append(parts[0])
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

    structure = Structure(lattice, species, coords)
    return structure

structure = get_structure(generated_text)
print(structure)
```
"""


def model_name_for_variant(variant_name: str) -> str:
    base_name = "Qwen2.5-3B-CrysReas"
    if not variant_name:
        return base_name
    return f"{base_name}-{variant_name}"


def build_readme(variant_name: str) -> str:
    return README_TEMPLATE.replace(
        "Qwen2.5-3B-CrysReas-[variant name]",
        model_name_for_variant(variant_name),
    ).replace("[variant name]", variant_name)


def repo_id_for_variant(variant_name: str) -> str:
    return f"{ORG}/{model_name_for_variant(variant_name)}"


def has_model_files(path: Path) -> bool:
    required_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ]
    return all((path / name).is_file() for name in required_files) and any(
        path.glob("*.safetensors")
    )


def read_readme_model_folders(readme_path: Path) -> list[str]:
    rows = []
    in_checkpoint_table = False

    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "Folder name" in line and "Model Name" in line:
            in_checkpoint_table = True
            continue
        if not in_checkpoint_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("| ---"):
            continue

        match = re.match(r"\| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", line)
        if not match:
            continue
        folder_name, _paper_name, _notes = match.groups()
        rows.append(folder_name)

    if not rows:
        raise ValueError(f"No checkpoint rows found in {readme_path}")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload README-listed CrystalReasoner checkpoints to Hugging Face Hub."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help="README containing the checkpoint table.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=CHECKPOINT_ROOT,
        help="Directory containing merged checkpoint folders.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Upload only these checkpoint folder names.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip folders that do not contain a complete Hugging Face model.",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload only the generated README.md model cards.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without creating repositories or uploading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    readme_folders = read_readme_model_folders(args.readme)
    selected = args.only or [name for name in readme_folders if name in VARIANT_NAMES]

    unknown_names = [name for name in selected if name not in readme_folders]
    if unknown_names:
        raise ValueError(
            "Requested checkpoint folders are not README checkpoint models: "
            + ", ".join(unknown_names)
        )
    missing_variants = [name for name in selected if name not in VARIANT_NAMES]
    if missing_variants:
        raise KeyError(
            "Missing Hub variant names for selected README checkpoint folders: "
            + ", ".join(missing_variants)
        )
    if not selected:
        raise ValueError("No README checkpoint rows match the configured Hub variants.")

    for folder_name in selected:
        variant_name = VARIANT_NAMES[folder_name]
        variant_label = variant_name or "main"
        checkpoint_dir = args.checkpoint_root / folder_name
        repo_id = repo_id_for_variant(variant_name)

        if not args.readme_only and not has_model_files(checkpoint_dir):
            message = (
                f"Missing complete model files for {folder_name}: {checkpoint_dir}"
            )
            if args.skip_missing:
                print(f"SKIP: {message}")
                continue
            raise FileNotFoundError(message)

        if args.readme_only:
            print(f"UPLOAD README: {repo_id}")
        else:
            print(f"UPLOAD: {checkpoint_dir} -> {repo_id}")
        if args.dry_run:
            continue

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        if not args.readme_only:
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=checkpoint_dir,
                ignore_patterns=[
                    "*.parquet",
                    "*.filepart",
                    "__pycache__/*",
                ],
                commit_message=f"Upload CrysReas {variant_label} checkpoint",
            )
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="README.md",
            path_or_fileobj=build_readme(variant_name).encode("utf-8"),
            commit_message=f"Add CrysReas {variant_label} model card",
        )

        print(f"DONE: {repo_id}")


if __name__ == "__main__":
    main()
