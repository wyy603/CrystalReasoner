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

CrystalReasoner (CrysReas) is an end-to-end LLM framework for generating crystal structures from natural language instructions. It uses supervised fine-tuning (SFT) to teach crystal-structure generation, thinking traces to introduce crystallographic and physical priors before coordinates, and reinforcement learning (RL) with verifiable rewards to improve validity, stability, and property conditioning. Please see our work at [crystalreasoner.github.io](https://crystalreasoner.github.io/).

<h4 align="center">
  
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github)](https://github.com/wyy603/CrystalReasoner/)
[![Website](https://img.shields.io/badge/🌐-Project_Website-2ea44f?style=flat-square&logoColor=white)](https://crystalreasoner.github.io/)
[![Dataset](https://img.shields.io/badge/📊-Checkpoints-005B99?style=flat-square)](https://nyu.app.box.com/folder/361279226287)
[![Paper](https://img.shields.io/badge/arXiv-2605.14344-blue.svg?logo=arxiv&logoColor=white.svg)](https://arxiv.org/abs/2605.14344)
</h4>

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
    {{"role": "user", "content": "Below is a description of a bulk material. The chemical formula is NaCl. The bulk_modulus is about 100 GPa. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"}},
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

# Citation

Check out our [paper](https://arxiv.org/abs/2605.14344) for more details. If you use our dataset or find our work useful, please cite

```
@article{{wu2026crysreas,
  title={{CrystalReasoner: Reasoning and RL for Property-Conditioned Crystal Structure Generation}},
  author={{Yuyang Wu and Stefano Falletta and Delia McGrath and Sherry Yang}},
  year={{2026}},
  journal={{arXiv preprint arXiv:2605.14344}},
  url={{https://arxiv.org/abs/2605.14344}}
}}
```
"""


def build_readme(variant_name: str) -> str:
    return README_TEMPLATE.replace("[variant name]", variant_name)


def repo_id_for_variant(variant_name: str) -> str:
    return f"{ORG}/Qwen2.5-3B-CrysReas-{variant_name}"


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
        if line.startswith("| Folder name |") or line.startswith("| `Folder name` |"):
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
        folder_name, _paper_name, notes = match.groups()
        if "prior-work" in notes.lower():
            continue
        rows.append(folder_name)

    if not rows:
        raise ValueError(f"No checkpoint rows found in {readme_path}")

    missing_variants = [name for name in rows if name not in VARIANT_NAMES]
    if missing_variants:
        raise KeyError(
            "Missing Hub variant names for README checkpoint folders: "
            + ", ".join(missing_variants)
        )

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
        "--dry-run",
        action="store_true",
        help="Print planned uploads without creating repositories or uploading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    readme_folders = read_readme_model_folders(args.readme)
    selected = args.only or readme_folders

    unknown_names = [name for name in selected if name not in readme_folders]
    if unknown_names:
        raise ValueError(
            "Requested checkpoint folders are not non-prior-work README models: "
            + ", ".join(unknown_names)
        )

    for folder_name in selected:
        variant_name = VARIANT_NAMES[folder_name]
        checkpoint_dir = args.checkpoint_root / folder_name
        repo_id = repo_id_for_variant(variant_name)

        if not has_model_files(checkpoint_dir):
            message = (
                f"Missing complete model files for {folder_name}: {checkpoint_dir}"
            )
            if args.skip_missing:
                print(f"SKIP: {message}")
                continue
            raise FileNotFoundError(message)

        print(f"UPLOAD: {checkpoint_dir} -> {repo_id}")
        if args.dry_run:
            continue

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=checkpoint_dir,
            ignore_patterns=[
                "*.parquet",
                "*.filepart",
                "__pycache__/*",
            ],
            commit_message=f"Upload CrysReas {variant_name} checkpoint",
        )
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="README.md",
            path_or_fileobj=build_readme(variant_name).encode("utf-8"),
            commit_message=f"Add CrysReas {variant_name} model card",
        )

        print(f"DONE: {repo_id}")


if __name__ == "__main__":
    main()
