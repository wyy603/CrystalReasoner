"""Generation + metric matrix for reproducing the paper's reported scores.

Each entry is one generation run followed by one metric pass. Sampling budget:
  * main conditional evaluation   -> 1024 prompts x 10 samples = 10,240 structures
  * space-group conditioned eval  -> ~1000 structures per target space group
  * elasticity / CTE conditioned  -> paper protocol (512x16 / 256x16)
"""
from __future__ import annotations
import json, shlex, subprocess, sys, time
from pathlib import Path
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PY_BIN = str(ROOT / ".venv" / "bin" / "python")
OUT = DATA_DIR / "out"
GPUS = "1"

MAIN_METRICS = [
    "simple_structure", "structure_validity", "smact_validity",
    "composition_consistency", "spacegroup_consistency", "stable_unique_novel",
]
SG_METRICS = ["simple_structure", "structure_validity", "smact_validity",
              "composition_consistency", "spacegroup_consistency"]
ELASTIC_METRICS = ["simple_structure", "structure_validity", "smact_validity",
                   "composition_consistency", "elastic_reward_all"]
CTE_METRICS = ["simple_structure", "structure_validity", "smact_validity", "cte_reward_all"]

# (tag, checkpoint dir, prompt_type, split path, n_samples, metrics)
MAIN_MODELS = [
    ("base",             "no_thinking",            "conditional+thinking"),
    ("thinking",         "thinking",               "conditional+thinking"),
    ("rl",               "rl_no_thinking",         "conditional+thinking"),
    ("full",             "rl_thinking_mix",        "conditional+thinking"),
    ("validity_only",    "thinking_only_validity", "conditional+thinking"),
    ("energy_only",      "thinking_only_energy",   "conditional+thinking"),
    ("crystaltextllm",   "crystaltextllm",         "crystaltextllm_generation+no_thinking"),
    ("plaid_wyckoff",    "plaid_wyckoff",          "plaid_wyckoff_generation+no_thinking"),
    ("crystaltextllm_8", "crystaltextllm_8",       "crystaltextllm_8_generation+no_thinking"),
    ("plaid_wyckoff_8",  "plaid_wyckoff_8",        "plaid_wyckoff_8_generation+no_thinking"),
]

def jobs() -> list[dict]:
    out = []
    for tag, ckpt, ptype in MAIN_MODELS:
        out.append(dict(tag=f"main.{tag}", ckpt=ckpt, ptype=ptype,
                        split="assets/MP/split_generation.json", n=10, metrics=MAIN_METRICS))
    sg_dir = DATA_DIR / "splits"
    for f in sorted(sg_dir.glob("split_sg_*.json")):
        keys = json.load(open(f))["test"]
        n = max(1, round(1000 / max(1, len(keys))))
        for tag, ckpt in (("sg_specialist", "spacegroup_thinking"), ("full", "rl_thinking_mix")):
            out.append(dict(tag=f"sg.{f.stem[9:]}.{tag}", ckpt=ckpt, ptype="spacegroup+thinking",
                            split=str(f.relative_to(ROOT)), n=n, metrics=SG_METRICS))
    for tag, ckpt in (("full", "rl_thinking_mix"), ("specialist", "rl_elastic_thinking_new")):
        out.append(dict(tag=f"elastic.{tag}", ckpt=ckpt, ptype="elastic+thinking",
                        split="assets/MP/split_generation_elastic.json", n=16, metrics=ELASTIC_METRICS))
    for tag, ckpt in (("full", "rl_thinking_mix"), ("specialist", "rl_cte_thinking")):
        out.append(dict(tag=f"cte.{tag}", ckpt=ckpt, ptype="cte+thinking",
                        split="assets/MP/split_generation_cte.json", n=16, metrics=CTE_METRICS))
    return out


def gen_cmd(j: dict, out_path: Path) -> str:
    return shlex.join([
        PY_BIN, "-m", "crysreas.trainer.main_generation",
        f"trainer.n_gpus_per_node={GPUS}",
        f"model.path=checkpoints_merged/{j['ckpt']}",
        f"data.custom_data.prompt_type={j['ptype']}",
        f"data.custom_data.split_path={j['split']}",
        f"data.n_samples={j['n']}",
        f"data.output_path={out_path}",
        "data.batch_size=256",
        "rollout.temperature=1.0", "rollout.top_p=0.7",
        "rollout.gpu_memory_utilization=0.85",
    ])


def metric_cmd(j: dict, out_path: Path) -> str:
    return shlex.join([
        PY_BIN, "-m", "crysreas.metric_process",
        "--path", str(out_path),
        "--metrics-name", *j["metrics"],
        "--prompt-type", j["ptype"],
        "--num-workers", "8",
        "--forced",
    ])


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    for j in jobs():
        if only and not j["tag"].startswith(only):
            continue
        out_path = OUT / f"{j['tag']}.parquet"
        for phase, cmd in (("generate", gen_cmd(j, out_path)), ("metric", metric_cmd(j, out_path))):
            if phase == "generate" and out_path.exists():
                print(f"[skip] {j['tag']} generation, {out_path} exists", flush=True)
                continue
            print(f"\n=== [{j['tag']}] {phase} ===\n{cmd}", flush=True)
            t0 = time.time()
            rc = subprocess.run(cmd, shell=True, cwd=ROOT).returncode
            print(f"=== [{j['tag']}] {phase} rc={rc} in {time.time()-t0:.0f}s ===", flush=True)
            if rc != 0:
                print(f"!! [{j['tag']}] {phase} FAILED, continuing", flush=True)
                break


if __name__ == "__main__":
    main()
