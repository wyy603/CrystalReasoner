"""De-novo S.U.N. re-evaluation: 8141 distinct test formulas x 2 samples.

Rationale (see REPORT.md): the paper's protocol is 1024 prompts x 16 samples, so
`uniqueness` is dominated by intra-prompt mode collapse and is sensitive to set
size. Holding the structure budget roughly constant but spreading it over every
distinct test formula decouples the two.

Novelty is NOT recomputed here -- `stable_unique_novel` still uses MatterGen's
MP+Alexandria reference. `pipeline/t1_novelty_train.py` adds the
novelty-vs-training-split variant afterwards as a post-processing pass, which
keeps the MP2020 convex hull (and therefore `is_stable`) untouched.
"""
from __future__ import annotations
import shlex, subprocess, sys, time
from pathlib import Path
# These scripts put the repository root on sys.path before `crysreas` becomes
# importable, so the paths must be available without importing it. Keep in sync
# with crysreas/experiment/paths.py.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PY_BIN = str(ROOT / ".venv" / "bin" / "python")
OUT = DATA_DIR / "denovo"
SPLIT = str(DATA_DIR / "splits/split_denovo_8141.json")
N_SAMPLES = 2
GPUS = "1"

MAIN_METRICS = [
    "simple_structure", "structure_validity", "smact_validity",
    "composition_consistency", "spacegroup_consistency", "stable_unique_novel",
]

# (tag, checkpoint dir, prompt_type) -- identical to run_matrix.py
MAIN_MODELS = [
    ("base",           "no_thinking",            "conditional+thinking"),
    ("thinking",       "thinking",               "conditional+thinking"),
    ("rl",             "rl_no_thinking",         "conditional+thinking"),
    ("full",           "rl_thinking_mix",        "conditional+thinking"),
    ("validity_only",  "thinking_only_validity", "conditional+thinking"),
    ("energy_only",    "thinking_only_energy",   "conditional+thinking"),
    ("crystaltextllm", "crystaltextllm",         "crystaltextllm_generation+no_thinking"),
    ("plaid_wyckoff",  "plaid_wyckoff",          "plaid_wyckoff_generation+no_thinking"),
]


def jobs() -> list[dict]:
    return [dict(tag=f"main.{tag}", ckpt=ckpt, ptype=ptype,
                 split=SPLIT, n=N_SAMPLES, metrics=MAIN_METRICS)
            for tag, ckpt, ptype in MAIN_MODELS]


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
