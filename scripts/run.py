#!/usr/bin/env python3
"""
Unified experiment runner (replaces root ``task_*.sh`` / ``run_*.sh``).

Usage:
    python scripts/run.py <experiment> [extra hydra overrides...]

Runs experiments directly in the current process environment without submitting
through an external scheduler.

Log level: pass ``--level=debug`` / ``--level=info`` / … (any standard ``logging`` level name), or
two-token ``--level debug``. ``RUNPY_LOG_LEVEL`` is also supported (legacy:
``RUNPY_DEBUG=1`` maps to ``DEBUG``). For ``run_metric`` at ``DEBUG``, the child
gets ``AI4SCI_METRIC_PROGRESS_DEBUG=1`` so batch ``tqdm`` uses stderr after ``ray.init()``.
For ``generate_*`` at ``DEBUG``, ``RUNPY_GENERATION_DEBUG=1`` runs the generation driver
in-process so ``tqdm`` reaches the terminal instead of a Ray worker log.

All worker output uses a primary log file ``logs/experiment.out``.

Examples:
    python scripts/run.py rl_cte_thinking
    python scripts/run.py rl_thinking trainer.max_steps=10
    python scripts/run.py run_task data.foo=1
    python scripts/run.py run_metric --path assets/MP/split_cdvae.json \\
        --output_path assets/MP/out.parquet --metrics_name simple_structure \\
        --prompt_type elastic+thinking
    python scripts/run.py merge --path checkpoints/20251227/global_step_812 --output_path checkpoints_merged/20251227/global_step_812
    python scripts/run.py generate_rl checkpoints_merged/rl_thinking
        # auto output: checkpoints_merged/rl_thinking/conditional+thinking.parquet
    python scripts/run.py generate checkpoints_merged/thinking trainer.n_gpus_per_node=4
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Project root (parent of ``scripts/``)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load project environment variables (``crysreas`` does this on import).
import crysreas  # noqa: F401

from scripts.config import EXPERIMENTS, ExperimentSpec


def _valid_log_level_names() -> str:
    """Comma-separated sorted names for error messages."""
    return ", ".join(sorted(logging._nameToLevel.keys()))


def _normalize_log_level(name: str) -> str:
    """Return canonical upper-case level name or exit with usage."""
    key = name.strip().upper()
    if key not in logging._nameToLevel:
        print(
            f"Invalid log level {name!r}. Choose one of: {_valid_log_level_names()}",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def _strip_log_level_from_argv(argv: list[str]) -> tuple[list[str], str | None]:
    """Remove ``--level`` / ``--level=`` from argv; return cleaned argv and last level if any."""
    out: list[str] = []
    level: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--level="):
            level = a[len("--level=") :].strip()
            i += 1
            continue
        if a == "--level":
            if i + 1 >= len(argv):
                print("error: --level requires a value", file=sys.stderr)
                sys.exit(2)
            level = argv[i + 1].strip()
            i += 2
            continue
        out.append(a)
        i += 1
    return out, level


def _argv_has_level_flag(argv: list[str]) -> bool:
    if any(a.startswith("--level=") for a in argv):
        return True
    return "--level" in argv


def _configure_global_logging(level_name: str) -> None:
    """Set root logger and basicConfig to the given standard level."""
    lvl = logging._nameToLevel[level_name]
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(lvl)

def _venv_ray() -> str:
    """Ray CLI from the project venv (PATH may not include ``.venv/bin``)."""
    return str(ROOT / ".venv" / "bin" / "ray")

def _redirect_stdio_to_log(log_path: Path) -> None:
    """Send process stdout/stderr (fd 1 and 2) to ``log_path`` (truncate)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

worker_log_path: str
def _resolve_worker_log_path() -> Path:
    """Primary log file for this worker run (under ``logs/``)."""
    global worker_log_path
    return worker_log_path


def _merge_with_extras(base: str, extras: list[str]) -> str:
    base = base.strip()
    if extras:
        rest = " ".join(extras).strip()
        if not base:
            return rest
        return f"{base} {rest}"
    return base


def _merge_ppo_args(spec: ExperimentSpec, extras: list[str]) -> str:
    return _merge_with_extras(spec.ppo_args, extras)


def _merge_sft_args(spec: ExperimentSpec, extras: list[str]) -> str:
    return _merge_with_extras(spec.sft_args, extras)


def _merge_dpo_args(spec: ExperimentSpec, extras: list[str]) -> str:
    return _merge_with_extras(spec.dpo_args, extras)


def _normalize_metric_cli_argv(extras: list[str]) -> list[str]:
    """Allow ``--output_path`` / ``--metrics_name`` style; tyro expects kebab-case flags."""
    aliases = {
        "--output_path": "--output-path",
        "--metrics_name": "--metrics-name",
        "--prompt_type": "--prompt-type",
        "--num_workers": "--num-workers",
    }
    return [aliases.get(a, a) for a in extras]


def _normalize_merge_cli_argv(extras: list[str]) -> list[str]:
    aliases = {
        "--output_path": "--output-path",
    }
    return [aliases.get(a, a) for a in extras]


def _extract_flag_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    prefix = f"{flag}="
    for token in argv:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _extract_hydra_value(args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for token in reversed(args):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _parse_generate_extras(extras: list[str]) -> tuple[str | None, list[str]]:
    model_dir: str | None = None
    hydra_extras: list[str] = []
    for token in extras:
        if (
            model_dir is None
            and "=" not in token
            and not token.startswith("-")
        ):
            model_dir = token
            continue
        hydra_extras.append(token)
    return model_dir, hydra_extras


def _run_metric(
    spec: ExperimentSpec,
    extras: list[str],
    log_level: str | None,
) -> None:
    if spec.metric_argv_from_cli:
        if not extras:
            print(
                "run_metric requires arguments after the experiment name, e.g.\n"
                "  python scripts/run.py run_metric --path assets/MP/x.json "
                "--output-path out.parquet --metrics-name simple_structure\n"
                "Elastic/GT buckets: set --prompt_type elastic+thinking (maps to --prompt-type).",
                file=sys.stderr,
            )
            sys.exit(2)
        mp_argv = _normalize_metric_cli_argv(extras)
        if log_level and not _argv_has_level_flag(mp_argv):
            mp_argv.append(f"--level={log_level.lower()}")
        argv = [sys.executable, "-m", "crysreas.metric_process", *mp_argv]
    else:
        preset = list(spec.metric_argv)
        if log_level and not _argv_has_level_flag(preset):
            preset.append(f"--level={log_level.lower()}")
        argv = [sys.executable, "-m", "crysreas.metric_process", *preset]
    cmd_s = " ".join(shlex.quote(a) for a in argv)

    env = os.environ.copy()
    if log_level == "DEBUG":
        # Child process: Ray may lower loggers after init; batch tqdm uses this env.
        env["AI4SCI_METRIC_PROGRESS_DEBUG"] = "1"

    try:
        subprocess.run([_venv_ray(), "stop", "--force"], cwd=ROOT, env=env, check=False)
        subprocess.run([_venv_ray(), "start", "--head"], cwd=ROOT, env=env, check=True)

        print(f"▶️  {cmd_s}")
        try:
            subprocess.run(argv, cwd=ROOT, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"❌ metric job failed: {e.returncode}")
            raise
        print("✅ metric job finished")
    finally:
        subprocess.run([_venv_ray(), "stop", "--force"], cwd=ROOT, env=env, check=False)


def _run_ppo_like(
    spec: ExperimentSpec,
    merged_args: str,
) -> None:
    env = os.environ.copy()
    try:
        subprocess.run([_venv_ray(), "stop", "--force"], cwd=ROOT, env=env, check=False)
        subprocess.run([_venv_ray(), "start", "--head"], cwd=ROOT, env=env, check=True)

        py = shlex.quote(str(ROOT / ".venv" / "bin" / "python"))
        inner = f"{py} scripts/run_pipeline.py ppo --gpu {spec.gpu} --args {shlex.quote(merged_args)}"

        print(f"▶️  {inner}")
        subprocess.run(inner, shell=True, cwd=ROOT, check=True, env=env)
    finally:
        subprocess.run([_venv_ray(), "stop", "--force"], cwd=ROOT, env=env, check=False)


def _run_sft(spec: ExperimentSpec, merged_args: str) -> None:
    env = os.environ.copy()

    py = shlex.quote(str(ROOT / ".venv" / "bin" / "python"))
    inner = (
        f"{py} scripts/run_pipeline.py sft --gpu {spec.sft_gpu} "
        f"--local-dir {shlex.quote(spec.sft_local_dir)} --args {shlex.quote(merged_args)}"
    )

    print(f"▶️  {inner}")
    subprocess.run(inner, shell=True, cwd=ROOT, check=True, env=env)


def _run_dpo(spec: ExperimentSpec, merged_args: str) -> None:
    env = os.environ.copy()

    py = shlex.quote(str(ROOT / ".venv" / "bin" / "python"))
    inner = (
        f"{py} scripts/run_pipeline.py dpo --gpu {spec.dpo_gpu} "
        f"--local-dir {shlex.quote(spec.dpo_local_dir)} --args {shlex.quote(merged_args)}"
    )

    print(f"▶️  {inner}")
    subprocess.run(inner, shell=True, cwd=ROOT, check=True, env=env)


def _run_generate(
    spec: ExperimentSpec,
    extras: list[str],
    log_level: str | None,
) -> None:
    env = os.environ.copy()
    if log_level == "DEBUG":
        # Run generation driver loop in-process so ``tqdm`` reaches the terminal (see ``main_generation``).
        env["RUNPY_GENERATION_DEBUG"] = "1"

    model_dir, hydra_extras = _parse_generate_extras(extras)
    base_args = shlex.split(spec.generate_args)

    base_model = _extract_hydra_value(base_args, "model.path")
    extra_model = _extract_hydra_value(hydra_extras, "model.path")
    model_path = model_dir or extra_model or base_model
    if not model_path:
        print(
            "generate requires a model directory as first arg or model.path=..., e.g.\n"
            "  python scripts/run.py generate_rl checkpoints_merged/rl_thinking",
            file=sys.stderr,
        )
        sys.exit(2)

    prompt_type = (
        _extract_hydra_value(hydra_extras, "data.custom_data.prompt_type")
        or _extract_hydra_value(base_args, "data.custom_data.prompt_type")
        or "conditional+thinking"
    )
    auto_output = f"{model_path.rstrip('/')}/{prompt_type}.parquet"

    final_args: list[str] = [*base_args, *hydra_extras, f"model.path={model_path}"]
    if _extract_hydra_value(final_args, "data.output_path") is None:
        final_args.append(f"data.output_path={auto_output}")
    print(f"generate output path: {auto_output}")

    cmd = [sys.executable, "-m", "crysreas.trainer.main_generation", *final_args]
    cmd_s = " ".join(shlex.quote(a) for a in cmd)
    print(f"▶️  {cmd_s}")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _run_merge(spec: ExperimentSpec, extras: list[str]) -> None:
    env = os.environ.copy()

    argv = _normalize_merge_cli_argv(extras)
    path = _extract_flag_value(argv, "--path") or spec.merge_path
    output_path = _extract_flag_value(argv, "--output-path") or spec.merge_output_path
    if not path or not output_path:
        print(
            "merge requires --path and --output_path (or --output-path), e.g.\n"
            "  python scripts/run.py merge --path checkpoints/xxx --output_path checkpoints_merged/xxx",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        path,
        "--target_dir",
        output_path,
    ]
    cmd_s = " ".join(shlex.quote(a) for a in cmd)
    print(f"▶️  {cmd_s}")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _run_worker(experiment: str, extras: list[str], log_level: str | None) -> None:
    primary_log = _resolve_worker_log_path()
    _redirect_stdio_to_log(Path(primary_log))
    print(f"run.py primary log: {primary_log}", flush=True)

    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment {experiment!r}. Valid: {', '.join(sorted(EXPERIMENTS))}")
        sys.exit(2)

    spec = EXPERIMENTS[experiment]

    if spec.kind == "ppo":
        merged = _merge_ppo_args(spec, extras)
        if experiment == "run_task" and not merged:
            print("run_task requires Hydra overrides after the experiment name (former run_task.sh).")
            sys.exit(2)
        _run_ppo_like(spec, merged)
        return

    if spec.kind == "sft":
        merged = _merge_sft_args(spec, extras)
        _run_sft(spec, merged)
        return

    if spec.kind == "dpo":
        merged = _merge_dpo_args(spec, extras)
        _run_dpo(spec, merged)
        return

    if spec.kind == "metric":
        if extras and not spec.metric_argv_from_cli:
            print(
                "warning: extra arguments are ignored for this metric experiment "
                "(use run_metric for custom metric_process CLI)",
                file=sys.stderr,
            )
        _run_metric(
            spec,
            extras,
            log_level,
        )
        return

    if spec.kind == "generate":
        _run_generate(spec, extras, log_level)
        return

    if spec.kind == "merge":
        _run_merge(spec, extras)
        return


def _resolve_log_level(argv_level: str | None) -> str | None:
    """Effective level: CLI wins, then ``RUNPY_LOG_LEVEL``, then legacy ``RUNPY_DEBUG``."""
    if argv_level:
        return _normalize_log_level(argv_level)
    raw = (os.environ.get("RUNPY_LOG_LEVEL") or "").strip()
    if raw:
        return _normalize_log_level(raw)
    if os.environ.get("RUNPY_DEBUG") == "1":
        return "DEBUG"
    return None


def main() -> None:
    argv = sys.argv[1:]
    argv, level_from_argv = _strip_log_level_from_argv(argv)
    log_level = _resolve_log_level(level_from_argv)
    if log_level:
        _configure_global_logging(log_level)

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Experiments:\n")
        for k, s in sorted(EXPERIMENTS.items()):
            print(f"  {k:24}  {s.description or s.kind}")
        sys.exit(0 if argv else 1)

    experiment = argv[0]
    extras = argv[1:]

    global worker_log_path
    log_rel = f"logs/{experiment}.out"
    worker_log_path = (ROOT / log_rel).resolve()

    try:
        _run_worker(experiment, extras, log_level)
    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
