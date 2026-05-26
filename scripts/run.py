#!/usr/bin/env python3
"""Run registered experiments with scripts/config_new.py."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import crysreas  # noqa: F401

from scripts.config import CLI_KINDS, EXPERIMENTS, RunSpec


@dataclass
class CliArgs:
    experiment: str = ""
    output: str = "default"
    dry_run: bool = False


def _parse_cli(argv: list[str]) -> tuple[CliArgs, list[str]]:
    argv = [
        "--dry_run" if arg == "--dry-run" else "--no_dry_run" if arg == "--no-dry-run" else arg
        for arg in argv
    ]
    args, extras = tyro.cli(
        CliArgs,
        args=argv,
        return_unknown_args=True,
        use_underscores=True,
        description=(
            "Run a registered CrysReas experiment.\n\n"
            f"Experiments: {', '.join(sorted(EXPERIMENTS))}"
        ),
    )
    if not args.experiment and extras and not extras[0].startswith("-"):
        args.experiment = extras.pop(0)
    return args, extras


def _cli_to_hydra(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                out.append(key)
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.append(f"{key}={tokens[i + 1]}")
                i += 1
            else:
                out.append(f"{key}=true")
        else:
            out.append(token)
        i += 1
    return out


def _redirect(output: str, experiment_name: str) -> Path | None:
    if output == "terminal":
        return None
    if output == "default":
        path = ROOT / "logs" / f"{experiment_name}.out"
    elif output == "time":
        path = ROOT / "logs" / f"{datetime.now():%Y%m%d_%H%M%S}.out"
    else:
        path = Path(output)
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    if fd > 2:
        os.close(fd)
    return path


def _run_command(command: str, dry_run: bool = False) -> None:
    print(f"Command: {command}", flush=True)
    if not dry_run:
        subprocess.run(command, shell=True, cwd=ROOT, check=True, text=True)


def _run_with_ray(command: str, dry_run: bool = False) -> None:
    if dry_run:
        _run_command(command, dry_run=True)
        return
    try:
        subprocess.run("ray stop --force", shell=True, cwd=ROOT, check=False)
        subprocess.run("ray start --head", shell=True, cwd=ROOT, check=True)
        _run_command(command)
    finally:
        subprocess.run("ray stop --force", shell=True, cwd=ROOT, check=False)


def main() -> None:
    cli, extras = _parse_cli(sys.argv[1:])
    if not cli.experiment:
        print("Usage: python scripts/run.py [--experiment EXPERIMENT] [--output default|time|terminal|PATH] [--dry_run] [extra overrides...]")
        print("\nExperiments:", ", ".join(sorted(EXPERIMENTS)))
        raise SystemExit(1)
    if cli.experiment not in EXPERIMENTS:
        raise SystemExit(f"Unknown experiment {cli.experiment!r}. Valid: {', '.join(sorted(EXPERIMENTS))}")

    base = EXPERIMENTS[cli.experiment]
    extra_args = extras if base.kind in CLI_KINDS else _cli_to_hydra(extras)
    spec = base | RunSpec(extra_args)

    log_path = _redirect(cli.output, cli.experiment)
    if log_path:
        print(f"log: {log_path}", flush=True)

    command = spec.command()
    if spec.kind in {"ppo", "metric"}:
        _run_with_ray(command, dry_run=cli.dry_run)
    else:
        _run_command(command, dry_run=cli.dry_run)


if __name__ == "__main__":
    main()
