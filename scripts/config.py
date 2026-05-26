"""Experiment specs for scripts/run_new.py."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterable

HYDRA_KINDS = {"ppo", "sft", "dpo", "generate"}
CLI_KINDS = {"metric", "merge"}


def _tokens(args: str | Iterable[str]) -> tuple[str, ...]:
    return tuple(shlex.split(args) if isinstance(args, str) else args)


def _hydra_key(token: str) -> str:
    return token.split("=", 1)[0].lstrip("+~")


def _value(args: tuple[str, ...], key: str) -> str | None:
    prefix = f"{key}="
    for arg in reversed(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


def _flag(args: tuple[str, ...], *names: str) -> str | None:
    aliases = {name.replace("_", "-") for name in names}
    for i, arg in enumerate(args):
        name = arg.split("=", 1)[0].replace("_", "-")
        if name in aliases:
            return arg.split("=", 1)[1] if "=" in arg else (args[i + 1] if i + 1 < len(args) else None)
    return None


def _merge_hydra(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    out = list(left)
    pos = {_hydra_key(arg): i for i, arg in enumerate(out) if "=" in arg or arg.startswith(("~", "+"))}
    for arg in right:
        key = _hydra_key(arg)
        if key in pos:
            out[pos[key]] = arg
        else:
            pos[key] = len(out)
            out.append(arg)
    return tuple(out)


def _cli_groups(args: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    groups: list[tuple[str, tuple[str, ...]]] = []
    i = 0
    pos = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i].split("=", 1)[0].replace("_", "-")
            j = i + 1
            while j < len(args) and not args[j].startswith("--"):
                j += 1
            groups.append((key, args[i:j]))
            i = j
        else:
            groups.append((f"@{pos}", (args[i],)))
            pos += 1
            i += 1
    return groups


def _merge_cli(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    groups = _cli_groups(left)
    pos = {key: i for i, (key, _) in enumerate(groups) if not key.startswith("@")}
    for key, tokens in _cli_groups(right):
        if key in pos:
            groups[pos[key]] = (key, tokens)
        else:
            groups.append((key, tokens))
    return tuple(token for _, tokens in groups for token in tokens)


@dataclass(frozen=True)
class RunSpec:
    kind: str
    args: tuple[str, ...] = ()

    def __init__(self, args: str | Iterable[str] = (), kind: str = ""):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "args", _tokens(args))

    def __or__(self, other: "RunSpec") -> "RunSpec":
        kind = other.kind or self.kind
        if kind in CLI_KINDS:
            args = _merge_cli(self.args, other.args)
        else:
            args = _merge_hydra(self.args, other.args)
        return RunSpec(args=args, kind=kind)

    def get(self, key: str, default: str = "") -> str:
        return _value(self.args, key) or default

    def hydra_args(self) -> list[str]:
        return [arg for arg in self.args if _hydra_key(arg) != "gpu"]

    def command(self) -> str:
        if self.kind == "ppo":
            return shlex.join(["python", "-m", "crysreas.trainer.main_ppo", f"trainer.n_gpus_per_node={self.get('gpu', '1')}", *self.hydra_args()])
        if self.kind == "sft":
            return self._torchrun("crysreas.trainer.fsdp_sft_trainer")
        if self.kind == "dpo":
            return self._torchrun("crysreas.trainer.fsdp_dpo_trainer")
        if self.kind == "generate":
            return self._generate_command()
        if self.kind == "metric":
            return shlex.join(["python", "-m", "crysreas.metric_process", *self.args])
        if self.kind == "merge":
            path = _flag(self.args, "--path")
            output = _flag(self.args, "--output-path", "--output_path")
            if not path or not output:
                raise SystemExit("merge requires --path and --output-path")
            return shlex.join(["python", "-m", "verl.model_merger", "merge", "--backend", "fsdp", "--local_dir", path, "--target_dir", output])
        raise SystemExit(f"Unknown config kind: {self.kind}")

    def _torchrun(self, module: str) -> str:
        return shlex.join(["torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={self.get('gpu', '1')}", "-m", module, *self.hydra_args()])

    def _generate_command(self) -> str:
        args = tuple(arg for arg in self.hydra_args() if "=" in arg or arg.startswith("-"))
        positional = [arg for arg in self.hydra_args() if "=" not in arg and not arg.startswith("-")]
        if positional and _value(args, "model.path") is None:
            args = (*args, f"model.path={positional[0]}")
        model_path = _value(args, "model.path")
        if not model_path:
            raise SystemExit("generate requires a model path, e.g. python scripts/run_new.py generate checkpoints_merged/thinking")
        prompt_type = _value(args, "data.custom_data.prompt_type") or "conditional+thinking"
        if _value(args, "data.output_path") is None:
            args = (*args, f"data.output_path={model_path.rstrip('/')}/{prompt_type}.parquet")
        return shlex.join(["python", "-m", "crysreas.trainer.main_generation", *args])


PPO = RunSpec("gpu=2", kind="ppo")
SFT = RunSpec("gpu=1 trainer.default_local_dir=checkpoints/thinking", kind="sft")
DPO = RunSpec("gpu=1 trainer.default_local_dir=checkpoints/dpo_plaid_wyckoff", kind="dpo")
GEN = RunSpec(
    "trainer.n_gpus_per_node=1 data.n_samples=16 "
    "data.custom_data.prompt_type=conditional+thinking "
    "data.custom_data.split_path=assets/MP/split_generation.json",
    kind="generate",
)
METRIC = RunSpec(kind="metric")
MERGE = RunSpec(kind="merge")


EXPERIMENTS: dict[str, RunSpec] = {
    "rl_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=conditional+thinking "
        "~custom_reward_function.weights.smooth_energy_reward "
        "+custom_reward_function.weights.energy_reward=10 "
        "data.custom_data.split_path=assets/MP/split_rl.json "
        "actor_rollout_ref.model.path=checkpoints_merged/thinking "
        "trainer.experiment_name=rl_thinking_mix"
    ),
    "rl_thinking_novel": PPO
    | RunSpec(
        "data.custom_data.prompt_type=conditional+thinking "
        "~custom_reward_function.weights.smooth_energy_reward "
        "+custom_reward_function.weights.sn_reward=10 "
        "data.custom_data.split_path=assets/MP/split_rl.json "
        "actor_rollout_ref.model.path=checkpoints_merged/thinking "
        "trainer.experiment_name=rl_thinking_novel"
    ),
    "rl_thinking_only_energy": PPO
    | RunSpec(
        "data.custom_data.prompt_type=conditional+thinking ~custom_reward_function.weights "
        "+custom_reward_function.weights.energy_reward=10 data.custom_data.split_path=assets/MP/split_rl.json "
        "actor_rollout_ref.model.path=checkpoints_merged/thinking trainer.experiment_name=rl_thinking_only_energy"
    ),
    "rl_thinking_only_validity": PPO
    | RunSpec(
        "data.custom_data.prompt_type=conditional+thinking ~custom_reward_function.weights.smooth_energy_reward "
        "data.custom_data.split_path=assets/MP/split_rl.json actor_rollout_ref.model.path=checkpoints_merged/thinking "
        "trainer.experiment_name=rl_thinking_only_validity"
    ),
    "rl_spacegroup_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=spacegroup+thinking data.custom_data.split_path=assets/MP/split_rl.json "
        "~custom_reward_function.weights.smooth_energy_reward +custom_reward_function.weights.spacegroup_consistency=1 "
        "actor_rollout_ref.model.path=checkpoints_merged/thinking trainer.experiment_name=rl_spacegroup_thinking"
    ),
    "rl_elastic_no_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=elastic+no_thinking data.custom_data.split_path=assets/MP/split_elastic.json "
        "~custom_reward_function.weights +custom_reward_function.weights.elastic_reward=1 "
        "actor_rollout_ref.model.path=checkpoints_merged/rl_no_thinking trainer.experiment_name=elastic_no_thinking"
    ),
    "rl_cte_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=cte+thinking data.custom_data.split_path=assets/MP/split_cte.json "
        "~custom_reward_function.weights +custom_reward_function.weights.cte_reward=5 "
        "+custom_reward_function.weights.energy_reward=1 actor_rollout_ref.model.path=checkpoints_merged/thinking "
        "trainer.experiment_name=cte_thinking"
    ),
    "rl_elastic_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=elastic+thinking data.custom_data.split_path=assets/MP/split_elastic.json "
        "~custom_reward_function.weights +custom_reward_function.weights.elastic_reward_all=1 "
        "actor_rollout_ref.model.path=checkpoints_merged/thinking trainer.experiment_name=elastic_thinking"
    ),
    "rl_elastic_thinking_new": PPO
    | RunSpec(
        "data.custom_data.prompt_type=elastic+thinking data.custom_data.split_path=assets/MP/split_elastic.json "
        "~custom_reward_function.weights +custom_reward_function.weights.new_elastic_reward=5 "
        "+custom_reward_function.weights.energy_reward=1 actor_rollout_ref.model.path=checkpoints_merged/thinking "
        "trainer.total_epochs=2 trainer.experiment_name=rl_elastic_thinking_new"
    ),
    "rl_no_thinking": PPO
    | RunSpec(
        "data.custom_data.prompt_type=conditional+thinking ~custom_reward_function.weights.smooth_energy_reward "
        "+custom_reward_function.weights.energy_reward=10 data.custom_data.split_path=assets/MP/split_rl.json "
        "actor_rollout_ref.model.path=checkpoints_merged/no_thinking trainer.experiment_name=rl_no_thinking"
    ),
    "sft_no_thinking": SFT | RunSpec("trainer.default_local_dir=checkpoints/no_thinking data.custom_data.prompt_type=conditional+no_thinking trainer.experiment_name=no_thinking data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "sft_thinking": SFT | RunSpec("trainer.default_local_dir=checkpoints/thinking data.custom_data.prompt_type=conditional+thinking trainer.experiment_name=thinking data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "sft_crystaltextllm": SFT | RunSpec("trainer.default_local_dir=checkpoints/crystaltextllm data.custom_data.prompt_type=crystaltextllm_train+no_thinking trainer.experiment_name=sft_crystaltextllm data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "sft_crystaltextllm_8": SFT | RunSpec("trainer.default_local_dir=checkpoints/crystaltextllm_8 data.custom_data.prompt_type=crystaltextllm_8_train+no_thinking trainer.experiment_name=sft_crystaltextllm_8 data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "sft_plaid_wyckoff": SFT | RunSpec("trainer.default_local_dir=checkpoints/plaid_wyckoff data.custom_data.prompt_type=plaid_wyckoff_train+no_thinking trainer.experiment_name=sft_plaid_wyckoff data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "sft_plaid_wyckoff_8": SFT | RunSpec("trainer.default_local_dir=checkpoints/plaid_wyckoff_8 data.custom_data.prompt_type=plaid_wyckoff_8_train+no_thinking trainer.experiment_name=sft_plaid_wyckoff_8 data.custom_data.split_path=assets/MP/split_cdvae.json"),
    "dpo_plaid_wyckoff": DPO | RunSpec("model.partial_pretrain=checkpoints_merged/plaid_wyckoff model.reference_pretrain=checkpoints_merged/plaid_wyckoff data.custom_data.prompt_type=plaid_wyckoff_train+no_thinking trainer.experiment_name=dpo_plaid_wyckoff"),
    "run_metric": METRIC,
    "metric_qha": METRIC | RunSpec(kind="metric", args=["--path", "assets/MP/split_rl.json", "--output-path", "assets/MP/split_rl_qha.pkl", "--metrics-name", "cte"]),
    "metric_elastic": METRIC | RunSpec(kind="metric", args=["--path", "assets/MP/split_rl.json", "--output-path", "assets/MP/split_rl_elastic.pkl", "--metrics-name", "elastic_reward"]),
    "merge": MERGE,
    "generate": GEN,
    "generate_rl": GEN,
    "test_generate": GEN | RunSpec("data.custom_data.split_path=assets/MP/split_generation_one.json rollout.calculate_log_probs=true"),
    "generate_elastic": GEN | RunSpec("data.custom_data.prompt_type=elastic+thinking data.custom_data.split_path=assets/MP/split_generation_elastic.json"),
    "generate_cte": GEN | RunSpec("data.custom_data.prompt_type=cte+thinking data.custom_data.split_path=assets/MP/split_generation_cte.json"),
    "generate_qha": GEN | RunSpec("data.custom_data.prompt_type=cte+thinking data.custom_data.split_path=assets/MP/split_generation_cte.json"),
    "generate_crystaltextllm": GEN | RunSpec("data.custom_data.prompt_type=crystaltextllm_generation+no_thinking"),
    "generate_crystaltextllm_8": GEN | RunSpec("data.custom_data.prompt_type=crystaltextllm_8_generation+no_thinking"),
    "generate_plaid_wyckoff": GEN | RunSpec("data.custom_data.prompt_type=plaid_wyckoff_generation+no_thinking"),
    "generate_plaid_wyckoff_8": GEN | RunSpec("data.custom_data.prompt_type=plaid_wyckoff_8_generation+no_thinking"),
}
