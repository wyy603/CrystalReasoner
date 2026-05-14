import subprocess
import sys
import os
from dataclasses import dataclass
from typing import Union, Annotated
import tyro

@dataclass
class SFTConfig:
    """SFT 训练配置"""
    nnodes: int = 1
    gpu: int = 1
    local_dir: str = "checkpoints/no_thinking"
    args: str = ""
    
    def to_command(self) -> str:
        return (f"torchrun --standalone --nnodes={self.nnodes} "
                f"--nproc_per_node={self.gpu} "
                f"-m crysreas.trainer.fsdp_sft_trainer "
                f"trainer.default_local_dir={self.local_dir} "
                f"{self.args}")

@dataclass
class DPOConfig:
    """Offline DPO 训练配置"""
    nnodes: int = 1
    gpu: int = 1
    local_dir: str = "checkpoints/dpo_plaid_wyckoff"
    args: str = ""

    def to_command(self) -> str:
        return (f"torchrun --standalone --nnodes={self.nnodes} "
                f"--nproc_per_node={self.gpu} "
                f"-m crysreas.trainer.fsdp_dpo_trainer "
                f"trainer.default_local_dir={self.local_dir} "
                f"{self.args}")

@dataclass
class GenConfig:
    """推理/生成配置"""
    gpu: int = 2
    model_path: str = "checkpoints_merged/20260102/global_step_1626"
    save_path: str = "outputs/inference_default.parquet"
    prompt_type: str = 'conditional+thinking'
    split_path: str = 'assets/MP/split_generation.json'
    args: str = ""
    
    def to_command(self) -> str:
        return (f"python -m crysreas.trainer.main_generation "
                f"trainer.n_gpus_per_node={self.gpu} "
                f"model.path={self.model_path} "
                f"data.output_path={self.save_path} "
                f"data.custom_data.prompt_type='{self.prompt_type}' "
                f"data.custom_data.split_path='{self.split_path}' "
                f"{self.args}")

@dataclass
class PPOConfig:
    """PPO 训练配置"""
    gpu: int = 2
    args: str = ""
    
    def to_command(self) -> str:
        return f"python -m crysreas.trainer.main_ppo trainer.n_gpus_per_node={self.gpu} {self.args}"

@dataclass
class Args:
    command: Union[
        Annotated[SFTConfig, tyro.conf.subcommand(name="sft")],
        Annotated[DPOConfig, tyro.conf.subcommand(name="dpo")],
        Annotated[GenConfig, tyro.conf.subcommand(name="gen")],
        Annotated[PPOConfig, tyro.conf.subcommand(name="ppo")],
    ]
    slurm: bool = False

from datetime import datetime
def run_workflow(args: Args):
    config = args.command
    
    command_text = config.to_command()
    exp_name = ""
    if isinstance(config, SFTConfig): exp_name = "sft"
    elif isinstance(config, DPOConfig): exp_name = "dpo"
    elif isinstance(config, GenConfig): exp_name = "gen"
    elif isinstance(config, PPOConfig): exp_name = "ppo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"{exp_name}_{timestamp}"
    
    if args.slurm:
        log_file = os.readlink('/proc/self/fd/1')
        final_cmd = f"{command_text} +log_path={log_file}"
    else:
        os.makedirs('logs', exist_ok=True)
        log_file = f"logs/pipeline_{exp_name}.log"
        final_cmd = f"{command_text} > {log_file} 2>&1"
    
    print(f"Experiment task: {exp_name.upper()}")
    print(f"Command: {final_cmd}")
    print(f"Log file: {log_file}")
    print("-" * 40)
    
    try:
        subprocess.run(final_cmd, shell=True, check=True, text=True)
        print("Task finished successfully.")
            
    except subprocess.CalledProcessError as e:
        print(f"Task failed with return code {e.returncode}.")
        sys.exit(1)

if __name__ == "__main__":
    # 关键点：使用 tyro.conf.OmitSubcommandPrefixes 来去掉 "command:" 前缀
    args = tyro.cli(
        Args, 
        config=(tyro.conf.OmitSubcommandPrefixes,)
    )
    run_workflow(args)
