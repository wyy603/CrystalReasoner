import subprocess
import sys
import os
from dataclasses import dataclass
from typing import Union, Annotated
import tyro

# ================= 指令配置定义 =================

@dataclass
class SFTConfig:
    """SFT 训练配置"""
    nnodes: int = 1
    gpu: int = 4
    local_dir: str = "checkpoints/20260202"
    
    def to_command(self) -> str:
        return (f"torchrun --standalone --nnodes={self.nnodes} "
                f"--nproc_per_node={self.gpu} "
                f"-m crysreas.trainer.fsdp_sft_trainer "
                f"trainer.default_local_dir={self.local_dir}")

@dataclass
class GenConfig:
    """推理/生成配置"""
    gpu: int = 2
    model_path: str = "checkpoints_merged/20260102/global_step_1626"
    save_path: str = "outputs/inference_default.parquet"
    prompt_type: str = 'conditional+thinking'
    
    def to_command(self) -> str:
        # 注意：这里对带有特殊字符的字符串加了引号处理
        return (f"python -m crysreas.trainer.run_generation "
                f"--n_gpus_per_node={self.gpu} "
                f"--model_path={self.model_path} "
                f"--save_path={self.save_path} "
                f"--prompt_type='{self.prompt_type}' "
                f"--max_length=3584")

@dataclass
class PPOConfig:
    """PPO 训练配置"""
    gpu: int = 2
    mode: str = "default"
    
    def to_command(self) -> str:
        return f"python -m crysreas.trainer.main_ppo trainer.n_gpus_per_node={self.gpu} custom_reward_function.mode={self.mode}"

# ================= 主解析与执行逻辑 =================

@dataclass
class Args:
    # 使用 Annotated 配合 tyro.conf.subcommand 来显式定义子命令名称
    command: Union[
        Annotated[SFTConfig, tyro.conf.subcommand(name="sft")],
        Annotated[GenConfig, tyro.conf.subcommand(name="gen")],
        Annotated[PPOConfig, tyro.conf.subcommand(name="ppo")],
    ]

def run_workflow(args: Args):
    config = args.command
    
    # 1. 构造指令
    command_text = config.to_command()
    # 获取子命令名称 (sft, gen, or ppo)
    # 在 tyro 中，config 现在已经是 SFTConfig 等实例了
    exp_name = ""
    if isinstance(config, SFTConfig): exp_name = "sft"
    elif isinstance(config, GenConfig): exp_name = "gen"
    elif isinstance(config, PPOConfig): exp_name = "ppo"
    
    final_cmd = f"{command_text}"
    log_file = os.environ.get('SLURM_STDOUT_FILE')
    
    print(f"Experiment task: {exp_name.upper()}")
    print(f"Command: {command_text}")
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
