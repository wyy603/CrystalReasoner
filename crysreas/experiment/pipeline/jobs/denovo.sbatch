#!/bin/bash
#SBATCH --job-name=crysdenovo
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yw10199/CrystalReasoner/logs/denovo_%A_%a.out
set -x
cd /scratch/yw10199/CrystalReasoner
export TMPDIR=/scratch/yw10199/tmp
export HF_HOME=/scratch/yw10199/hf_cache
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SWANLAB_MODE=offline
export RAY_DEDUP_LOGS=0
TAG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" repro/tags_denovo.txt)
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID TAG=$TAG START=$(date)"
nvidia-smi
ray stop --force 2>/dev/null
ray start --head --num-cpus=16 --num-gpus=1 --temp-dir=$TMPDIR/ray_$SLURM_JOB_ID
.venv/bin/python repro/run_matrix_denovo.py "$TAG"
rc=$?
ray stop --force 2>/dev/null
echo "DONE rc=$rc TAG=$TAG $(date)"
