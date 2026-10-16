#!/bin/bash
#SBATCH --job-name=crysshuf
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yw10199/CrystalReasoner/logs/shuffle_%j.out
set -x
cd /scratch/yw10199/CrystalReasoner
export TMPDIR=/scratch/yw10199/tmp
export HF_HOME=/scratch/yw10199/hf_cache
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DISABLED=true
export SWANLAB_MODE=offline
export RAY_DEDUP_LOGS=0
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID START=$(date)"
nvidia-smi
.venv/bin/ray stop --force 2>/dev/null
.venv/bin/ray start --head --num-cpus=16 --num-gpus=1 --temp-dir=$TMPDIR/ray_$SLURM_JOB_ID
.venv/bin/python repro/t1_shuffle_ablation.py \
  --args.num-source-rows 256 --args.num-attempts-per-condition 4
rc=$?
.venv/bin/ray stop --force 2>/dev/null
echo "DONE rc=$rc $(date)"
