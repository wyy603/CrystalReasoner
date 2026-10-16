#!/bin/bash
#SBATCH --job-name=crysfixctl
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/yw10199/CrystalReasoner/logs/fixctl_%j.out
set -x
cd /scratch/yw10199/CrystalReasoner
export TMPDIR=/scratch/yw10199/tmp
export HF_HOME=/scratch/yw10199/hf_cache
export TOKENIZERS_PARALLELISM=true
export RAY_DEDUP_LOGS=0
.venv/bin/ray stop --force 2>/dev/null
.venv/bin/ray start --head --num-cpus=16 --num-gpus=1 --temp-dir=$TMPDIR/ray_$SLURM_JOB_ID
.venv/bin/python -m crysreas.metric_process \
  --path repro/denovo/main.crystaltextllm.parquet \
  --metrics-name simple_structure structure_validity smact_validity \
                 composition_consistency spacegroup_consistency stable_unique_novel \
  --prompt-type crystaltextllm_generation+no_thinking --num-workers 8 --forced
rc=$?
.venv/bin/ray stop --force 2>/dev/null
echo "DONE rc=$rc $(date)"
