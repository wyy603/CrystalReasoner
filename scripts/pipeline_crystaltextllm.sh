#!/bin/bash

# 获取起始步数，默认为 1
START_STEP=${1:-1}
MERGED_DIR="checkpoints_merged/crystaltextllm"

source .venv/bin/activate

# ----------------------- [1/3] Merge -----------------------
if [ "$START_STEP" -le 1 ]; then
    echo "======================= [1/3] python scripts/run.py merge"
    python scripts/run.py merge \
      --path checkpoints/crystaltextllm/global_step_1514/ \
      --output_path "${MERGED_DIR}/"
else
    echo ">>> Skip Step [1/3]"
fi

# ----------------------- [2/3] Generate RL -----------------------
if [ "$START_STEP" -le 2 ]; then
    echo "======================= [2/3] python scripts/run.py generate_crystaltextllm"
    python scripts/run.py generate_crystaltextllm "${MERGED_DIR}" --level=debug
else
    echo ">>> Skip Step [2/3]"
fi

# ----------------------- [3/3] Run Metric -----------------------
if [ "$START_STEP" -le 3 ]; then
    echo "======================= [3/3] python scripts/run.py run_metric"
    python -m crysreas.metric_process \
      --path "${MERGED_DIR}/crystaltextllm_generation+no_thinking.parquet" \
      --metrics-name simple_structure smact_validity structure_validity composition_consistency relaxed_structures energy_above_hull stable_unique_novel \
      --level=debug \
      --prompt-type=crystaltextllm_generation+no_thinking \
      --forced
else
    echo ">>> Skip Step [3/3]"
fi

echo "Mission Complete."