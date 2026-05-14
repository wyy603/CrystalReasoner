#!/bin/bash

# Start from a specific step; default is 1.
START_STEP=${1:-1}
MERGED_DIR="checkpoints_merged/thinking"

source .venv/bin/activate

# ----------------------- [1/4] SFT -----------------------
if [ "$START_STEP" -le 1 ]; then
    echo "======================= [1/4] python scripts/run.py sft_thinking"
    python scripts/run.py sft_thinking --level=debug
else
    echo ">>> Skip Step [1/4]"
fi

# ----------------------- [2/4] Merge -----------------------
if [ "$START_STEP" -le 2 ]; then
    echo "======================= [2/4] python scripts/run.py merge"
    python scripts/run.py merge \
      --path checkpoints/thinking/global_step_1514/ \
      --output_path "${MERGED_DIR}/"
else
    echo ">>> Skip Step [2/4]"
fi

# ----------------------- [3/4] Generate -----------------------
if [ "$START_STEP" -le 3 ]; then
    echo "======================= [3/4] python scripts/run.py generate"
    python scripts/run.py generate "${MERGED_DIR}" --level=debug
else
    echo ">>> Skip Step [3/4]"
fi

# ----------------------- [4/4] Run Metric -----------------------
if [ "$START_STEP" -le 4 ]; then
    echo "======================= [4/4] python -m crysreas.metric_process"
    python -m crysreas.metric_process \
      --path "${MERGED_DIR}/conditional+thinking.parquet" \
      --metrics-name simple_structure smact_validity structure_validity composition_consistency spacegroup_consistency relaxed_structures energy_above_hull stable_unique_novel \
      --level=debug \
      --prompt-type=conditional+thinking \
      --forced
else
    echo ">>> Skip Step [4/4]"
fi

echo "Mission Complete."
