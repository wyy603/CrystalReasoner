#!/bin/bash

# Start from a specific step; default is 1.
START_STEP=${1:-1}
MERGED_DIR="checkpoints_merged/plaid_wyckoff_8"

source .venv/bin/activate

# ----------------------- [1/4] SFT -----------------------
if [ "$START_STEP" -le 1 ]; then
    echo "======================= [1/4] python scripts/run.py sft_plaid_wyckoff_8"
    python scripts/run.py sft_plaid_wyckoff_8 --level=debug
else
    echo ">>> Skip Step [1/4]"
fi

# ----------------------- [2/4] Merge -----------------------
if [ "$START_STEP" -le 2 ]; then
    echo "======================= [2/4] python scripts/run.py merge"
    python scripts/run.py merge \
      --path checkpoints/plaid_wyckoff_8/global_step_1514/ \
      --output_path "${MERGED_DIR}/"
else
    echo ">>> Skip Step [2/4]"
fi

# ----------------------- [3/4] Generate -----------------------
if [ "$START_STEP" -le 3 ]; then
    echo "======================= [3/4] python scripts/run.py generate_plaid_wyckoff_8"
    python scripts/run.py generate_plaid_wyckoff_8 "${MERGED_DIR}" --level=debug
else
    echo ">>> Skip Step [3/4]"
fi

# ----------------------- [4/4] Run Metric -----------------------
if [ "$START_STEP" -le 4 ]; then
    echo "======================= [4/4] python scripts/run.py run_metric"
    python -m crysreas.metric_process \
      --path "${MERGED_DIR}/plaid_wyckoff_8_generation+no_thinking.parquet" \
      --metrics-name composition_consistency stable_unique_novel \
      --level=debug \
      --prompt-type=plaid_wyckoff_8_generation+no_thinking \
      --forced
else
    echo ">>> Skip Step [4/4]"
fi

echo "Mission Complete."
