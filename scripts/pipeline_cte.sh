source .venv/bin/activate

MERGED_DIR="checkpoints_merged/rl_cte_thinking"

# echo "======================= [1/4] python scripts/run.py rl_cte_thinking"
# python scripts/run.py rl_cte_thinking --level=debug

echo "======================= [2/4] python scripts/run.py merge"
python scripts/run.py merge \
  --path checkpoints/crystal_rl/cte_thinking/global_step_62/actor/ \
  --output_path "${MERGED_DIR}/"

echo "======================= [3/4] python scripts/run.py generate_cte"
python scripts/run.py generate_cte "${MERGED_DIR}" --level=debug

echo "======================= [4/4] python scripts/run.py run_metric"
python scripts/run.py run_metric \
  --path "${MERGED_DIR}/cte+thinking.parquet" --metrics-name cte_reward_all --level=debug
