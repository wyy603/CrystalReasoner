source .venv/bin/activate

MERGED_DIR="checkpoints_merged/rl_elastic_thinking_new2"

echo "======================= [1/4] python scripts/run.py rl_elastic_thinking_new"
python scripts/run.py rl_elastic_thinking_new --level=debug

echo "======================= [2/4] python scripts/run.py merge"
python scripts/run.py merge \
  --path checkpoints/crystal_rl/rl_elastic_thinking_new2/global_step_124/actor/ \
  --output_path "${MERGED_DIR}/"

echo "======================= [3/4] python scripts/run.py generate_elastic"
python scripts/run.py generate_elastic "${MERGED_DIR}" --level=debug

echo "======================= [4/4] python scripts/run.py run_metric"
python scripts/run.py run_metric \
  --path "${MERGED_DIR}/elastic+thinking.parquet" --metrics-name elastic_reward_all --level=debug
