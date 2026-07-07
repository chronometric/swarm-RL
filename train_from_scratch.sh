#!/usr/bin/env bash
# Full training pipeline: demos → BC warm-start → RL fine-tune
set -euo pipefail
cd "$(dirname "$0")/.."
source miner_env/bin/activate

EPISODES="${1:-64}"
TIMESTEPS="${2:-500000}"

echo "=== 1/3 Collect heuristic demos (${EPISODES} episodes) ==="
python RL/collect_demos.py --episodes "$EPISODES" --stage 0 --out RL/demos_open.npz

echo "=== 2/3 Behavioral cloning warm-start ==="
python RL/pretrain_bc.py --demos RL/demos_open.npz --out RL/checkpoints/bc_warmstart.zip --epochs 8

echo "=== 3/3 RL fine-tune (Stage 0 open) ==="
python RL/train_sota.py \
  --timesteps "$TIMESTEPS" \
  --eval-freq 10000 \
  --stage 0 \
  --resume RL/checkpoints/bc_warmstart.zip \
  --run-name stage0_bc_rl

echo "Done. Best model: RL/checkpoints/stage0_bc_rl/best/best_model.zip"
