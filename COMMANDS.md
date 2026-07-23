# Commands — do NOT eval/package until close_mae ≪ 0.5 m

```bash
source miner_env/bin/activate
cd ~/workspace/swarm
```

## Retrain (required — previous run plateaued at 2.9 m)

GPS `search_rel` leak is now blocked; train close-range only:

```bash
python -u RL/train_pad_estimator.py \
  --demos RL/demos_pad_labels.shards \
  --epochs 30 \
  --batch-size 64 \
  --stride 1 \
  --max-dist 6 \
  --out RL/checkpoints/pad_estimator.pt \
  2>&1 | tee RL/checkpoints/pad_estimator_train.log
```

Gate: **`close_mae ≪ 0.5`** and finite (not nan). If still >1 m after epoch ~15, stop and ping me.

## Only after gate passes

```bash
python RL/check_progress.py \
  --hybrid \
  --model RL/checkpoints/landing_v2/best/best_model.zip \
  --pad-estimator RL/checkpoints/pad_estimator.pt
```

Target ≥2/4, then:

```bash
python RL/run_pipeline.py \
  --model RL/checkpoints/landing_v2/best/best_model.zip \
  --pad-estimator RL/checkpoints/pad_estimator.pt
```
