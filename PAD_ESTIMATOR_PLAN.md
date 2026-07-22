# Pad Estimator Pipeline (handoff)

**Goal:** ≥2–3/4 open landings (seeds `2002002, 10012, 10033, 10052`), mean score ≳ 0.35.

**Do not** resume full-episode / landing PPO. Bottleneck is pad XY under GPS noise.

## Kept assets

| Path | Why |
|------|-----|
| `keep/landing_v2_best.zip` | Best historical RL land weights (optional assist later) |
| `checkpoints/landing_v2/best/best_model.zip` | Same, for hybrid packaging |
| `keep/bc_oracle_v6.zip` / `checkpoints/bc_oracle_v6.zip` | Oracle BC (collapsed under RL; reference only) |
| `demos_oracle_land.npz` | Successful oracle land trajectories (no `goal_xy` labels — **do not** use alone for pad supervised training) |
| `search_pilot.py` + `hybrid_controller.py` | Deploy baseline (USE_RL_LAND=False) |
| `archive/` | Old train/collect scripts (safe to ignore) |

## Architecture

```
CRUISE → SPIRAL → [pad estimator predicts pad_xy] → soft_land(target_xy)
```

1. **Collect** labeled frames: `(depth, state) → goal_xy_rel` (privileged `GOAL_POS` at train time only).
2. **Train** small CNN+MLP pad estimator.
3. **Plug** estimator into `SearchLandPilot` lock (replace fragile altitude-peak lock when confident).
4. **Eval** hybrid with RL land assist **off**.
5. Optional later: DAgger + tiny residual on soft-land.

---

## Scripts ready for you

| Script | Role |
|--------|------|
| `collect_pad_labels.py` | Step 1 — labeled `(depth, state → goal_xy_rel)` |
| `pad_estimator.py` | Model + `predict_pad_xy()` |
| `train_pad_estimator.py` | Step 2 — supervised train |
| `PAD_ESTIMATOR_PLAN.md` | This handoff |

**You still must do Step 3:** wire `predict_pad_xy` into `search_pilot.py` / `my_agent/drone_agent.py`.

---

## Commands (run from repo root)

```bash
source miner_env/bin/activate
cd ~/workspace/swarm
```

### Step 0 — Baseline (current hybrid)

```bash
python RL/check_progress.py --hybrid --model RL/checkpoints/landing_v2/best/best_model.zip
# Expect ~1/4 (10052). This is the bar to beat.
```

### Step 1 — Collect pad labels (~1–3 h)

```bash
python RL/collect_pad_labels.py \
  --episodes 400 \
  --out RL/demos_pad_labels.npz \
  --distance-min 2.0 \
  --distance-max 25.0
```

Success check:

```bash
python -c "
import numpy as np
d=np.load('RL/demos_pad_labels.npz', allow_pickle=True)
print(list(d.keys()), 'N=', len(d['goal_xy_rel']))
err=np.linalg.norm(d['goal_xy_rel'], axis=1)
print('mean|goal_xy_rel|', err.mean(), 'pct<2m', (err<2).mean())
"
```

### Step 2 — Train pad estimator (~10–40 min)

```bash
python RL/train_pad_estimator.py \
  --demos RL/demos_pad_labels.npz \
  --epochs 30 \
  --batch-size 256 \
  --out RL/checkpoints/pad_estimator.pt
```

Look for val XY error **≪ 0.5 m** when altitude ray < ~4 m (close-range subset).

### Step 3 — Wire into pilot + eval

Edit `RL/search_pilot.py` (and sync `my_agent/drone_agent.py`):

- Load `pad_estimator.pt`
- During SEARCH/REFINE, if estimator confidence high → set `_pad_xy` and enter LAND
- Keep soft-land heuristic; **do not** blend RL

```bash
python RL/check_progress.py --hybrid --model RL/checkpoints/landing_v2/best/best_model.zip
# Target: ≥2/4 open landings

python RL/run_pipeline.py --model RL/checkpoints/landing_v2/best/best_model.zip
# Only after ≥2/4
```

### Step 4 (optional) — DAgger / residual

Only if locks are good but soft-land still fails stable contact:

1. Roll out student; relabel pad_xy / actions with oracle `GOAL_POS`
2. Retrain estimator (and optional tiny residual Δu on soft-land)
3. Re-eval

---

## Constants (do not fight these)

- `GOAL_TOL ≈ 0.5088`, platform radius `0.6`
- `LANDING_MAX_VZ=0.5`, `LANDING_MAX_VXY_REL=0.6`, `LANDING_MAX_TILT_RAD=0.26`, `LANDING_STABLE_SEC=0.5`
- Horizon 60 s @ 50 Hz (~3000 steps) — keep cruise+search efficient

## Stop conditions

| Signal | Action |
|--------|--------|
| Val pad error > 1 m close-range | More/diverse labels; filter to SEARCH phase near pad |
| Locks good, land fails tilt/vel | Soft-land only; optional residual — not full PPO |
| Still 1/4 after good estimator | Expand spiral for large GPS noise (2002002); check time budget |
