# Pad Estimator — implemented path

**Goal:** ≥2–3/4 open landings. Do **not** resume landing PPO.

## What is implemented

| Piece | Status |
|-------|--------|
| RAM-safe shard-sequential trainer | ✅ drops NaN depth frames (was killing training) |
| `SearchLandPilot` pad lock via estimator EMA | ✅ |
| `HybridController` auto-loads `pad_estimator.pt` | ✅ RL land OFF |
| `my_agent/drone_agent.py` + package `.pt` | ✅ |
| `check_progress` / `run_pipeline` flags | ✅ |

## Commands

See **`RL/COMMANDS.md`** — run those only.
