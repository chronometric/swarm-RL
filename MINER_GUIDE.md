# Swarm Subnet 124 — Miner Training Guide

This guide covers the **SOTA RecurrentPPO pipeline** in `RL/` for training a competitive miner on [Swarm SN124](https://github.com/swarm-subnet/swarm).

---

## What You Are Building

| Component | Choice | Why |
|-----------|--------|-----|
| Algorithm | **Recurrent PPO** (sb3-contrib) | Stable on-policy RL + LSTM memory for 50 Hz partial observability |
| Vision | **NatureCNN** on 128×128 depth | Default MultiInputLstmPolicy encoder — proven for pixel/depth RL |
| Memory | **LSTM (256 hidden)** | Tracks momentum & hidden obstacles; replaces frame stacking |
| Curriculum | open → mountain → village → **all 6 types** | Avoids early failure in warehouse/forest |
| Shaping | progress + time penalty + landing bonus | Dense signal on top of validator `flight_reward` |
| Domain rand. | Gaussian noise on state vector | Robustness across procedural seeds |

**Scoring (same as validators):**

```
score = 0.45 × success + 0.45 × time + 0.10 × safety
```

---

## One-Time Setup

```bash
cd /home/ubuntu/workspace/swarm

# System deps (optional if Python 3.11 already installed)
chmod +x scripts/miner/install_dependencies.sh scripts/miner/setup.sh
./scripts/miner/install_dependencies.sh   # needs sudo
./scripts/miner/setup.sh

# Extra: sb3-contrib for RecurrentPPO (whitelisted for submission)
source miner_env/bin/activate
pip install sb3-contrib tensorboard

pip install -e .
swarm doctor
```

---

## Training (Long-Running)

### Stage 1 — Open terrain (easiest)

```bash
source miner_env/bin/activate
cd /home/ubuntu/workspace/swarm

python RL/train_sota.py \
  --timesteps 200000 \
  --eval-freq 10000 \
  --stage 0 \
  --run-name stage0_open
```

### Stage 2–4 — Resume with harder curriculum

When validation **mean score ≥ 0.35** (stage 0 threshold), continue:

```bash
python RL/train_sota.py \
  --timesteps 500000 \
  --eval-freq 10000 \
  --stage 1 \
  --resume RL/checkpoints/stage0_open/best/best_model.zip \
  --run-name stage1_open_mountain
```

Repeat with `--stage 2` then `--stage 3` (full benchmark mix) as scores improve.

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--timesteps` | 500000 | Total env steps |
| `--eval-freq` | 10000 | Run validation every N steps |
| `--stage` | 0 | Curriculum stage (0=open … 3=all types) |
| `--resume` | — | Continue from a `.zip` checkpoint |
| `--device` | auto | `cuda` if GPU available |

---

## Tracking Results During Training

### 1. Console (every `--eval-freq` steps)

```
[validation @ 100,000 steps | stage=open] mean=0.4123 | success=66.7% | city=... | open=0.85 | ...
  → new best model saved (0.4123) → RL/checkpoints/.../best/best_model.zip
```

### 2. TensorBoard (recommended — leave running in another terminal)

```bash
source miner_env/bin/activate
tensorboard --logdir RL/runs --port 6006
```

Open `http://localhost:6006` and watch:
- `validation/mean_score` — **primary metric**; stop/advance when plateaued
- `validation/success_rate`
- `validation/score_open`, `score_city`, etc.

### 3. JSONL log

```bash
tail -f RL/checkpoints/<run_name>/logs/validation.jsonl
```

### When to stop training

- **mean_score plateaus** for 3+ eval cycles with no improvement
- **Per-type scores** are balanced (no env type stuck at 0.01)
- Local benchmark (below) beats champion + dynamic floor (~0.005–0.015)

---

## Watch the Simulation (Visual)

```bash
# Random environment
python RL/watch_agent.py --model RL/checkpoints/stage0_open/best/best_model.zip --random

# Specific type: 1=city 2=open 3=mountain 4=village 5=warehouse 6=forest
python RL/watch_agent.py --model RL/checkpoints/.../best/best_model.zip --type 2 --seed 2002002
```

Or explore maps without an agent:

```bash
swarm visualize --type 1          # city
swarm visualize --type 5 --seed 323518   # warehouse
```

---

## Validation & Benchmark

### Fast local validation (6 fixed seeds, no Docker)

```bash
python -c "
from sb3_contrib import RecurrentPPO
from RL.validate import evaluate_model
m = RecurrentPPO.load('RL/checkpoints/stage0_open/best/best_model.zip')
r = evaluate_model(m)
print(r.summary_line())
"
```

### Package + validator-faithful benchmark

```bash
python RL/run_pipeline.py \
  --model RL/checkpoints/stage3_full/best/best_model.zip \
  --benchmark \
  --seeds-per-group 3 \
  --workers 4

swarm report
```

Quick smoke test (1 seed per env type):

```bash
python RL/run_pipeline.py --model RL/checkpoints/.../best/best_model.zip --benchmark --seeds-per-group 1
```

### Docker RPC test (closest to validator)

```bash
python RL/test_RL.py --model Submission/submission.zip --num-seeds 6 --workers 2
```

---

## Submit to Subnet (One Shot!)

Only after local benchmark consistently beats the champion:

1. `python RL/run_pipeline.py --model .../best_model.zip`
2. Copy `swarm/templates/README.md` to your GitHub repo (byte-exact)
3. Push `submission.zip` to a **private** repo
4. Commit URL on-chain, wait ~30s, make repo **public**

```bash
python neurons/miner.py \
  --netuid 124 \
  --subtensor.network finney \
  --wallet.name my_cold \
  --wallet.hotkey my_hot \
  --github_url "https://github.com/YOUR_USER/YOUR_REPO"
```

---

## File Layout

```
RL/
  train_sota.py      ← main training
  watch_agent.py     ← PyBullet GUI demo
  run_pipeline.py    ← deploy → package → benchmark
  validate.py        ← epoch validation logic
  callbacks.py       ← TensorBoard + best-model saving
  env_utils.py       ← curriculum + task sampling
  wrappers.py        ← reward shaping + domain rand.
  checkpoints/       ← saved models (gitignored)
  runs/              ← TensorBoard logs

my_agent/
  drone_agent.py     ← DroneFlightController (loads model.zip)
  model.zip          ← copied by run_pipeline.py
  requirements.txt   ← sb3-contrib (whitelisted)
```

---

## Tips for High Emissions

1. **Never register early** — one hotkey, one submission, forever.
2. Train through **all 6 environment types** before benchmarking seriously.
3. Watch **warehouse** and **forest** scores — they often lag open terrain.
4. Moving platforms appear in city/open/mountain/village — train stage 3 long enough.
5. Keep submission **≤ 50 MiB** — RecurrentPPO zip is typically ~5–15 MiB.
6. Use GPU for training (`--device cuda`); inference is lightweight on validators.

---

## Official Docs

- [Miner guide](docs/miner.md)
- [CLI reference](docs/CLI_readme.md)
- [King of the Hill scoring](docs/king_of_the_hill.md)
- [Leaderboard](https://swarm124.com/benchmark)
