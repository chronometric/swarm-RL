#!/usr/bin/env python3
"""
Watch your trained agent fly in the PyBullet GUI.

Usage:
  source miner_env/bin/activate
  python RL/watch_agent.py --model RL/checkpoints/sota_open_*/best/best_model.zip
  python RL/watch_agent.py --model RL/checkpoints/.../best/best_model.zip --type 1 --seed 1001001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.validate import load_recurrent_model, rollout_episode
from swarm.constants import SIM_DT
from swarm.validator.task_gen import random_task, task_for_seed_and_type


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Swarm agent in PyBullet GUI")
    parser.add_argument("--model", type=Path, required=True, help="Path to RecurrentPPO .zip")
    parser.add_argument("--type", type=int, default=None, help="Challenge type 1-6 (default: random)")
    parser.add_argument("--seed", type=int, default=42, help="Map seed")
    parser.add_argument("--random", action="store_true", help="Random task instead of fixed type/seed")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    print(f"Loading {args.model} ...")
    model = load_recurrent_model(args.model)

    if args.random:
        task = random_task(sim_dt=SIM_DT, seed=args.seed)
    elif args.type is not None:
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=args.seed, challenge_type=args.type)
    else:
        task = random_task(sim_dt=SIM_DT, seed=args.seed)

    print(
        f"Flying: type={task.challenge_type} seed={task.map_seed} "
        f"start={task.start} goal={task.goal}"
    )
    print("Close the PyBullet window or press Ctrl+C when done.\n")

    result = rollout_episode(model, task, gui=True, deterministic=True)
    print(f"\nEpisode result: score={result['score']:.4f} success={result['success']}")


if __name__ == "__main__":
    main()
