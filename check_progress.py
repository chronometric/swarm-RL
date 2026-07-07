#!/usr/bin/env python3
"""Quick validator-faithful progress check for a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.validate import evaluate_model, load_recurrent_model, rollout_episode
from swarm.constants import SIM_DT
from swarm.validator.task_gen import task_for_seed_and_type


def main():
    parser = argparse.ArgumentParser(description="Check distance/score for one or more open-terrain seeds")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=[2002002, 10012, 10033, 10052])
    parser.add_argument("--type", type=int, default=2, help="Challenge type (2=open)")
    parser.add_argument("--all-types", action="store_true", help="Run fixed validation seeds for all env types")
    args = parser.parse_args()

    model = load_recurrent_model(args.model)
    if args.all_types:
        result = evaluate_model(model, challenge_types=None)
        print(result.summary_line())
        for ep in result.episodes:
            print(
                f"  type={ep['challenge_type']} seed={ep['map_seed']} "
                f"dist={ep['distance_to_goal']:.1f}m success={ep['success']} score={ep['score']:.4f}"
            )
        return

    print(f"Model: {args.model}")
    for seed in args.seed:
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=args.type)
        ep = rollout_episode(model, task)
        print(
            f"  seed={seed} dist={ep['distance_to_goal']:.1f}m "
            f"success={ep['success']} score={ep['score']:.4f}"
        )


if __name__ == "__main__":
    main()
