#!/usr/bin/env python3
"""Quick validator-faithful progress check for a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.validate import (
    evaluate_hybrid_model,
    evaluate_model,
    load_recurrent_model,
    rollout_episode,
    rollout_episode_hybrid,
)
from swarm.constants import SIM_DT
from swarm.validator.task_gen import task_for_seed_and_type


def main():
    parser = argparse.ArgumentParser(description="Check distance/score for one or more open-terrain seeds")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", default=[2002002, 10012, 10033, 10052])
    parser.add_argument("--type", type=int, default=2, help="Challenge type (2=open)")
    parser.add_argument("--all-types", action="store_true", help="Run fixed validation seeds for all env types")
    parser.add_argument("--hybrid", action="store_true", help="Heuristic cruise + RL landing (deployment mode)")
    parser.add_argument("--handoff-m", type=float, default=8.0)
    parser.add_argument(
        "--pad-estimator",
        type=Path,
        default=Path("RL/checkpoints/pad_estimator.pt"),
        help="Pad XY estimator checkpoint (used when present)",
    )
    parser.add_argument("--no-pad-estimator", action="store_true", help="Disable pad estimator even if file exists")
    args = parser.parse_args()

    pad_path = None if args.no_pad_estimator else args.pad_estimator
    if pad_path is not None and not pad_path.exists():
        print(f"[warn] pad estimator not found at {pad_path} — running without it")
        pad_path = None

    model = load_recurrent_model(args.model)
    if args.all_types:
        eval_fn = evaluate_hybrid_model if args.hybrid else evaluate_model
        if args.hybrid:
            result = eval_fn(model, challenge_types=None, handoff_m=args.handoff_m, pad_estimator_path=pad_path)
        else:
            result = eval_fn(model)
        print(result.summary_line())
        for ep in result.episodes:
            print(
                f"  type={ep['challenge_type']} seed={ep['map_seed']} "
                f"dist={ep['distance_to_goal']:.1f}m success={ep['success']} score={ep['score']:.4f}"
            )
        return

    print(f"Model: {args.model}" + (" [HYBRID]" if args.hybrid else "") + (f" [PAD={pad_path}]" if pad_path else ""))
    for seed in args.seed:
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=args.type)
        if args.hybrid:
            ep = rollout_episode_hybrid(model, task, handoff_m=args.handoff_m, pad_estimator_path=pad_path)
        else:
            ep = rollout_episode(model, task)
        print(
            f"  seed={seed} dist={ep['distance_to_goal']:.1f}m "
            f"success={ep['success']} score={ep['score']:.4f}"
        )


if __name__ == "__main__":
    main()
