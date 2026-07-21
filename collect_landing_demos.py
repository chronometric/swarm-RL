#!/usr/bin/env python3
"""Collect heuristic demos on near-goal (landing) episodes for BC / DAgger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.collect_demos import _merge_shards, _shard_dir_for
from RL.env_utils import CURRICULUM, sample_landing_task
from RL.search_pilot import SearchLandPilot
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env
from RL.action_utils import prepare_swarm_action


def collect_landing_episode(task, *, speed: float = 0.5):
    env = SwarmActionWrapper(make_env(task, gui=False))
    pilot = SearchLandPilot(cruise_speed=speed)
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        pilot.reset()
        depth_steps, state_steps, actions = [], [], []
        done = False
        while not done:
            act = pilot.act(obs)
            depth_steps.append(np.asarray(obs["depth"], dtype=np.float32))
            state_steps.append(np.asarray(obs["state"], dtype=np.float32))
            actions.append(np.asarray(act, dtype=np.float32).reshape(-1))
            obs, _r, terminated, truncated, info = env.step(prepare_swarm_action(act, env))
            done = bool(terminated or truncated)
        meta = {
            "map_seed": int(task.map_seed),
            "challenge_type": int(task.challenge_type),
            "score": float(info.get("score", 0.0)),
            "success": bool(info.get("success", False)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
            "steps": len(actions),
        }
        return np.stack(depth_steps), np.stack(state_steps), np.stack(actions), meta
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect landing-phase heuristic demos")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("RL/demos_landing.npz"))
    parser.add_argument("--distance-min", type=float, default=4.0)
    parser.add_argument("--distance-max", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    stage = CURRICULUM[0]
    distance_range = (args.distance_min, args.distance_max)
    shard_dir = _shard_dir_for(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    successes = 0
    for i in range(args.episodes):
        seed = args.base_seed + i * 7919
        task = sample_landing_task(stage=stage, seed=seed, distance_range=distance_range)
        depth, state, actions, meta = collect_landing_episode(task, speed=args.speed)
        shard_path = shard_dir / f"ep_{i:05d}.npz"
        np.savez_compressed(shard_path, depth=depth, state=state, actions=actions)
        metas.append(meta)
        successes += int(meta["success"])
        if (i + 1) % 32 == 0 or i == 0:
            print(
                f"  ep {i+1}/{args.episodes}  steps={meta['steps']}  "
                f"dist={meta['distance_to_goal']:.1f}m  success={meta['success']}"
            )

    total = _merge_shards(shard_dir, args.out, metas, use_float16=True)
    summary = {
        "episodes": args.episodes,
        "transitions": total,
        "success_rate": successes / max(1, args.episodes),
        "distance_range": distance_range,
        "mean_final_dist": float(np.mean([m["distance_to_goal"] for m in metas])),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Landing demos → {args.out}  ({total:,} transitions)")
    print(f"   success rate: {summary['success_rate']:.1%}  mean final dist: {summary['mean_final_dist']:.1f}m")


if __name__ == "__main__":
    main()
