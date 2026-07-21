#!/usr/bin/env python3
"""Collect spiral-search pilot demos (many true landings) for BC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.action_utils import prepare_swarm_action
from RL.collect_demos import _merge_shards, _shard_dir_for
from RL.env_utils import CURRICULUM, sample_landing_task, sample_task
from RL.search_pilot import SearchLandPilot
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env


def collect_episode(task, *, cruise_speed: float = 0.55):
    env = SwarmActionWrapper(make_env(task, gui=False))
    pilot = SearchLandPilot(cruise_speed=cruise_speed)
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
            "collision": bool(info.get("collision", False)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
            "steps": len(actions),
            "phase": pilot.phase.name,
        }
        return np.stack(depth_steps), np.stack(state_steps), np.stack(actions), meta
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect spiral-search landing demos")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("RL/demos_spiral.npz"))
    parser.add_argument("--full-ratio", type=float, default=0.35, help="Fraction of full-distance missions")
    parser.add_argument("--distance-min", type=float, default=8.0)
    parser.add_argument("--distance-max", type=float, default=35.0)
    parser.add_argument("--success-only", action="store_true", help="Keep only successful landings")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=0.55)
    args = parser.parse_args()

    stage = CURRICULUM[0]
    shard_dir = _shard_dir_for(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    kept = 0
    successes = 0
    for i in range(args.episodes):
        seed = args.base_seed + i * 7919
        if np.random.random() < args.full_ratio:
            task = sample_task(stage=stage, seed=seed)
        else:
            task = sample_landing_task(
                stage=stage,
                seed=seed,
                distance_range=(args.distance_min, args.distance_max),
            )
        depth, state, actions, meta = collect_episode(task, cruise_speed=args.speed)
        keep = meta["success"] if args.success_only else True
        if keep:
            np.savez_compressed(
                shard_dir / f"ep_{kept:05d}.npz",
                depth=depth,
                state=state,
                actions=actions,
            )
            metas.append(meta)
            kept += 1
            successes += int(meta["success"])
        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"  tried {i+1}/{args.episodes} kept={kept} successes={successes} "
                f"last dist={meta['distance_to_goal']:.1f}m success={meta['success']} phase={meta['phase']}"
            )

    if kept == 0:
        raise SystemExit("No demos kept — spiral pilot found zero successes. Check pad detection.")

    total = _merge_shards(shard_dir, args.out, metas, use_float16=True)
    summary = {
        "tried": args.episodes,
        "kept": kept,
        "successes": successes,
        "success_rate": successes / max(1, args.episodes),
        "transitions": total,
        "mean_final_dist": float(np.mean([m["distance_to_goal"] for m in metas])),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Spiral demos → {args.out}")
    print(f"   success {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)  transitions={total:,}")


if __name__ == "__main__":
    main()
