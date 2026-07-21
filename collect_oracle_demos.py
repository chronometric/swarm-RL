#!/usr/bin/env python3
"""
Collect privileged (oracle) landing demos for BC.

Teacher knows GOAL_POS (training only). Cruise/search uses the public spiral
pilot; once within the search disk, soft-land aims at the true pad. The student
policy only sees depth+state, so BC transfers the landing skill to deployable
observations.
"""

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
from RL.env_utils import CURRICULUM, sample_landing_task
from RL.search_pilot import PilotPhase, SearchLandPilot, soft_land_action
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env


def collect_oracle_episode(task, *, speed: float = 0.42, enter_m: float = 10.0):
    env = SwarmActionWrapper(make_env(task, gui=False))
    pilot = SearchLandPilot(cruise_speed=speed, search_enter_m=enter_m)
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        pilot.reset()
        goal_xy = np.asarray(env.unwrapped.GOAL_POS[:2], dtype=np.float64)
        depth_steps, state_steps, actions = [], [], []
        done = False
        while not done:
            # Public spiral until LAND, then privileged soft-land to true pad.
            base = pilot.act(obs)
            if pilot.phase == PilotPhase.LAND:
                # Override lock with true goal (oracle).
                pilot._pad_xy = goal_xy.copy()
                act = soft_land_action(obs, target_xy=goal_xy)
            elif float(np.linalg.norm(np.asarray(obs["state"][:2], dtype=np.float64) - goal_xy)) < 6.0:
                # Early privileged assist when already near the true pad.
                act = soft_land_action(obs, target_xy=goal_xy)
                pilot.phase = PilotPhase.LAND
                pilot._pad_xy = goal_xy.copy()
            else:
                act = base

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
        }
        return np.stack(depth_steps), np.stack(state_steps), np.stack(actions), meta
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Collect oracle (privileged-goal) landing demos")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("RL/demos_oracle_land.npz"))
    parser.add_argument("--distance-min", type=float, default=3.0)
    parser.add_argument("--distance-max", type=float, default=12.0)
    parser.add_argument("--base-seed", type=int, default=1337)
    parser.add_argument("--success-only", action="store_true")
    args = parser.parse_args()

    stage = CURRICULUM[0]
    shard_dir = _shard_dir_for(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)

    metas = []
    kept = 0
    successes = 0
    for i in range(args.episodes):
        seed = args.base_seed + i * 7919
        task = sample_landing_task(
            stage=stage,
            seed=seed,
            distance_range=(args.distance_min, args.distance_max),
        )
        depth, state, actions, meta = collect_oracle_episode(task)
        keep = meta["success"] if args.success_only else (
            meta["success"] or (not meta["collision"] and meta["distance_to_goal"] < 2.0)
        )
        if keep:
            shard_path = shard_dir / f"ep_{kept:05d}.npz"
            np.savez_compressed(shard_path, depth=depth, state=state, actions=actions)
            metas.append(meta)
            kept += 1
            successes += int(meta["success"])
        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"  tried {i+1}/{args.episodes}  kept={kept}  successes={successes}  "
                f"last_dist={meta['distance_to_goal']:.1f}m success={meta['success']}"
            )

    if kept == 0:
        raise SystemExit("No oracle successes/near-misses — aborting")

    total = _merge_shards(shard_dir, args.out, metas, use_float16=True)
    summary = {
        "tried": args.episodes,
        "kept": kept,
        "successes": successes,
        "transitions": total,
        "success_rate_among_kept": successes / max(1, kept),
        "mean_final_dist": float(np.mean([m["distance_to_goal"] for m in metas])),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Oracle demos → {args.out}")
    print(f"   kept {kept}/{args.episodes}  successes={successes}  transitions={total:,}")


if __name__ == "__main__":
    main()
