#!/usr/bin/env python3
"""Collect successful hybrid landings for BC / DAgger (Stage 0 unlock)."""

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
from RL.hybrid_controller import HybridConfig, HybridController
from RL.validate import load_recurrent_model
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env


def collect_hybrid_episode(model, task, *, use_rl_land: bool = True):
    env = SwarmActionWrapper(make_env(task, gui=False))
    ctrl = HybridController(
        model,
        config=HybridConfig(use_rl_land=use_rl_land, deterministic=True),
    )
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        ctrl.reset()
        depth_steps, state_steps, actions = [], [], []
        done = False
        while not done:
            act = ctrl.act(obs)
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
    parser = argparse.ArgumentParser(description="Collect successful hybrid landing demos")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("RL/demos_success.npz"))
    parser.add_argument("--distance-min", type=float, default=3.0)
    parser.add_argument("--distance-max", type=float, default=10.0)
    parser.add_argument("--handoff-m", type=float, default=8.0)
    parser.add_argument("--soft-land-m", type=float, default=2.5)
    parser.add_argument("--base-seed", type=int, default=777)
    parser.add_argument("--keep-near-miss", action="store_true", help="Also keep dist<2m failures")
    parser.add_argument("--no-rl-land", action="store_true")
    args = parser.parse_args()

    model = load_recurrent_model(args.model)
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
        depth, state, actions, meta = collect_hybrid_episode(
            model,
            task,
            use_rl_land=not args.no_rl_land,
        )
        keep = meta["success"] or (
            args.keep_near_miss
            and (not meta["collision"])
            and meta["distance_to_goal"] < 2.0
        )
        if keep:
            shard_path = shard_dir / f"ep_{kept:05d}.npz"
            np.savez_compressed(shard_path, depth=depth, state=state, actions=actions)
            metas.append(meta)
            kept += 1
            successes += int(meta["success"])
        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  tried {i+1}/{args.episodes}  kept={kept}  "
                f"successes={successes}  last_dist={meta['distance_to_goal']:.1f}m "
                f"success={meta['success']}"
            )

    if kept == 0:
        raise SystemExit("No successful/near-miss episodes collected — aborting")

    total = _merge_shards(shard_dir, args.out, metas, use_float16=True)
    summary = {
        "tried": args.episodes,
        "kept": kept,
        "successes": successes,
        "transitions": total,
        "success_rate_among_kept": successes / max(1, kept),
        "mean_final_dist": float(np.mean([m["distance_to_goal"] for m in metas])),
        "model": str(args.model),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n✅ Success demos → {args.out}")
    print(f"   kept {kept}/{args.episodes}  successes={successes}  transitions={total:,}")


if __name__ == "__main__":
    main()
