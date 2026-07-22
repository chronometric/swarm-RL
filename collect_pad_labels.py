#!/usr/bin/env python3
"""
Collect privileged pad-localization labels for supervised training.

Teacher-only label: goal_xy_rel = GOAL_POS[:2] - drone_xy
Student inputs at deploy: depth + state (no GOAL_POS).

Runs the public spiral pilot so observations match SEARCH/REFINE distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.action_utils import prepare_swarm_action
from RL.collect_demos import _shard_dir_for
from RL.env_utils import CURRICULUM, sample_landing_task
from RL.search_pilot import SearchLandPilot
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env


def collect_pad_label_episode(
    task,
    *,
    speed: float = 0.42,
    enter_m: float = 10.0,
    max_goal_dist_m: float = 22.0,
):
    """Return depth, state, goal_xy_rel stacks + episode meta."""
    env = SwarmActionWrapper(make_env(task, gui=False))
    pilot = SearchLandPilot(cruise_speed=speed, search_enter_m=enter_m)
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        pilot.reset()
        goal_xy = np.asarray(env.unwrapped.GOAL_POS[:2], dtype=np.float64)

        depth_steps, state_steps, rel_steps = [], [], []
        done = False
        while not done:
            act = pilot.act(obs)
            pos_xy = np.asarray(obs["state"][:2], dtype=np.float64)
            goal_rel = (goal_xy - pos_xy).astype(np.float32)
            dist = float(np.linalg.norm(goal_rel))

            # Dense labels near the pad / inside the search disk (where estimator matters).
            if dist <= max_goal_dist_m:
                depth_steps.append(np.asarray(obs["depth"], dtype=np.float32))
                state_steps.append(np.asarray(obs["state"], dtype=np.float32))
                rel_steps.append(goal_rel)

            obs, _r, terminated, truncated, info = env.step(prepare_swarm_action(act, env))
            done = bool(terminated or truncated)

        meta = {
            "map_seed": int(task.map_seed),
            "challenge_type": int(task.challenge_type),
            "score": float(info.get("score", 0.0)),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
            "steps": len(rel_steps),
            "raw_steps": int(info.get("step", 0)) if "step" in info else -1,
        }
        if not rel_steps:
            return None
        return (
            np.stack(depth_steps),
            np.stack(state_steps),
            np.stack(rel_steps),
            meta,
        )
    finally:
        env.close()


def _merge_pad_shards(shard_dir: Path, out: Path, metas: list[dict]) -> int:
    shard_paths = sorted(shard_dir.glob("ep_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards in {shard_dir}")
    total = sum(int(m["steps"]) for m in metas)
    sample = np.load(shard_paths[0])
    depth_shape = sample["depth"].shape[1:]
    state_dim = int(sample["state"].shape[1])
    del sample

    with tempfile.TemporaryDirectory(prefix="pad_merge_") as tmp:
        tmp_path = Path(tmp)
        depth_mm = np.lib.format.open_memmap(
            tmp_path / "depth.dat", mode="w+", dtype=np.float16, shape=(total, *depth_shape)
        )
        state_mm = np.lib.format.open_memmap(
            tmp_path / "state.dat", mode="w+", dtype=np.float32, shape=(total, state_dim)
        )
        rel_mm = np.lib.format.open_memmap(
            tmp_path / "goal_xy_rel.dat", mode="w+", dtype=np.float32, shape=(total, 2)
        )
        offset = 0
        for shard_path in shard_paths:
            data = np.load(shard_path)
            n = int(data["depth"].shape[0])
            depth_mm[offset : offset + n] = data["depth"].astype(np.float16, copy=False)
            state_mm[offset : offset + n] = data["state"]
            rel_mm[offset : offset + n] = data["goal_xy_rel"]
            offset += n
            del data
        depth_mm.flush()
        state_mm.flush()
        rel_mm.flush()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            depth=np.asarray(depth_mm),
            state=np.asarray(state_mm),
            goal_xy_rel=np.asarray(rel_mm),
            meta=np.asarray(metas, dtype=object),
        )
    return total


def main():
    parser = argparse.ArgumentParser(description="Collect pad XY labels (privileged GOAL_POS)")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("RL/demos_pad_labels.npz"))
    parser.add_argument("--distance-min", type=float, default=2.0)
    parser.add_argument("--distance-max", type=float, default=25.0)
    parser.add_argument("--max-goal-dist", type=float, default=22.0)
    parser.add_argument("--base-seed", type=int, default=9001)
    args = parser.parse_args()

    stage = CURRICULUM[0]
    shard_dir = _shard_dir_for(args.out)
    if shard_dir.exists():
        for p in shard_dir.glob("ep_*.npz"):
            p.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []
    kept = 0
    successes = 0
    for i in range(args.episodes):
        seed = args.base_seed + i * 7919
        task = sample_landing_task(
            stage=stage,
            seed=seed,
            distance_range=(args.distance_min, args.distance_max),
        )
        result = collect_pad_label_episode(task, max_goal_dist_m=args.max_goal_dist)
        if result is None:
            print(f"  skip seed={seed}: no near-pad frames")
            continue
        depth, state, goal_xy_rel, meta = result
        shard_path = shard_dir / f"ep_{kept:05d}.npz"
        np.savez_compressed(
            shard_path,
            depth=depth,
            state=state,
            goal_xy_rel=goal_xy_rel,
        )
        metas.append(meta)
        kept += 1
        successes += int(meta["success"])
        if (i + 1) % 20 == 0 or i == 0:
            print(
                f"  ep {i+1}/{args.episodes}  shards={kept}  "
                f"frames={meta['steps']}  dist={meta['distance_to_goal']:.1f}m  "
                f"success={meta['success']}"
            )

    if kept == 0:
        raise SystemExit("No pad-label episodes collected")

    total = _merge_pad_shards(shard_dir, args.out, metas)
    errs = []
    for shard_path in sorted(shard_dir.glob("ep_*.npz")):
        data = np.load(shard_path)
        errs.append(np.linalg.norm(data["goal_xy_rel"], axis=1).mean())
    summary = {
        "episodes": len(metas),
        "transitions": total,
        "successes": successes,
        "mean_frame_goal_dist": float(np.mean(errs)) if errs else None,
        "mean_final_dist": float(np.mean([m["distance_to_goal"] for m in metas])),
        "success_rate": successes / max(1, len(metas)),
    }
    (args.out.parent / f"{args.out.stem}_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nPad labels → {args.out}")
    print(
        f"  episodes={len(metas)}  transitions={total:,}  "
        f"success_rate={summary['success_rate']:.1%}  "
        f"mean_final_dist={summary['mean_final_dist']:.2f}m"
    )


if __name__ == "__main__":
    main()
