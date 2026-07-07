#!/usr/bin/env python3
"""Collect heuristic pilot demonstrations for behavioral cloning."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.env_utils import CURRICULUM, sample_task
from RL.heuristic_pilot import heuristic_action
from RL.wrappers import SwarmActionWrapper
from swarm.utils.env_factory import make_env


def collect_episode(task, *, speed: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
  """Run one episode; return stacked arrays (no per-step Python lists retained)."""
  env = SwarmActionWrapper(make_env(task, gui=False))
  try:
    obs, info = env.reset(seed=int(task.map_seed))
    depth_steps: list[np.ndarray] = []
    state_steps: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    done = False

    while not done:
      act = heuristic_action(obs, speed=speed)
      depth_steps.append(np.asarray(obs["depth"], dtype=np.float32))
      state_steps.append(np.asarray(obs["state"], dtype=np.float32))
      actions.append(act.reshape(-1).astype(np.float32))
      obs, _r, terminated, truncated, info = env.step(act)
      done = bool(terminated or truncated)

    meta = {
      "map_seed": int(task.map_seed),
      "challenge_type": int(task.challenge_type),
      "score": float(info.get("score", 0.0)),
      "success": bool(info.get("success", False)),
      "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
      "steps": len(actions),
    }
    return (
      np.stack(depth_steps),
      np.stack(state_steps),
      np.stack(actions),
      meta,
    )
  finally:
    env.close()


def _shard_dir_for(out: Path) -> Path:
  return out.parent / f"{out.stem}.shards"


def _merge_shards(shard_dir: Path, out: Path, metas: list[dict], *, use_float16: bool) -> int:
  """
  Merge per-episode shard .npz files into one dataset without holding all
  transitions in Python lists (avoids multi‑GB RAM spike at the end).
  """
  shard_paths = sorted(shard_dir.glob("ep_*.npz"))
  if not shard_paths:
    raise FileNotFoundError(f"No shards found in {shard_dir}")

  total = sum(int(m["steps"]) for m in metas)
  sample = np.load(shard_paths[0])
  depth_shape = sample["depth"].shape[1:]
  state_dim = int(sample["state"].shape[1])
  action_dim = int(sample["actions"].shape[1])
  del sample

  depth_dtype = np.float16 if use_float16 else np.float32
  with tempfile.TemporaryDirectory(prefix="demos_merge_") as tmp:
    tmp_path = Path(tmp)
    depth_mm = np.lib.format.open_memmap(
      tmp_path / "depth.dat",
      mode="w+",
      dtype=depth_dtype,
      shape=(total, *depth_shape),
    )
    state_mm = np.lib.format.open_memmap(
      tmp_path / "state.dat",
      mode="w+",
      dtype=np.float32,
      shape=(total, state_dim),
    )
    actions_mm = np.lib.format.open_memmap(
      tmp_path / "actions.dat",
      mode="w+",
      dtype=np.float32,
      shape=(total, action_dim),
    )

    offset = 0
    for shard_path in shard_paths:
      data = np.load(shard_path)
      n = int(data["depth"].shape[0])
      depth_mm[offset : offset + n] = data["depth"].astype(depth_dtype, copy=False)
      state_mm[offset : offset + n] = data["state"]
      actions_mm[offset : offset + n] = data["actions"]
      offset += n
      del data

    depth_mm.flush()
    state_mm.flush()
    actions_mm.flush()

    out.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed np.savez reads memmaps page-by-page; savez_compressed would
    # load entire arrays into RAM and can OOM on large datasets.
    np.savez(
      out,
      depth=depth_mm,
      state=state_mm,
      actions=actions_mm,
      meta=np.array(metas, dtype=object),
    )

  return total


def main():
  parser = argparse.ArgumentParser(description="Collect heuristic demos for BC warm-start")
  parser.add_argument("--episodes", type=int, default=64)
  parser.add_argument("--stage", type=int, default=0)
  parser.add_argument("--speed", type=float, default=0.6)
  parser.add_argument("--out", type=Path, default=Path("RL/demos_open.npz"))
  parser.add_argument(
    "--shard-only",
    action="store_true",
    help="Keep per-episode shards only; skip final merge (lowest RAM at end)",
  )
  parser.add_argument(
    "--fp32",
    action="store_true",
    help="Store depth as float32 (default: float16 in merged file to save RAM/disk)",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Continue from existing shards in <out>.shards/ (skip finished episodes)",
  )
  args = parser.parse_args()

  stage = CURRICULUM[max(0, min(args.stage, len(CURRICULUM) - 1))]
  shard_dir = _shard_dir_for(args.out)
  shard_dir.mkdir(parents=True, exist_ok=True)

  metas: list[dict] = []
  meta_path = shard_dir / "meta.json"
  if args.resume and meta_path.exists():
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Resuming: {len(metas)} episodes already in {shard_dir}")

  start_ep = len(metas)
  if start_ep >= args.episodes:
    print(f"All {args.episodes} episodes already collected in {shard_dir}")
  else:
    print(
      f"Collecting {args.episodes} demos | stage={stage.name} | speed={args.speed} | "
      f"shards→{shard_dir}"
    )

  for ep in range(start_ep, args.episodes):
    task = sample_task(stage=stage, seed=10_000 + ep)
    depth, state, actions, meta = collect_episode(task, speed=args.speed)
    shard_path = shard_dir / f"ep_{ep:04d}.npz"
    np.savez_compressed(shard_path, depth=depth, state=state, actions=actions)
    metas.append(meta)
    meta_path.write_text(json.dumps(metas, indent=2), encoding="utf-8")
    print(
      f"  ep {ep+1:3d}/{args.episodes} seed={meta['map_seed']} "
      f"type={meta['challenge_type']} steps={meta['steps']} "
      f"dist={meta['distance_to_goal']:.1f} success={meta['success']}"
    )
    del depth, state, actions

  success_rate = float(np.mean([m["success"] for m in metas]))
  mean_dist = float(np.mean([m["distance_to_goal"] for m in metas]))
  total_steps = sum(int(m["steps"]) for m in metas)

  if args.shard_only:
    est_gb = total_steps * 128 * 128 * (4 if args.fp32 else 2) / 1e9
    print(f"\nShard-only mode: {len(metas)} episodes, {total_steps:,} transitions")
    print(f"  dir: {shard_dir}")
    print(f"  success={success_rate:.1%} | mean dist={mean_dist:.1f}m")
    print(f"  (~{est_gb:.1f} GB depth if merged; pretrain_bc accepts shard dirs)")
    return

  print(f"\nMerging {len(metas)} shards ({total_steps:,} transitions) → {args.out}")
  print("  (uses memmap; peak RAM stays low — do not kill during merge)")
  n = _merge_shards(shard_dir, args.out, metas, use_float16=not args.fp32)
  size_mb = args.out.stat().st_size / (1024 * 1024)
  print(f"Saved {n:,} transitions → {args.out} ({size_mb:.0f} MiB)")
  print(f"Episode success rate: {success_rate:.1%} | mean final dist: {mean_dist:.1f}m")

  if args.out.exists():
    shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"Removed temp shards {shard_dir}")


if __name__ == "__main__":
  main()
