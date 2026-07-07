#!/usr/bin/env python3
"""Headless flight video for trained SB3 models (no Docker, no X11)."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.action_utils import prepare_swarm_action
from RL.validate import load_recurrent_model
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import task_for_seed_and_type

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


def _chase_frame(cli, drone_pos, drone_quat, width, height):
    import pybullet as p

    rot = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
    forward = rot[:, 0]
    up = rot[:, 2]
    cam_pos = drone_pos - 4.0 * forward + 1.5 * up
    target = drone_pos + 2.0 * forward

    view = p.computeViewMatrix(cam_pos.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(60, width / height, 0.05, 200.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_TINY_RENDERER,
        shadow=0,
        physicsClientId=cli,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]


def render_flight(model_path: Path, seed: int, challenge_type: int, out_path: Path, fps: int = 25):
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=challenge_type)
    model = load_recurrent_model(model_path)
    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)
    cli = env.CLIENT

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    act_lo = env.action_space.low.flatten()
    act_hi = env.action_space.high.flatten()

    width, height = 960, 540
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    frame_dt = 1.0 / fps
    next_frame_t = 0.0
    t_sim = 0.0
    success = False
    score = 0.0

    try:
        while t_sim < task.horizon:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
            act = prepare_swarm_action(action, env)
            obs, _r, terminated, truncated, info = env.step(act)
            t_sim += 1.0 / env.CTRL_FREQ
            success = bool(info.get("success", False))
            score = float(info.get("score", 0.0))

            while t_sim >= next_frame_t:
                pos = np.asarray(env._getDroneStateVector(0)[:3])
                quat = np.asarray(env.quat[0])
                writer.append_data(_chase_frame(cli, pos, quat, width, height))
                next_frame_t += frame_dt

            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()

    return {"success": success, "score": score, "sim_time": t_sim, "path": str(out_path)}


def main():
    parser = argparse.ArgumentParser(description="Render headless chase video of trained agent")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2002002)
    parser.add_argument("--type", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("videos/flight_chase.mp4"))
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = render_flight(args.model, args.seed, args.type, args.out, fps=args.fps)
    print(f"Video: {result['path']}")
    print(f"Score: {result['score']:.4f}  success: {result['success']}  sim: {result['sim_time']:.1f}s")
    print(f"Wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
