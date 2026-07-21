#!/usr/bin/env python3
"""Behavioral cloning warm-start for RecurrentPPO before RL fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.env_utils import CURRICULUM, make_landing_env_fn, make_training_env_fn
from RL.train_sota import _build_model, _wrap_env
from RL.wrappers import LandingFocusedWrapper, SwarmActionWrapper, StateNoiseWrapper, EpisodeScoreWrapper


def _wrap_landing_env(raw_env):
    env = SwarmActionWrapper(raw_env)
    env = LandingFocusedWrapper(env)
    env = StateNoiseWrapper(env, std=0.0)
    env = EpisodeScoreWrapper(env)
    return env


def _load_demos(path: Path):
    if path.is_dir():
        shard_paths = sorted(path.glob("ep_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"No ep_*.npz shards in {path}")
        depth_parts, state_parts, action_parts = [], [], []
        for shard_path in shard_paths:
            data = np.load(shard_path)
            depth_parts.append(data["depth"])
            state_parts.append(data["state"])
            action_parts.append(data["actions"])
        return (
            np.concatenate(depth_parts, axis=0),
            np.concatenate(state_parts, axis=0),
            np.concatenate(action_parts, axis=0),
        )
    data = np.load(path, allow_pickle=True)
    return data["depth"], data["state"], data["actions"]


def pretrain_bc(
    demos_path: Path,
    output_path: Path,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    landing: bool = False,
) -> Path:
    depth, state, actions = _load_demos(demos_path)
    n = len(actions)
    print(f"BC dataset: {n} transitions from {demos_path}")

    stage = CURRICULUM[0]

    def factory():
        if landing:
            raw = make_landing_env_fn(stage)()
            return _wrap_landing_env(raw)
        raw = SwarmActionWrapper(make_training_env_fn(stage)())
        return _wrap_env(raw, state_noise=0.0)

    vec_env = DummyVecEnv([factory])
    model = _build_model(
        vec_env,
        learning_rate=lr,
        n_steps=512,
        batch_size=batch_size,
        device=device,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=4,
        target_kl=None,
        tensorboard_log=None,
    )
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    depth_t = torch.as_tensor(depth, device=device)
    state_t = torch.as_tensor(state, device=device)
    actions_t = torch.as_tensor(actions, device=device)

    policy.train()
    hidden = policy.lstm_actor.hidden_size
    n_layers = policy.lstm_actor.num_layers
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            obs = {"depth": depth_t[idx], "state": state_t[idx]}
            batch_len = len(idx)
            episode_starts = torch.ones((batch_len,), dtype=torch.float32, device=device)
            lstm_states = (
                torch.zeros(n_layers, batch_len, hidden, device=device),
                torch.zeros(n_layers, batch_len, hidden, device=device),
            )
            dist, _ = policy.get_distribution(obs, lstm_states=lstm_states, episode_starts=episode_starts)
            pred = dist.distribution.mean
            loss = F.mse_loss(pred, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"  epoch {epoch}/{epochs}  mse={np.mean(losses):.5f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    vec_env.close()
    print(f"BC model saved → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="BC warm-start from heuristic demos")
    parser.add_argument("--demos", type=Path, default=Path("RL/demos_open.npz"))
    parser.add_argument("--out", type=Path, default=Path("RL/checkpoints/bc_warmstart.zip"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--landing",
        action="store_true",
        help="Use landing env factory (auto-enabled when demos path contains 'landing')",
    )
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    landing = args.landing or "landing" in args.demos.stem.lower()
    pretrain_bc(
        args.demos,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        landing=landing,
    )


if __name__ == "__main__":
    main()
