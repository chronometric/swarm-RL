"""Small BC anchor updates to prevent PPO from drifting away from demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_demos(path: Path):
    """Load depth/state/actions from a single .npz or sharded demo directory."""
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


def run_bc_anchor(
    model,
    demos_path: Path,
    *,
    steps: int = 512,
    batch_size: int = 128,
    lr: float = 5e-6,
    device: str | None = None,
) -> float:
    """
    One mini BC pass on a random subset of demos. Returns mean MSE loss.
    Keeps the policy near demonstrated flight without full offline training.
    """
    from RL.bc_anchor import load_demos

    depth, state, actions = load_demos(demos_path)
    n = len(actions)
    if n == 0:
        return 0.0

    dev = device or str(model.device)
    policy = model.policy
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
    )

    depth_t = torch.as_tensor(depth, device=dev)
    state_t = torch.as_tensor(state, device=dev)
    actions_t = torch.as_tensor(actions, device=dev)

    hidden = policy.lstm_actor.hidden_size
    n_layers = policy.lstm_actor.num_layers
    policy.train()

    losses: list[float] = []
    count = min(steps, max(1, n // batch_size))
    for _ in range(count):
        idx = torch.randint(0, n, (min(batch_size, n),), device=dev)
        obs = {"depth": depth_t[idx], "state": state_t[idx]}
        episode_starts = torch.ones((len(idx),), dtype=torch.float32, device=dev)
        lstm_states = (
            torch.zeros(n_layers, len(idx), hidden, device=dev),
            torch.zeros(n_layers, len(idx), hidden, device=dev),
        )
        dist, _ = policy.get_distribution(obs, lstm_states=lstm_states, episode_starts=episode_starts)
        pred = dist.distribution.mean
        loss = F.mse_loss(pred, actions_t[idx])
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        losses.append(float(loss.item()))

    policy.eval()
    return float(np.mean(losses)) if losses else 0.0
