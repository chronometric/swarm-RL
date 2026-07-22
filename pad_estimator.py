"""Small CNN+MLP: depth + state → pad XY offset in world meters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PadEstimator(nn.Module):
    """
    Predicts goal_xy_rel = (goal_x - pos_x, goal_y - pos_y).

    At deploy: pad_xy = drone_xy + predict(depth, state)
    """

    def __init__(self, state_dim: int = 40, *, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        cnn_out = 64 * 4 * 4
        self.head = nn.Sequential(
            nn.Linear(cnn_out + state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.state_dim = state_dim

    def forward(self, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        elif depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
        feat = self.cnn(depth)
        x = torch.cat([feat, state], dim=-1)
        return self.head(x)


def load_pad_estimator(path: Path | str, *, device: str | torch.device = "cpu") -> PadEstimator:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PadEstimator(state_dim=int(ckpt["state_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_pad_xy(
    model: PadEstimator,
    observation: dict,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return absolute pad XY [2] from an observation dict."""
    depth = np.asarray(observation["depth"], dtype=np.float32)
    state = np.asarray(observation["state"], dtype=np.float32)
    if depth.ndim == 2:
        depth_t = torch.from_numpy(depth)[None, None, ...].to(device)
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth_t = torch.from_numpy(depth).permute(2, 0, 1)[None, ...].to(device)
    else:
        depth_t = torch.from_numpy(depth)[None, ...].to(device)
    state_t = torch.from_numpy(state)[None, ...].to(device)
    rel = model(depth_t, state_t).cpu().numpy().reshape(2)
    pos_xy = state[:2]
    return (pos_xy + rel).astype(np.float64)
