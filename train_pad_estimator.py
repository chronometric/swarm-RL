#!/usr/bin/env python3
"""Train a small depth+state → pad XY offset estimator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.pad_estimator import PadEstimator


class PadLabelDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=True)
        self.depth = data["depth"]
        self.state = data["state"].astype(np.float32)
        self.goal = data["goal_xy_rel"].astype(np.float32)
        assert len(self.depth) == len(self.state) == len(self.goal)

    def __len__(self) -> int:
        return len(self.goal)

    def __getitem__(self, idx: int):
        d = self.depth[idx]
        if d.dtype != np.float32:
            d = d.astype(np.float32)
        if d.ndim == 2:
            d = d[None, ...]
        elif d.ndim == 3 and d.shape[-1] == 1:
            d = np.transpose(d, (2, 0, 1))
        return (
            torch.from_numpy(np.ascontiguousarray(d)),
            torch.from_numpy(self.state[idx]),
            torch.from_numpy(self.goal[idx]),
        )


@torch.no_grad()
def eval_split(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    errs = []
    close_errs = []
    for depth, state, goal in loader:
        depth = depth.to(device)
        state = state.to(device)
        goal = goal.to(device)
        pred = model(depth, state)
        e = torch.linalg.norm(pred - goal, dim=-1)
        errs.append(e.cpu())
        alt = state[:, -4] * 20.0 if state.shape[1] >= 4 else torch.full((state.shape[0],), 99.0)
        mask = (torch.linalg.norm(goal, dim=-1) < 4.0) & (alt < 5.0)
        if mask.any():
            close_errs.append(e[mask].cpu())
    all_e = torch.cat(errs)
    out = {
        "mae_m": float(all_e.mean()),
        "p50_m": float(all_e.median()),
        "p90_m": float(torch.quantile(all_e, 0.9)),
        "frac_lt_0_5m": float((all_e < 0.5).float().mean()),
    }
    if close_errs:
        ce = torch.cat(close_errs)
        out["close_mae_m"] = float(ce.mean())
        out["close_frac_lt_0_5m"] = float((ce < 0.5).float().mean())
    return out


def main():
    parser = argparse.ArgumentParser(description="Train pad XY estimator")
    parser.add_argument("--demos", type=Path, default=Path("RL/demos_pad_labels.npz"))
    parser.add_argument("--out", type=Path, default=Path("RL/checkpoints/pad_estimator.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    ds = PadLabelDataset(args.demos)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    sample_state = ds.state[0]
    model = PadEstimator(state_dim=int(sample_state.shape[0])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_close = 1e9
    history = []

    print(f"Train {n_train:,} / val {n_val:,} on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for depth, state, goal in train_loader:
            depth = depth.to(device)
            state = state.to(device)
            goal = goal.to(device)
            pred = model(depth, state)
            loss = F.huber_loss(pred, goal, delta=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        metrics = eval_split(model, val_loader, device)
        metrics["train_huber"] = float(np.mean(losses))
        history.append({"epoch": epoch, **metrics})
        close = metrics.get("close_mae_m", metrics["mae_m"])
        print(
            f"  epoch {epoch}/{args.epochs}  huber={metrics['train_huber']:.4f}  "
            f"val_mae={metrics['mae_m']:.3f}m  p50={metrics['p50_m']:.3f}m  "
            f"frac<0.5={metrics['frac_lt_0_5m']:.2%}  "
            f"close_mae={metrics.get('close_mae_m', float('nan')):.3f}m"
        )
        if close < best_close:
            best_close = close
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "state_dim": int(sample_state.shape[0]),
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out,
            )
            print(f"    saved best → {args.out} (close_mae={close:.3f}m)")

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps({"best_close_mae_m": best_close, "history": history}, indent=2))
    print(f"Done. Best close-range MAE={best_close:.3f}m  meta→{meta_path}")


if __name__ == "__main__":
    main()
