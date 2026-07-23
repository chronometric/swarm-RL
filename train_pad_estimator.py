#!/usr/bin/env python3
"""Train depth+state → pad XY estimator (RAM-safe, shard-sequential).

WHY THE PREVIOUS TRAINER OOM'D
------------------------------
Each ep_*.npz stores ~3000×128×128 float32 depth (~197 MB uncompressed).
Files are DEFLATE-compressed, so np.load(..., mmap_mode='r') still fully
decompresses depth into RAM. WeightedRandomSampler jumped randomly across
400 shards → nearly every sample reloaded ~197 MB → OOM / instance kill
on 15 GB RAM with no swap.

FIX
---
Train one shard at a time (peak ~0.3–0.5 GB data). Weight close-pad frames
in the Huber loss instead of a global WeightedRandomSampler.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.pad_estimator import PadEstimator


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _resolve_shard_dir(path: Path) -> Path | None:
    if path.is_dir() and any(path.glob("ep_*.npz")):
        return path
    candidate = path.parent / f"{path.stem}.shards"
    if candidate.is_dir() and any(candidate.glob("ep_*.npz")):
        return candidate
    return None


def _list_shards(shard_dir: Path) -> list[Path]:
    paths = sorted(shard_dir.glob("ep_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No ep_*.npz in {shard_dir}")
    return paths


def _load_shard_meta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only state + goal (lazy npz — does not decompress depth)."""
    with np.load(path) as data:
        state = np.asarray(data["state"], dtype=np.float32)
        goal = np.asarray(data["goal_xy_rel"], dtype=np.float32)
    return state, goal


def _load_shard_depth(path: Path) -> np.ndarray:
    """Decompress one shard's depth (~197 MB). Caller must delete promptly."""
    with np.load(path) as data:
        depth = np.asarray(data["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, None, ...]
    elif depth.ndim == 4 and depth.shape[-1] == 1:
        depth = np.transpose(depth, (0, 3, 1, 2))
    return np.ascontiguousarray(depth)


def _frame_weights(goal: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Strongly up-weight near-pad / low-altitude frames (lock quality)."""
    dist = np.linalg.norm(goal, axis=1)
    alt = state[:, -4] * 20.0 if state.shape[1] >= 4 else np.full(len(state), 5.0)
    w = np.ones(len(goal), dtype=np.float32)
    w[dist < 8.0] *= 2.0
    w[dist < 4.0] *= 4.0
    w[dist < 2.0] *= 6.0
    w[dist < 1.0] *= 4.0
    w[alt < 5.0] *= 3.0
    w[alt < 3.0] *= 2.0
    return w


def _filter_indices(
    goal: np.ndarray,
    state: np.ndarray | None = None,
    *,
    max_dist_m: float | None,
    stride: int,
) -> np.ndarray:
    n = len(goal)
    keep = np.arange(n)
    if max_dist_m is not None:
        dist = np.linalg.norm(goal, axis=1)
        keep = keep[dist <= max_dist_m]
    if state is not None and len(keep):
        finite_state = np.isfinite(state[keep]).all(axis=1)
        finite_goal = np.isfinite(goal[keep]).all(axis=1)
        keep = keep[finite_state & finite_goal]
    if stride > 1:
        keep = keep[::stride]
    return keep


def _drop_bad_depth(keep: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Remove frames with any non-finite depth pixel (sim glitches)."""
    if len(keep) == 0:
        return keep
    flat = depth[keep].reshape(len(keep), -1)
    return keep[np.isfinite(flat).all(axis=1)]


def _sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """Belt-and-suspenders: replace any remaining non-finite values."""
    return np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


@torch.no_grad()
def eval_shards(
    model: nn.Module,
    shard_paths: list[Path],
    device: str,
    *,
    max_dist_m: float | None,
    batch_size: int,
    max_frames: int | None = None,
) -> dict:
    """Evaluate held-out shards one-at-a-time (RAM-safe)."""
    model.eval()
    errs: list[torch.Tensor] = []
    close_errs: list[torch.Tensor] = []
    seen = 0

    for path in shard_paths:
        state, goal = _load_shard_meta(path)
        keep = _filter_indices(goal, state, max_dist_m=max_dist_m, stride=1)
        if len(keep) == 0:
            continue
        if max_frames is not None and seen >= max_frames:
            break
        if max_frames is not None and seen + len(keep) > max_frames:
            keep = keep[: max(1, max_frames - seen)]

        depth = _load_shard_depth(path)
        try:
            keep = _drop_bad_depth(keep, depth)
            if len(keep) == 0:
                continue
            depth = _sanitize_depth(depth)
            for start in range(0, len(keep), batch_size):
                idx = keep[start : start + batch_size]
                d = torch.from_numpy(np.ascontiguousarray(depth[idx])).to(device)
                s = torch.from_numpy(np.ascontiguousarray(state[idx])).to(device)
                g = torch.from_numpy(np.ascontiguousarray(goal[idx])).to(device)
                pred = model(d, s)
                e = torch.linalg.norm(pred - g, dim=-1)
                if not torch.isfinite(e).all():
                    continue
                errs.append(e.cpu())
                alt = (
                    s[:, -4] * 20.0
                    if s.shape[1] >= 4
                    else torch.full((s.shape[0],), 99.0, device=device)
                )
                mask = (torch.linalg.norm(g, dim=-1) < 4.0) & (alt < 5.0)
                if mask.any():
                    close_errs.append(e[mask].cpu())
                seen += len(idx)
        finally:
            del depth
            gc.collect()

    if not errs:
        return {"mae_m": float("nan"), "p50_m": float("nan"), "p90_m": float("nan"),
                "frac_lt_0_5m": 0.0}

    all_e = torch.cat(errs)
    out = {
        "mae_m": float(all_e.mean()),
        "p50_m": float(all_e.median()),
        "p90_m": float(torch.quantile(all_e, 0.9)),
        "frac_lt_0_5m": float((all_e < 0.5).float().mean()),
        "n_eval": int(all_e.numel()),
    }
    if close_errs:
        ce = torch.cat(close_errs)
        out["close_mae_m"] = float(ce.mean())
        out["close_frac_lt_0_5m"] = float((ce < 0.5).float().mean())
        out["n_close"] = int(ce.numel())
    return out


def train_one_shard(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    path: Path,
    device: str,
    *,
    batch_size: int,
    max_dist_m: float | None,
    stride: int,
    rng: np.random.Generator,
) -> list[float]:
    """Train on a single shard; return per-batch losses."""
    state, goal = _load_shard_meta(path)
    keep = _filter_indices(goal, state, max_dist_m=max_dist_m, stride=stride)
    if len(keep) < 8:
        return []

    depth = _load_shard_depth(path)
    losses: list[float] = []
    try:
        keep = _drop_bad_depth(keep, depth)
        if len(keep) < 8:
            return []
        depth = _sanitize_depth(depth)

        weights = _frame_weights(goal[keep], state[keep])
        order = rng.permutation(len(keep))
        model.train()
        for start in range(0, len(order), batch_size):
            sel = order[start : start + batch_size]
            if len(sel) < 2:
                continue
            idx = keep[sel]
            d = torch.from_numpy(np.ascontiguousarray(depth[idx])).to(device)
            s = torch.from_numpy(np.ascontiguousarray(state[idx])).to(device)
            g = torch.from_numpy(np.ascontiguousarray(goal[idx])).to(device)
            w = torch.from_numpy(weights[sel]).to(device)
            if not (torch.isfinite(d).all() and torch.isfinite(s).all() and torch.isfinite(g).all()):
                continue

            pred = model(d, s)
            per = F.huber_loss(pred, g, delta=1.0, reduction="none").mean(dim=-1)
            loss = (per * w).sum() / w.sum().clamp_min(1e-6)
            if not torch.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bad_grad = False
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break
            if bad_grad:
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            losses.append(float(loss.item()))
    finally:
        del depth
        gc.collect()
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train pad XY estimator (RAM-safe)")
    parser.add_argument(
        "--demos",
        type=Path,
        default=Path("RL/demos_pad_labels.shards"),
        help="Shard dir (required). Merged .npz will OOM on 16GB RAM.",
    )
    parser.add_argument("--out", type=Path, default=Path("RL/checkpoints/pad_estimator.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-dist", type=float, default=6.0, help="Drop frames farther than this (close-range focus)")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth in-range frame (1 = all close frames)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-val-frames",
        type=int,
        default=40_000,
        help="Cap validation frames per epoch (keeps eval fast + light)",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="If >0, use only the first N shards (smoke test)",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    shard_dir = _resolve_shard_dir(args.demos)
    if shard_dir is None:
        raise SystemExit(
            f"No shards found for {args.demos}. "
            "Pass RL/demos_pad_labels.shards (merged .npz is too large for 16GB RAM)."
        )

    all_shards = _list_shards(shard_dir)
    if args.limit_shards and args.limit_shards > 0:
        all_shards = all_shards[: args.limit_shards]
        print(f"Smoke/limit mode: using {len(all_shards)} shards")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Hold out whole shards for validation (no random frame mix across RAM).
    n_val = max(1, int(round(len(all_shards) * args.val_frac)))
    n_val = min(n_val, max(1, len(all_shards) - 1))
    perm = rng.permutation(len(all_shards))
    val_shards = [all_shards[i] for i in perm[:n_val]]
    train_shards = [all_shards[i] for i in perm[n_val:]]

    # Infer state_dim from first non-empty shard meta.
    state_dim = None
    n_train_est = 0
    for p in train_shards:
        st, gl = _load_shard_meta(p)
        state_dim = int(st.shape[1])
        n_train_est += len(_filter_indices(gl, st, max_dist_m=args.max_dist, stride=args.stride))
    assert state_dim is not None

    model = PadEstimator(state_dim=state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_close = 1e9
    history: list[dict] = []

    print(
        f"Shards: {len(train_shards)} train / {len(val_shards)} val | "
        f"~{n_train_est:,} train frames (max_dist={args.max_dist}, stride={args.stride}) | "
        f"batch={args.batch_size} device={device} rss={_rss_mb():.0f}MB"
    )
    print(
        "Mode: shard-sequential (loads ≤1 depth shard ≈200MB at a time). "
        "Do NOT use WeightedRandomSampler across shards on this box."
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        order = train_shards[:]
        random.shuffle(order)
        losses: list[float] = []
        for si, path in enumerate(order):
            losses.extend(
                train_one_shard(
                    model,
                    opt,
                    path,
                    device,
                    batch_size=args.batch_size,
                    max_dist_m=args.max_dist,
                    stride=args.stride,
                    rng=rng,
                )
            )
            if (si + 1) % 50 == 0 or si == 0:
                print(
                    f"    shard {si + 1}/{len(order)}  "
                    f"rss={_rss_mb():.0f}MB  last_loss={losses[-1] if losses else float('nan'):.4f}",
                    flush=True,
                )

        metrics = eval_shards(
            model,
            val_shards,
            device,
            max_dist_m=args.max_dist,
            batch_size=args.batch_size,
            max_frames=args.max_val_frames,
        )
        metrics["train_huber"] = float(np.mean(losses)) if losses else float("nan")
        metrics["epoch_sec"] = time.time() - t0
        metrics["rss_mb"] = _rss_mb()
        history.append({"epoch": epoch, **metrics})
        close = metrics.get("close_mae_m", metrics["mae_m"])
        print(
            f"  epoch {epoch}/{args.epochs}  huber={metrics['train_huber']:.4f}  "
            f"val_mae={metrics['mae_m']:.3f}m  p50={metrics['p50_m']:.3f}m  "
            f"frac<0.5={metrics['frac_lt_0_5m']:.2%}  "
            f"close_mae={metrics.get('close_mae_m', float('nan')):.3f}m  "
            f"({metrics['epoch_sec']:.0f}s, rss={metrics['rss_mb']:.0f}MB)",
            flush=True,
        )
        if close < best_close:
            best_close = close
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "state_dim": state_dim,
                    "metrics": metrics,
                    "epoch": epoch,
                },
                args.out,
            )
            print(f"    saved best → {args.out} (close_mae={close:.3f}m)", flush=True)

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"best_close_mae_m": best_close, "history": history}, indent=2)
    )
    print(f"Done. Best close-range MAE={best_close:.3f}m  meta→{meta_path}")


if __name__ == "__main__":
    main()
