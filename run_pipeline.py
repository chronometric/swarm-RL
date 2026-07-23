#!/usr/bin/env python3
"""
End-to-end miner pipeline: deploy model → package → optional benchmark.

Usage:
  source miner_env/bin/activate
  python RL/run_pipeline.py --model RL/checkpoints/.../best/best_model.zip
  python RL/run_pipeline.py --model ... --benchmark --seeds-per-group 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MY_AGENT = ROOT / "my_agent"
SUBMISSION = ROOT / "Submission" / "submission.zip"
DEFAULT_BENCH_LOG = Path("/tmp/bench_full_eval.log")


def _run(cmd: list[str], *, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}\n")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=ROOT, check=check, env=merged)


def deploy_model(model_path: Path, *, pad_estimator: Path | None = None) -> None:
    """Copy checkpoint to my_agent/, stripping optimizer state to stay under 50 MiB."""
    import importlib.util
    import shutil

    from sb3_contrib import RecurrentPPO

    agent_path = MY_AGENT / "drone_agent.py"
    spec = importlib.util.spec_from_file_location("drone_agent", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    MY_AGENT.mkdir(parents=True, exist_ok=True)
    dest = MY_AGENT / "model.zip"

    model = RecurrentPPO.load(
        str(model_path),
        custom_objects={"SwarmDepthCNN": mod.SwarmDepthCNN},
    )
    model.save(str(dest), exclude=["optimizer"])
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Deployed {model_path} → {dest} ({size_mb:.1f} MiB, hybrid agent, optimizer stripped)")

    pad_src = pad_estimator
    if pad_src is None:
        candidate = ROOT / "RL" / "checkpoints" / "pad_estimator.pt"
        if candidate.exists():
            pad_src = candidate
    if pad_src is not None and Path(pad_src).exists():
        pad_dst = MY_AGENT / "pad_estimator.pt"
        shutil.copy2(pad_src, pad_dst)
        print(f"Deployed pad estimator {pad_src} → {pad_dst} ({pad_dst.stat().st_size / 1024:.1f} KiB)")
    else:
        print("[warn] no pad_estimator.pt found — packaging spiral-only agent")

    if dest.stat().st_size > 50 * 1024 * 1024:
        raise SystemExit(
            f"Submission too large ({size_mb:.1f} MiB > 50 MiB). "
            "Retrain with a smaller net or strip more state."
        )


def main():
    parser = argparse.ArgumentParser(description="Package and benchmark Swarm miner agent")
    parser.add_argument("--model", type=Path, required=True, help="RecurrentPPO checkpoint .zip")
    parser.add_argument("--benchmark", action="store_true", help="Run swarm Docker benchmark after packaging")
    parser.add_argument("--seeds-per-group", type=int, default=1, help="Benchmark seeds per env type")
    parser.add_argument("--workers", type=int, default=2, help="Benchmark Docker workers")
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="(compat) Hybrid controller is always deployed from my_agent/drone_agent.py",
    )
    parser.add_argument(
        "--strict-lockdown",
        action="store_true",
        help="Require Docker network lockdown (needs root/CAP_NET_ADMIN). Default: skip for local bench.",
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=DEFAULT_BENCH_LOG,
        help=f"Benchmark log path (default: {DEFAULT_BENCH_LOG})",
    )
    parser.add_argument(
        "--local-check",
        action="store_true",
        help="Also run fast local hybrid check_progress before Docker benchmark",
    )
    parser.add_argument(
        "--pad-estimator",
        type=Path,
        default=Path("RL/checkpoints/pad_estimator.pt"),
        help="Pad XY estimator .pt to include in my_agent/",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    deploy_model(args.model, pad_estimator=args.pad_estimator if args.pad_estimator.exists() else None)

    _run([sys.executable, "-m", "swarm", "model", "test", "--source", str(MY_AGENT)])
    _run([
        sys.executable, "-m", "swarm", "model", "package",
        "--source", str(MY_AGENT),
        "--output", str(SUBMISSION),
        "--overwrite",
    ])
    _run([sys.executable, "-m", "swarm", "model", "verify", "--model", str(SUBMISSION)])

    if args.local_check:
        _run([
            sys.executable,
            "RL/check_progress.py",
            "--model",
            str(args.model),
            "--hybrid",
        ], check=False)

    if args.benchmark:
        bench_env = {}
        if not args.strict_lockdown:
            # Local machines often lack nsenter/iptables privileges; without this
            # every seed fails with network_lockdown_failed and simT=0.
            bench_env["SWARM_SKIP_NETWORK_LOCKDOWN"] = "1"
            print(
                "\n[pipeline] SWARM_SKIP_NETWORK_LOCKDOWN=1 "
                "(local Docker bench). Use --strict-lockdown on a root-capable host.\n"
            )
        args.log_out.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "-m",
                "swarm",
                "benchmark",
                "--model",
                str(SUBMISSION),
                "--seeds-per-group",
                str(args.seeds_per_group),
                "--workers",
                str(args.workers),
                "--log-out",
                str(args.log_out),
            ],
            env=bench_env,
        )
        report = _run(
            [sys.executable, "-m", "swarm", "report", "--input", str(args.log_out)],
            check=False,
        )
        if report.returncode != 0:
            print(
                f"\n[pipeline] report failed (exit {report.returncode}). "
                f"Inspect log: {args.log_out}\n"
            )

    print(f"\n✅ Pipeline complete. Submission: {SUBMISSION}")
    print("Register on-chain only when benchmark scores consistently beat the champion.")


if __name__ == "__main__":
    main()
