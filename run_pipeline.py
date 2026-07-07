#!/usr/bin/env python3
"""
End-to-end miner pipeline: deploy model → package → quick benchmark.

Usage:
  source miner_env/bin/activate
  python RL/run_pipeline.py --model RL/checkpoints/.../best/best_model.zip
  python RL/run_pipeline.py --model ... --benchmark --seeds-per-group 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MY_AGENT = ROOT / "my_agent"
SUBMISSION = ROOT / "Submission" / "submission.zip"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=ROOT, check=check)


def deploy_model(model_path: Path) -> None:
    """Copy checkpoint to my_agent/, stripping optimizer state to stay under 50 MiB."""
    import importlib.util

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
    print(f"Deployed {model_path} → {dest} ({size_mb:.1f} MiB, optimizer stripped)")
    if dest.stat().st_size > 50 * 1024 * 1024:
        raise SystemExit(
            f"Submission too large ({size_mb:.1f} MiB > 50 MiB). "
            "Retrain with normalized_image=True (see RL/train_sota.py) or reduce net size."
        )


def main():
    parser = argparse.ArgumentParser(description="Package and benchmark Swarm miner agent")
    parser.add_argument("--model", type=Path, required=True, help="RecurrentPPO checkpoint .zip")
    parser.add_argument("--benchmark", action="store_true", help="Run swarm benchmark after packaging")
    parser.add_argument("--seeds-per-group", type=int, default=1, help="Benchmark seeds per env type")
    parser.add_argument("--workers", type=int, default=2, help="Benchmark Docker workers")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    deploy_model(args.model)

    _run([sys.executable, "-m", "swarm", "model", "test", "--source", str(MY_AGENT)])
    _run([
        sys.executable, "-m", "swarm", "model", "package",
        "--source", str(MY_AGENT),
        "--output", str(SUBMISSION),
        "--overwrite",
    ])
    _run([sys.executable, "-m", "swarm", "model", "verify", "--model", str(SUBMISSION)])

    if args.benchmark:
        _run([
            sys.executable, "-m", "swarm", "benchmark",
            "--model", str(SUBMISSION),
            "--seeds-per-group", str(args.seeds_per_group),
            "--workers", str(args.workers),
        ])
        _run([sys.executable, "-m", "swarm", "report"])

    print(f"\n✅ Pipeline complete. Submission: {SUBMISSION}")
    print("Register on-chain only when benchmark scores consistently beat the champion.")


if __name__ == "__main__":
    main()
