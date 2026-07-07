#!/usr/bin/env python3
"""
SOTA miner training for Swarm Subnet 124.

Architecture: Recurrent PPO (PPO-LSTM) with MultiInputLstmPolicy
  - NatureCNN on 128×128 depth + fused state vector
  - GRU/LSTM memory for temporal context (replaces frame stacking)

Training stack:
  - Curriculum: open → open+mountain → +village → all 6 env types
  - Reward shaping: progress + time penalty + landing bonus (on top of flight_reward)
  - Domain randomization: Gaussian noise on state vector
  - Epoch validation: 1 fixed seed per env type, logged to TensorBoard + JSONL

Usage:
  source miner_env/bin/activate
  python RL/train_sota.py --timesteps 500000 --eval-freq 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.callbacks import EpochValidationCallback  # noqa: E402
from RL.env_utils import CURRICULUM, CurriculumStage, make_training_env_fn  # noqa: E402
from RL.policy_net import load_swarm_depth_cnn_class  # noqa: E402
from RL.validate import load_recurrent_model  # noqa: E402
from RL.wrappers import (
    EpisodeScoreWrapper,
    ShapedProgressWrapper,
    StateNoiseWrapper,
    SwarmActionWrapper,
)


def _wrap_env(raw_env, *, state_noise: float):
    env = SwarmActionWrapper(raw_env)
    env = ShapedProgressWrapper(env)
    env = StateNoiseWrapper(env, std=state_noise)
    env = EpisodeScoreWrapper(env)
    return env


class _StageEnvFactory:
    """Mutable env factory so curriculum can swap task distribution mid-run."""

    def __init__(self, stage: CurriculumStage, *, state_noise: float, gui: bool = False):
        self.stage = stage
        self.state_noise = state_noise
        self.gui = gui

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage = stage

    def __call__(self):
        raw = make_training_env_fn(self.stage, gui=self.gui)()
        return _wrap_env(raw, state_noise=self.state_noise)


def _build_model(
    env,
    *,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    device: str,
    ent_coef: float,
    clip_range: float,
    n_epochs: int,
):
    SwarmDepthCNN = load_swarm_depth_cnn_class()
    policy_kwargs = dict(
        lstm_hidden_size=256,
        n_lstm_layers=1,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
        features_extractor_class=SwarmDepthCNN,
        features_extractor_kwargs=dict(features_dim=256),
        log_std_init=-0.5,
    )
    return RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=str(ROOT / "RL" / "runs"),
    )


def _apply_train_hparams(model, *, learning_rate: float, ent_coef: float, clip_range: float) -> None:
    """Apply CLI hyperparameters to a loaded model (resume-safe).

    SB3 expects ``clip_range`` and ``lr_schedule`` to be callables, not raw floats.
    """
    model.lr_schedule = FloatSchedule(learning_rate)
    model.ent_coef = float(ent_coef)
    model.clip_range = FloatSchedule(clip_range)
    if not callable(model.clip_range):
        raise TypeError("clip_range must be callable after applying hyperparameters")
    if not callable(model.lr_schedule):
        raise TypeError("lr_schedule must be callable after applying hyperparameters")


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO miner for Swarm SN124")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Validation every N steps")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint every N steps")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coef (default: 0.005 if --resume, else 0.02)")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range (default: 0.1 if --resume, else 0.2)")
    parser.add_argument("--n-epochs", type=int, default=None, help="PPO epochs per rollout (default: 5 if --resume, else 10)")
    parser.add_argument("--state-noise", type=float, default=0.02, help="Domain rand. std on state")
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage index (0=open)")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    stage_idx = max(0, min(args.stage, len(CURRICULUM) - 1))
    stage = CURRICULUM[stage_idx]

    fine_tuning = args.resume is not None
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.005 if fine_tuning else 0.02)
    clip_range = args.clip_range if args.clip_range is not None else (0.1 if fine_tuning else 0.2)
    n_epochs = args.n_epochs if args.n_epochs is not None else (5 if fine_tuning else 10)

    run_name = args.run_name or f"sota_{stage.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = ROOT / "RL" / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in (
            vars(args)
            | {
                "stage": stage.name,
                "device": device,
                "run_name": run_name,
                "ent_coef": ent_coef,
                "clip_range": clip_range,
                "n_epochs": n_epochs,
            }
        ).items()
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Swarm SN124 SOTA Training — RecurrentPPO")
    print(f"  stage     : {stage.name} ({stage.challenge_types})")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  lr        : {args.lr}")
    print(f"  ent_coef  : {ent_coef}")
    print(f"  clip_range: {clip_range}")
    print(f"  n_epochs  : {n_epochs}")
    print(f"  device    : {device}")
    print(f"  output    : {output_dir}")
    print(f"{'='*60}\n")

    factory = _StageEnvFactory(stage, state_noise=args.state_noise)
    vec_env = DummyVecEnv([factory])

    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from {args.resume}")
        model = load_recurrent_model(args.resume)
        model.set_env(vec_env)
        if device != "auto":
            model.device = torch.device(device)
        _apply_train_hparams(
            model,
            learning_rate=args.lr,
            ent_coef=ent_coef,
            clip_range=clip_range,
        )
        model.n_epochs = n_epochs
    else:
        model = _build_model(
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=device,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=n_epochs,
        )

    def on_stage_change(new_stage: CurriculumStage, new_idx: int) -> None:
        factory.set_stage(new_stage)
        print(f"[curriculum] env sampling now uses types {new_stage.challenge_types}")

    callbacks = CallbackList([
        EpochValidationCallback(
            args.eval_freq,
            best_model_save_path=output_dir / "best",
            log_path=output_dir / "logs",
            curriculum_stage=stage,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="rppo",
        ),
    ])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
    finally:
        final_path = output_dir / "final_model.zip"
        model.save(str(final_path))
        vec_env.close()
        print(f"\n✅ Training complete.")
        print(f"   final model : {final_path}")
        print(f"   best model  : {output_dir / 'best' / 'best_model.zip'}")
        print(f"\nNext steps:")
        print(f"  python RL/run_pipeline.py --model {output_dir / 'best' / 'best_model.zip'}")
        print(f"  tensorboard --logdir RL/runs")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
