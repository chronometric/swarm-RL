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

NOTE: Full-episode PPO consistently diverges (26m/48m modes). Use the hybrid
landing pipeline instead:
  python RL/collect_landing_demos.py
  python RL/pretrain_bc.py --demos RL/demos_landing.npz --out RL/checkpoints/bc_landing.zip
  python RL/train_landing.py --resume RL/checkpoints/bc_landing.zip
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
from sb3_contrib import RecurrentPPO  # noqa: F401 — kept for type hints / SB3 compat
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.callbacks import BCAnchorCallback, EpochValidationCallback  # noqa: E402
from RL.env_utils import CURRICULUM, CurriculumStage, make_training_env_fn  # noqa: E402
from RL.policy_net import load_swarm_depth_cnn_class  # noqa: E402
from RL.safe_rppo import SafeRecurrentPPO  # noqa: E402
from RL.validate import evaluate_model  # noqa: E402
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

    def __init__(
        self,
        stage: CurriculumStage,
        *,
        state_noise: float,
        gui: bool = False,
        val_mix_ratio: float = 0.0,
    ):
        self.stage = stage
        self.state_noise = state_noise
        self.gui = gui
        self.val_mix_ratio = val_mix_ratio

    def set_stage(self, stage: CurriculumStage) -> None:
        self.stage = stage

    def __call__(self):
        raw = make_training_env_fn(
            self.stage,
            gui=self.gui,
            val_mix_ratio=self.val_mix_ratio,
        )()
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
    target_kl: float | None,
    tensorboard_log: str | None = None,
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
    return SafeRecurrentPPO(
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
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log,
    )


def _apply_train_hparams(
    model,
    *,
    learning_rate: float,
    ent_coef: float,
    clip_range: float,
    target_kl: float | None,
) -> None:
    """Apply CLI hyperparameters to a loaded model (resume-safe).

    SB3 expects ``clip_range`` and ``lr_schedule`` to be callables, not raw floats.
    """
    model.lr_schedule = FloatSchedule(learning_rate)
    model.ent_coef = float(ent_coef)
    model.clip_range = FloatSchedule(clip_range)
    model.target_kl = target_kl
    if not callable(model.clip_range):
        raise TypeError("clip_range must be callable after applying hyperparameters")
    if not callable(model.lr_schedule):
        raise TypeError("lr_schedule must be callable after applying hyperparameters")


def bootstrap_best_checkpoint(
    model,
    output_dir: Path,
    stage: CurriculumStage,
    *,
    label: str = "resume",
) -> "ValidationResult":
    """Validate before any training steps and seed best/ with the baseline."""
    from RL.validate import ValidationResult  # noqa: F811

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / "best_model.zip"

    result = evaluate_model(model, challenge_types=stage.challenge_types)
    print(f"[bootstrap] {label} baseline: {result.summary_line()}")
    model.save(str(best_path))
    meta = {
        "timesteps": 0,
        "mean_score": result.mean_score,
        "success_rate": result.success_rate,
        "mean_distance_to_goal": result.mean_distance_to_goal,
        "stage": stage.name,
        "source": label,
    }
    (best_dir / "best_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


def _maybe_freeze_encoder(model, *, freeze: bool) -> None:
    if not freeze:
        return
    frozen = 0
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
        frozen += param.numel()
    print(f"Frozen vision encoder ({frozen:,} params) — RL updates LSTM + policy heads only")


def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO miner for Swarm SN124")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Validation every N steps")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint every N steps")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 3e-5 fine-tune, 3e-4 scratch)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coef (default: 0.001 if --resume, else 0.02)")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip range (default: 0.08 if --resume, else 0.2)")
    parser.add_argument("--n-epochs", type=int, default=None, help="PPO epochs per rollout (default: 4 if --resume, else 10)")
    parser.add_argument("--target-kl", type=float, default=None, help="Early-stop PPO epoch if KL exceeds this (default: 0.02 fine-tune)")
    parser.add_argument("--state-noise", type=float, default=0.02, help="Domain rand. std on state")
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze SwarmDepthCNN during fine-tune (default: on when --resume)",
    )
    parser.add_argument("--stage", type=int, default=0, help="Curriculum stage index (0=open)")
    parser.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name")
    parser.add_argument(
        "--val-mix",
        type=float,
        default=None,
        help="Fraction of episodes sampled near validation seeds (default: 0.35 fine-tune, 0.15 scratch)",
    )
    parser.add_argument(
        "--bc-anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Periodic BC anchor updates during RL (default: on when --resume)",
    )
    parser.add_argument(
        "--demos-path",
        type=Path,
        default=ROOT / "RL" / "demos_open_v2.npz",
        help="Demonstrations for BC anchor",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    stage_idx = max(0, min(args.stage, len(CURRICULUM) - 1))
    stage = CURRICULUM[stage_idx]

    fine_tuning = args.resume is not None
    learning_rate = args.lr if args.lr is not None else (5e-6 if fine_tuning else 3e-4)
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.0002 if fine_tuning else 0.02)
    clip_range = args.clip_range if args.clip_range is not None else (0.03 if fine_tuning else 0.2)
    n_epochs = args.n_epochs if args.n_epochs is not None else (2 if fine_tuning else 10)
    target_kl = args.target_kl if args.target_kl is not None else (0.01 if fine_tuning else None)
    freeze_encoder = args.freeze_encoder if args.freeze_encoder is not None else fine_tuning
    state_noise = 0.0 if fine_tuning else args.state_noise
    val_mix_ratio = args.val_mix if args.val_mix is not None else (0.35 if fine_tuning else 0.15)
    use_bc_anchor = args.bc_anchor if args.bc_anchor is not None else fine_tuning

    run_name = args.run_name or f"sota_{stage.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = ROOT / "RL" / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

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
                "target_kl": target_kl,
                "freeze_encoder": freeze_encoder,
                "learning_rate": learning_rate,
                "state_noise_effective": state_noise,
                "val_mix_ratio": val_mix_ratio,
                "bc_anchor": use_bc_anchor,
                "tensorboard_log": str(tb_dir),
            }
        ).items()
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Swarm SN124 SOTA Training — RecurrentPPO")
    print(f"  stage     : {stage.name} ({stage.challenge_types})")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  lr        : {learning_rate}")
    print(f"  ent_coef  : {ent_coef}")
    print(f"  clip_range: {clip_range}")
    print(f"  n_epochs  : {n_epochs}")
    print(f"  target_kl : {target_kl}")
    print(f"  freeze_enc: {freeze_encoder}")
    print(f"  val_mix   : {val_mix_ratio}")
    print(f"  bc_anchor : {use_bc_anchor}")
    print(f"  device    : {device}")
    print(f"  output    : {output_dir}")
    print(f"  tensorboard: {tb_dir}")
    print(f"{'='*60}\n")

    factory = _StageEnvFactory(stage, state_noise=state_noise, val_mix_ratio=val_mix_ratio)
    vec_env = DummyVecEnv([factory])

    bootstrap_result = None
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from {args.resume}")
        SwarmDepthCNN = load_swarm_depth_cnn_class()
        model = SafeRecurrentPPO.load(
            str(args.resume),
            custom_objects={"SwarmDepthCNN": SwarmDepthCNN},
        )
        model.set_env(vec_env)
        if device != "auto":
            model.device = torch.device(device)
        _apply_train_hparams(
            model,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            clip_range=clip_range,
            target_kl=target_kl,
        )
        model.n_epochs = n_epochs
        _maybe_freeze_encoder(model, freeze=freeze_encoder)
        model.tensorboard_log = str(tb_dir)
        bootstrap_result = bootstrap_best_checkpoint(
            model,
            output_dir,
            stage,
            label=str(args.resume),
        )
    else:
        model = _build_model(
            vec_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=device,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_epochs=n_epochs,
            target_kl=target_kl,
            tensorboard_log=str(tb_dir),
        )
        _maybe_freeze_encoder(model, freeze=freeze_encoder)
        bootstrap_result = bootstrap_best_checkpoint(
            model,
            output_dir,
            stage,
            label="scratch",
        )

    best_path = output_dir / "best" / "best_model.zip"
    model.recovery_checkpoint = best_path
    model.freeze_encoder_on_recovery = freeze_encoder

    def on_stage_change(new_stage: CurriculumStage, new_idx: int) -> None:
        nonlocal stage
        stage = new_stage
        factory.set_stage(new_stage)
        print(f"[curriculum] env sampling now uses types {new_stage.challenge_types}")

    callback_list = [
        EpochValidationCallback(
            args.eval_freq,
            best_model_save_path=output_dir / "best",
            log_path=output_dir / "logs",
            curriculum_stage=stage,
            bootstrap_result=bootstrap_result,
            freeze_encoder=freeze_encoder,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="rppo",
        ),
    ]
    if use_bc_anchor:
        callback_list.append(
            BCAnchorCallback(
                args.demos_path,
                anchor_freq=args.eval_freq,
                verbose=1,
            )
        )
    callbacks = CallbackList(callback_list)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="train",
            progress_bar=True,
            reset_num_timesteps=True,
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
        print(f"  tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
