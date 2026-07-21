#!/usr/bin/env python3
"""
Landing-specialist training for Swarm SN124.

Strategy (decisive change from full-episode PPO):
  1. Train on SHORT episodes (6–18 m start→goal) so every rollout is landing-focused.
  2. Deploy with HYBRID inference: heuristic cruise → RL landing (see hybrid_controller.py).

Full-episode PPO destroys cruise behaviour; this script teaches only approach + landing.

Usage:
  # Step 1 — landing demos (heuristic near-goal episodes)
  python RL/collect_landing_demos.py --episodes 256

  # Step 2 — BC warm-start on landing demos
  python RL/pretrain_bc.py --demos RL/demos_landing.npz --out RL/checkpoints/bc_landing.zip

  # Step 3 — landing RL (resume from BC or prior best)
  python RL/train_landing.py \\
    --resume RL/checkpoints/bc_landing.zip \\
    --run-name landing_v1
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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.callbacks import BCAnchorCallback, EpochValidationCallback  # noqa: E402
from RL.env_utils import CURRICULUM, CurriculumStage, make_landing_env_fn  # noqa: E402
from RL.policy_net import load_swarm_depth_cnn_class  # noqa: E402
from RL.safe_rppo import SafeRecurrentPPO  # noqa: E402
from RL.train_sota import (  # noqa: E402
    _apply_train_hparams,
    _build_model,
    _maybe_freeze_encoder,
    bootstrap_best_checkpoint,
)
from RL.validate import evaluate_hybrid_model, evaluate_hybrid_open_benchmark  # noqa: E402
from RL.wrappers import (
    EpisodeScoreWrapper,
    LandingFocusedWrapper,
    StateNoiseWrapper,
    SwarmActionWrapper,
)


def _wrap_landing_env(raw_env, *, state_noise: float):
    env = SwarmActionWrapper(raw_env)
    env = LandingFocusedWrapper(env)
    env = StateNoiseWrapper(env, std=state_noise)
    env = EpisodeScoreWrapper(env)
    return env


class _LandingEnvFactory:
    def __init__(
        self,
        stage: CurriculumStage,
        *,
        state_noise: float,
        full_episode_ratio: float,
        distance_range: tuple[float, float],
        gui: bool = False,
    ):
        self.stage = stage
        self.state_noise = state_noise
        self.full_episode_ratio = full_episode_ratio
        self.distance_range = distance_range
        self.gui = gui

    def __call__(self):
        raw = make_landing_env_fn(
            self.stage,
            gui=self.gui,
            full_episode_ratio=self.full_episode_ratio,
            distance_range=self.distance_range,
        )()
        return _wrap_landing_env(raw, state_noise=self.state_noise)


def bootstrap_hybrid(
    model,
    output_dir: Path,
    stage: CurriculumStage,
    *,
    handoff_m: float,
    label: str,
):
    """Validate with hybrid controller (matches deployment)."""
    from RL.validate import ValidationResult  # noqa: F811

    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / "best_model.zip"

    result = evaluate_hybrid_open_benchmark(model, handoff_m=handoff_m)
    print(f"[bootstrap-hybrid] {label}: {result.summary_line()}")
    model.save(str(best_path))
    meta = {
        "timesteps": 0,
        "mean_score": result.mean_score,
        "success_rate": result.success_rate,
        "mean_distance_to_goal": result.mean_distance_to_goal,
        "stage": stage.name,
        "source": label,
        "handoff_m": handoff_m,
        "eval_mode": "hybrid",
    }
    (best_dir / "best_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return result


class HybridValidationCallback(EpochValidationCallback):
    """Epoch validation using hybrid cruise+landing (deployment-faithful)."""

    def __init__(self, *args, handoff_m: float = 8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.handoff_m = handoff_m

    def _run_validation(self, timesteps: int):
        from RL.callbacks import _policy_weights_valid

        if not _policy_weights_valid(self.model):
            self._nan_events += 1
            self._recover_from_best(f"NaN/Inf in policy weights (event #{self._nan_events})")
            return None

        try:
            result = evaluate_hybrid_open_benchmark(
                self.model,
                gui=False,
                handoff_m=self.handoff_m,
            )
        except (ValueError, RuntimeError) as exc:
            msg = str(exc).lower()
            if "nan" in msg or "invalid values" in msg:
                self._nan_events += 1
                self._recover_from_best(f"validation predict failed: {exc}")
                return None
            raise

        if self.logger is not None:
            self.logger.record("validation/mean_score", result.mean_score)
            self.logger.record("validation/success_rate", result.success_rate)
            self.logger.record("validation/mean_distance_to_goal", result.mean_distance_to_goal)
            for name, score in result.per_type.items():
                self.logger.record(f"validation/score_{name}", score)

        record = {
            "timesteps": timesteps,
            "stage": self.curriculum_stage.name,
            "mean_score": result.mean_score,
            "success_rate": result.success_rate,
            "mean_distance_to_goal": result.mean_distance_to_goal,
            "per_type": result.per_type,
            "eval_mode": "hybrid",
        }
        self.validation_history.append(record)
        from RL.validate import save_validation_log

        save_validation_log(
            self.log_path / "validation.jsonl",
            result,
            extra=record,
        )

        if self.verbose:
            per_seed = " | ".join(
                f"s{ep['map_seed']}={ep['distance_to_goal']:.1f}m"
                for ep in result.episodes
            )
            print(
                f"\n[validation-hybrid @ {timesteps:,} | handoff={self.handoff_m}m] "
                f"{result.summary_line()}\n  {per_seed}\n"
            )

        if self._should_save_best(result):
            self._save_best(result, timesteps)
        elif (
            self.regression_guard
            and self.best_mean_distance < 25.0
            and result.mean_distance_to_goal > self.best_mean_distance + self.regression_margin_m
        ):
            self._recover_from_best(
                f"dist={result.mean_distance_to_goal:.1f}m regressed from best={self.best_mean_distance:.1f}m"
            )

        if hasattr(self.model, "recovery_checkpoint"):
            self.model.recovery_checkpoint = self._best_path

        return result


def main():
    parser = argparse.ArgumentParser(description="Landing-specialist RecurrentPPO training")
    parser.add_argument("--timesteps", type=int, default=800_000)
    parser.add_argument("--eval-freq", type=int, default=8_000)
    parser.add_argument("--checkpoint-freq", type=int, default=16_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ent-coef", type=float, default=0.008)
    parser.add_argument("--clip-range", type=float, default=0.08)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--state-noise", type=float, default=0.01)
    parser.add_argument("--handoff-m", type=float, default=8.0, help="Hybrid validation handoff distance")
    parser.add_argument("--full-episode-ratio", type=float, default=0.05)
    parser.add_argument("--distance-min", type=float, default=3.0)
    parser.add_argument("--distance-max", type=float, default=10.0)
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep vision encoder trainable for depth landing (default: off)",
    )
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--demos-path",
        type=Path,
        default=ROOT / "RL" / "demos_landing.npz",
    )
    parser.add_argument(
        "--no-bc-anchor",
        action="store_true",
        help="Disable periodic BC anchor on landing demos",
    )
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    stage_idx = max(0, min(args.stage, len(CURRICULUM) - 1))
    stage = CURRICULUM[stage_idx]
    distance_range = (args.distance_min, args.distance_max)

    run_name = args.run_name or f"landing_{stage.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = ROOT / "RL" / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args) | {
        "stage": stage.name,
        "device": device,
        "run_name": run_name,
        "distance_range": distance_range,
        "tensorboard_log": str(tb_dir),
        "training_mode": "landing_specialist",
        "eval_mode": "hybrid",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print("Swarm SN124 — Landing Specialist Training")
    print(f"  stage          : {stage.name}")
    print(f"  distance range : {distance_range[0]:.0f}–{distance_range[1]:.0f} m")
    print(f"  full-ep ratio  : {args.full_episode_ratio}")
    print(f"  hybrid handoff : {args.handoff_m} m (validation only)")
    print(f"  timesteps      : {args.timesteps:,}")
    print(f"  lr / clip      : {args.lr} / {args.clip_range}")
    print(f"  output         : {output_dir}")
    print(f"{'='*60}\n")

    factory = _LandingEnvFactory(
        stage,
        state_noise=args.state_noise,
        full_episode_ratio=args.full_episode_ratio,
        distance_range=distance_range,
    )
    vec_env = DummyVecEnv([factory])

    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"Checkpoint not found: {args.resume}")
        SwarmDepthCNN = load_swarm_depth_cnn_class()
        model = SafeRecurrentPPO.load(
            str(args.resume),
            custom_objects={"SwarmDepthCNN": SwarmDepthCNN},
        )
        model.set_env(vec_env)
        _apply_train_hparams(
            model,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
        )
        model.n_epochs = args.n_epochs
        model.tensorboard_log = str(tb_dir)
    else:
        model = _build_model(
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=device,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
            target_kl=args.target_kl,
            tensorboard_log=str(tb_dir),
        )

    _maybe_freeze_encoder(model, freeze=args.freeze_encoder)
    bootstrap_result = bootstrap_hybrid(
        model,
        output_dir,
        stage,
        handoff_m=args.handoff_m,
        label=str(args.resume) if args.resume else "scratch",
    )

    best_path = output_dir / "best" / "best_model.zip"
    model.recovery_checkpoint = best_path
    model.freeze_encoder_on_recovery = args.freeze_encoder

    callback_list = [
        HybridValidationCallback(
            args.eval_freq,
            best_model_save_path=output_dir / "best",
            log_path=output_dir / "logs",
            curriculum_stage=stage,
            bootstrap_result=bootstrap_result,
            freeze_encoder=args.freeze_encoder,
            handoff_m=args.handoff_m,
            regression_margin_m=5.0,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="landing",
        ),
    ]
    if not args.no_bc_anchor and args.demos_path.exists():
        callback_list.append(
            BCAnchorCallback(
                args.demos_path,
                anchor_freq=args.eval_freq,
                lr=3e-6,
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
        print(f"\n✅ Landing training complete.")
        print(f"   best model : {best_path}")
        print(f"   hybrid eval: python RL/check_progress.py --model {best_path} --hybrid")
        print(f"   package    : python RL/run_pipeline.py --model {best_path} --hybrid")
        print(f"   tensorboard: tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
