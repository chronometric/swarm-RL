"""Stable-Baselines3 callbacks for epoch validation and curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

from RL.env_utils import CURRICULUM, CurriculumStage
from RL.validate import evaluate_model, load_recurrent_model, save_validation_log


def _policy_weights_valid(model) -> bool:
    for param in model.policy.parameters():
        if param is None:
            continue
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False
    return True


def reload_policy_weights(model, checkpoint_path: Path, *, freeze_encoder: bool = False) -> None:
    """Reload policy weights from disk and clear optimizer momentum."""
    best = load_recurrent_model(checkpoint_path)
    model.policy.load_state_dict(best.policy.state_dict())
    del best
    if freeze_encoder:
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
    opt = getattr(model.policy, "optimizer", None)
    if opt is not None:
        opt.state.clear()


class EpochValidationCallback(BaseCallback):
    """
    Run validator-faithful local evaluation every ``eval_freq`` steps.

    Logs to TensorBoard and JSONL. Saves best checkpoint by:
      1. higher mean validator score, or
      2. same score (typical 0.01 pre-landing) with lower mean distance to goal.
    """

    def __init__(
        self,
        eval_freq: int,
        *,
        best_model_save_path: Path,
        log_path: Path,
        curriculum_stage: CurriculumStage,
        bootstrap_result=None,
        freeze_encoder: bool = False,
        regression_guard: bool = True,
        regression_margin_m: float = 3.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.curriculum_stage = curriculum_stage
        self.freeze_encoder = freeze_encoder
        self.regression_guard = regression_guard
        self.regression_margin_m = regression_margin_m
        self.best_mean_score = -1.0
        self.best_mean_distance = float("inf")
        self._regression_events = 0
        self._nan_events = 0
        self.validation_history: list[dict] = []

        self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        if bootstrap_result is not None:
            self.best_mean_score = bootstrap_result.mean_score
            self.best_mean_distance = bootstrap_result.mean_distance_to_goal

    @property
    def _best_path(self) -> Path:
        return self.best_model_save_path / "best_model.zip"

    def _should_save_best(self, result) -> bool:
        score = result.mean_score
        dist = result.mean_distance_to_goal
        if score > self.best_mean_score:
            return True
        if abs(score - self.best_mean_score) < 1e-9 and dist < self.best_mean_distance:
            return True
        return False

    def _save_best(self, result, timesteps: int) -> None:
        self.best_mean_score = result.mean_score
        self.best_mean_distance = result.mean_distance_to_goal
        self.model.save(str(self._best_path))
        meta = {
            "timesteps": timesteps,
            "mean_score": result.mean_score,
            "success_rate": result.success_rate,
            "mean_distance_to_goal": result.mean_distance_to_goal,
            "stage": self.curriculum_stage.name,
        }
        (self.best_model_save_path / "best_model_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        if self.verbose:
            print(
                f"  → new best saved score={result.mean_score:.4f} "
                f"dist={result.mean_distance_to_goal:.1f}m → {self._best_path}"
            )

    def _recover_from_best(self, reason: str) -> None:
        if not self._best_path.exists():
            if self.verbose:
                print(f"[recovery] {reason} but no best checkpoint at {self._best_path}")
            return
        self._regression_events += 1
        if self.verbose:
            print(
                f"\n[recovery] {reason} → reload {self._best_path} "
                f"(best dist={self.best_mean_distance:.1f}m, event #{self._regression_events})\n"
            )
        reload_policy_weights(
            self.model,
            self._best_path,
            freeze_encoder=self.freeze_encoder,
        )
        current_lr = float(self.model.lr_schedule(1.0))
        new_lr = max(current_lr * 0.5, 5e-7)
        self.model.lr_schedule = FloatSchedule(new_lr)
        if self.verbose:
            print(f"[recovery] lr {current_lr:.2e} → {new_lr:.2e}\n")

    def _run_validation(self, timesteps: int):
        if not _policy_weights_valid(self.model):
            self._nan_events += 1
            self._recover_from_best(f"NaN/Inf in policy weights (event #{self._nan_events})")
            return None

        try:
            result = evaluate_model(
                self.model,
                gui=False,
                challenge_types=self.curriculum_stage.challenge_types,
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
        }
        self.validation_history.append(record)
        save_validation_log(
            self.log_path / "validation.jsonl",
            result,
            extra={
                "timesteps": timesteps,
                "stage": self.curriculum_stage.name,
                "mean_distance_to_goal": result.mean_distance_to_goal,
            },
        )

        if self.verbose:
            print(
                f"\n[validation @ {timesteps:,} steps | stage={self.curriculum_stage.name}] "
                f"{result.summary_line()}\n"
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

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        self._run_validation(self.num_timesteps)
        return True


class BCAnchorCallback(BaseCallback):
    """Periodic BC mini-updates to anchor policy near demonstrations during RL."""

    def __init__(
        self,
        demos_path: Path,
        *,
        anchor_freq: int = 10_000,
        steps: int = 256,
        batch_size: int = 128,
        lr: float = 5e-6,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.demos_path = Path(demos_path)
        self.anchor_freq = anchor_freq
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.anchor_freq != 0:
            return True
        if not self.demos_path.exists():
            if self.verbose:
                print(f"[bc-anchor] demos not found: {self.demos_path}")
            return True

        from RL.bc_anchor import run_bc_anchor

        loss = run_bc_anchor(
            self.model,
            self.demos_path,
            steps=self.steps,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        if self.logger is not None:
            self.logger.record("bc_anchor/mse", loss)
        if self.verbose:
            print(f"[bc-anchor @ {self.num_timesteps:,}] mse={loss:.6f}")
        return True


class CurriculumCallback(BaseCallback):
    """Advance curriculum when validation mean score crosses stage threshold."""

    def __init__(
        self,
        *,
        stage_index: int,
        eval_freq: int,
        on_stage_change,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stage_index = stage_index
        self.eval_freq = eval_freq
        self.on_stage_change = on_stage_change
        self._last_mean: Optional[float] = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        result = evaluate_model(
            self.model,
            gui=False,
            challenge_types=self.curriculum_stage.challenge_types,
        )
        self._last_mean = result.mean_score

        if self.stage_index >= len(CURRICULUM) - 1:
            return True

        stage = CURRICULUM[self.stage_index]
        if result.mean_score >= stage.min_mean_score:
            self.stage_index += 1
            new_stage = CURRICULUM[self.stage_index]
            if self.verbose:
                print(
                    f"\n[curriculum] advancing to stage '{new_stage.name}' "
                    f"(mean={result.mean_score:.4f} >= {stage.min_mean_score})\n"
                )
            self.on_stage_change(new_stage, self.stage_index)

        return True

    @property
    def current_stage(self) -> CurriculumStage:
        return CURRICULUM[self.stage_index]
