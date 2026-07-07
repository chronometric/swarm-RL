"""Stable-Baselines3 callbacks for epoch validation and curriculum."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from RL.env_utils import CURRICULUM, CurriculumStage
from RL.validate import evaluate_model, save_validation_log


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
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_model_save_path = Path(best_model_save_path)
        self.log_path = Path(log_path)
        self.curriculum_stage = curriculum_stage
        self.best_mean_score = -1.0
        self.best_mean_distance = float("inf")
        self.validation_history: list[dict] = []

        self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

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
        best_path = self.best_model_save_path / "best_model.zip"
        self.model.save(str(best_path))
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
                f"dist={result.mean_distance_to_goal:.1f}m → {best_path}"
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        result = evaluate_model(
            self.model,
            gui=False,
            challenge_types=self.curriculum_stage.challenge_types,
        )
        timesteps = self.num_timesteps

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
