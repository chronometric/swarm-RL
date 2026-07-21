"""RecurrentPPO with rollout/train NaN guards and optional checkpoint recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from sb3_contrib import RecurrentPPO

from RL.callbacks import _policy_weights_valid, reload_policy_weights


class SafeRecurrentPPO(RecurrentPPO):
    """
    RecurrentPPO that reloads a known-good checkpoint when weights become NaN/Inf.

    Set ``recovery_checkpoint`` before ``learn()`` (done automatically in train_sota).
    """

    recovery_checkpoint: Optional[Path] = None
    freeze_encoder_on_recovery: bool = False
    _recovery_events: int = 0

    def _recover_if_needed(self, reason: str) -> bool:
        if _policy_weights_valid(self):
            return False
        path = self.recovery_checkpoint
        if path is None or not Path(path).exists():
            raise RuntimeError(f"Policy weights invalid ({reason}) and no recovery checkpoint set")
        self._recovery_events += 1
        print(f"\n[safe-rppo] {reason} → reload {path} (event #{self._recovery_events})\n")
        reload_policy_weights(
            self,
            Path(path),
            freeze_encoder=self.freeze_encoder_on_recovery,
        )
        current_lr = float(self.lr_schedule(1.0))
        new_lr = max(current_lr * 0.5, 5e-7)
        from stable_baselines3.common.utils import FloatSchedule

        self.lr_schedule = FloatSchedule(new_lr)
        print(f"[safe-rppo] lr {current_lr:.2e} → {new_lr:.2e}\n")
        return True

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        self._recover_if_needed("pre-rollout NaN/Inf")
        try:
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        except (ValueError, RuntimeError) as exc:
            msg = str(exc).lower()
            if "nan" in msg or "invalid values" in msg:
                if self._recover_if_needed(f"rollout failed: {exc}"):
                    return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
            raise

    def train(self) -> None:
        if not _policy_weights_valid(self):
            self._recover_if_needed("pre-train NaN/Inf")
        super().train()
        if not _policy_weights_valid(self):
            self._recover_if_needed("post-train NaN/Inf")
