"""Hybrid deployment: spiral disk-search (primary) + optional RL soft-assist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from RL.search_pilot import PilotPhase, SearchLandPilot, soft_land_action


def horizontal_dist_to_search(observation: dict) -> float:
    state = np.asarray(observation["state"], dtype=np.float64)
    search_rel = state[-3:].astype(np.float64)
    horiz = search_rel.copy()
    horiz[2] = 0.0
    return float(np.linalg.norm(horiz))


@dataclass
class HybridConfig:
    # Spiral search is primary. RL only assists during LAND if enabled.
    use_rl_land: bool = True
    rl_land_alt_m: float = 3.5  # engage RL soft-assist when altitude ray below this
    cruise_speed: float = 0.55
    search_enter_m: float = 10.0
    deterministic: bool = True


class HybridController:
    """
    Most reliable Swarm Stage-0 controller:

      1. SearchLandPilot cruise + spiral disk search finds the elevated pad
      2. Soft-land heuristic completes touchdown
      3. Optional: RL predicts action when very close / low altitude (assist only)

    LSTM is reset when entering the land phase.
    """

    def __init__(self, model, *, config: Optional[HybridConfig] = None):
        self.model = model
        self.config = config or HybridConfig()
        self.pilot = SearchLandPilot(cruise_speed=self.config.cruise_speed, search_enter_m=self.config.search_enter_m)
        self._lstm_states: Any = None
        self._episode_start = np.ones((1,), dtype=bool)
        self._using_rl = False

    def reset(self) -> None:
        self.pilot.reset()
        self._lstm_states = None
        self._episode_start = np.ones((1,), dtype=bool)
        self._using_rl = False

    def act(self, observation: dict) -> np.ndarray:
        cfg = self.config
        base = self.pilot.act(observation)

        # Only consider RL assist after pad lock (LAND phase) and when low.
        if (
            cfg.use_rl_land
            and self.pilot.phase == PilotPhase.LAND
            and self.model is not None
        ):
            state = np.asarray(observation["state"], dtype=np.float64)
            alt_ray = float(state[-4]) * 20.0 if state.shape[0] >= 4 else 99.0
            if alt_ray <= cfg.rl_land_alt_m:
                if not self._using_rl:
                    self._using_rl = True
                    self._lstm_states = None
                    self._episode_start = np.ones((1,), dtype=bool)
                try:
                    rl_action, self._lstm_states = self.model.predict(
                        observation,
                        state=self._lstm_states,
                        episode_start=self._episode_start,
                        deterministic=cfg.deterministic,
                    )
                    self._episode_start = np.zeros((1,), dtype=bool)
                    rl_action = np.asarray(rl_action, dtype=np.float64).reshape(-1)
                    if not np.all(np.isfinite(rl_action)):
                        raise ValueError("non-finite RL action")
                    rl_action[3] = float(np.clip(rl_action[3], 0.0, 1.0))
                    # Blend: prefer low thrust / soft descent from heuristic.
                    out = 0.55 * soft_land_action(observation, target_xy=self.pilot._pad_xy) + 0.45 * rl_action[:5]
                    out[3] = float(np.clip(min(out[3], 0.35), 0.0, 1.0))
                    return out
                except (ValueError, RuntimeError):
                    self._using_rl = False
                    self._lstm_states = None
                    self._episode_start = np.ones((1,), dtype=bool)
                    return soft_land_action(observation, target_xy=self.pilot._pad_xy)

        self._using_rl = False
        return np.asarray(base, dtype=np.float64).reshape(-1)


def load_hybrid_controller(
    checkpoint: Path,
    *,
    use_rl_land: bool = True,
    device: str | None = None,
) -> HybridController:
    from sb3_contrib import RecurrentPPO

    from RL.policy_net import load_swarm_depth_cnn_class

    SwarmDepthCNN = load_swarm_depth_cnn_class()
    model = RecurrentPPO.load(
        str(checkpoint),
        custom_objects={"SwarmDepthCNN": SwarmDepthCNN},
        device=device or "auto",
    )
    return HybridController(model, config=HybridConfig(use_rl_land=use_rl_land))
