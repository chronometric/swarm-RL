"""Hybrid deployment: spiral disk-search + pad estimator + soft-land (RL assist off)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from RL.search_pilot import PilotPhase, SearchLandPilot, soft_land_action

DEFAULT_PAD_ESTIMATOR = Path("RL/checkpoints/pad_estimator.pt")


def horizontal_dist_to_search(observation: dict) -> float:
    state = np.asarray(observation["state"], dtype=np.float64)
    search_rel = state[-3:].astype(np.float64)
    horiz = search_rel.copy()
    horiz[2] = 0.0
    return float(np.linalg.norm(horiz))


@dataclass
class HybridConfig:
    # Spiral search + pad estimator + soft-land. RL land assist off by default.
    use_rl_land: bool = False
    rl_land_alt_m: float = 3.5
    cruise_speed: float = 0.55
    search_enter_m: float = 10.0
    deterministic: bool = True
    pad_estimator_path: Optional[Path] = None
    pad_estimator_device: str = "cpu"


class HybridController:
    """
    Stage-0 controller:

      1. SearchLandPilot cruise + spiral
      2. Pad estimator locks pad XY when confident
      3. Soft-land heuristic completes touchdown
      4. Optional RL assist (off by default — often hurts)
    """

    def __init__(self, model, *, config: Optional[HybridConfig] = None):
        self.model = model
        self.config = config or HybridConfig()
        pad_model = None
        device = self.config.pad_estimator_device
        path = self.config.pad_estimator_path
        if path is None and DEFAULT_PAD_ESTIMATOR.exists():
            path = DEFAULT_PAD_ESTIMATOR
        if path is not None and Path(path).exists():
            from RL.pad_estimator import load_pad_estimator

            pad_model = load_pad_estimator(path, device=device)
            print(f"[hybrid] pad estimator loaded: {path}")
        self.pilot = SearchLandPilot(
            cruise_speed=self.config.cruise_speed,
            search_enter_m=self.config.search_enter_m,
            pad_estimator=pad_model,
            pad_estimator_device=device,
        )
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
                    out = 0.35 * soft_land_action(
                        observation, target_xy=self.pilot._pad_xy
                    ) + 0.65 * rl_action[:5]
                    out[3] = float(np.clip(min(out[3], 0.32), 0.0, 1.0))
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
    use_rl_land: bool = False,
    device: str | None = None,
    pad_estimator_path: Path | None = None,
) -> HybridController:
    from sb3_contrib import RecurrentPPO

    from RL.policy_net import load_swarm_depth_cnn_class

    SwarmDepthCNN = load_swarm_depth_cnn_class()
    model = RecurrentPPO.load(
        str(checkpoint),
        custom_objects={"SwarmDepthCNN": SwarmDepthCNN},
        device=device or "auto",
    )
    return HybridController(
        model,
        config=HybridConfig(
            use_rl_land=use_rl_land,
            pad_estimator_path=pad_estimator_path,
            pad_estimator_device="cpu",
        ),
    )
