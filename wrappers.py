"""Gymnasium wrappers for reward shaping and domain randomization."""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from RL.action_utils import prepare_swarm_action


class SwarmActionWrapper(gym.Wrapper):
    """Apply validator-faithful action preprocessing before env.step."""

    def step(self, action):
        action = prepare_swarm_action(action, self.env)
        return self.env.step(action)


class ShapedProgressWrapper(gym.Wrapper):
    """
    Dense reward on top of incremental flight_reward:
      - horizontal progress toward landing platform
      - altitude progress toward goal height
      - proximity bonus when close to platform
      - alignment with GPS search-area vector
      - tilt penalty (validator truncates at 60° roll/pitch)
      - dive penalty at spawn altitude
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        progress_scale: float = 0.12,
        altitude_scale: float = 0.06,
        proximity_scale: float = 0.05,
        alignment_scale: float = 0.02,
        time_penalty: float = 0.0003,
        landing_bonus: float = 10.0,
        dive_penalty: float = 0.03,
        tilt_penalty_scale: float = 0.02,
        landing_radius: float = 5.0,
    ):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.altitude_scale = altitude_scale
        self.proximity_scale = proximity_scale
        self.alignment_scale = alignment_scale
        self.time_penalty = time_penalty
        self.landing_bonus = landing_bonus
        self.dive_penalty = dive_penalty
        self.tilt_penalty_scale = tilt_penalty_scale
        self.landing_radius = landing_radius
        self._prev_dist: Optional[float] = None
        self._prev_alt_err: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_dist = float(info.get("distance_to_goal", 0.0))
        base = self.env.unwrapped
        goal_z = float(getattr(getattr(base, "task", None), "goal", (0, 0, 0))[2])
        self._prev_alt_err = abs(float(obs["state"][2]) - goal_z)
        return obs, info

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward) - self.time_penalty

        dist = float(info.get("distance_to_goal", 0.0))
        if self._prev_dist is not None:
            shaped += self.progress_scale * (self._prev_dist - dist)
        self._prev_dist = dist

        state = np.asarray(obs["state"], dtype=np.float64)
        base = self.env.unwrapped
        goal_z = float(getattr(getattr(base, "task", None), "goal", (0, 0, 0))[2])
        alt_err = abs(float(state[2]) - goal_z)
        if self._prev_alt_err is not None:
            shaped += self.altitude_scale * (self._prev_alt_err - alt_err)
        self._prev_alt_err = alt_err

        if dist < self.landing_radius:
            shaped += self.proximity_scale * (self.landing_radius - dist) / self.landing_radius

        # Stronger signal inside 10 m — where depth-based landing must be learned.
        if dist < 10.0:
            shaped += 0.04 * (10.0 - dist) / 10.0
        if dist < 5.0:
            shaped += 0.06 * (5.0 - dist) / 5.0
            # Reward slowing down near the pad.
            if float(raw[3]) < 0.45:
                shaped += 0.02 * (0.45 - float(raw[3]))

        search_rel = state[-3:].copy()
        search_horiz = search_rel.copy()
        search_horiz[2] = 0.0
        sh_norm = float(np.linalg.norm(search_horiz))
        if sh_norm > 1e-3:
            target_dir = search_horiz / sh_norm
            act_dir = raw[:3].copy()
            act_dir[2] = 0.0
            a_norm = float(np.linalg.norm(act_dir))
            if a_norm > 1e-3:
                shaped += self.alignment_scale * float(np.dot(target_dir, act_dir / a_norm))

        if float(state[2]) > goal_z + 3.0:
            if raw[2] < -0.3 and raw[3] > 0.3:
                shaped -= self.dive_penalty

        # Discourage excessive tilt (env truncates at MAX_TILT_RAD ≈ 60°).
        drone = getattr(base, "_getDroneStateVector", lambda _: None)(0)
        if drone is not None:
            roll, pitch = float(drone[7]), float(drone[8])
            max_tilt = float(getattr(base, "MAX_TILT_RAD", 1.047))
            tilt_excess = max(0.0, max(abs(roll), abs(pitch)) - 0.7 * max_tilt)
            shaped -= self.tilt_penalty_scale * tilt_excess

        if info.get("success"):
            shaped += self.landing_bonus

        return obs, shaped, terminated, truncated, info


class StateNoiseWrapper(gym.ObservationWrapper):
    """Mild Gaussian noise on the state vector for domain randomization."""

    def __init__(self, env: gym.Env, *, std: float = 0.02):
        super().__init__(env)
        self.std = std

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(observation)
        state = np.asarray(out["state"], dtype=np.float32).copy()
        if self.std > 0:
            state += np.random.normal(0.0, self.std, size=state.shape).astype(np.float32)
        out["state"] = state
        return out


class LandingFocusedWrapper(ShapedProgressWrapper):
    """
    Landing-specialist reward shaping for short near-goal episodes.

    Adds exponential time pressure and stronger pad-proximity / slow-down signals.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        progress_scale: float = 0.18,
        proximity_scale: float = 0.12,
        landing_bonus: float = 25.0,
        time_penalty: float = 0.0008,
        late_time_scale: float = 4.0,
    ):
        super().__init__(
            env,
            progress_scale=progress_scale,
            proximity_scale=proximity_scale,
            landing_bonus=landing_bonus,
            time_penalty=time_penalty,
        )
        self.late_time_scale = late_time_scale
        self._horizon_steps: Optional[int] = None
        self._step_idx = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        horizon_sec = float(getattr(getattr(base, "task", None), "horizon", 60.0))
        sim_dt = float(getattr(base, "_sim_dt", 1.0 / 50.0))
        self._horizon_steps = max(1, int(horizon_sec / sim_dt))
        self._step_idx = 0
        self._prev_dist = float(info.get("distance_to_goal", 0.0))
        goal_z = float(getattr(getattr(base, "task", None), "goal", (0, 0, 0))[2])
        self._prev_alt_err = abs(float(obs["state"][2]) - goal_z)
        return obs, info

    def step(self, action):
        self._step_idx += 1
        obs, reward, terminated, truncated, info = super().step(action)
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        dist = float(info.get("distance_to_goal", 0.0))

        if self._horizon_steps and self._step_idx > self._horizon_steps // 2:
            frac = (self._step_idx - self._horizon_steps // 2) / max(1, self._horizon_steps // 2)
            reward -= self.late_time_scale * self.time_penalty * (1.5**frac)

        if dist < 3.0:
            reward += 0.08 * (3.0 - dist) / 3.0
            if float(raw[3]) < 0.4:
                reward += 0.04 * (0.4 - float(raw[3]))

        state = np.asarray(obs["state"], dtype=np.float64)
        base = self.env.unwrapped
        drone = getattr(base, "_getDroneStateVector", lambda _: None)(0)
        if drone is not None and dist < 8.0:
            vz = float(drone[12])
            if abs(vz) < 0.8:
                reward += 0.01
            # Soft-landing: penalize hard descents near the pad.
            if dist < 3.0 and abs(vz) > 0.5:
                reward -= 0.05 * (abs(vz) - 0.5)

        if info.get("collision"):
            reward -= 5.0

        return obs, reward, terminated, truncated, info


class EpisodeScoreWrapper(gym.Wrapper):
    """Track the latest validator-style score in info for callbacks."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["validator_score"] = float(info.get("score", 0.0))
        return obs, reward, terminated, truncated, info
