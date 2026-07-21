"""Spiral disk-search pilot: find the pad inside the noisy GPS search radius."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

MAX_RAY_M = 20.0


class PilotPhase(Enum):
    CRUISE = auto()
    SEARCH = auto()
    LAND = auto()


def _state_parts(observation: dict):
    state = np.asarray(observation["state"], dtype=np.float64)
    pos = state[0:3]
    search_rel = state[-3:].astype(np.float64)
    alt_ray = float(state[-4]) * MAX_RAY_M if state.shape[0] >= 4 else 5.0
    horiz = search_rel.copy()
    horiz[2] = 0.0
    horiz_dist = float(np.linalg.norm(horiz))
    return state, pos, search_rel, horiz, horiz_dist, alt_ray


def soft_land_action(
    observation: dict,
    *,
    target_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Gentle vertical touchdown suitable for LANDING_MAX_VZ / tilt gates."""
    state, pos, search_rel, horiz, horiz_dist, alt_ray = _state_parts(observation)

    if target_xy is not None:
        err = np.asarray(target_xy[:2], dtype=np.float64) - pos[:2]
        err_dist = float(np.linalg.norm(err))
        if err_dist > 1e-3:
            horiz_dir = np.array([err[0] / err_dist, err[1] / err_dist, 0.0])
            horiz_dist = err_dist
        else:
            horiz_dir = np.zeros(3, dtype=np.float64)
            horiz_dist = 0.0
    else:
        if horiz_dist > 1e-3:
            horiz_dir = np.array([horiz[0] / horiz_dist, horiz[1] / horiz_dist, 0.0])
        else:
            horiz_dir = np.zeros(3, dtype=np.float64)

    direction = np.zeros(3, dtype=np.float64)
    if horiz_dist > 0.35:
        scale = float(np.clip(0.35 * min(horiz_dist, 3.0) / 3.0, 0.12, 0.35))
        direction[0] = scale * horiz_dir[0]
        direction[1] = scale * horiz_dir[1]

    # Slow descent — critical for static-platform landing criteria.
    if alt_ray > 2.0:
        direction[2] = -0.18
        thrust = 0.26
    elif alt_ray > 1.0:
        direction[2] = -0.12
        thrust = 0.18
    else:
        direction[2] = -0.06
        thrust = 0.12

    return np.array(
        [direction[0], direction[1], direction[2], thrust, 0.0],
        dtype=np.float64,
    )


def _depth_pad_score(observation: dict) -> float:
    """
    Higher score ⇒ elevated object under the drone (landing pad cue).

    Compares center depth (closer = smaller normalized depth? wait: depth is
    normalized distance 0..1 for 0.5..20m, so closer ⇒ smaller value).
    """
    depth = np.asarray(observation.get("depth"), dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size < 16:
        return 0.0
    h, w = depth.shape
    cy, cx = h // 2, w // 2
    r = max(4, min(h, w) // 10)
    center = depth[cy - r : cy + r, cx - r : cx + r].mean()
    # Outer ring sample
    border = np.concatenate(
        [
            depth[0:r, :].ravel(),
            depth[-r:, :].ravel(),
            depth[:, 0:r].ravel(),
            depth[:, -r:].ravel(),
        ]
    ).mean()
    # Closer center than border ⇒ positive score (meters-ish via *20).
    return float(max(0.0, (border - center) * MAX_RAY_M))


class SearchLandPilot:
    """
    Stateful three-phase pilot for Swarm GPS noise:

      CRUISE  — fly horizontally to the noisy search centre
      SEARCH  — Archimedean spiral inside the search disk; detect pad via
                downward altitude-ray discontinuity (elevated platform)
      LAND    — soft touchdown at the locked pad XY

    This is the reliable Stage-0 approach: RL alone cannot recover from
    GPS offset; disk search finds the real pad.
    """

    def __init__(
        self,
        *,
        cruise_speed: float = 0.42,
        search_enter_m: float = 12.0,
        hover_clearance_m: float = 2.8,
        spiral_pitch_m: float = 1.8,
        spiral_step_rad: float = 0.18,
        max_spiral_radius_m: float = 18.0,
        pad_drop_m: float = 0.25,
        pad_confirm_steps: int = 4,
        cruise_horiz_scale: float = 0.65,
    ):
        self.cruise_speed = cruise_speed
        self.search_enter_m = search_enter_m
        self.hover_clearance_m = hover_clearance_m
        self.spiral_pitch_m = spiral_pitch_m
        self.spiral_step_rad = spiral_step_rad
        self.max_spiral_radius_m = max_spiral_radius_m
        self.pad_drop_m = pad_drop_m
        self.pad_confirm_steps = pad_confirm_steps
        self.cruise_horiz_scale = cruise_horiz_scale
        self.reset()

    def reset(self) -> None:
        self.phase = PilotPhase.CRUISE
        self._theta = 0.0
        self._baseline_alt: float | None = None
        self._pad_hits = 0
        self._pad_xy: np.ndarray | None = None
        self._search_center_xy: np.ndarray | None = None
        self._hover_z: float | None = None
        self._search_steps = 0
        self._best_pad_score = 0.0
        self._best_pad_xy: np.ndarray | None = None
        self._spiral_laps = 0

    def act(self, observation: dict) -> np.ndarray:
        state, pos, search_rel, horiz, horiz_dist, alt_ray = _state_parts(observation)
        search_xy = pos[:2] + search_rel[:2]
        search_z = float(pos[2] + search_rel[2])

        if self.phase == PilotPhase.CRUISE:
            if horiz_dist <= self.search_enter_m:
                self.phase = PilotPhase.SEARCH
                self._theta = 0.0
                self._baseline_alt = None
                self._pad_hits = 0
                self._pad_xy = None
                self._search_center_xy = search_xy.copy()
                self._hover_z = max(float(pos[2]), search_z + self.hover_clearance_m)
                self._search_steps = 0
                self._best_pad_score = 0.0
                self._best_pad_xy = None
                self._spiral_laps = 0
            else:
                return self._cruise(horiz, horiz_dist, pos[2], search_z)

        if self.phase == PilotPhase.SEARCH:
            return self._search(observation, pos, search_xy, search_z, alt_ray, horiz_dist)

        # LAND
        return soft_land_action(observation, target_xy=self._pad_xy)

    def _cruise(
        self,
        horiz: np.ndarray,
        horiz_dist: float,
        z: float,
        search_z: float,
    ) -> np.ndarray:
        # Keep horizontal command soft — aggressive unit vectors tip the drone past MAX_TILT.
        direction = np.zeros(3, dtype=np.float64)
        if horiz_dist > 1e-3:
            direction[0] = self.cruise_horiz_scale * horiz[0] / horiz_dist
            direction[1] = self.cruise_horiz_scale * horiz[1] / horiz_dist
        target_z = max(z, search_z + self.hover_clearance_m)
        # Prefer holding altitude during long cruise (avoid diving into start platform).
        direction[2] = float(np.clip((target_z - z) / 12.0, -0.15, 0.25))
        thrust = float(np.clip(self.cruise_speed, 0.30, 0.50))
        return np.array(
            [direction[0], direction[1], direction[2], thrust, 0.0],
            dtype=np.float64,
        )

    def _search(
        self,
        observation: dict,
        pos: np.ndarray,
        search_xy: np.ndarray,
        search_z: float,
        alt_ray: float,
        horiz_dist: float,
    ) -> np.ndarray:
        self._search_steps += 1
        if self._search_center_xy is None:
            self._search_center_xy = search_xy.copy()
        else:
            self._search_center_xy = 0.95 * self._search_center_xy + 0.05 * search_xy

        if self._hover_z is None:
            self._hover_z = max(float(pos[2]), search_z + self.hover_clearance_m)

        if self._baseline_alt is None:
            self._baseline_alt = alt_ray
        else:
            self._baseline_alt = 0.99 * self._baseline_alt + 0.01 * max(self._baseline_alt, alt_ray)

        depth_score = _depth_pad_score(observation)
        alt_drop = 0.0 if self._baseline_alt is None else max(0.0, self._baseline_alt - alt_ray)
        pad_score = alt_drop + 0.6 * depth_score

        if pad_score > self._best_pad_score and pad_score > 0.25:
            self._best_pad_score = pad_score
            self._best_pad_xy = pos[:2].copy()

        # Confirm pad when altitude drops OR strong depth cue persists.
        if pad_score > self.pad_drop_m or depth_score > 0.8:
            self._pad_hits += 1
            if self._pad_hits >= self.pad_confirm_steps:
                self._pad_xy = pos[:2].copy()
                self.phase = PilotPhase.LAND
                return soft_land_action(observation, target_xy=self._pad_xy)
        else:
            self._pad_hits = max(0, self._pad_hits - 1)

        # After enough spiral coverage, land on best cue or search centre.
        radius = self.spiral_pitch_m * self._theta / (2.0 * np.pi)
        if radius > self.max_spiral_radius_m:
            self._theta = 0.0
            radius = 0.0
            self._spiral_laps += 1
            self.spiral_pitch_m = max(1.0, self.spiral_pitch_m * 0.85)
            if self._spiral_laps >= 1 and self._best_pad_xy is not None:
                self._pad_xy = self._best_pad_xy
                self.phase = PilotPhase.LAND
                return soft_land_action(observation, target_xy=self._pad_xy)
            if self._spiral_laps >= 2:
                # Last resort: soft-land toward GPS search centre.
                self._pad_xy = self._search_center_xy.copy() if self._search_center_xy is not None else pos[:2].copy()
                self.phase = PilotPhase.LAND
                return soft_land_action(observation, target_xy=self._pad_xy)

        # If already very close to search centre with a cue, commit.
        if horiz_dist < 2.0 and self._search_steps > 80 and self._best_pad_xy is not None:
            self._pad_xy = self._best_pad_xy
            self.phase = PilotPhase.LAND
            return soft_land_action(observation, target_xy=self._pad_xy)

        wp = self._search_center_xy + radius * np.array(
            [np.cos(self._theta), np.sin(self._theta)],
            dtype=np.float64,
        )
        self._theta += self.spiral_step_rad

        err = wp - pos[:2]
        err_dist = float(np.linalg.norm(err))
        direction = np.zeros(3, dtype=np.float64)
        if err_dist > 1e-3:
            scale = 0.45 if err_dist > 2.0 else 0.30
            direction[0] = scale * err[0] / err_dist
            direction[1] = scale * err[1] / err_dist

        direction[2] = float(np.clip((self._hover_z - float(pos[2])) / 6.0, -0.20, 0.25))
        thrust = 0.32 if err_dist > 1.5 else 0.26
        return np.array(
            [direction[0], direction[1], direction[2], thrust, 0.0],
            dtype=np.float64,
        )


def heuristic_action(observation: dict, *, speed: float = 0.55) -> np.ndarray:
    """
    Stateless fallback (used only for one-shot demo collectors without a pilot object).
    Prefer SearchLandPilot for real episodes.
    """
    _, _, _, horiz, horiz_dist, _ = _state_parts(observation)
    if horiz_dist < 3.0:
        return soft_land_action(observation)
    if horiz_dist > 1e-3:
        direction = np.array([horiz[0] / horiz_dist, horiz[1] / horiz_dist, 0.0])
    else:
        direction = np.zeros(3, dtype=np.float64)
    thrust = float(np.clip(speed, 0.35, 0.65))
    return np.array(
        [direction[0], direction[1], direction[2], thrust, 0.0],
        dtype=np.float64,
    )
