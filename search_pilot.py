"""Spiral disk-search pilot: find the pad inside the noisy GPS search radius."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

MAX_RAY_M = 20.0


class PilotPhase(Enum):
    CRUISE = auto()
    SEARCH = auto()
    REFINE = auto()
    LAND = auto()


def _state_parts(observation: dict):
    state = np.asarray(observation["state"], dtype=np.float64)
    pos = state[0:3]
    vel = state[10:13] if state.shape[0] >= 13 else np.zeros(3, dtype=np.float64)
    search_rel = state[-3:].astype(np.float64)
    alt_ray = float(state[-4]) * MAX_RAY_M if state.shape[0] >= 4 else 5.0
    horiz = search_rel.copy()
    horiz[2] = 0.0
    horiz_dist = float(np.linalg.norm(horiz))
    return state, pos, vel, search_rel, horiz, horiz_dist, alt_ray


def soft_land_action(
    observation: dict,
    *,
    target_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Gentle vertical touchdown suitable for LANDING_MAX_VZ / tilt gates."""
    state, pos, vel, search_rel, horiz, horiz_dist, alt_ray = _state_parts(observation)

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
    # Translate toward target while always sinking — holding altitude forever
    # when the lock is wrong prevents any landing.
    if horiz_dist > 0.80:
        scale = float(np.clip(0.28 * min(horiz_dist, 3.0) / 3.0, 0.10, 0.28))
        direction[0] = scale * horiz_dir[0]
        direction[1] = scale * horiz_dir[1]
    elif horiz_dist > 0.35:
        direction[0] = 0.10 * horiz_dir[0]
        direction[1] = 0.10 * horiz_dir[1]
    # else: kill XY — already over the pad; any tilt here causes edge collisions.

    if alt_ray > 2.5:
        direction[2] = -0.12
        thrust = 0.22
    elif alt_ray > 1.2:
        direction[2] = -0.07
        thrust = 0.15
    elif alt_ray > 0.55:
        direction[2] = -0.035
        thrust = 0.10
    elif alt_ray > 0.25:
        # Near contact: almost hover to satisfy LANDING_STABLE_SEC.
        direction[0] *= 0.3
        direction[1] *= 0.3
        direction[2] = -0.015
        thrust = 0.06
    else:
        direction[0] = 0.0
        direction[1] = 0.0
        direction[2] = -0.008
        thrust = 0.05

    # Strong velocity damping — LANDING_MAX_VXY_REL is 0.6 m/s.
    direction[0] -= float(np.clip(0.25 * vel[0], -0.20, 0.20))
    direction[1] -= float(np.clip(0.25 * vel[1], -0.20, 0.20))
    direction[2] -= float(np.clip(0.12 * vel[2], -0.06, 0.06))

    xy_norm = float(np.linalg.norm(direction[:2]))
    if xy_norm > 0.28:
        direction[:2] *= 0.28 / xy_norm

    return np.array(
        [direction[0], direction[1], direction[2], float(thrust), 0.0],
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

    Optional pad_estimator (supervised depth→pad XY) overrides fragile
    altitude-peak lock when temporally consistent.
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
        pad_estimator=None,
        pad_estimator_device: str = "cpu",
        pad_est_confirm: int = 4,
        pad_est_max_center_m: float = 16.0,
        pad_est_stability_m: float = 1.2,
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
        self.pad_estimator = pad_estimator
        self.pad_estimator_device = pad_estimator_device
        self.pad_est_confirm = pad_est_confirm
        self.pad_est_max_center_m = pad_est_max_center_m
        self.pad_est_stability_m = pad_est_stability_m
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
        self._refine_theta = 0.0
        self._refine_steps = 0
        self._best_surface_z = -1e9
        self._best_surface_xy: np.ndarray | None = None
        self._refine_targets: list[np.ndarray] = []
        self._refine_ti = 0
        self._terrain_z: float | None = None
        self._surface_samples = 0
        self._est_ema: np.ndarray | None = None
        self._est_hits = 0
        self._est_lock_xy: np.ndarray | None = None

    def _update_pad_estimator(self, observation: dict, pos: np.ndarray) -> np.ndarray | None:
        """EMA-smoothed pad XY from learned estimator; None if unusable."""
        if self.pad_estimator is None:
            return None
        try:
            from RL.pad_estimator import predict_pad_xy

            pred = predict_pad_xy(
                self.pad_estimator,
                observation,
                device=self.pad_estimator_device,
            )
        except Exception:
            return None
        if pred is None or not np.isfinite(pred).all():
            return None
        pred = np.asarray(pred[:2], dtype=np.float64)

        # Reject predictions far outside the GPS search disk.
        if self._search_center_xy is not None:
            d_center = float(np.linalg.norm(pred - self._search_center_xy))
            if d_center > self.pad_est_max_center_m:
                self._est_hits = max(0, self._est_hits - 1)
                return None

        if self._est_ema is None:
            self._est_ema = pred.copy()
            self._est_hits = 1
        else:
            jump = float(np.linalg.norm(pred - self._est_ema))
            self._est_ema = 0.65 * self._est_ema + 0.35 * pred
            if jump <= self.pad_est_stability_m:
                self._est_hits += 1
            else:
                self._est_hits = max(0, self._est_hits - 1)

        if self._est_hits >= self.pad_est_confirm:
            self._est_lock_xy = self._est_ema.copy()
            return self._est_lock_xy
        return None

    def _commit_land(self, observation: dict, pad_xy: np.ndarray) -> np.ndarray:
        self._pad_xy = np.asarray(pad_xy[:2], dtype=np.float64).copy()
        self._finalize_pad_xy()
        self.phase = PilotPhase.LAND
        self._land_settle = 0
        self._land_peak_z = -1e9
        self._land_peak_xy = None
        return soft_land_action(observation, target_xy=self._pad_xy)

    def act(self, observation: dict) -> np.ndarray:
        state, pos, vel, search_rel, horiz, horiz_dist, alt_ray = _state_parts(observation)
        search_xy = pos[:2] + search_rel[:2]
        search_z = float(pos[2] + search_rel[2])

        if self.phase == PilotPhase.CRUISE:
            if horiz_dist <= self.search_enter_m:
                self._pad_xy = None
                self._search_center_xy = search_xy.copy()
                self._hover_z = max(float(pos[2]), search_z + self.hover_clearance_m)
                self._search_steps = 0
                self._best_pad_score = 0.0
                self._best_pad_xy = None
                self._spiral_laps = 0
                self._terrain_z = None
                self._surface_samples = 0
                self._best_surface_z = -1e9
                self._best_surface_xy = None
                self._est_ema = None
                self._est_hits = 0
                self._est_lock_xy = None
                # Skip long spiral when GPS is already close — spend budget on refine+land.
                if horiz_dist <= 6.0:
                    self._begin_refine()
                else:
                    self.phase = PilotPhase.SEARCH
                    self._theta = 0.0
                    self._baseline_alt = None
                    self._pad_hits = 0
            else:
                return self._cruise(horiz, horiz_dist, pos[2], search_z)

        if self.phase == PilotPhase.SEARCH:
            return self._search(observation, pos, search_xy, search_z, alt_ray, horiz_dist)

        if self.phase == PilotPhase.REFINE:
            return self._refine(observation, pos, alt_ray)

        # LAND with on-pad surface tracking — if we slip off the elevated pad, steer back.
        # Keep refining lock with estimator while descending if confident.
        est = self._update_pad_estimator(observation, pos)
        if est is not None and alt_ray > 0.8:
            self._pad_xy = 0.7 * np.asarray(
                self._pad_xy if self._pad_xy is not None else est
            ) + 0.3 * est

        surface_z = float(pos[2] - alt_ray)
        if not hasattr(self, "_land_peak_z"):
            self._land_peak_z = -1e9
            self._land_peak_xy = None
        if np.isfinite(surface_z) and surface_z > self._land_peak_z:
            self._land_peak_z = surface_z
            self._land_peak_xy = pos[:2].copy()
        elif (
            self._land_peak_xy is not None
            and np.isfinite(surface_z)
            and surface_z < self._land_peak_z - 0.12
            and alt_ray < 3.0
        ):
            self._pad_xy = 0.7 * np.asarray(
                self._pad_xy if self._pad_xy is not None else self._land_peak_xy
            ) + 0.3 * self._land_peak_xy
        return soft_land_action(observation, target_xy=self._pad_xy)
    def _finalize_pad_xy(self) -> None:
        """Nudge lock away from GPS centre through the elevated peak (toward pad core)."""
        if self._pad_xy is None or self._search_center_xy is None:
            return
        delta = self._pad_xy - self._search_center_xy
        d = float(np.linalg.norm(delta))
        if d > 0.15:
            # Mild overshoot — GOAL_TOL is 0.51 m so lock must be well inside the pad.
            self._pad_xy = self._pad_xy + 0.30 * delta / d

    def _begin_refine(self, center_xy: np.ndarray | None = None) -> None:
        """Polar scan around GPS / estimator centre to peak-pick elevated pad surface."""
        self.phase = PilotPhase.REFINE
        self._refine_theta = 0.0
        self._refine_steps = 0
        self._refine_ti = 0
        self._dwell = 0
        if center_xy is not None:
            c = np.asarray(center_xy[:2], dtype=np.float64).copy()
        else:
            c = self._search_center_xy.copy() if self._search_center_xy is not None else None
        self._best_surface_z = -1e9
        self._best_surface_xy = c.copy() if c is not None else None
        self._refine_targets = []
        self._refine_center = c
        self._refine_zs = []
        self._max_surf_z = -1e9
        self._max_surf_xy = None
        self._elev_xy = []
        self._elev_z = []
        self._did_micro = False
        if c is not None:
            self._refine_targets = [c.copy()]
            for r in (0.6, 1.2, 2.0):
                for k in range(6):
                    ang = 2.0 * np.pi * k / 6.0
                    self._refine_targets.append(
                        c + r * np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
                    )

    def _refine(self, observation: dict, pos: np.ndarray, alt_ray: float) -> np.ndarray:
        self._refine_steps += 1
        # Prefer confident estimator lock → soft-land immediately (skip remaining scan).
        est = self._update_pad_estimator(observation, pos)
        if est is not None and self._refine_steps >= 8:
            return self._commit_land(observation, est)

        center = getattr(self, "_refine_center", None)
        if center is None:
            center = self._search_center_xy if self._search_center_xy is not None else pos[:2]
        surface_z = float(pos[2] - alt_ray)
        d_center = float(np.linalg.norm(pos[:2] - center))
        if np.isfinite(surface_z) and d_center <= 3.5:
            self._refine_zs.append(surface_z)
            if not hasattr(self, "_elev_xy") or self._elev_xy is None:
                self._elev_xy = []
                self._elev_z = []
            self._elev_xy.append(pos[:2].copy())
            self._elev_z.append(surface_z)
            if surface_z > self._max_surf_z:
                self._max_surf_z = surface_z
                self._max_surf_xy = pos[:2].copy()

        terrain = (
            float(np.percentile(self._refine_zs, 30))
            if len(self._refine_zs) >= 12
            else (self._terrain_z if self._terrain_z is not None else surface_z - 0.5)
        )
        if (
            np.isfinite(surface_z)
            and d_center <= 3.5
            and surface_z >= terrain + 0.12
            and surface_z > self._best_surface_z
        ):
            self._best_surface_z = surface_z
            self._best_surface_xy = pos[:2].copy()

        if not self._refine_targets:
            pad = self._est_lock_xy if self._est_lock_xy is not None else center
            return self._commit_land(observation, pad)

        if self._refine_ti >= len(self._refine_targets):
            # Prefer estimator lock; else elevated peak; else GPS centre.
            if self._est_lock_xy is not None:
                return self._commit_land(observation, self._est_lock_xy)
            peak = self._max_surf_z if self._max_surf_z > -1e8 else self._best_surface_z
            if getattr(self, "_elev_z", None) and peak > -1e8:
                xs = [xy for xy, z in zip(self._elev_xy, self._elev_z) if z >= peak - 0.15]
                if xs:
                    self._pad_xy = np.mean(np.stack(xs, axis=0), axis=0)
                    self._best_surface_z = peak
                elif self._max_surf_xy is not None:
                    self._pad_xy = self._max_surf_xy
                    self._best_surface_z = peak
                else:
                    self._pad_xy = np.asarray(center, dtype=np.float64).copy()
            elif self._best_surface_xy is not None and self._best_surface_z > -1e8:
                self._pad_xy = self._best_surface_xy
            else:
                self._pad_xy = np.asarray(center, dtype=np.float64).copy()
            return self._commit_land(observation, self._pad_xy)

        wp = self._refine_targets[self._refine_ti]
        # Bias current waypoint toward live estimator EMA when available.
        if self._est_ema is not None:
            wp = 0.55 * wp + 0.45 * self._est_ema
        err = wp - pos[:2]
        err_dist = float(np.linalg.norm(err))
        if err_dist < 0.40:
            self._dwell += 1
        else:
            self._dwell = 0
        if self._dwell >= 8 or self._refine_steps > (self._refine_ti + 1) * 30:
            self._refine_ti += 1
            self._dwell = 0

        direction = np.zeros(3, dtype=np.float64)
        if err_dist > 1e-3:
            scale = 0.28 if err_dist > 1.0 else 0.16
            direction[0] = scale * err[0] / err_dist
            direction[1] = scale * err[1] / err_dist
        hover = (self._hover_z - 0.8) if self._hover_z is not None else None
        if hover is not None:
            direction[2] = float(np.clip((hover - float(pos[2])) / 6.0, -0.15, 0.22))
        return np.array(
            [direction[0], direction[1], direction[2], 0.28, 0.0],
            dtype=np.float64,
        )

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

        # Learned pad lock — primary unlock for borderline GPS noise.
        est = self._update_pad_estimator(observation, pos)
        if est is not None and self._search_steps > 20:
            self._begin_refine(center_xy=est)
            return self._refine(observation, pos, alt_ray)

        if self._baseline_alt is None:
            self._baseline_alt = alt_ray
        else:
            self._baseline_alt = 0.99 * self._baseline_alt + 0.01 * max(self._baseline_alt, alt_ray)

        depth_score = _depth_pad_score(observation)
        alt_drop = 0.0 if self._baseline_alt is None else max(0.0, self._baseline_alt - alt_ray)
        pad_score = alt_drop + 0.6 * depth_score

        # Track elevated surface height across the spiral (true pad cue).
        surface_z = float(pos[2] - alt_ray)
        self._surface_samples += 1
        if self._terrain_z is None:
            self._terrain_z = surface_z
        else:
            # Slow low-pass toward lower surfaces (= ground).
            self._terrain_z = 0.995 * self._terrain_z + 0.005 * min(self._terrain_z, surface_z)
        elev = surface_z - self._terrain_z
        # GPS prior: when noise is small this helps; when large, still keep global max.
        gps_w = 1.0
        d = 0.0
        if self._search_center_xy is not None:
            d = float(np.linalg.norm(pos[:2] - self._search_center_xy))
            gps_w = float(np.exp(-0.5 * (d / 8.0) ** 2))
        surf_score = elev * (0.15 + 0.85 * gps_w)
        # Only accept spiral surface peaks reasonably near the GPS centre.
        if (
            surf_score > self._best_pad_score
            and elev > 0.18
            and (self._search_center_xy is None or d <= 6.0)
        ):
            self._best_pad_score = surf_score
            self._best_pad_xy = pos[:2].copy()
            self._best_surface_z = surface_z
            self._best_surface_xy = pos[:2].copy()

        if pad_score > self._best_pad_score and pad_score > 0.25:
            # Keep legacy depth/alt score only if stronger than surface score.
            pass

        # Once close to the GPS centre, refine immediately (save episode budget).
        if horiz_dist <= 5.5 and self._search_steps > 15:
            center = self._est_ema if self._est_ema is not None else None
            self._begin_refine(center_xy=center)
            return self._refine(observation, pos, alt_ray)

        # Confirm pad when elevated surface persists near the GPS centre.
        near_center = (
            self._search_center_xy is not None
            and float(np.linalg.norm(pos[:2] - self._search_center_xy)) < 5.0
        )
        if elev >= 0.28 and near_center and self._search_steps > 30:
            self._pad_hits += 1
            if self._pad_hits >= self.pad_confirm_steps:
                center = self._est_ema if self._est_ema is not None else pos[:2]
                self._begin_refine(center_xy=center)
                return self._refine(observation, pos, alt_ray)
        else:
            self._pad_hits = max(0, self._pad_hits - 1)

        # After enough spiral coverage, land on best cue or search centre.
        radius = self.spiral_pitch_m * self._theta / (2.0 * np.pi)
        if radius > self.max_spiral_radius_m:
            self._theta = 0.0
            radius = 0.0
            self._spiral_laps += 1
            self.spiral_pitch_m = max(1.0, self.spiral_pitch_m * 0.85)
            center = self._est_lock_xy or self._est_ema or self._best_pad_xy
            if self._spiral_laps >= 1 and center is not None:
                self._begin_refine(center_xy=center)
                return self._refine(observation, pos, alt_ray)
            if self._spiral_laps >= 2:
                self._begin_refine(center_xy=center)
                return self._refine(observation, pos, alt_ray)

        # If already very close to search centre with a cue, commit.
        if horiz_dist < 2.0 and self._search_steps > 80 and self._best_pad_xy is not None:
            center = self._est_ema if self._est_ema is not None else self._best_pad_xy
            self._begin_refine(center_xy=center)
            return self._refine(observation, pos, alt_ray)

        # Spiral waypoint; bias toward live estimator when available.
        wp = self._search_center_xy + radius * np.array(
            [np.cos(self._theta), np.sin(self._theta)],
            dtype=np.float64,
        )
        if self._est_ema is not None and self._est_hits >= 2:
            wp = 0.5 * wp + 0.5 * self._est_ema
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
    _, _, _, _, horiz, horiz_dist, _ = _state_parts(observation)
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
