"""Search-area-guided pilot — re-exports spiral disk search (reliable Stage 0 path)."""

from __future__ import annotations

import numpy as np

from RL.search_pilot import SearchLandPilot, heuristic_action, soft_land_action

__all__ = ["SearchLandPilot", "heuristic_action", "soft_land_action", "heuristic_action_legacy"]


def heuristic_action_legacy(observation: dict, *, speed: float = 0.55) -> np.ndarray:
    """Old non-search heuristic kept for reference / A-B tests."""
    state = np.asarray(observation["state"], dtype=np.float64)
    pos = state[0:3]
    search_rel = state[-3:].astype(np.float64)
    horiz_rel = search_rel.copy()
    horiz_rel[2] = 0.0
    horiz_dist = float(np.linalg.norm(horiz_rel))
    if horiz_dist > 1e-3:
        horiz_dir = horiz_rel / horiz_dist
    else:
        horiz_dir = np.zeros(3, dtype=np.float64)
    search_center_z = float(pos[2] + search_rel[2])
    alt_above_search = float(pos[2] - search_center_z)
    if horiz_dist < 2.5:
        return soft_land_action(observation)
    if horiz_dist < 8.0:
        direction = search_rel / max(float(np.linalg.norm(search_rel)), 1e-3)
        if alt_above_search > 1.0:
            direction[2] = float(np.clip(-alt_above_search / 15.0, -0.5, 0.0))
        thrust = float(np.clip(0.22 + 0.04 * horiz_dist, 0.22, 0.42))
    elif horiz_dist < 25.0:
        direction = horiz_dir.copy()
        if alt_above_search > 1.5:
            direction[2] = float(np.clip(-alt_above_search / 20.0, -0.6, 0.0))
        else:
            direction[2] = float(np.clip(search_rel[2] / max(horiz_dist, 1.0), -0.5, 0.3))
        norm = float(np.linalg.norm(direction))
        if norm > 1e-3:
            direction /= norm
        thrust = float(np.clip(0.4 + 0.2 * (horiz_dist / 25.0), 0.35, 0.6))
    else:
        direction = horiz_dir.copy()
        direction[2] = 0.0
        thrust = float(np.clip(speed, 0.35, 0.65))
    return np.array(
        [[direction[0], direction[1], direction[2], thrust, 0.0]],
        dtype=np.float32,
    )
