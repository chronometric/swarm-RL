"""Search-area-guided pilot with cruise → approach → land phases."""

from __future__ import annotations

import numpy as np


def heuristic_action(observation: dict, *, speed: float = 0.55) -> np.ndarray:
    """
    Three-phase pilot:
      1. Cruise horizontally toward the GPS search area (stay at altitude).
      2. Approach: when horizontally close, descend toward goal altitude.
      3. Land: slow down inside the landing radius and align vertically.
    """
    state = np.asarray(observation["state"], dtype=np.float64)
    pos = state[0:3]
    search_rel = state[-3:].astype(np.float64)

    horiz_rel = search_rel.copy()
    horiz_rel[2] = 0.0
    horiz_dist = float(np.linalg.norm(horiz_rel))

    # search_rel points to noisy search centre; goal is within search_radius of it.
    if horiz_dist > 1e-3:
        horiz_dir = horiz_rel / horiz_dist
    else:
        horiz_dir = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    search_center_z = float(pos[2] + search_rel[2])
    alt_above_search = float(pos[2] - search_center_z)

    # Phase 3 — very close: slow final approach + descend
    if horiz_dist < 3.0:
        direction = search_rel / max(float(np.linalg.norm(search_rel)), 1e-3)
        thrust = float(np.clip(0.25 + 0.15 * horiz_dist, 0.2, 0.45))
    # Phase 2 — within ~25 m horizontally: add descent
    elif horiz_dist < 25.0:
        direction = horiz_dir.copy()
        # Blend in vertical component toward search-centre altitude
        if alt_above_search > 1.5:
            direction[2] = float(np.clip(-alt_above_search / 20.0, -0.6, 0.0))
        else:
            direction[2] = float(np.clip(search_rel[2] / max(horiz_dist, 1.0), -0.5, 0.3))
        norm = float(np.linalg.norm(direction))
        if norm > 1e-3:
            direction /= norm
        thrust = float(np.clip(0.4 + 0.2 * (horiz_dist / 25.0), 0.35, 0.6))
    # Phase 1 — cruise: horizontal only (avoid driving into start platform)
    else:
        direction = horiz_dir
        direction[2] = 0.0
        thrust = float(np.clip(speed, 0.35, 0.65))

    return np.array(
        [[direction[0], direction[1], direction[2], thrust, 0.0]],
        dtype=np.float32,
    )
