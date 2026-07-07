"""Validator-faithful action preprocessing (matches docker RPC evaluator)."""

from __future__ import annotations

import numpy as np

from gym_pybullet_drones.utils.enums import ActionType
from swarm.constants import SPEED_LIMIT


def prepare_swarm_action(action, env) -> np.ndarray:
    """
    Convert a model action to the shape/format expected by MovingDroneAviary.step().

    Mirrors swarm/validator/docker/docker_evaluator_parts/rpc.py.
    """
    lo = env.action_space.low.flatten()
    hi = env.action_space.high.flatten()

    raw = np.nan_to_num(
        np.asarray(action, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if raw.size != 5:
        raw = np.zeros(5, dtype=np.float32)

    act = np.clip(raw, lo, hi)

    if hasattr(env, "ACT_TYPE") and getattr(env, "ACT_TYPE", None) == ActionType.VEL:
        n = max(float(np.linalg.norm(act[:3])), 1e-6)
        scale = min(1.0, float(SPEED_LIMIT) / n)
        act[:3] *= scale
        act = np.clip(act, lo, hi)

    return act.reshape(1, -1)
