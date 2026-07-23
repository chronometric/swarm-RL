"""Fast local validation using the same env + flight_reward as validators."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from RL.action_utils import prepare_swarm_action
from RL.env_utils import CHALLENGE_NAMES, DEFAULT_VAL_SEEDS, OPEN_BENCHMARK_SEEDS, validation_tasks
from RL.hybrid_controller import HybridController, HybridConfig
from RL.policy_net import load_swarm_depth_cnn_class
from swarm.utils.env_factory import make_env


def load_recurrent_model(path: str | Path):
    """Load RecurrentPPO with SwarmDepthCNN custom extractor."""
    from sb3_contrib import RecurrentPPO

    SwarmDepthCNN = load_swarm_depth_cnn_class()
    return RecurrentPPO.load(str(path), custom_objects={"SwarmDepthCNN": SwarmDepthCNN})


@dataclass
class ValidationResult:
    mean_score: float
    success_rate: float
    per_type: dict[str, float] = field(default_factory=dict)
    per_type_success: dict[str, float] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_distance_to_goal(self) -> float:
        if not self.episodes:
            return float("inf")
        return float(np.mean([ep["distance_to_goal"] for ep in self.episodes]))

    def summary_line(self) -> str:
        parts = [
            f"mean={self.mean_score:.4f}",
            f"success={self.success_rate:.1%}",
            f"dist={self.mean_distance_to_goal:.1f}m",
        ]
        for name, score in sorted(self.per_type.items()):
            parts.append(f"{name}={score:.3f}")
        return " | ".join(parts)


def _init_recurrent_state(model) -> tuple[Any, np.ndarray]:
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    return lstm_states, episode_start


def rollout_episode(model, task, *, gui: bool = False, deterministic: bool = True) -> dict[str, Any]:
    """Run one episode and return validator-style metrics."""
    env = make_env(task, gui=gui)
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        lstm_states, episode_start = _init_recurrent_state(model)
        done = False

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start = np.zeros((1,), dtype=bool)
            act = prepare_swarm_action(action, env)
            if not np.all(np.isfinite(act)):
                return {
                    "challenge_type": int(task.challenge_type),
                    "map_seed": int(task.map_seed),
                    "score": 0.01,
                    "success": False,
                    "collision": True,
                    "distance_to_goal": float(info.get("distance_to_goal", 999.0)),
                }
            obs, _reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)

        score = float(info.get("score", 0.0))
        return {
            "challenge_type": int(task.challenge_type),
            "map_seed": int(task.map_seed),
            "score": score,
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
        }
    finally:
        env.close()


def rollout_episode_hybrid(
    model,
    task,
    *,
    gui: bool = False,
    handoff_m: float = 8.0,  # kept for CLI compat; spiral search ignores it
    heuristic_speed: float = 0.55,
    use_rl_land: bool = False,
    pad_estimator_path=None,
) -> dict[str, Any]:
    """Full mission: spiral disk-search + pad estimator + soft-land."""
    from pathlib import Path

    env = make_env(task, gui=gui)
    controller = HybridController(
        model,
        config=HybridConfig(
            use_rl_land=use_rl_land,
            cruise_speed=heuristic_speed,
            search_enter_m=max(8.0, float(handoff_m)),
            deterministic=True,
            pad_estimator_path=Path(pad_estimator_path) if pad_estimator_path else None,
        ),
    )
    try:
        obs, info = env.reset(seed=int(task.map_seed))
        controller.reset()
        done = False
        while not done:
            act = prepare_swarm_action(controller.act(obs), env)
            if not np.all(np.isfinite(act)):
                return {
                    "challenge_type": int(task.challenge_type),
                    "map_seed": int(task.map_seed),
                    "score": 0.01,
                    "success": False,
                    "collision": True,
                    "distance_to_goal": float(info.get("distance_to_goal", 999.0)),
                }
            obs, _reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)
        return {
            "challenge_type": int(task.challenge_type),
            "map_seed": int(task.map_seed),
            "score": float(info.get("score", 0.0)),
            "success": bool(info.get("success", False)),
            "collision": bool(info.get("collision", False)),
            "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
        }
    finally:
        env.close()


def evaluate_hybrid_model(
    model,
    *,
    challenge_types: Optional[Sequence[int]] = None,
    seeds: Optional[dict[int, int]] = None,
    gui: bool = False,
    handoff_m: float = 8.0,
    pad_estimator_path=None,
) -> ValidationResult:
    """Validator-faithful eval with hybrid cruise + pad estimator + soft-land."""
    tasks = validation_tasks(seeds=seeds, challenge_types=challenge_types)
    episodes = [
        rollout_episode_hybrid(
            model,
            task,
            gui=gui,
            handoff_m=handoff_m,
            pad_estimator_path=pad_estimator_path,
        )
        for _, task in tasks
    ]
    per_type: dict[str, list[float]] = {}
    per_type_success: dict[str, list[bool]] = {}
    for ep in episodes:
        name = CHALLENGE_NAMES.get(ep["challenge_type"], str(ep["challenge_type"]))
        per_type.setdefault(name, []).append(ep["score"])
        per_type_success.setdefault(name, []).append(ep["success"])
    all_scores = [ep["score"] for ep in episodes]
    all_success = [ep["success"] for ep in episodes]
    return ValidationResult(
        mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
        success_rate=float(np.mean(all_success)) if all_success else 0.0,
        per_type={k: float(np.mean(v)) for k, v in per_type.items()},
        per_type_success={k: float(np.mean(v)) for k, v in per_type_success.items()},
        episodes=episodes,
    )


def evaluate_hybrid_open_benchmark(
    model,
    *,
    seeds: Sequence[int] | None = None,
    gui: bool = False,
    handoff_m: float = 8.0,
) -> ValidationResult:
    """Hybrid eval on multiple fixed open-terrain seeds (landing progress metric)."""
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import task_for_seed_and_type

    seed_list = list(seeds or OPEN_BENCHMARK_SEEDS)
    episodes = []
    for seed in seed_list:
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=2)
        episodes.append(rollout_episode_hybrid(model, task, gui=gui, handoff_m=handoff_m))
    all_scores = [ep["score"] for ep in episodes]
    all_success = [ep["success"] for ep in episodes]
    return ValidationResult(
        mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
        success_rate=float(np.mean(all_success)) if all_success else 0.0,
        per_type={"open": float(np.mean(all_scores)) if all_scores else 0.0},
        per_type_success={"open": float(np.mean(all_success)) if all_success else 0.0},
        episodes=episodes,
    )


def evaluate_model(
    model,
    *,
    challenge_types: Optional[Sequence[int]] = None,
    seeds: Optional[dict[int, int]] = None,
    gui: bool = False,
) -> ValidationResult:
    """Evaluate on one fixed seed per challenge type (fast epoch check)."""
    tasks = validation_tasks(seeds=seeds, challenge_types=challenge_types)
    episodes = [rollout_episode(model, task, gui=gui) for _, task in tasks]

    per_type: dict[str, list[float]] = {}
    per_type_success: dict[str, list[bool]] = {}
    for ep in episodes:
        name = CHALLENGE_NAMES.get(ep["challenge_type"], str(ep["challenge_type"]))
        per_type.setdefault(name, []).append(ep["score"])
        per_type_success.setdefault(name, []).append(ep["success"])

    all_scores = [ep["score"] for ep in episodes]
    all_success = [ep["success"] for ep in episodes]

    return ValidationResult(
        mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
        success_rate=float(np.mean(all_success)) if all_success else 0.0,
        per_type={k: float(np.mean(v)) for k, v in per_type.items()},
        per_type_success={k: float(np.mean(v)) for k, v in per_type_success.items()},
        episodes=episodes,
    )


def evaluate_random_seeds(
    model,
    *,
    n_seeds: int = 12,
    base_seed: int = 42,
    gui: bool = False,
) -> ValidationResult:
    """Evaluate on random tasks (mixed env types) for broader coverage."""
    from RL.env_utils import random_benchmark_tasks

    tasks = random_benchmark_tasks(n_seeds, base_seed=base_seed)
    episodes = [rollout_episode(model, task, gui=gui) for task in tasks]

    per_type: dict[str, list[float]] = {}
    per_type_success: dict[str, list[bool]] = {}
    for ep in episodes:
        name = CHALLENGE_NAMES.get(ep["challenge_type"], str(ep["challenge_type"]))
        per_type.setdefault(name, []).append(ep["score"])
        per_type_success.setdefault(name, []).append(ep["success"])

    all_scores = [ep["score"] for ep in episodes]
    all_success = [ep["success"] for ep in episodes]

    return ValidationResult(
        mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
        success_rate=float(np.mean(all_success)) if all_success else 0.0,
        per_type={k: float(np.mean(v)) for k, v in per_type.items()},
        per_type_success={k: float(np.mean(v)) for k, v in per_type_success.items()},
        episodes=episodes,
    )


def save_validation_log(path: Path, result: ValidationResult, *, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean_score": result.mean_score,
        "success_rate": result.success_rate,
        "per_type": result.per_type,
        "per_type_success": result.per_type_success,
        "episodes": result.episodes,
    }
    if extra:
        payload.update(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
