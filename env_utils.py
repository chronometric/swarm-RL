"""Environment construction and curriculum task sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from swarm.constants import CHALLENGE_TYPE_DISTRIBUTION, SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task, screening_task, task_for_seed_and_type

CHALLENGE_NAMES = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}

OPEN_BENCHMARK_SEEDS: tuple[int, ...] = (2_002_002, 10_012, 10_033, 10_052)

# Fixed seeds for reproducible validation across training epochs.
DEFAULT_VAL_SEEDS: dict[int, int] = {
    1: 1_001_001,
    2: 2_002_002,
    3: 3_003_003,
    4: 4_004_004,
    5: 5_005_005,
    6: 6_006_006,
}

@dataclass(frozen=True)
class CurriculumStage:
    name: str
    challenge_types: tuple[int, ...]
    min_mean_score: float  # advance when validation mean >= this


CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("open", (2,), 0.35),
    CurriculumStage("open_mountain", (2, 3), 0.45),
    CurriculumStage("open_mountain_village", (2, 3, 4), 0.55),
    CurriculumStage("full", tuple(CHALLENGE_TYPE_DISTRIBUTION.keys()), 0.65),
)


def sample_task(
    *,
    stage: CurriculumStage,
    seed: Optional[int] = None,
) -> "MapTask":
    """Sample a MapTask restricted to the curriculum stage's challenge types."""
    from swarm.protocol import MapTask  # noqa: F401 — re-export for type checkers

    if seed is None:
        seed = random.randrange(1, 2_147_483_647)

    type_rng = random.Random(seed + 999_999)
    chosen = type_rng.choice(stage.challenge_types)
    return task_for_seed_and_type(
        sim_dt=SIM_DT,
        seed=seed,
        challenge_type=chosen,
    )


def make_training_env_fn(
    stage: CurriculumStage,
    *,
    gui: bool = False,
    seed_offset: int = 0,
    val_mix_ratio: float = 0.0,
) -> Callable[[], object]:
    """Return a zero-arg factory suitable for DummyVecEnv.

    When ``val_mix_ratio`` > 0, a fraction of episodes use seeds near the fixed
    validation seeds so the policy trains on the same distribution it is scored on.
    """

    def _factory():
        if val_mix_ratio > 0.0 and random.random() < val_mix_ratio:
            ctype = random.choice(stage.challenge_types)
            base_seed = DEFAULT_VAL_SEEDS.get(ctype, 2_002_002)
            # Neighbourhood around the fixed validation seed.
            seed = base_seed + random.randint(-100, 100)
            task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=ctype)
        else:
            seed = random.randrange(1, 2_147_483_647) ^ seed_offset
            task = sample_task(stage=stage, seed=seed)
        return make_env(task, gui=gui)

    return _factory


def sample_landing_task(
    *,
    stage: CurriculumStage,
    seed: Optional[int] = None,
    distance_range: tuple[float, float] = (6.0, 18.0),
) -> "MapTask":
    """Task with start 6–18 m from goal — landing-specialist curriculum."""
    if seed is None:
        seed = random.randrange(1, 2_147_483_647)
    ctype = random.choice(stage.challenge_types)
    return screening_task(
        SIM_DT,
        seed,
        challenge_type=ctype,
        distance_range=distance_range,
        goal_height_range=None,
        moving_platform=False,
    )


def make_landing_env_fn(
    stage: CurriculumStage,
    *,
    gui: bool = False,
    full_episode_ratio: float = 0.15,
    distance_range: tuple[float, float] = (6.0, 18.0),
) -> Callable[[], object]:
    """Mostly short landing episodes; occasional full-distance for generalization."""

    def _factory():
        if random.random() < full_episode_ratio:
            seed = random.randrange(1, 2_147_483_647)
            task = sample_task(stage=stage, seed=seed)
        else:
            task = sample_landing_task(stage=stage, distance_range=distance_range)
        return make_env(task, gui=gui)

    return _factory


def make_task_env(task, *, gui: bool = False):
    return make_env(task, gui=gui)


def validation_tasks(
    seeds: Optional[dict[int, int]] = None,
    challenge_types: Optional[Sequence[int]] = None,
) -> list[tuple[int, object]]:
    """Build (challenge_type, MapTask) pairs for epoch validation."""
    seed_map = seeds or DEFAULT_VAL_SEEDS
    types = challenge_types or tuple(seed_map.keys())
    tasks = []
    for ctype in types:
        map_seed = seed_map[ctype]
        task = task_for_seed_and_type(
            sim_dt=SIM_DT,
            seed=map_seed,
            challenge_type=ctype,
        )
        tasks.append((ctype, task))
    return tasks


def random_benchmark_tasks(n: int, *, base_seed: int = 42) -> list[object]:
    rng = random.Random(base_seed)
    return [
        random_task(sim_dt=SIM_DT, seed=rng.randrange(1, 2_147_483_647))
        for _ in range(n)
    ]
