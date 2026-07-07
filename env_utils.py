"""Environment construction and curriculum task sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from swarm.constants import CHALLENGE_TYPE_DISTRIBUTION, SIM_DT
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task, task_for_seed_and_type

CHALLENGE_NAMES = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}

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
) -> Callable[[], object]:
    """Return a zero-arg factory suitable for DummyVecEnv."""

    def _factory():
        seed = random.randrange(1, 2_147_483_647) ^ seed_offset
        task = sample_task(stage=stage, seed=seed)
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
