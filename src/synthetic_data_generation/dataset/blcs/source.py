"""Deterministic production BLCS trajectory source over the physics generator."""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Protocol, Self

import numpy as np
import torch

from src.synthetic_data_generation.dataset.blcs.contracts import BLCSTrajectory
from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneGenerator,
    GeneratorConfig,
)

_SPLITS = ("train", "validation", "test")
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class BLCSTrajectoryProvider(Protocol):
    """Application source for deterministic, complete physics trajectories."""

    def preflight(self, *, scene_id: str, seed: int) -> None:
        """Validate source configuration without generating trajectories."""

    def load(self, *, scene_id: str, seed: int) -> Sequence[BLCSTrajectory]:
        """Return all configured scenes without frame subsampling."""


@dataclass(frozen=True, slots=True)
class BLCSTrajectorySourceSettings:
    """No-default Hydra-facing settings for the production physics source."""

    scene_count: int
    split_scene_counts: Mapping[str, int]
    multi_object: bool
    maximum_physics_attempts_per_object: int
    timeline: TimelineConfig
    device: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.scene_count, bool)
            or not isinstance(self.scene_count, int)
            or self.scene_count <= 0
        ):
            raise ValueError("BLCS source scene_count must be a positive integer.")
        if not isinstance(self.split_scene_counts, Mapping):
            raise TypeError("BLCS split_scene_counts must be a mapping.")
        counts = dict(self.split_scene_counts)
        if set(counts) != set(_SPLITS):
            raise ValueError(
                "BLCS split_scene_counts must declare train, validation, and test."
            )
        for split, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"BLCS split count for {split!r} must be a non-negative integer."
                )
        if sum(counts.values()) != self.scene_count:
            raise ValueError("BLCS split counts must sum exactly to scene_count.")
        if self.multi_object is not True:
            raise ValueError(
                "Production BLCS trajectory generation must be multi-object."
            )
        if (
            isinstance(self.maximum_physics_attempts_per_object, bool)
            or not isinstance(self.maximum_physics_attempts_per_object, int)
            or self.maximum_physics_attempts_per_object <= 0
        ):
            raise ValueError(
                "maximum_physics_attempts_per_object must be a positive integer."
            )
        if not isinstance(self.timeline, TimelineConfig):
            raise TypeError("BLCS source timeline must be a TimelineConfig.")
        if self.timeline.min_tracks < 2:
            raise ValueError("Production BLCS timelines must contain multiple objects.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("BLCS physics device must be an explicit non-empty string.")
        device = torch.device(self.device)
        if device.type != "cpu":
            raise ValueError(
                "Canonical BLCS physics generation requires explicit CPU execution."
            )
        object.__setattr__(
            self,
            "split_scene_counts",
            {split: counts[split] for split in _SPLITS},
        )
        object.__setattr__(self, "device", str(device))

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse Hydra/plain mappings without implicit source settings."""
        if not isinstance(value, Mapping):
            raise TypeError("BLCS trajectory source settings must be a mapping.")
        required = {
            "scene_count",
            "split_scene_counts",
            "multi_object",
            "maximum_physics_attempts_per_object",
            "timeline",
            "device",
        }
        if set(value) != required:
            raise ValueError(
                "BLCS trajectory source setting keys do not match; "
                f"missing={sorted(required - set(value))}, "
                f"unknown={sorted(set(value) - required)}."
            )
        timeline_value = value["timeline"]
        if isinstance(timeline_value, TimelineConfig):
            timeline = timeline_value
        elif isinstance(timeline_value, Mapping):
            timeline = TimelineConfig.from_mapping(timeline_value)
        else:
            raise TypeError("BLCS source timeline must be a mapping or TimelineConfig.")
        split_counts = value["split_scene_counts"]
        if not isinstance(split_counts, Mapping):
            raise TypeError("BLCS split_scene_counts must be a mapping.")
        return cls(
            scene_count=value["scene_count"],
            split_scene_counts=split_counts,
            multi_object=value["multi_object"],
            maximum_physics_attempts_per_object=value[
                "maximum_physics_attempts_per_object"
            ],
            timeline=timeline,
            device=value["device"],
        )

    def split_sequence(self) -> tuple[str, ...]:
        """Expand canonical split counts into deterministic scene order."""
        return tuple(
            split for split in _SPLITS for _ in range(self.split_scene_counts[split])
        )


@dataclass(frozen=True, slots=True)
class PhysicsBLCSTrajectoryProvider:
    """Generate deterministic multi-object scenes via the existing BLCS simulator."""

    generator_config: GeneratorConfig
    settings: BLCSTrajectorySourceSettings

    def __post_init__(self) -> None:
        if not isinstance(self.generator_config, GeneratorConfig):
            raise TypeError(
                "generator_config must be the existing BLCS GeneratorConfig."
            )
        if not isinstance(self.settings, BLCSTrajectorySourceSettings):
            raise TypeError("settings must be BLCSTrajectorySourceSettings.")

    def preflight(self, *, scene_id: str, seed: int) -> None:
        """Validate the exact production request without manufacturing a fixture."""
        _validate_request(scene_id=scene_id, seed=seed)

    def load(self, *, scene_id: str, seed: int) -> Sequence[BLCSTrajectory]:
        """Generate every configured physics scene and preserve its complete timeline."""
        self.preflight(scene_id=scene_id, seed=seed)
        trajectories: list[BLCSTrajectory] = []
        for index, split in enumerate(self.settings.split_sequence()):
            source_scene_id = f"{scene_id}-blcs-{index:06d}"
            scene_seed = seed + index
            with _deterministic_random_state(scene_seed):
                base = BLCSSceneGenerator(
                    config=self.generator_config,
                    device=self.settings.device,
                )
                generator = MultiBallSceneGenerator(
                    base,
                    timeline=self.settings.timeline,
                    maximum_physics_attempts_per_object=(
                        self.settings.maximum_physics_attempts_per_object
                    ),
                    rng=random.Random(scene_seed),
                )
                scene = generator.generate_scene(source_scene_id)
            if scene.scene_id != source_scene_id:
                raise ValueError(
                    "BLCS physics generator changed the requested scene ID."
                )
            trajectory = BLCSTrajectory.from_scene(scene, split=split)
            trajectory = replace(
                trajectory,
                source_metadata={
                    **trajectory.source_metadata,
                    "physics_proposals": scene.physics_proposal_diagnostics,
                },
            )
            if trajectory.frame_count != int(scene.ball_pos_world.shape[0]):
                raise ValueError(
                    "BLCS physics source did not preserve every generated frame."
                )
            if trajectory.object_count < 2:
                raise ValueError(
                    "Production BLCS physics source returned a single-object scene."
                )
            trajectories.append(trajectory)
        if len(trajectories) != self.settings.scene_count:
            raise ValueError(
                "BLCS physics source returned an incomplete scene inventory."
            )
        if len({trajectory.trajectory_id for trajectory in trajectories}) != len(
            trajectories
        ):
            raise ValueError("BLCS physics source returned duplicate scene IDs.")
        return tuple(trajectories)


def _validate_request(*, scene_id: str, seed: int) -> None:
    if not isinstance(scene_id, str) or _PORTABLE_ID.fullmatch(scene_id) is None:
        raise ValueError("BLCS source scene_id must be a portable identifier.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("BLCS source seed must be a non-negative integer.")


@contextmanager
def _deterministic_random_state(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


__all__ = [
    "BLCSTrajectoryProvider",
    "BLCSTrajectorySourceSettings",
    "PhysicsBLCSTrajectoryProvider",
]
