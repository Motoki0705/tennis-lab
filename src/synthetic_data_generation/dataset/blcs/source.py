"""Canonical adapter for the public BLCS physics-source boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, cast

from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSTrack,
    BLCSTrajectory,
)
from src.tasks.blcs.generate_dataset.source_api import (
    BLCSGeneratorConfiguration,
    BLCSPhysicsSourceSettings,
    BLCSPhysicsTrajectorySource,
    BLCSSourceScene,
    BLCSTimelineSpec,
)

_SPLITS = ("train", "validation", "test")


class BLCSTrajectoryProvider(Protocol):
    """Application source for deterministic, complete physics trajectories."""

    def preflight(self, *, scene_id: str, seed: int) -> None:
        """Validate source configuration without generating trajectories."""

    def load(self, *, scene_id: str, seed: int) -> Sequence[BLCSTrajectory]:
        """Return all configured scenes without frame subsampling."""


@dataclass(frozen=True, slots=True)
class BLCSTrajectorySourceSettings:
    """Canonical inventory policy around the task-owned physics source."""

    scene_count: int
    split_scene_counts: Mapping[str, int]
    multi_object: bool
    maximum_physics_attempts_per_object: int
    timeline: BLCSTimelineSpec
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
            raise ValueError("Production BLCS trajectory generation must be multi-object.")
        if not isinstance(self.timeline, BLCSTimelineSpec):
            raise TypeError("BLCS source timeline must be a BLCSTimelineSpec.")
        physics_settings = self.physics_settings()
        object.__setattr__(
            self,
            "split_scene_counts",
            {split: counts[split] for split in _SPLITS},
        )
        object.__setattr__(
            self,
            "maximum_physics_attempts_per_object",
            physics_settings.maximum_physics_attempts_per_object,
        )
        object.__setattr__(self, "device", physics_settings.device)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """Parse canonical inventory plus explicit public task-source settings."""
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
        timeline = (
            timeline_value
            if isinstance(timeline_value, BLCSTimelineSpec)
            else BLCSTimelineSpec.from_mapping(timeline_value)
        )
        split_counts = value["split_scene_counts"]
        if not isinstance(split_counts, Mapping):
            raise TypeError("BLCS split_scene_counts must be a mapping.")
        scene_count = value["scene_count"]
        multi_object = value["multi_object"]
        maximum_attempts = value["maximum_physics_attempts_per_object"]
        device = value["device"]
        return cls(
            scene_count=cast(int, scene_count),
            split_scene_counts=cast(Mapping[str, int], split_counts),
            multi_object=cast(bool, multi_object),
            maximum_physics_attempts_per_object=cast(int, maximum_attempts),
            timeline=timeline,
            device=cast(str, device),
        )

    def physics_settings(self) -> BLCSPhysicsSourceSettings:
        """Return the complete public task-source configuration."""
        return BLCSPhysicsSourceSettings(
            timeline=self.timeline,
            maximum_physics_attempts_per_object=(
                self.maximum_physics_attempts_per_object
            ),
            device=self.device,
        )

    def split_sequence(self) -> tuple[str, ...]:
        """Expand canonical split counts into deterministic scene order."""
        return tuple(
            split for split in _SPLITS for _ in range(self.split_scene_counts[split])
        )


@dataclass(frozen=True, slots=True)
class PhysicsBLCSTrajectoryProvider:
    """Adapt task-owned public physics scenes into canonical trajectories."""

    generator_config: BLCSGeneratorConfiguration
    settings: BLCSTrajectorySourceSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, BLCSTrajectorySourceSettings):
            raise TypeError("settings must be BLCSTrajectorySourceSettings.")

    def _source(self) -> BLCSPhysicsTrajectorySource:
        return BLCSPhysicsTrajectorySource(
            generator_config=self.generator_config,
            settings=self.settings.physics_settings(),
        )

    def preflight(self, *, scene_id: str, seed: int) -> None:
        """Validate every deterministic public source request without generation."""
        source = self._source()
        for index in range(self.settings.scene_count):
            source.preflight(
                scene_id=f"{scene_id}-blcs-{index:06d}",
                seed=seed + index,
            )

    def load(self, *, scene_id: str, seed: int) -> Sequence[BLCSTrajectory]:
        """Generate through the public API and retain each complete source scene."""
        source = self._source()
        trajectories = tuple(
            _adapt_source_scene(
                source.generate(
                    scene_id=f"{scene_id}-blcs-{index:06d}",
                    seed=seed + index,
                ),
                split=split,
            )
            for index, split in enumerate(self.settings.split_sequence())
        )
        if len(trajectories) != self.settings.scene_count:
            raise ValueError("BLCS physics source returned an incomplete scene inventory.")
        if len({trajectory.trajectory_id for trajectory in trajectories}) != len(
            trajectories
        ):
            raise ValueError("BLCS physics source returned duplicate scene IDs.")
        return trajectories


def _adapt_source_scene(
    scene: BLCSSourceScene,
    *,
    split: str,
) -> BLCSTrajectory:
    """Adapt one validated task source without dropping any source semantics."""
    if not isinstance(scene, BLCSSourceScene):
        raise TypeError("BLCS public source returned an unsupported scene value.")
    if scene.frame_indices != tuple(range(scene.frame_count)):
        raise ValueError("BLCS public source omitted or reordered frame identities.")
    trajectory = BLCSTrajectory(
        trajectory_id=scene.scene_id,
        split=split,
        fps=scene.fps,
        positions_court_m=scene.positions_court_m,
        velocities_court_mps=scene.velocities_court_mps,
        present=scene.present,
        tracks=tuple(
            BLCSTrack(
                object_id=track.object_id,
                source_trajectory_id=track.source_trajectory_id,
                source_frame_indices=track.source_frame_indices,
            )
            for track in scene.tracks
        ),
        source_metadata=scene.to_metadata(),
    )
    if trajectory.frame_count != scene.frame_count:
        raise ValueError("BLCS public source did not preserve every generated frame.")
    if trajectory.object_count != scene.object_count or trajectory.object_count < 2:
        raise ValueError("Production BLCS source must preserve every physical object.")
    if trajectory.fps != scene.fps:
        raise ValueError("BLCS public source output FPS changed during adaptation.")
    return trajectory


__all__ = [
    "BLCSTrajectoryProvider",
    "BLCSTrajectorySourceSettings",
    "PhysicsBLCSTrajectoryProvider",
]
