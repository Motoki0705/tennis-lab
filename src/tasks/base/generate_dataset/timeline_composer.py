"""Compose independent source tracks on one fixed global timeline.

The composer owns only lifecycle placement. Task-specific generators remain
responsible for producing physical trajectories and projecting their composed
3D values into each camera.

All interval ends in this module are exclusive. A placement with
``birth_frame=10`` and ``death_frame=20`` is present on frames ``[10, 20)``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class TimelineConfig:
    """Validated common BLCS/PLCS lifecycle generation schema."""

    num_frames: int = 1024
    min_tracks: int = 3
    max_tracks: int = 10
    max_concurrent: int = 4
    min_reuse_gap_frames: int = 4
    start_index_range: tuple[int, int] = (-128, 992)
    min_active_frames: int = 32
    overlap_probability: float = 0.3
    min_gap_frames: int = 8
    max_gap_frames: int = 256

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("timeline.num_frames must be positive.")
        if self.min_tracks <= 0 or self.max_tracks < self.min_tracks:
            raise ValueError(
                "timeline track count must satisfy 1 <= min_tracks <= max_tracks."
            )
        if self.max_concurrent <= 0 or self.max_concurrent > self.max_tracks:
            raise ValueError(
                "timeline.max_concurrent must be in [1, max_tracks]."
            )
        if self.min_reuse_gap_frames < 0:
            raise ValueError("timeline.min_reuse_gap_frames must be non-negative.")
        if self.start_index_range[0] > self.start_index_range[1]:
            raise ValueError("timeline.start_index_range must be increasing.")
        if self.min_active_frames <= 0 or self.min_active_frames > self.num_frames:
            raise ValueError(
                "timeline.min_active_frames must be in [1, num_frames]."
            )
        if not 0.0 <= self.overlap_probability <= 1.0:
            raise ValueError("timeline.overlap_probability must be in [0, 1].")
        if self.min_gap_frames < 0 or self.max_gap_frames < self.min_gap_frames:
            raise ValueError(
                "timeline gaps must satisfy 0 <= min_gap_frames <= max_gap_frames."
            )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TimelineConfig:
        """Build the schema from a Hydra/plain mapping without silent defaults."""
        required = {
            "num_frames",
            "min_tracks",
            "max_tracks",
            "max_concurrent",
            "min_reuse_gap_frames",
            "start_index_range",
            "min_active_frames",
            "overlap_probability",
            "min_gap_frames",
            "max_gap_frames",
        }
        missing = sorted(required.difference(config))
        if missing:
            raise KeyError(f"Missing timeline config keys: {missing}")
        start_range = config["start_index_range"]
        if len(start_range) != 2:
            raise ValueError("timeline.start_index_range must contain two values.")
        return cls(
            num_frames=int(config["num_frames"]),
            min_tracks=int(config["min_tracks"]),
            max_tracks=int(config["max_tracks"]),
            max_concurrent=int(config["max_concurrent"]),
            min_reuse_gap_frames=int(config["min_reuse_gap_frames"]),
            start_index_range=(int(start_range[0]), int(start_range[1])),
            min_active_frames=int(config["min_active_frames"]),
            overlap_probability=float(config["overlap_probability"]),
            min_gap_frames=int(config["min_gap_frames"]),
            max_gap_frames=int(config["max_gap_frames"]),
        )


@dataclass(frozen=True)
class TrackPlacement:
    """One source track's visible interval on the global timeline."""

    track_id: int
    source_scene_id: str
    source_start: int
    source_end: int
    birth_frame: int
    death_frame: int

    @property
    def num_active_frames(self) -> int:
        return self.death_frame - self.birth_frame

    def to_metadata(self) -> dict[str, int | str]:
        """Return the canonical JSON metadata record."""
        return {
            "track_id": self.track_id,
            "source_scene_id": self.source_scene_id,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "birth_frame": self.birth_frame,
            "death_frame": self.death_frame,
        }


@dataclass(frozen=True)
class TimelineComposition:
    """A validated placement plan and its physical-track presence matrix."""

    config: TimelineConfig
    placements: tuple[TrackPlacement, ...]
    present: NDArray[np.bool_]

    def compose_numpy(
        self,
        sources: Sequence[NDArray[Any]],
        *,
        fill_value: float | int | bool = 0,
    ) -> NDArray[Any]:
        """Place numpy sources into ``(T, max_tracks, ...)`` output."""
        if len(sources) != len(self.placements):
            raise ValueError("sources must have one array per placement.")
        if not sources:
            raise ValueError("sources cannot be empty.")
        trailing_shape = sources[0].shape[1:]
        if any(source.shape[1:] != trailing_shape for source in sources):
            raise ValueError("All source arrays must share trailing dimensions.")
        output = np.full(
            (self.config.num_frames, self.config.max_tracks, *trailing_shape),
            fill_value,
            dtype=sources[0].dtype,
        )
        for placement, source in zip(self.placements, sources, strict=True):
            if source.shape[0] < placement.source_end:
                raise ValueError(
                    f"Source {placement.source_scene_id} is shorter than its placement."
                )
            output[
                placement.birth_frame : placement.death_frame,
                placement.track_id,
            ] = source[placement.source_start : placement.source_end]
        return output

    def compose_tensor(
        self,
        sources: Sequence[Tensor],
        *,
        fill_value: float | int | bool = 0,
    ) -> Tensor:
        """Place torch sources into ``(T, max_tracks, ...)`` output."""
        if len(sources) != len(self.placements):
            raise ValueError("sources must have one tensor per placement.")
        if not sources:
            raise ValueError("sources cannot be empty.")
        trailing_shape = tuple(sources[0].shape[1:])
        if any(tuple(source.shape[1:]) != trailing_shape for source in sources):
            raise ValueError("All source tensors must share trailing dimensions.")
        output = torch.full(
            (self.config.num_frames, self.config.max_tracks, *trailing_shape),
            fill_value,
            dtype=sources[0].dtype,
            device=sources[0].device,
        )
        for placement, source in zip(self.placements, sources, strict=True):
            if source.shape[0] < placement.source_end:
                raise ValueError(
                    f"Source {placement.source_scene_id} is shorter than its placement."
                )
            output[
                placement.birth_frame : placement.death_frame,
                placement.track_id,
            ] = source[placement.source_start : placement.source_end]
        return output


class TimelineComposer:
    """Sample track count/start indices and enforce simultaneous-track limits."""

    def __init__(
        self,
        config: TimelineConfig,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.rng = rng or random.Random()

    def sample_num_tracks(self) -> int:
        """Sample the number of physical lifecycle instances in one scene."""
        return self.rng.randint(self.config.min_tracks, self.config.max_tracks)

    def compose(
        self,
        source_scene_ids: Sequence[str],
        source_lengths: Sequence[int],
    ) -> TimelineComposition:
        """Sample a valid global placement for the supplied source tracks."""
        if len(source_scene_ids) != len(source_lengths):
            raise ValueError("source_scene_ids and source_lengths must have equal length.")
        if not self.config.min_tracks <= len(source_lengths) <= self.config.max_tracks:
            raise ValueError(
                "Number of source tracks must be within timeline min/max_tracks."
            )
        if any(int(length) < self.config.min_active_frames for length in source_lengths):
            raise ValueError(
                "Every source track must contain at least timeline.min_active_frames."
            )

        placements: list[TrackPlacement] | None = None
        last_error: RuntimeError | None = None
        for _ in range(64):
            candidate_placements: list[TrackPlacement] = []
            slot_occupancy: NDArray[np.int16] = np.zeros(
                self.config.num_frames, dtype=np.int16
            )
            try:
                for track_id, (scene_id, source_length) in enumerate(
                    zip(source_scene_ids, source_lengths, strict=True)
                ):
                    placement = self._sample_placement(
                        track_id=track_id,
                        source_scene_id=str(scene_id),
                        source_length=int(source_length),
                        existing=candidate_placements,
                        slot_occupancy=slot_occupancy,
                    )
                    candidate_placements.append(placement)
                    occupied_until = min(
                        self.config.num_frames,
                        placement.death_frame + self.config.min_reuse_gap_frames,
                    )
                    slot_occupancy[placement.birth_frame:occupied_until] += 1
            except RuntimeError as error:
                last_error = error
                continue
            placements = candidate_placements
            break
        if placements is None:
            raise RuntimeError(
                "Could not compose a valid lifecycle timeline after 64 complete "
                "placement attempts."
            ) from last_error

        present: NDArray[np.bool_] = np.zeros(
            (self.config.num_frames, self.config.max_tracks), dtype=np.bool_
        )
        for placement in placements:
            present[
                placement.birth_frame : placement.death_frame,
                placement.track_id,
            ] = True
        if int(present.sum(axis=1).max(initial=0)) > self.config.max_concurrent:
            raise RuntimeError("Timeline composer produced an illegal overlap.")
        return TimelineComposition(
            config=self.config,
            placements=tuple(placements),
            present=present,
        )

    def _sample_placement(
        self,
        *,
        track_id: int,
        source_scene_id: str,
        source_length: int,
        existing: Sequence[TrackPlacement],
        slot_occupancy: NDArray[np.int16],
    ) -> TrackPlacement:
        for _ in range(256):
            active_length = self.rng.randint(
                self.config.min_active_frames,
                min(source_length, self.config.num_frames),
            )
            start = self._candidate_start(active_length, existing)
            placement = self._placement_from_start(
                track_id,
                source_scene_id,
                source_length,
                start,
                active_length,
            )
            if placement is not None and self._can_place(placement, slot_occupancy):
                return placement

        start_min, start_max = self.config.start_index_range
        starts = list(range(start_min, start_max + 1))
        self.rng.shuffle(starts)
        for start in starts:
            active_length = self.rng.randint(
                self.config.min_active_frames,
                min(source_length, self.config.num_frames),
            )
            placement = self._placement_from_start(
                track_id,
                source_scene_id,
                source_length,
                start,
                active_length,
            )
            if placement is not None and self._can_place(placement, slot_occupancy):
                return placement
        raise RuntimeError(
            "Could not place track without exceeding timeline.max_concurrent; "
            f"track_id={track_id}, source_length={source_length}."
        )

    def _candidate_start(
        self,
        source_length: int,
        existing: Sequence[TrackPlacement],
    ) -> int:
        start_min, start_max = self.config.start_index_range
        if not existing:
            return self.rng.randint(start_min, start_max)

        anchor = self.rng.choice(existing)
        if self.rng.random() < self.config.overlap_probability:
            lower = max(
                start_min,
                anchor.birth_frame - source_length + self.config.min_active_frames,
            )
            upper = min(
                start_max,
                anchor.death_frame - self.config.min_active_frames,
            )
            if lower <= upper:
                return self.rng.randint(lower, upper)

        gap = self.rng.randint(
            self.config.min_gap_frames, self.config.max_gap_frames
        )
        if self.rng.random() < 0.5:
            return min(start_max, anchor.death_frame + gap)
        return max(start_min, anchor.birth_frame - gap - source_length)

    def _placement_from_start(
        self,
        track_id: int,
        source_scene_id: str,
        source_length: int,
        start: int,
        requested_active_length: int,
    ) -> TrackPlacement | None:
        birth = max(0, start)
        source_start = -start if start < 0 else 0
        active = min(
            requested_active_length,
            self.config.num_frames - birth,
            source_length - source_start,
        )
        if active < self.config.min_active_frames:
            return None
        if start >= 0 and source_length > active:
            source_start = self.rng.randint(0, source_length - active)
        death = birth + active
        source_end = source_start + active
        return TrackPlacement(
            track_id=track_id,
            source_scene_id=source_scene_id,
            source_start=source_start,
            source_end=source_end,
            birth_frame=birth,
            death_frame=death,
        )

    def _can_place(
        self,
        placement: TrackPlacement,
        slot_occupancy: NDArray[np.int16],
    ) -> bool:
        occupied_until = min(
            self.config.num_frames,
            placement.death_frame + self.config.min_reuse_gap_frames,
        )
        interval = slot_occupancy[placement.birth_frame:occupied_until]
        return bool(np.all(interval < self.config.max_concurrent))


__all__ = [
    "TimelineComposer",
    "TimelineComposition",
    "TimelineConfig",
    "TrackPlacement",
]
