"""Compose independent source tracks on one variable-length global timeline.

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
    """Full-source lifetimes; only births are optimized for uniform occupancy."""

    min_tracks: int
    max_tracks: int
    max_concurrent: int
    min_reuse_gap_frames: int
    min_scene_frames: int
    planning_iterations: int

    def __post_init__(self) -> None:
        for name in (
            "min_tracks",
            "max_tracks",
            "max_concurrent",
            "min_scene_frames",
            "planning_iterations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"timeline.{name} must be a positive integer.")
        if (
            not self.min_tracks <= self.max_tracks
            or self.max_concurrent > self.max_tracks
        ):
            raise ValueError("Invalid timeline track count or max_concurrent.")
        if type(self.min_reuse_gap_frames) is not int or self.min_reuse_gap_frames < 0:
            raise ValueError("timeline.min_reuse_gap_frames must be non-negative.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> TimelineConfig:
        from dataclasses import fields

        required = {field.name for field in fields(cls)}
        if set(config) != required:
            raise ValueError(
                f"timeline requires exactly {sorted(required)}; got {sorted(config)}"
            )
        return cls(**dict(config))


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
            (self.present.shape[0], self.config.max_tracks, *trailing_shape),
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
            (self.present.shape[0], self.config.max_tracks, *trailing_shape),
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


def occupancy_durations(
    births: NDArray[np.int64], lengths: NDArray[np.int64], *, bins: int
) -> NDArray[np.float64]:
    """Exact event-sweep histogram; no allocation proportional to scene length."""
    events = np.concatenate((births, births + lengths))
    order = np.argsort(events, kind="stable")
    counts = np.cumsum(
        np.concatenate(
            (
                np.ones(len(births), dtype=np.int64),
                -np.ones(len(births), dtype=np.int64),
            )
        )[order]
    )[:-1]
    spans = np.diff(events[order])
    return np.bincount(counts, weights=spans, minlength=bins).astype(np.float64)


class TimelineComposer:
    """Deterministic source-length-aware birth planning, independent of worker order.

    Minimize the positive-frame occupancy histogram's distance from uniform.
    Deaths always equal birth plus the entire source length. Dataset publication
    audits the aggregate durations (seconds), since per-scene optima need not
    realize every count, especially for short or unequal source sets.
    """

    def __init__(
        self, config: TimelineConfig, *, rng: random.Random | None = None
    ) -> None:
        self.config = config
        self.rng = rng or random.Random()
        self.occupancy_seconds: NDArray[np.float64] = np.zeros(
            config.max_concurrent + 1
        )

    def sample_num_tracks(self) -> int:
        return self.rng.randint(self.config.min_tracks, self.config.max_tracks)

    def compose(
        self,
        source_scene_ids: Sequence[str],
        source_lengths: Sequence[int],
        *,
        fps: float = 1.0,
        balance_dataset: bool = False,
    ) -> TimelineComposition:
        from scipy.optimize import differential_evolution

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        n = len(source_lengths)
        if (
            len(source_scene_ids) != n
            or not self.config.min_tracks <= n <= self.config.max_tracks
        ):
            raise ValueError(
                "Source IDs/lengths must match and obey timeline track counts."
            )
        if any(type(x) is not int or x <= 0 for x in source_lengths):
            raise ValueError("Every complete source must have a positive frame count.")
        lengths = np.asarray(source_lengths, dtype=np.int64)
        capacity = self.config.max_concurrent
        gap = self.config.min_reuse_gap_frames
        # A serial schedule is always valid and consumes all sources.
        serial = np.concatenate(([0], np.cumsum(lengths[:-1] + gap))).astype(np.int64)
        horizon = int(serial[-1])

        def objective(values: NDArray[np.float64]) -> float:
            births = np.rint(values).astype(np.int64)
            births -= births.min()
            durations = occupancy_durations(births, lengths, bins=max(n, capacity) + 1)
            reserved = occupancy_durations(
                births, lengths + gap, bins=max(n, capacity) + 1
            )
            overflow = reserved[capacity + 1 :].sum()
            positive = durations[1:].sum()
            current_seconds = durations[1 : capacity + 1] / fps
            aggregate = current_seconds + (
                self.occupancy_seconds[1:] if balance_dataset else 0
            )
            score = float(
                np.square(
                    (aggregate - aggregate.mean()) / max(current_seconds.sum(), 1 / fps)
                ).sum()
            )
            score += 10 * float(overflow / max(positive, 1))
            scene_length = int((births + lengths).max())
            score += (
                10
                * max(0, self.config.min_scene_frames - scene_length)
                / self.config.min_scene_frames
            )
            # Avoid rewarding unobserved gaps: they do not count toward the 1..K quota.
            score += 0.01 * durations[0] / max(positive, 1)
            return float(score)

        if n == 1:
            births: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        else:
            result = differential_evolution(
                objective,
                [(0, max(horizon, 1))] * n,
                seed=self.rng.getrandbits(32),
                maxiter=self.config.planning_iterations,
                popsize=8,
                polish=False,
                integrality=True,
                x0=serial,
                tol=1e-5,
                atol=1e-6,
                workers=1,
            )
            births = np.rint(result.x).astype(np.int64)
            births -= births.min()
            # Repair a rare soft-penalty overflow by delaying births, never trimming sources.
            order = np.argsort(births, kind="stable")
            available: NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
            for i in order:
                slot = int(np.argmin(available))
                births[i] = max(births[i], available[slot])
                available[slot] = births[i] + lengths[i] + gap
        scene_frames = int((births + lengths).max())
        if scene_frames < self.config.min_scene_frames:
            raise ValueError(
                "Complete sources cannot satisfy min_scene_frames; resample the source set."
            )
        placements = tuple(
            TrackPlacement(
                i,
                str(source_scene_ids[i]),
                0,
                int(lengths[i]),
                int(births[i]),
                int(births[i] + lengths[i]),
            )
            for i in range(n)
        )
        present: NDArray[np.bool_] = np.zeros(
            (scene_frames, self.config.max_tracks), dtype=np.bool_
        )
        for placement in placements:
            present[
                placement.birth_frame : placement.death_frame, placement.track_id
            ] = True
        if not present[0].any() or present.sum(1).max() > capacity:
            raise RuntimeError("Invalid full-source birth plan.")
        if balance_dataset:
            self.occupancy_seconds += (
                np.bincount(present.sum(1), minlength=capacity + 1) / fps
            )
        return TimelineComposition(self.config, placements, present)


__all__ = [
    "TimelineComposer",
    "TimelineComposition",
    "TimelineConfig",
    "TrackPlacement",
]
