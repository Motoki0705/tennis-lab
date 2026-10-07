"""Typed in-memory inputs and exact, mergeable measurements."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import AnnotationStates
from src.tasks.ball_detection.data.store import ClipRecord


@dataclass(frozen=True)
class PoseInput:
    points: NDArray[np.float64]
    scores: NDArray[np.float64]
    observed: NDArray[np.bool_]
    boxes: NDArray[np.float64]
    player_ids: tuple[str, ...]
    raw_tracks: NDArray[np.int64]


@dataclass(frozen=True)
class AnnotationDetails:
    availability: str
    reason: str | None = None
    issues: tuple[str, ...] = ()
    notes: dict[int, str] = field(default_factory=dict)
    endpoints: dict[int, tuple[int, int]] = field(default_factory=dict)
    sha256: str | None = None
    notes_supported: bool = False
    source_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClipInput:
    clip: ClipRecord
    states: AnnotationStates
    durations: NDArray[np.float64]
    breaks: NDArray[np.bool_]
    details: AnnotationDetails
    pose: PoseInput | None
    pose_reason: str | None


@dataclass(frozen=True)
class Samples:
    values: NDArray[np.float64]
    unit: str
    eligible: int


@dataclass
class Measurements:
    samples: dict[str, Samples] = field(default_factory=dict)
    # Numerator/denominator pairs preserve pooled rates (including zero denominator).
    rates: dict[str, tuple[int, int]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    views: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def sample(self, name: str, values: Any, unit: str, eligible: int | None = None) -> None:
        array = np.asarray(values, np.float64).reshape(-1)
        self.samples[name] = Samples(array, unit, int(array.size) if eligible is None else eligible)

    def merge(self, other: Measurements) -> None:
        for name in ('samples', 'rates', 'counts', 'views'):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine.keys() & theirs.keys():
                raise ValueError(f'Duplicate metric keys: {mine.keys() & theirs.keys()}')
            mine.update(theirs)
        self.findings.extend(other.findings)
