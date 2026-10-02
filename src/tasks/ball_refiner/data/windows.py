"""Real-time windows, explicit detector-only input and source-balanced sampling."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Sampler

from src.tasks.ball_detection.data.store import ClipRecord
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.inputs import CANDIDATE_FIELDS as CANDIDATE_FIELDS
from src.tasks.ball_refiner.data.inputs import (
    collate_inputs,
    detector_only_input,
    input_to,
)
from src.tasks.ball_refiner.data.targets import ClipTargets
from src.tasks.ball_refiner.data.temporal import window_owners as window_owners
from src.tasks.ball_refiner.data.temporal import window_starts
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput, Refiner2DTarget


@dataclass(frozen=True)
class LoadedClip:
    record: ClipRecord
    evidence: ClipEvidence
    targets: ClipTargets

    def __post_init__(self) -> None:
        if len(self.evidence.frame_index) != self.record.frame_count or any(
            not np.array_equal(getattr(self.evidence, key), getattr(self.targets, key))
            for key in ("frame_index", "pts", "timestamps_seconds")
        ):
            raise ValueError("Input/target timelines must agree exactly")


@dataclass(frozen=True)
class WindowSample:
    inputs: Refiner2DInput
    target: Refiner2DTarget
    clip_id: str
    start: int


@dataclass(frozen=True)
class RefinerBatch:
    inputs: Refiner2DInput
    target: Refiner2DTarget

    def to(self, device: torch.device) -> RefinerBatch:
        target = Refiner2DTarget(**{
            field.name: getattr(self.target, field.name).to(device) for field in fields(self.target)
        })
        return RefinerBatch(input_to(self.inputs, device), target)


def detector_only_window(
    clip: LoadedClip, start: int, length: int, config: Refiner2DConfig,
) -> WindowSample:
    """Attach labels only at the training boundary."""
    inputs = detector_only_input(clip.evidence, start, length, config)
    return WindowSample(inputs, clip.targets.target(start, start + length), clip.record.clip_id, start)


def collate_windows(samples: Sequence[WindowSample]) -> RefinerBatch:
    """Pad only the unordered person axis; differing temporal lengths fail."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = collate_inputs([sample.inputs for sample in samples])
    target = Refiner2DTarget(**{
        field.name: torch.cat([getattr(x.target, field.name) for x in samples])
        for field in fields(Refiner2DTarget)
    })
    return RefinerBatch(batch, target)


class RefinerWindowDataset(Dataset[WindowSample]):
    """Only supervised train windows. Excluded clips/windows remain in the audit."""

    def __init__(
        self, clips: Sequence[LoadedClip], *, length: int, stride: int, config: Refiner2DConfig,
    ) -> None:
        if config.use_pose or config.use_court or not config.use_detector:
            raise ValueError("This dataset supports only the explicit detector-only baseline")
        self.clips, self.length, self.config = tuple(clips), length, config
        self.windows: list[tuple[int, int]] = []
        self.source_indices: dict[str, list[int]] = {}
        self.excluded: list[dict[str, object]] = []
        for index, clip in enumerate(clips):
            if clip.record.split != "train":
                raise ValueError("Training dataset received a non-train clip")
            if clip.record.frame_count < length:
                self.excluded.append({"clip_id": clip.record.clip_id, "reason": "short_clip",
                                      "frames": clip.record.frame_count})
                continue
            skipped = 0
            for start in window_starts(clip.record.frame_count, length, stride):
                if not clip.targets.presence_valid[start:start + length].any():
                    skipped += 1
                    continue
                source = clip.record.source
                if source not in self.source_indices:
                    self.source_indices[source] = []
                self.source_indices[source].append(len(self.windows))
                self.windows.append((index, start))
            if skipped:
                self.excluded.append({"clip_id": clip.record.clip_id, "reason": "no_known_target",
                                      "windows": skipped})
        if not self.windows or set(self.source_indices) != {c.record.source for c in clips}:
            raise ValueError("Every requested train source needs supervised, unpadded windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> WindowSample:
        clip_index, start = self.windows[index]
        return detector_only_window(self.clips[clip_index], start, self.length, self.config)


class BalancedSourceSampler(Sampler[int]):
    """Seeded uniform source/window sampling; per-epoch source counts differ <=1."""

    def __init__(self, source_indices: dict[str, list[int]], *, draws: int, seed: int) -> None:
        if draws < 1 or not source_indices or any(not x for x in source_indices.values()):
            raise ValueError("Balanced sampling requires nonempty sources and positive draws")
        self.groups = [source_indices[key] for key in sorted(source_indices)]
        self.draws, self.seed, self.epoch = draws, seed, 0

    def __len__(self) -> int:
        return self.draws

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        sources: NDArray[np.int64] = np.resize(rng.permutation(len(self.groups)), self.draws)
        rng.shuffle(sources)
        for source in sources:
            group = self.groups[int(source)]
            yield group[int(rng.integers(len(group)))]
