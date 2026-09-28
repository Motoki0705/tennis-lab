"""Real-time windows, explicit detector-only input and source-balanced sampling."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Sampler

from src.tasks.ball_detection.data.store import ClipRecord
from src.tasks.ball_detection.model_io.contracts import BallCandidates
from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.targets import ClipTargets
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DInput, Refiner2DTarget

CANDIDATE_FIELDS = ("coords", "scores", "valid", "cells", "patches", "patch_valid")


def window_starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
    """Backfill the tail with real frames; a short clip is an explicit error."""
    if length < 1 or not 1 <= stride <= length or frames < length:
        raise ValueError("Windows require frames >= length >= stride >= 1; no padding")
    starts = list(range(0, frames - length + 1, stride))
    if starts[-1] != frames - length:
        starts.append(frames - length)
    return tuple(starts)


def window_owners(frames: int, starts: Sequence[int], length: int) -> NDArray[np.int64]:
    """One prediction per frame: nearest centre, ties resolved by earlier start."""
    owners: NDArray[np.int64] = np.full(frames, -1, dtype=np.int64)
    distances = np.full(frames, np.inf)
    if not starts or list(starts) != sorted(set(starts)):
        raise ValueError("Window starts must be nonempty, sorted and unique")
    for index, start in enumerate(starts):
        if start < 0 or start + length > frames:
            raise ValueError("Window exceeds the source timeline")
        rows = np.arange(start, start + length)
        distance = np.abs(rows - (start + (length - 1) / 2))
        take = distance < distances[rows]
        owners[rows[take]] = index
        distances[rows[take]] = distance[take]
    if (owners < 0).any():
        raise ValueError("Window policy leaves source frames uncovered")
    return owners


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
        source = self.inputs
        candidates = BallCandidates(
            **{name: getattr(source.candidates, name).to(device) for name in CANDIDATE_FIELDS},
            config=source.candidates.config,
        )
        inputs = Refiner2DInput(candidates=candidates, **{
            field.name: getattr(source, field.name).to(device)
            for field in fields(source) if field.name != "candidates"
        })
        target = Refiner2DTarget(**{
            field.name: getattr(self.target, field.name).to(device) for field in fields(self.target)
        })
        return RefinerBatch(inputs, target)


def detector_only_window(
    clip: LoadedClip, start: int, length: int, config: Refiner2DConfig,
) -> WindowSample:
    """An explicit ablation, never a fallback for missing generated context."""
    if config.use_pose or config.use_court or not config.use_detector:
        raise ValueError("Detector-only dataset requires use_detector=true, use_pose/use_court=false")
    if not 0 <= start < start + length <= clip.record.frame_count:
        raise ValueError("Window must contain only real source frames")
    evidence = clip.evidence
    if config.patch_size != evidence.candidates.config.patch_size:
        raise ValueError("Model and evidence patch sizes differ")
    candidates = BallCandidates(**{
        name: getattr(evidence.candidates, name)[:, start:start + length].clone()
        for name in CANDIDATE_FIELDS
    }, config=evidence.candidates.config)
    inputs = Refiner2DInput(
        candidates=candidates,
        timestamps_seconds=torch.from_numpy(evidence.timestamps_seconds[start:start + length].copy())[None],
        pose_uv=torch.zeros(1, length, 0, 4, 2),
        pose_confidence=torch.zeros(1, length, 0, 4),
        pose_valid=torch.zeros(1, length, 0, 4, dtype=torch.bool),
        court_uv=torch.zeros(1, config.court_keypoints, 2),
        court_confidence=torch.zeros(1, config.court_keypoints),
        court_valid=torch.zeros(1, config.court_keypoints, dtype=torch.bool),
    )
    return WindowSample(inputs, clip.targets.target(start, start + length), clip.record.clip_id, start)


def collate_windows(samples: Sequence[WindowSample]) -> RefinerBatch:
    """Pad only the unordered person axis; differing temporal lengths fail."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    inputs = [sample.inputs for sample in samples]
    length = inputs[0].timestamps_seconds.shape[1]
    config = inputs[0].candidates.config
    if any(x.timestamps_seconds.shape != (1, length) or x.candidates.config != config for x in inputs):
        raise ValueError("Collation requires matching real time windows and candidate settings")
    people = max(x.pose_uv.shape[2] for x in inputs)
    pose: dict[str, torch.Tensor] = {}
    for name in ("pose_uv", "pose_confidence", "pose_valid"):
        values = []
        for item in inputs:
            value = getattr(item, name)
            padded = value.new_zeros((1, length, people, *value.shape[3:]))
            padded[:, :, :value.shape[2]] = value
            values.append(padded)
        pose[name] = torch.cat(values)
    candidates = BallCandidates(**{
        name: torch.cat([getattr(x.candidates, name) for x in inputs]) for name in CANDIDATE_FIELDS
    }, config=config)
    batch = Refiner2DInput(candidates=candidates, **pose, **{
        name: torch.cat([getattr(x, name) for x in inputs])
        for name in ("timestamps_seconds", "court_uv", "court_confidence", "court_valid")
    })
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
