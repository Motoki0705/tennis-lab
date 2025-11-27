from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class HumanAssetSample:
    joints: np.ndarray
    pelvis: np.ndarray


@dataclass(slots=True)
class HumanAssetClip:
    path: Path
    fps: float
    joints_rel: np.ndarray
    pelvis_rel: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joints_rel.shape[0])

    @property
    def duration_sec(self) -> float:
        if self.num_frames <= 1:
            return 0.0
        return float(self.num_frames - 1) / float(self.fps)

    def sample_sequence(
        self,
        frames_total: int,
        target_fps: float,
        rng: random.Random | None = None,
    ) -> HumanAssetSample:
        if frames_total <= 0:
            raise ValueError("frames_total must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")

        seq_j = self.joints_rel
        seq_p = self.pelvis_rel

        needed_span = 0.0 if frames_total == 1 else (frames_total - 1) / target_fps
        if self.duration_sec + 1e-6 < needed_span and self.num_frames > 0:
            repeats = max(1, math.ceil((needed_span * self.fps + 1) / self.num_frames))
            seq_j = np.tile(seq_j, (repeats, 1, 1))
            seq_p = np.tile(seq_p, (repeats, 1))

        total_frames = seq_j.shape[0]
        times = np.arange(total_frames, dtype=np.float32) / float(self.fps)
        if frames_total == 1:
            target_times = np.array([0.0], dtype=np.float32)
        else:
            target_times = np.linspace(0.0, needed_span, frames_total, dtype=np.float32)

        max_start = float(max(0.0, times[-1] - target_times[-1]))
        start = 0.0
        if max_start > 1e-6:
            rand = rng.random() if rng is not None else random.random()
            start = rand * max_start
        target_times = target_times + start

        joints = _interp_sequence(seq_j, times, target_times)
        pelvis = _interp_sequence(seq_p, times, target_times)
        return HumanAssetSample(joints=joints, pelvis=pelvis)


class HumanAssetLibrary:
    def __init__(
        self,
        root: str | Path,
        *,
        min_frames: int = 30,
        max_files: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            msg = f"asset root not found: {self.root}"
            raise FileNotFoundError(msg)
        self.min_frames = int(min_frames)
        candidate_paths = list(sorted(self.root.rglob("*.npz")))
        if max_files is not None:
            candidate_paths = candidate_paths[: int(max_files)]
        if not candidate_paths:
            msg = f"no NPZ files found under: {self.root}"
            raise FileNotFoundError(msg)
        self._paths: list[Path] = candidate_paths
        self._cache: dict[Path, HumanAssetClip] = {}
        self._invalid: set[Path] = set()

    def sample_sequence(
        self,
        frames_total: int,
        target_fps: float,
        rng: random.Random | None = None,
    ) -> HumanAssetSample:
        attempts = 0
        while attempts < len(self._paths):
            attempts += 1
            rand = rng.random() if rng is not None else random.random()
            idx = int(rand * len(self._paths))
            path = self._paths[idx]
            if path in self._invalid:
                continue
            clip = self._cache.get(path)
            if clip is None:
                clip = self._load_clip(path)
            if clip is None:
                self._invalid.add(path)
                continue
            return clip.sample_sequence(frames_total, target_fps, rng)
        raise RuntimeError("no usable human motion clips were found")

    def _load_clip(self, path: Path) -> HumanAssetClip | None:
        try:
            data = np.load(path, allow_pickle=False)
        except Exception:
            return None
        if "joints" not in data or "pelvis" not in data or "fps" not in data:
            return None
        joints = data["joints"]
        pelvis = data["pelvis"]
        fps = float(data["fps"])
        if joints.ndim != 3 or pelvis.ndim != 2:
            return None
        if joints.shape[0] != pelvis.shape[0]:
            return None
        if joints.shape[0] < self.min_frames:
            return None
        clip = HumanAssetClip(
            path=path,
            fps=fps,
            joints_rel=joints.astype(np.float32, copy=False),
            pelvis_rel=pelvis.astype(np.float32, copy=False),
        )
        self._cache[path] = clip
        return clip


def _interp_sequence(
    seq: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    seq_flat = seq.reshape(seq.shape[0], -1)
    out = np.empty((target_times.shape[0], seq_flat.shape[1]), dtype=np.float32)
    for col in range(seq_flat.shape[1]):
        out[:, col] = np.interp(target_times, times, seq_flat[:, col])
    return out.reshape(target_times.shape[0], *seq.shape[1:])
