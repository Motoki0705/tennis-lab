"""Utilities for loading and sampling 3DTennisDS motion clips."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.tennis.geometry.skeleton import RACKET_3_NAMES, VITPOSE_17_NAMES

if TYPE_CHECKING:  # pragma: no cover
    pass

try:  # pragma: no cover - exercised in integration tests
    import ezc3d as _ezc3d
except ImportError as exc:  # pragma: no cover - surfaced with actionable hint
    _EZC3D: Any | None = None
    EZC3D_IMPORT_ERROR: ImportError | None = exc
else:  # pragma: no cover
    _EZC3D = _ezc3d
    EZC3D_IMPORT_ERROR = None


COCO_TO_MARKERS: dict[str, Sequence[str]] = {
    "nose": ("LFHD", "RFHD"),
    "left_eye": ("LFHD",),
    "right_eye": ("RFHD",),
    "left_ear": ("LBHD",),
    "right_ear": ("RBHD",),
    "left_shoulder": ("LSHO",),
    "right_shoulder": ("RSHO",),
    "left_elbow": ("LELB",),
    "right_elbow": ("RELB",),
    "left_wrist": ("LWRA", "LWRB"),
    "right_wrist": ("RWRA", "RWRB"),
    "left_hip": ("LASI", "LTHI"),
    "right_hip": ("RASI", "RTHI"),
    "left_knee": ("LKNE",),
    "right_knee": ("RKNE",),
    "left_ankle": ("LANK",),
    "right_ankle": ("RANK",),
}

RACKET_MARKERS: dict[str, Sequence[str]] = {
    "racket_handle": ("DOL",),
    "racket_throat": ("RH2", "RH1"),
    "racket_head_top": ("RH3", "RH4", "RH5", "RH6"),
}


@dataclass(slots=True)
class AssetSample:
    """A resampled motion clip ready for placement on the court."""

    joints: np.ndarray  # shape (T, 17, 3)
    racket: np.ndarray  # shape (T, 3, 3)
    pelvis: np.ndarray  # shape (T, 3)


@dataclass(slots=True)
class AssetClip:
    """Processed clip cached in memory (local coordinates, pelvis-relative)."""

    path: Path
    fps: float
    joints_rel: np.ndarray
    racket_rel: np.ndarray
    pelvis_rel: np.ndarray

    @property
    def num_frames(self) -> int:
        """Return the number of cached frames."""
        return int(self.joints_rel.shape[0])

    @property
    def duration_sec(self) -> float:
        """Return clip duration in seconds."""
        if self.num_frames <= 1:
            return 0.0
        return float(self.num_frames - 1) / float(self.fps)

    def sample_sequence(
        self,
        frames_total: int,
        target_fps: float,
        rng: random.Random | None = None,
    ) -> AssetSample:
        """Resample the clip to match the requested timeline.

        Args:
            frames_total (int): Desired output frame count.
            target_fps (float): Frames per second in the target timeline.
            rng (random.Random | None): Optional RNG for start offset.

        Returns:
            AssetSample: Resampled joints/racket/pelvis trajectories.

        Raises:
            ValueError: If frames_total or target_fps is not positive.

        """
        if frames_total <= 0:
            raise ValueError("frames_total must be positive")
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")

        seq_j = self.joints_rel
        seq_r = self.racket_rel
        seq_p = self.pelvis_rel

        needed_span = 0.0 if frames_total == 1 else (frames_total - 1) / target_fps
        if self.duration_sec + 1e-6 < needed_span and self.num_frames > 0:
            repeats = max(1, math.ceil((needed_span * self.fps + 1) / self.num_frames))
            seq_j = np.tile(seq_j, (repeats, 1, 1))
            seq_r = np.tile(seq_r, (repeats, 1, 1))
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
        racket = _interp_sequence(seq_r, times, target_times)
        pelvis = _interp_sequence(seq_p, times, target_times)
        return AssetSample(joints=joints, racket=racket, pelvis=pelvis)


class TennisAssetLibrary:
    """Helper that lazily loads C3D clips and serves resampled sequences."""

    def __init__(
        self,
        root: str | Path,
        *,
        min_frames: int = 30,
        max_files: int | None = None,
    ) -> None:
        if EZC3D_IMPORT_ERROR is not None:  # pragma: no cover - import guard
            msg = (
                "ezc3d is required to load 3DTennisDS assets. Install dependencies "
                "or run `uv pip install ezc3d`."
            )
            raise RuntimeError(msg) from EZC3D_IMPORT_ERROR
        self.root = Path(root)
        if not self.root.exists():
            msg = f"asset root not found: {self.root}"
            raise FileNotFoundError(msg)
        self.min_frames = int(min_frames)
        candidate_paths = list(sorted(self.root.rglob("*.c3d")))
        if max_files is not None:
            candidate_paths = candidate_paths[: int(max_files)]
        if not candidate_paths:
            msg = f"no C3D files found under: {self.root}"
            raise FileNotFoundError(msg)
        self._paths: list[Path] = candidate_paths
        self._cache: dict[Path, AssetClip] = {}
        self._invalid: set[Path] = set()

    def sample_sequence(
        self,
        frames_total: int,
        target_fps: float,
        rng: random.Random | None = None,
    ) -> AssetSample:
        """Return a resampled clip that matches the requested timeline.

        Args:
            frames_total (int): Desired number of frames.
            target_fps (float): Target frames per second.
            rng (random.Random | None): Optional RNG for start offsets.

        Returns:
            AssetSample: Resampled clip guaranteed to have ``frames_total`` frames.

        Raises:
            RuntimeError: If no usable 3DTennisDS clips are found.

        """
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
        raise RuntimeError("no usable 3DTennisDS clips were found")

    def _load_clip(self, path: Path) -> AssetClip | None:
        """Load and cache a single clip if it meets minimum requirements."""
        try:
            coords, labels, fps = _load_c3d_points(path)
        except Exception as exc:  # pragma: no cover - defensive against bad files
            # Some files under the asset root may not be valid C3D blobs even
            # if they match the *.c3d pattern. Treat such files as unusable and
            # let the caller mark them as invalid instead of aborting.
            import warnings

            warnings.warn(
                f"Skipping invalid 3DTennisDS C3D file {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        joints = _build_named_points(coords, labels, VITPOSE_17_NAMES, COCO_TO_MARKERS)
        racket = _build_named_points(coords, labels, RACKET_3_NAMES, RACKET_MARKERS)
        if joints.size == 0 or racket.size == 0:
            return None
        mask = (~np.isnan(joints).any(axis=(1, 2))) & (
            ~np.isnan(racket).any(axis=(1, 2))
        )
        joints = joints[mask]
        racket = racket[mask]
        if joints.shape[0] < self.min_frames:
            return None
        pelvis = 0.5 * (joints[:, 11] + joints[:, 12])
        pelvis = pelvis - pelvis[0]
        joints_rel = joints - pelvis[:, None, :]
        racket_rel = racket - pelvis[:, None, :]
        clip = AssetClip(
            path=path,
            fps=fps,
            joints_rel=joints_rel.astype(np.float32, copy=False),
            racket_rel=racket_rel.astype(np.float32, copy=False),
            pelvis_rel=pelvis.astype(np.float32, copy=False),
        )
        self._cache[path] = clip
        return clip


def _load_c3d_points(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Load marker coordinates/labels/fps from a C3D file."""
    if _EZC3D is None:  # pragma: no cover - guarded earlier
        raise RuntimeError("ezc3d is not available")
    reader = _EZC3D.c3d(str(path))
    points = reader["data"]["points"]  # (4, M, T)
    coords = np.transpose(points[:3, :, :], (2, 1, 0)).astype(np.float32) / 1000.0
    residual = points[3, :, :]
    invalid = residual < 0
    if invalid.any():
        coords[invalid.T] = np.nan
    raw_labels = reader["parameters"]["POINT"]["LABELS"]["value"]
    labels = [str(lab).split(":")[-1].strip().upper() for lab in raw_labels]
    fps = float(
        reader["header"]["points"].get(
            "frameRate", reader["parameters"]["POINT"]["RATE"]["value"][0]
        )
    )
    return coords, labels, fps


def _build_named_points(
    coords: np.ndarray,
    labels: Sequence[str],
    target_names: Sequence[str],
    mapping: dict[str, Sequence[str]],
) -> np.ndarray:
    """Aggregate named joints from raw marker coordinates."""
    num_frames = coords.shape[0]
    result = np.full((num_frames, len(target_names), 3), np.nan, dtype=np.float32)
    label_to_index = {label.upper(): idx for idx, label in enumerate(labels)}
    for idx, name in enumerate(target_names):
        marker_names = mapping.get(name)
        if not marker_names:
            continue
        indices = [
            label_to_index[m.upper()]
            for m in marker_names
            if m.upper() in label_to_index
        ]
        if not indices:
            continue
        pts = coords[:, indices, :]
        result[:, idx, :] = np.nanmean(pts, axis=1)
    return result


def _interp_sequence(
    seq: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Interpolate a sequence along the temporal axis."""
    seq_flat = seq.reshape(seq.shape[0], -1)
    out = np.empty((target_times.shape[0], seq_flat.shape[1]), dtype=np.float32)
    for col in range(seq_flat.shape[1]):
        out[:, col] = np.interp(target_times, times, seq_flat[:, col])
    return out.reshape(target_times.shape[0], *seq.shape[1:])
