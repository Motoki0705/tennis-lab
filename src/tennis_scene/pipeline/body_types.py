"""Body parameter and geometry value contracts shared by separate components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class BodyParameters:
    body_pose: NDArray[np.float32]
    global_orient: NDArray[np.float32]
    betas: NDArray[np.float32]
    transl: NDArray[np.float32]

    def __post_init__(self) -> None:
        count = len(self.body_pose)
        for name, width in (("body_pose", 63), ("global_orient", 3), ("betas", 10), ("transl", 3)):
            value = getattr(self, name)
            if value.shape != (count, width) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid body parameter {name}")

    def tensors(self) -> dict[str, torch.Tensor]:
        return {name: torch.from_numpy(np.array(getattr(self, name), copy=True)) for name in ("body_pose", "global_orient", "betas", "transl")}


@dataclass(frozen=True)
class BodyRecoveryRequest:
    video_path: Path
    source_frames: NDArray[np.int64]
    keypoints: NDArray[np.float32]  # (L,17,3) pixel/confidence
    boxes_xys: NDArray[np.float32]  # (L,3) pixel boxes
    size: tuple[int, int]
    intrinsic: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = len(self.source_frames)
        if count < 2 or self.source_frames.dtype != np.int64 or (self.source_frames < 0).any() or (np.diff(self.source_frames) <= 0).any():
            raise ValueError("Body recovery requires at least two ordered source frames")
        if self.keypoints.shape != (count, 17, 3) or self.boxes_xys.shape != (count, 3) or self.intrinsic.shape != (3, 3):
            raise ValueError("Body recovery pose/box/camera shape mismatch")
        if not all(np.isfinite(x).all() for x in (self.keypoints, self.boxes_xys, self.intrinsic)) or (self.boxes_xys[:, 2] <= 0).any():
            raise ValueError("Body recovery requires finite coordinates and positive boxes")


@dataclass(frozen=True)
class BodyGeometry:
    vertices: NDArray[np.float32]
    coco17: NDArray[np.float32]
