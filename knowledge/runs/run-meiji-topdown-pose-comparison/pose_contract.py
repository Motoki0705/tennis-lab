"""The image/BBox and native-pixel COCO17 contracts for the pose comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from src.utils.inference_benchmark import Backend, Stage
from src.utils.schema.player import NUM_HUMAN_KP

Value = TypeVar("Value")


class PoseStages(Protocol):
    """The subset of BatchStages needed by a pose adapter."""

    def stage(
        self, stage: Stage, operation: Callable[[], Value], *, name: str = ""
    ) -> Value: ...

    def forward(
        self, backend: Backend, operation: Callable[[], Value], *, name: str = ""
    ) -> Value: ...


@dataclass(frozen=True)
class PoseInput:
    frame_bgr: NDArray[np.uint8]
    bbox_xyxy: NDArray[np.float32]

    def __post_init__(self) -> None:
        frame, box = self.frame_bgr, self.bbox_xyxy
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or min(frame.shape[:2]) < 2
        ):
            raise ValueError("Pose input requires a native uint8 BGR image")
        if (
            box.dtype != np.float32
            or box.shape != (4,)
            or not np.isfinite(box).all()
            or (box[2:] <= box[:2]).any()
        ):
            raise ValueError("Pose input requires a float32 positive-area xyxy box")


@dataclass(frozen=True)
class PoseBatch:
    keypoints_xyc: NDArray[np.float32]
    extras: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keypoints = self.keypoints_xyc
        if (
            keypoints.dtype != np.float32
            or keypoints.ndim != 3
            or keypoints.shape[1:] != (NUM_HUMAN_KP, 3)
        ):
            raise ValueError(
                "Pose output requires float32 native-pixel (N,17,3) COCO17"
            )
        for name, value in self.extras.items():
            if (
                not name.isidentifier()
                or value.ndim < 1
                or len(value) != len(keypoints)
                or value.dtype.kind not in "biuf"
            ):
                raise ValueError(f"Invalid person-aligned numeric pose output {name!r}")
