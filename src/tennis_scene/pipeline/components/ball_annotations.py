"""Scene projections of task-owned v3 annotations, separate from detector points.

No state table lives here: the task reader and usage API own the meaning of
every row. The original document is retained to preserve support frames,
occlusion, notes, source PTS and target semantics without another schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.annotation.video import VideoBallAnnotation, usage_masks
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionInput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort

ANNOTATION_PORT = InputPort("ball_annotations", 1)
ANNOTATION_POINTS_PORT = InputPort("ball_annotation_points", 1)


@dataclass(frozen=True)
class BallAnnotationOutput:
    camera_id: str
    source_size_wh: tuple[int, int]
    frame_indices: NDArray[np.int64]
    pts: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    position_usable: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    estimated: NDArray[np.bool_]
    point_kind: tuple[str, ...]  # task provenance.kind, empty only without position
    annotation_path: str
    annotation_sha256: str
    annotation: dict[str, Any]

    def __post_init__(self) -> None:
        saved = VideoBallAnnotation.model_validate_json(json.dumps(self.annotation, allow_nan=False))
        masks = usage_masks(saved)
        n = len(saved.frames)
        if (self.camera_id != saved.source.camera_id
                or self.source_size_wh != (saved.source.width, saved.source.height)
                or self.frame_indices.dtype != np.int64
                or not np.array_equal(self.frame_indices, np.arange(n))
                or saved.frame_range.start != 0 or saved.frame_range.stop != n
                or self.pts.dtype != np.int64
                or not np.array_equal(self.pts, [f.pts for f in saved.frames])):
            raise ValueError("Ball annotation camera/source frame/PTS mismatch")
        for name, expected in (("position_usable", masks.position_usable),
                               ("observed", masks.direct_observation),
                               ("estimated", masks.estimated)):
            mask = getattr(self, name)
            if mask.dtype != np.bool_ or not np.array_equal(mask, expected):
                raise ValueError(f"Ball annotation {name} differs from task usage")
        kinds = tuple("" if f.position is None else f.position.provenance.kind for f in saved.frames)
        coordinates = np.asarray([(0., 0.) if f.position is None else (f.position.x, f.position.y)
                                  for f in saved.frames], np.float32)
        if (self.uv_px.dtype != np.float32 or self.uv_px.shape != (n, 2)
                or not np.array_equal(self.uv_px, coordinates)
                or self.confidence.dtype != np.float32
                or not np.array_equal(self.confidence, np.asarray(masks.position_usable, np.float32))
                or self.point_kind != kinds):
            raise ValueError("Ball annotation coordinates/weights/provenance differ from saved v3")
        if not self.annotation_path or len(self.annotation_sha256) != 64:
            raise ValueError("Ball annotations require an immutable source reference")


class ImportedBallAnnotationModule:
    io = ComponentIO("ball_detection", BallDetectionInput, BallAnnotationOutput,
                     {}, ANNOTATION_PORT.schema, ANNOTATION_PORT.version)

    def process(self, inputs: BallDetectionInput) -> BallAnnotationOutput:
        raise RuntimeError("Saved ball annotations must be imported into a load-only node")


@dataclass(frozen=True)
class BallAnnotationPointsInput:
    annotations: BallAnnotationOutput


class BallAnnotationPointsModule:
    """Pass accepted task positions through without smoothing or observed-only gating."""

    io = ComponentIO("ball_points", BallAnnotationPointsInput, BallAnnotationOutput,
                     {"detections": ANNOTATION_PORT}, ANNOTATION_POINTS_PORT.schema,
                     ANNOTATION_POINTS_PORT.version)

    def process(self, inputs: BallAnnotationPointsInput) -> BallAnnotationOutput:
        return inputs.annotations
