"""Court keypoint detection on the first frame of one camera."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np

from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyConfig as CourtKPPostprocessConfig,
)
from src.tasks.court_detection.inference.contracts import (
    validate_hybrid_inference_config,
)
from src.tasks.court_detection.inference.regions import (
    CourtRegionSearchConfig,
    Region,
    predict_court_region,
    select_court_region,
)
from src.tennis_scene.pipeline.components.base import (
    BasePipelineModule,
    release_inference_memory,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, SourceVideo
from src.utils.configuration import PathResolver
from src.utils.geometry.keypoints import normalize_grid_keypoints
from src.utils.video import read_video_frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tasks.court_detection.inference.predictor import CourtPredictor

LOGGER = logging.getLogger(__name__)

NUM_COURT_KEYPOINTS = 14
KEYPOINT_CONTRACT = "camera_view_v2"
"""Camera-local KP14 order: the only court keypoint contract of the pipeline."""


@dataclass(frozen=True)
class CourtDetectionInput:
    video: SourceVideo


@dataclass(frozen=True, slots=True)
class CourtKPConfig:
    """Court detector checkpoint and its hybrid KP+LINE postprocess."""

    checkpoint: Path
    device: str
    subpixel_refine: bool
    postprocess: CourtKPPostprocessConfig
    region_search: CourtRegionSearchConfig
    resolver: PathResolver

    def __post_init__(self) -> None:
        validate_hybrid_inference_config(self.postprocess)


@dataclass
class CourtKPResult:
    """Court keypoints ``(N, T, 14, 2)`` normalised by ``(W-1, H-1)``."""

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_indices: NDArray[np.int32]
    diagnostics: dict[str, Any] | None = None

    def validate(self) -> tuple[bool, list[str]]:
        """Return ``(is_valid, errors)`` for the shape and value contract."""
        errors: list[str] = []
        if self.keypoints.ndim != 4 or self.keypoints.shape[2:] != (NUM_COURT_KEYPOINTS, 2):
            errors.append(f"keypoints shape must be (N, T, {NUM_COURT_KEYPOINTS}, 2), got {self.keypoints.shape}")
        if self.visibility.shape != self.keypoints.shape[:3]:
            errors.append(f"visibility shape must match keypoints[:3], got {self.visibility.shape} for {self.keypoints.shape}")
        if self.keypoints.ndim >= 2 and self.frame_indices.shape != (self.keypoints.shape[1],):
            errors.append(f"frame_indices shape must match (T,), got {self.frame_indices.shape}")
        if not np.isfinite(self.keypoints).all():
            errors.append("keypoints contain non-finite values")
        if not np.isfinite(self.visibility).all():
            errors.append("visibility contains non-finite values")
        tol = 1e-6
        if self.visibility.shape == self.keypoints.shape[:3]:
            visible_points = self.keypoints[self.visibility > 0]
            if np.any(visible_points < -tol) or np.any(visible_points > 1.0 + tol):
                errors.append("visible keypoints must be normalized to [0, 1]")
        if np.any(self.visibility < -tol) or np.any(self.visibility > 1.0 + tol):
            errors.append("visibility must be normalized to [0, 1]")
        return len(errors) == 0, errors


class CourtKPModule(BasePipelineModule):
    """Detect camera-local KP14 on frame 0 of one static camera.

    Only frame 0 is inferred; ``court_calibration`` broadcasts the static
    geometry explicitly and records that later frames were not inferred.
    """

    io = ComponentIO("court_detection", CourtDetectionInput, CourtKPResult, {}, "court_observations", 2)

    def __init__(self, config: CourtKPConfig) -> None:
        self.config = config
        self._predictor: CourtPredictor | None = None

    def load(self) -> None:
        if self._predictor is not None:
            return
        LOGGER.info("Loading Court KP model from %s", self.config.checkpoint)
        from src.tasks.court_detection.inference.predictor import CourtPredictor

        predictor = CourtPredictor.load_from_checkpoint(
            self.config.checkpoint,
            resolver=self.config.resolver,
            device=self.config.device,
            subpixel_refine=self.config.subpixel_refine,
            hybrid_config=self.config.postprocess,
        )
        _require_camera_view_schema(predictor.adapter.spec.target_bundle.targets["kp"].schema)
        self._predictor = predictor

    def unload(self) -> None:
        self._predictor = None
        release_inference_memory(self.config.device)

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def process(self, inputs: CourtDetectionInput) -> CourtKPResult:
        video = inputs.video
        try:
            self.load()
            packet = read_video_frame(video.path, 0)
            if tuple(packet.original_size) != (video.width, video.height):
                raise ValueError(f"Court frame size {packet.original_size} disagrees with source {(video.width, video.height)}")
            rgb = cast("NDArray[np.uint8]", cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB))
            region, selection = self._select_region(rgb)
            points_px, visible, frame_diagnostic = self._predict_frame_geometry(rgb, region=region)
            assert self._predictor is not None
            checkpoint = self._predictor.checkpoint_identity
        finally:
            self.unload()
        keypoints = normalize_grid_keypoints(points_px, video.width, video.height)
        result = CourtKPResult(
            keypoints[None, None].astype(np.float32), visible[None, None].astype(np.float32), np.array([0], np.int32),
            {"schema": "court_kp_hybrid_v1", "checkpoint": checkpoint,
             "postprocess": asdict(self.config.postprocess), "region_search": asdict(self.config.region_search),
             "output_keypoint_contract": KEYPOINT_CONTRACT,
             "cameras": [{"camera_id": video.camera_id, "video_path": str(video.path), "region_selection": selection,
                          "frames": [{"frame_index": 0, **frame_diagnostic}]}],
             "temporal_policy": "static_first_frame", "observed_frame_indices": [0], "source_frame_count": video.num_frames},
        )
        valid, errors = result.validate()
        if not valid:
            raise ValueError(f"Invalid CourtKP result: {errors}")
        return result

    def _select_region(self, rgb: NDArray[np.uint8]) -> tuple[Region | None, dict[str, Any] | None]:
        if not self.config.region_search.enabled:
            return None, None
        if self._predictor is None:
            raise RuntimeError("Court predictor is not loaded")
        selection = select_court_region(self._predictor, rgb, self.config.region_search)
        return selection.region, {"frame_index": 0, "region_xyxy": list(selection.region),
            "candidates": list(selection.candidates), "original_size_hw": list(rgb.shape[:2])}

    def _predict_frame_geometry(
        self,
        frame_rgb: NDArray[np.uint8],
        *,
        region: Region | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
        if self._predictor is None:
            raise RuntimeError("Court predictor is not loaded")
        if region is None:
            prediction = self._predictor.predict(frame_rgb, postprocess="hybrid", heads=("kp", "line"))
            points, visible = prediction.downstream_keypoints()
            diagnostic = prediction.geometry_diagnostics()
            schema = prediction.keypoint_schema
        else:
            region_prediction = predict_court_region(self._predictor, frame_rgb, region)
            points, visible = region_prediction.downstream_keypoints()
            diagnostic = region_prediction.geometry_diagnostics()
            schema = region_prediction.cropped.keypoint_schema
        if points.shape != (NUM_COURT_KEYPOINTS, 2) or visible.shape != (NUM_COURT_KEYPOINTS,):
            raise ValueError("Hybrid court geometry must contain ordered KP14")
        _require_camera_view_schema(schema)
        diagnostic["output_keypoint_contract"] = KEYPOINT_CONTRACT
        return points, visible, diagnostic


def _require_camera_view_schema(schema: str | None) -> None:
    camera_view = schema is not None and (
        schema.startswith("synthetic_camera_view_kp14_") or schema == "tennis_court_detector_kp14:gaussian_max_v1"
    )
    if not camera_view:
        raise ValueError(f"{KEYPOINT_CONTRACT} requires a camera-view KP14 checkpoint schema, got {schema!r}")
