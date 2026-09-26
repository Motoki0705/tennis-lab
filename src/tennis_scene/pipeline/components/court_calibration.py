"""Initial camera-local court calibration, before cross-camera association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryConfig,
    calibrate_local_courts,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.utils.geometry.court_roi import court_footpoint_polygon_px
from src.utils.geometry.keypoints import denormalize_grid_keypoints


@dataclass(frozen=True)
class CourtCalibrationInput:
    source: ClipSource
    courts: tuple[CourtKPResult, ...]


@dataclass(frozen=True)
class CourtCalibrationOutput:
    court: CourtKPResult
    calibration: CalibrationSet
    reference_camera: str
    footpoint_polygons: dict[str, tuple[tuple[float, float], ...] | None]
    """Court ROI per source camera; ``None`` marks a camera excluded from calibration."""


class CourtCalibrationModule:
    def __init__(self, camera_ids: tuple[str, ...], config: CameraGeometryConfig, *, roi_margins: tuple[float, float] = (1., 5.)) -> None:
        self.config, self.roi_margins = config, roi_margins
        self.io = ComponentIO("court_calibration", CourtCalibrationInput, CourtCalibrationOutput,
            {camera: InputPort("court_observations", 2) for camera in camera_ids}, "local_court_calibration")

    def process(self, inputs: CourtCalibrationInput) -> CourtCalibrationOutput:
        if len(inputs.courts) != len(inputs.source.videos):
            raise ValueError("Court calibration camera count mismatch")
        diagnostics = []
        for court in inputs.courts:
            if court.keypoints.shape != (1, 1, 14, 2) or not np.array_equal(court.frame_indices, [0]):
                raise ValueError("Static court calibration requires exactly the first source frame")
            if court.diagnostics is None or court.diagnostics.get("output_keypoint_contract") != "camera_view_v2":
                raise ValueError("Court output must declare camera-local schema")
            if court.diagnostics.get("temporal_policy") != "static_first_frame":
                raise ValueError("Court output must declare its static first-frame policy")
            diagnostics.extend(court.diagnostics["cameras"])
        result = CourtKPResult(np.concatenate([c.keypoints for c in inputs.courts]),
            np.concatenate([c.visibility for c in inputs.courts]), inputs.courts[0].frame_indices,
            {"output_keypoint_contract": "camera_view_v2", "cameras": diagnostics})
        calibration = calibrate_local_courts(result, inputs.source.camera_ids, size=inputs.source.size, config=self.config)
        by_camera = {view.camera.camera_id: view for view in calibration.views}
        polygons = {camera: court_footpoint_polygon_px(
            denormalize_grid_keypoints(result.keypoints[by_camera[camera].source_index, by_camera[camera].frame_index], *inputs.source.size),
            size=inputs.source.size, sideline_margin_m=self.roi_margins[0], baseline_margin_m=self.roi_margins[1])
            if camera in by_camera else None for camera in inputs.source.camera_ids}
        # Fixed-camera court geometry is broadcast explicitly; these are derived
        # coordinates, not a claim that later frames were independently inferred.
        expanded = CourtKPResult(np.repeat(result.keypoints, inputs.source.num_frames, axis=1),
            np.repeat(result.visibility, inputs.source.num_frames, axis=1),
            np.arange(inputs.source.num_frames, dtype=np.int32),
            {**(result.diagnostics or {}), "temporal_policy": "static_first_frame_broadcast",
             "observed_frame_indices": [0], "source_frame_count": inputs.source.num_frames})
        return CourtCalibrationOutput(expanded, calibration, calibration.reference(self.config), polygons)
