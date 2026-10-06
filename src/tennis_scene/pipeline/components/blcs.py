"""Learned single-ball reconstruction in the aligned physical court frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.blcs.configuration import parse_model_config
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io.checkpoints import load_checkpoint_runtime
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.contracts import ClipSource, ComponentIO, InputPort
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.configuration import PathResolver
from src.utils.geometry.keypoints import denormalize_grid_keypoints, normalize_keypoints
from src.utils.geometry.triangulation import (
    PointRejection,
    TriangulatedPoints,
    reject_excessive_speed,
)
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX


@dataclass(frozen=True)
class BallReconstructionInput:
    source: ClipSource
    alignment: CameraAlignmentOutput
    observations: GroupedObservations
    court: CourtKPResult
    camera_ids: tuple[str, ...]


@dataclass(frozen=True)
class BallReconstructionOutput:
    ball: BallReconstructionResult | None


class BLCSReconstructionModule:
    def __init__(self, camera_ids: tuple[str, ...], *, checkpoint: Path, resolver: PathResolver,
                 device: str, window_size: int, reprojection_px: float, min_frames: int,
                 enabled: bool = True) -> None:
        self.checkpoint, self.resolver, self.device = checkpoint, resolver, device
        self.window_size, self.reprojection_px, self.min_frames = window_size, reprojection_px, min_frames
        if window_size < 1 or min_frames < 1 or not np.isfinite(reprojection_px) or reprojection_px <= 0:
            raise ValueError("BLCS reconstruction thresholds and window must be positive")
        self.enabled = enabled
        self._predictor: BLCSPredictor | None = None
        self._model_fps: float | None = None
        self.io = ComponentIO("ball_reconstruction", BallReconstructionInput, BallReconstructionOutput,
            {"alignment": InputPort("aligned_cameras"), "calibration": InputPort("local_court_calibration"),
             **{f"ball_{c}": InputPort("ball_points", version=3) for c in camera_ids}}, "ball_trajectory", 2)

    def unload(self) -> None:
        self._predictor = None
        self._model_fps = None

    def _load_predictor(self) -> BLCSPredictor:
        if self._predictor is None:
            runtime = load_checkpoint_runtime(self.checkpoint, runtime_court_keypoints="physical_v1")
            model_config = parse_model_config(runtime.config)
            if model_config.num_court_tokens != 14 or self.window_size > model_config.max_seq_len:
                raise ValueError("Scene BLCS requires KP14 and a window within checkpoint capacity")
            self._model_fps = float(runtime.config["rally"]["output_fps"])
            self._predictor = BLCSPredictor.load_from_checkpoint(
                self.checkpoint, resolver=self.resolver, device=self.device, court_keypoints="physical_v1")
        return self._predictor

    def process(self, inputs: BallReconstructionInput) -> BallReconstructionOutput:
        geometry = inputs.alignment.geometry
        if geometry is None or not self.enabled:
            return BallReconstructionOutput(None)
        count, views, frames, joints, _ = inputs.observations.uv_px.shape
        if count > 1 or joints != 1 or geometry.camera_ids != tuple(c.camera_id for c in geometry.cameras):
            raise ValueError("BLCS requires one point stream in aligned camera order")
        if views != len(geometry.cameras) or inputs.court.keypoints.shape != (views, frames, 14, 2):
            raise ValueError("BLCS ball, court and camera axes must agree")
        if inputs.camera_ids != geometry.camera_ids or len(geometry.view_half_turns) != views:
            raise ValueError("BLCS observation IDs and side assignments must match aligned cameras")
        if geometry.view_half_turns[geometry.camera_ids.index(geometry.reference_camera)]:
            raise ValueError("BLCS physical court output requires an unturned reference camera")
        if inputs.court.diagnostics is None or inputs.court.diagnostics.get("output_keypoint_contract") != "camera_view_v2":
            raise ValueError("BLCS scene input requires explicit camera-local CourtKP14 observations")
        uv = inputs.observations.uv_px[0, :, :, 0] if count else np.zeros((views, frames, 2), np.float32)
        visible = inputs.observations.visibility[0, :, :, 0] if count else np.zeros((views, frames), bool)
        confidence = inputs.observations.confidence[0, :, :, 0] if count else np.zeros((views, frames), np.float32)
        observed = visible & (confidence > 0)
        if int((observed.sum(0) >= 2).sum()) < self.min_frames:
            empty = TriangulatedPoints(np.zeros((frames, 3), np.float32), np.zeros(frames, bool),
                np.full(frames, int(PointRejection.INSUFFICIENT_VIEWS), np.uint8), np.zeros((views, frames), bool),
                np.zeros((views, frames), np.float32))
            return BallReconstructionOutput(BallReconstructionResult(uv, visible, empty, "ball_insufficient_support"))
        width, height = inputs.source.size
        court = normalize_keypoints(denormalize_grid_keypoints(inputs.court.keypoints[:, 0], width, height), width, height)
        court_visibility = inputs.court.visibility[:, 0].copy()
        court = np.asarray(court, np.float32).copy()
        permutation = np.asarray(CAMERA_VIEW_HALF_TURN_INDEX)
        for index, turned in enumerate(geometry.view_half_turns):
            if turned:
                court[index] = court[index, permutation]
                court_visibility[index] = court_visibility[index, permutation]
        predictor = self._load_predictor()
        if predictor.input_profile != "multiview":
            raise ValueError("Scene BLCS requires the physical multiview KP14 model")
        # The checkpoint's temporal grid is explicit. Repeated nearest input
        # samples are model inputs only; source observation masks stay unchanged.
        if self._model_fps is None or not np.isfinite(self._model_fps) or self._model_fps <= 0:
            raise RuntimeError("BLCS checkpoint must declare a positive output frame rate")
        model_fps = self._model_fps
        model_count = int(np.ceil((frames - 1) * model_fps / inputs.source.fps)) + 1
        model_times = np.arange(model_count, dtype=np.float64) / model_fps
        indices = np.minimum(np.rint(model_times * inputs.source.fps).astype(np.int64), frames - 1)
        model_uv = normalize_keypoints(uv[:, indices], width, height).astype(np.float32)
        model_visible = observed[:, indices]
        document: dict[str, Any] = {}
        write_model_artifact_court_keypoint_contract(document, resolve_court_keypoint_contract("physical_v1"))
        predicted: NDArray[np.float32] = np.empty((model_count, 3), np.float32)
        for start in range(0, model_count, self.window_size):
            stop = min(start + self.window_size, model_count)
            output = predictor.predict_multiview_arrays(ball_uv=model_uv[:, start:stop], court_kp=court,
                ball_vis=model_visible[:, start:stop], court_vis=court_visibility, denormalize=True,
                court_keypoint_document=document)
            positions = output.position[0].numpy()
            if positions.shape != (stop - start, 3) or not np.isfinite(positions).all():
                raise RuntimeError("BLCS returned an invalid trajectory")
            predicted[start:stop] = positions
        times = np.arange(frames, dtype=np.float64) / inputs.source.fps
        positions = np.column_stack([np.interp(times, model_times, predicted[:, axis]) for axis in range(3)]).astype(np.float32)
        threshold = self.reprojection_px * inputs.source.pixel_threshold_scale
        errors = np.zeros((views, frames), np.float32)
        front = np.zeros((views, frames), bool)
        for index, camera in enumerate(geometry.cameras):
            projection, front[index] = camera.project(positions)
            errors[index] = np.where(observed[index], np.linalg.norm(projection - uv[index], axis=-1), 0)
        inliers = observed & front & (errors <= threshold)
        supported = observed.sum(0) >= 2
        reasons = np.full(frames, int(PointRejection.INSUFFICIENT_VIEWS), np.uint8)
        reasons[supported] = int(PointRejection.REPROJECTION)
        reasons[supported & ((observed & front).sum(0) < 2)] = int(PointRejection.BEHIND_CAMERA)
        valid = inliers.sum(0) >= 2
        bounds = (positions[:, 2] >= -.2) & (positions[:, 2] <= 20.) & (np.abs(positions[:, :2]) <= 40.).all(-1)
        reasons[valid & ~bounds] = int(PointRejection.OUTSIDE_BOUNDS)
        valid &= bounds
        reasons[valid] = int(PointRejection.VALID)
        trajectory = reject_excessive_speed(TriangulatedPoints(np.where(valid[:, None], positions, 0).astype(np.float32),
            valid, reasons, inliers & valid[None], errors), fps=inputs.source.fps, max_speed_mps=65.)
        status = "ok"
        if int(trajectory.valid.sum()) < self.min_frames:
            trajectory = TriangulatedPoints(np.zeros((frames, 3), np.float32), np.zeros(frames, bool),
                np.full(frames, int(PointRejection.INSUFFICIENT_VIEWS), np.uint8), np.zeros((views, frames), bool), errors)
            status = "ball_insufficient_support"
        return BallReconstructionOutput(BallReconstructionResult(uv, visible, trajectory, status))
