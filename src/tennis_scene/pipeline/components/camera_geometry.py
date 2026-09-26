"""Camera-local calibration and one shared, geometrically validated side decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.utilts.court_reference import fit_camera
from src.utils.geometry.triangulation import PinholeCamera, solve_homogeneous_dlt
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    CourtConfig,
    court_keypoints_3d,
)


@dataclass(frozen=True)
class CameraGeometryConfig:
    reference_camera: str | None = None
    calibration_samples: int = 9
    consensus_ratio: float = 0.01
    calibration_error_ratio: float = 0.005
    side_min_frames: int = 8
    side_max_cost: float = 0.5
    side_min_support: float = 0.5
    side_min_margin: float = 0.1

    def __post_init__(self) -> None:
        if self.reference_camera is not None and not self.reference_camera.strip():
            raise ValueError("Reference camera must be a nonempty ID")
        if self.calibration_samples < 1 or self.side_min_frames < 1:
            raise ValueError("Camera geometry sample counts must be positive")
        for value in (self.consensus_ratio, self.calibration_error_ratio, self.side_max_cost, self.side_min_support, self.side_min_margin):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Camera geometry thresholds must be in (0,1]")


@dataclass(frozen=True)
class LocalCourtCalibration:
    camera: PinholeCamera
    source_index: int
    frame_index: int
    homography: NDArray[np.float64]
    rmse_px: float
    support_frames: tuple[int, ...]

    def footpoint_polygon(self, *, sideline_margin_m: float = 1., baseline_margin_m: float = 5.) -> tuple[tuple[float, float], ...]:
        x, y = HALF_DOUBLES_WIDTH + sideline_margin_m, HALF_LENGTH + baseline_margin_m
        rectangle = np.array([[-x, y], [x, y], [x, -y], [-x, -y]], np.float64)
        pixels = cv2.perspectiveTransform(rectangle[None], self.homography)[0]
        if not np.isfinite(pixels).all():
            raise ValueError("Non-finite court ROI")
        return tuple((float(p[0]), float(p[1])) for p in pixels)


@dataclass(frozen=True)
class CalibrationSet:
    views: tuple[LocalCourtCalibration, ...]
    excluded: dict[str, str]

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(v.camera.camera_id for v in self.views)

    def reference(self, config: CameraGeometryConfig) -> str:
        if len(self.views) < 3:
            raise ReconstructionUnavailable("calibration_insufficient_views", "At least three calibrated model views are required", diagnostics={"excluded": self.excluded})
        reference = min(self.camera_ids) if config.reference_camera is None else config.reference_camera
        if reference not in self.camera_ids:
            raise ReconstructionUnavailable("reference_unavailable", f"Reference camera {reference!r} has no accepted calibration")
        return reference


@dataclass(frozen=True)
class SideEvidence:
    task: str
    uv_px: NDArray[np.float32]  # (identity,V,L,J,2)
    visibility: NDArray[np.bool_]
    reprojection_threshold_px: float

    def __post_init__(self) -> None:
        if self.uv_px.ndim != 5 or self.visibility.shape != self.uv_px.shape[:-1] or self.visibility.dtype != np.bool_:
            raise ValueError("Side evidence requires (ID,V,L,J,2) UV")
        if self.reprojection_threshold_px <= 0 or not np.isfinite(self.uv_px[self.visibility]).all():
            raise ValueError("Invalid side evidence")


@dataclass(frozen=True)
class CameraGeometryResult:
    camera_ids: tuple[str, ...]
    reference_camera: str
    view_half_turns: tuple[bool, ...]
    cameras: tuple[PinholeCamera, ...]
    selection: ReferenceViewSelection
    document: dict[str, Any]


def calibrate_local_courts(
    result: CourtKPResult,
    camera_ids: tuple[str, ...],
    *,
    size: tuple[int, int],
    config: CameraGeometryConfig,
) -> CalibrationSet:
    """Use accepted H provenance, never require all projected points in-image."""
    if result.keypoints.shape[0] != len(camera_ids) or result.keypoints.shape[2:] != (14, 2):
        raise ValueError("CourtKP14 observations must match camera IDs")
    diagnostics = result.diagnostics
    if not isinstance(diagnostics, dict) or diagnostics.get("output_keypoint_contract") != "camera_view_v2":
        raise ValueError("Automatic calibration requires recorded camera_view_v2 hybrid observations")
    camera_diagnostics = diagnostics.get("cameras")
    if not isinstance(camera_diagnostics, list) or len(camera_diagnostics) != len(camera_ids):
        raise ValueError("Automatic calibration requires per-camera hybrid diagnostics")
    frames = result.keypoints.shape[1]
    candidates = np.unique(np.rint(np.linspace(0, frames - 1, min(config.calibration_samples, frames))).astype(int))
    diagonal = math.hypot(*size)
    court_xyz = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14].astype(np.float64)
    views: list[LocalCourtCalibration] = []
    excluded: dict[str, str] = {}
    for view, camera_id in enumerate(camera_ids):
        records = camera_diagnostics[view].get("frames", [])
        by_frame = {int(r["frame_index"]): r for r in records}
        eligible = [int(t) for t in candidates if int(t) in by_frame and by_frame[int(t)].get("status") == "ok" and by_frame[int(t)].get("homography_court_metres_to_image_pixels") is not None]
        if not eligible:
            excluded[camera_id] = "no_accepted_homography"
            continue
        # CourtKPModule's documented legacy normalization is by W-1,H-1.
        pixel_points = result.keypoints[view, eligible].astype(np.float64) * np.maximum(np.asarray(size) - 1, 1)
        if not np.isfinite(pixel_points).all():
            excluded[camera_id] = "nonfinite_homography_points"
            continue
        distances = np.linalg.norm(pixel_points[:, None] - pixel_points[None], axis=-1).mean(-1)
        representative = int(np.argmin(distances.sum(-1)))
        supported = distances[representative] <= config.consensus_ratio * diagonal
        if int(supported.sum()) < min(3, len(candidates)):
            excluded[camera_id] = "homography_consensus_insufficient"
            continue
        frame = eligible[representative]
        homography = np.asarray(by_frame[frame]["homography_court_metres_to_image_pixels"], np.float64)
        if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
            excluded[camera_id] = "invalid_homography"
            continue
        projected_template = cv2.perspectiveTransform(court_xyz[None, :, :2], homography)[0]
        if not np.allclose(projected_template, pixel_points[representative], rtol=1e-6, atol=.01):
            excluded[camera_id] = "homography_keypoint_contract_mismatch"
            continue
        try:
            fit = fit_camera(pixel_points[representative] / np.asarray(size), size, False)
            camera = PinholeCamera(camera_id, np.asarray(fit["K"], np.float64), np.asarray(fit["R"], np.float64), np.asarray(fit["t"], np.float64))
        except (ValueError, cv2.error) as exc:
            excluded[camera_id] = f"pinhole_fit_failed: {exc}"
            continue
        if float(fit["rmse_px"]) > config.calibration_error_ratio * diagonal:
            excluded[camera_id] = "pinhole_reprojection"
            continue
        if not camera.project(court_xyz)[1].all():
            excluded[camera_id] = "court_behind_camera"
            continue
        views.append(LocalCourtCalibration(camera, view, frame, homography, float(fit["rmse_px"]), tuple(np.asarray(eligible)[supported].tolist())))
    return CalibrationSet(tuple(views), excluded)


def _check_side_evidence(evidence: tuple[SideEvidence, ...], views: int, reference: int, config: CameraGeometryConfig) -> None:
    if not evidence or not any(e.visibility.any() for e in evidence):
        raise ReconstructionUnavailable("no_reliable_association", "No accepted identity observations")
    lengths = {e.uv_px.shape[2] for e in evidence}
    if len(lengths) != 1 or any(e.uv_px.shape[1] != views for e in evidence):
        raise ValueError("Side evidence camera/timeline mismatch")
    length = next(iter(lengths))
    pair_frames = np.zeros((views, views, length), bool)
    usable = np.zeros(length, bool)
    for e in evidence:
        usable |= (e.visibility.sum(1) >= 2).any(axis=(0, 2))
        for a, b in combinations(range(views), 2):
            pair_frames[a, b] |= (e.visibility[:, a] & e.visibility[:, b]).any(axis=(0, 2))
            pair_frames[b, a] = pair_frames[a, b]
    if int(usable.sum()) < config.side_min_frames:
        raise ReconstructionUnavailable("side_evidence_insufficient", "Too few frames with cross-view identity evidence")
    connected = {reference}
    graph = pair_frames.sum(-1) >= config.side_min_frames
    while True:
        expanded = connected | {v for old in connected for v in range(views) if graph[old, v]}
        if expanded == connected:
            break
        connected = expanded
    if len(connected) != views:
        raise ReconstructionUnavailable("side_evidence_disconnected", "Not all camera sides have evidence connected to the reference")


def _score_candidate(evidence: tuple[SideEvidence, ...], cameras: tuple[PinholeCamera, ...]) -> tuple[float, float]:
    matrices = np.stack([c.matrix for c in cameras])
    costs: list[float] = []
    fractions: list[float] = []
    for e in evidence:
        object_costs, object_support = [], []
        for obj in range(len(e.uv_px)):
            mask = e.visibility[obj].reshape(len(cameras), -1)
            chosen = np.flatnonzero(mask.sum(0) >= 2)
            if not len(chosen):
                continue
            mask = mask[:, chosen]
            uv = e.uv_px[obj].reshape(len(cameras), -1, 2)[:, chosen]
            design = np.stack((uv[..., 0, None] * matrices[:, None, 2] - matrices[:, None, 0], uv[..., 1, None] * matrices[:, None, 2] - matrices[:, None, 1]), -2)
            design *= mask[..., None, None]
            xyz, valid = solve_homogeneous_dlt(design.transpose(1, 0, 2, 3).reshape(len(chosen), -1, 4))
            error = np.zeros(mask.shape, np.float64)
            for view, camera in enumerate(cameras):
                projected, front = camera.project(xyz)
                valid &= ~mask[view] | front
                error[view] = np.linalg.norm(projected - uv[view], axis=-1)
            rays = xyz[None] - np.stack([c.center for c in cameras])[:, None]
            rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
            angular: NDArray[np.bool_] = np.zeros(len(chosen), bool)
            for a, b in combinations(range(len(cameras)), 2):
                angular |= mask[a] & mask[b] & (np.abs((rays[a] * rays[b]).sum(-1)) <= math.cos(math.radians(1)))
            valid &= angular
            valid &= (np.abs(xyz[:, :2]) <= 40).all(-1)
            height = (-0.5, 4.) if e.task == "plcs" else (-0.2, 20.)
            valid &= (xyz[:, 2] >= height[0]) & (xyz[:, 2] <= height[1])
            normalized = (np.minimum((error / e.reprojection_threshold_px) ** 2, 1) * mask).sum(0) / mask.sum(0)
            normalized[~valid] = 1.
            support = valid & ((error <= e.reprojection_threshold_px) | ~mask).all(0)
            object_costs.append(float(normalized.mean()))
            object_support.append(float(support.mean()))
        if object_costs:
            costs.append(float(np.mean(object_costs)))
            fractions.append(float(np.mean(object_support)))
    if not costs:
        raise ReconstructionUnavailable("side_evidence_insufficient", "No paired observations for geometric scoring")
    return float(np.mean(costs)), float(np.mean(fractions))


def resolve_camera_geometry(
    calibration: CalibrationSet,
    reference_camera: str,
    side_predictions: tuple[NDArray[np.bool_], ...],
    evidence: tuple[SideEvidence, ...],
    *,
    config: CameraGeometryConfig,
) -> CameraGeometryResult:
    ids = calibration.camera_ids
    reference = ids.index(reference_camera)
    if not side_predictions or any(x.shape != (len(ids),) or x.dtype != np.bool_ for x in side_predictions):
        raise ValueError("Side predictions must match calibrated cameras")
    _check_side_evidence(evidence, len(ids), reference, config)
    choices = [(False,) if view == reference else tuple(sorted({bool(x[view]) for x in side_predictions})) for view in range(len(ids))]
    scored = []
    for sides in product(*choices):
        cameras = tuple(v.camera.half_turned(turn) for v, turn in zip(calibration.views, sides, strict=True))
        cost, support = _score_candidate(evidence, cameras)
        scored.append((cost, tuple(sides), support, cameras))
    scored.sort(key=lambda item: (item[0], item[1]))
    cost, sides, support, cameras = scored[0]
    receipt = [{"view_half_turns": list(x[1]), "cost": x[0], "support": x[2]} for x in scored]
    if cost > config.side_max_cost or support < config.side_min_support:
        raise ReconstructionUnavailable("side_geometry_rejected", "Side candidate lacks absolute geometric support", diagnostics={"candidates": receipt})
    if len(scored) > 1 and scored[1][0] - cost < config.side_min_margin:
        raise ReconstructionUnavailable("side_ambiguous", "Side candidate margin is insufficient", diagnostics={"candidates": receipt})
    contract = resolve_court_keypoint_contract("camera_view_v2")
    records = tuple(build_court_view_record(camera_id=c.camera_id, camera_center_court_m=c.center.tolist(), contract=contract) for c in cameras)
    selection = ReferenceViewSelection.create(stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(ids), selected_views=records, reference_camera_id=reference_camera)
    document: dict[str, Any] = {
        "camera_ids": list(ids), "reference_camera": reference_camera,
        "view_half_turns": list(sides), "side_candidates": receipt,
        "court_keypoint_views": [r.to_dict() for r in records],
        "court_reference_provenance": selection.provenance.to_dict(),
        "camera_fits": [{"K": c.intrinsic.tolist(), "R": c.rotation.tolist(), "t": c.translation.tolist(), "camera_center_court_m": c.center.tolist(), "rmse_px": local.rmse_px, "calibration_frame_index": local.frame_index, "support_frames": list(local.support_frames), "calibration": "approximate single-plane pinhole; no distortion correction"} for c, local in zip(cameras, calibration.views, strict=True)],
        "excluded_cameras": calibration.excluded,
    }
    write_model_artifact_court_keypoint_contract(document, contract)
    return CameraGeometryResult(ids, reference_camera, sides, cameras, selection, document)
