"""Shared CourtKP reference-frame preparation for integrated inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_physical_court_provenance,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.blcs.model_io.contracts import BLCSReferenceMetadata
from src.tasks.plcs.model_io.contracts import PLCSReferenceMetadata
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    CourtConfig,
    court_keypoints_3d,
)


@dataclass(frozen=True, slots=True)
class CourtReferenceRuntimeConfig:
    """Explicit camera-side declarations needed by camera-view CourtKP."""

    reference_camera: str | None
    view_half_turns: tuple[bool, ...] | None


@dataclass(frozen=True, slots=True)
class CourtReferenceContext:
    """Camera-local observations and exact geometry provenance for one clip."""

    keypoints: np.ndarray
    visibility: np.ndarray
    provenance: CourtReferenceFrameProvenance
    document: dict[str, Any] | None
    selection: ReferenceViewSelection | None


def court_footpoint_polygon_px(
    keypoints: np.ndarray,
    *,
    size: tuple[int, int],
    sideline_margin_m: float,
    baseline_margin_m: float,
) -> tuple[tuple[float, float], ...]:
    """Clip the expanded court's visible image region, including large margins.

    Projecting four corners directly can fold the polygon when a margin reaches
    behind the camera. Clipping the image by inverse-homography halfplanes keeps
    the branch containing the court centre and never includes that folded region.
    The expanded rectangle is invariant under the camera-local half-turn, so
    CourtKP14 in physical or camera-local order defines the same image region.
    """
    if keypoints.shape != (14, 2) or not np.isfinite(keypoints).all():
        raise ValueError("Court footpoint filtering requires 14 finite keypoints.")
    if (
        not np.isfinite(sideline_margin_m)
        or not np.isfinite(baseline_margin_m)
        or sideline_margin_m < 0
        or baseline_margin_m < 0
    ):
        raise ValueError(
            "Court footpoint filter margins must be finite and non-negative."
        )
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Court footpoint filtering requires a positive image size.")
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    pixels = keypoints.astype(np.float32) * np.array([width, height], np.float32)
    physical_to_pixels, _ = cv2.findHomography(
        physical.astype(np.float32), pixels, method=0
    )
    if physical_to_pixels is None:
        raise ValueError("Court footpoint filter homography fit failed.")
    x = HALF_DOUBLES_WIDTH + sideline_margin_m
    y = HALF_LENGTH + baseline_margin_m
    try:
        inverse = np.linalg.inv(physical_to_pixels)
    except np.linalg.LinAlgError as error:
        raise ValueError("Court footpoint filter homography is singular.") from error
    if not np.isfinite(inverse).all() or abs(physical_to_pixels[2, 2]) < 1e-12:
        raise ValueError("Court footpoint filter has no finite court-centre branch.")
    inverse *= np.sign(physical_to_pixels[2, 2])
    world_x, world_y, denominator = inverse
    halfplanes = (
        denominator,
        x * denominator + world_x,
        x * denominator - world_x,
        y * denominator + world_y,
        y * denominator - world_y,
    )
    polygon = np.array(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    for halfplane in halfplanes:
        polygon = _clip_image_halfplane(polygon, halfplane)
        if len(polygon) < 3:
            raise ValueError("Court footpoint filter does not intersect the image.")
    if abs(cv2.contourArea(polygon.astype(np.float32))) < 1e-6:
        raise ValueError("Court footpoint filter has no visible image area.")
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def _clip_image_halfplane(polygon: np.ndarray, halfplane: np.ndarray) -> np.ndarray:
    """Retain pixel points satisfying a*u + b*v + c >= 0."""
    result: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(previous @ halfplane[:2] + halfplane[2])
    for current in polygon:
        current_value = float(current @ halfplane[:2] + halfplane[2])
        if (previous_value >= 0) != (current_value >= 0):
            fraction = previous_value / (previous_value - current_value)
            result.append(previous + fraction * (current - previous))
        if current_value >= 0:
            result.append(current)
        previous, previous_value = current, current_value
    unique: list[np.ndarray] = []
    for point in result:
        if not unique or not np.allclose(point, unique[-1], rtol=0.0, atol=1e-8):
            unique.append(point)
    if len(unique) > 1 and np.allclose(unique[0], unique[-1], rtol=0.0, atol=1e-8):
        unique.pop()
    clipped: np.ndarray = np.asarray(unique, dtype=np.float64).reshape(-1, 2)
    return clipped


def fit_camera(
    keypoints: np.ndarray,
    size: tuple[int, int],
    half_turn: bool,
) -> dict[str, Any]:
    """Fit approximate camera parameters used only as audited side provenance."""
    width, height = size
    xyz = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14].astype(np.float64)
    if half_turn:
        xyz[:, :2] *= -1
    pixels = keypoints.astype(np.float64) * [width, height]

    def solve(focal: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        intrinsic = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]],
            np.float64,
        )
        ok, rotation_vector, translation = cv2.solvePnP(
            xyz,
            pixels,
            intrinsic,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("Court camera pose fit failed")
        projected, _ = cv2.projectPoints(
            xyz,
            rotation_vector,
            translation,
            intrinsic,
            None,
        )
        mse = float(np.square(projected[:, 0] - pixels).mean())
        return mse, rotation_vector, translation, intrinsic

    fit = minimize_scalar(
        lambda focal: solve(float(focal))[0],
        bounds=(width * 0.2, width * 3),
        method="bounded",
    )
    if not fit.success:
        raise ValueError("Court camera focal fit did not converge")
    mse, rotation_vector, translation, intrinsic = solve(float(fit.x))
    rotation = cv2.Rodrigues(rotation_vector)[0]
    center = (-rotation.T @ translation).ravel()
    if not np.isfinite(center).all() or center[2] <= 0 or (center[1] > 0) != half_turn:
        raise ValueError(f"Camera fit contradicts explicit court side: {center}")
    return {
        "camera_center_court_m": center.tolist(),
        "K": intrinsic.tolist(),
        "R": rotation.tolist(),
        "t": translation.ravel().tolist(),
        "rmse_px": float(np.sqrt(mse)),
        "calibration": "approximate single-plane pinhole; no distortion correction",
    }


def prepare_court_reference(
    *,
    camera_ids: tuple[str, ...],
    keypoints: np.ndarray,
    visibility: np.ndarray,
    contract: CourtKeypointContract,
    config: CourtReferenceRuntimeConfig,
    size: tuple[int, int],
    frame_index: int,
) -> CourtReferenceContext:
    """Preserve camera-local CourtKP slots and build downstream geometry context."""
    if keypoints.ndim != 4 or keypoints.shape[0] != len(camera_ids):
        raise ValueError(
            "Court reference keypoints must have shape (N,T,K,2) matching cameras."
        )
    if visibility.shape != keypoints.shape[:-1]:
        raise ValueError("Court reference visibility must match keypoints[:3].")
    if frame_index < 0 or frame_index >= keypoints.shape[1]:
        raise ValueError("Court reference frame_index is outside the clip timeline.")
    if contract.selector == PHYSICAL_V1_SELECTOR:
        if config.reference_camera is not None or config.view_half_turns is not None:
            raise ValueError(
                "physical_v1 forbids camera-view reference_camera/view_half_turns."
            )
        return CourtReferenceContext(
            keypoints=keypoints,
            visibility=visibility,
            provenance=build_physical_court_provenance(),
            document=None,
            selection=None,
        )

    reference_camera = config.reference_camera
    half_turns = config.view_half_turns
    if reference_camera is None or half_turns is None:
        raise ValueError(
            "camera_view_v2 requires reference_camera and one view_half_turn per camera."
        )
    if len(half_turns) != len(camera_ids):
        raise ValueError("camera_view_v2 requires one view_half_turn per camera.")
    if reference_camera not in camera_ids:
        raise ValueError("camera_view_v2 reference_camera is not in camera_ids.")
    if half_turns[camera_ids.index(reference_camera)]:
        raise ValueError("The reference camera must define the unrotated court gauge.")
    calibration_points = keypoints[:, frame_index]
    calibration_visibility = visibility[:, frame_index]
    if calibration_points.shape[1:] != (14, 2):
        raise ValueError("camera_view_v2 integrated inference requires CourtKP14.")
    if (
        not np.isfinite(calibration_points).all()
        or not (calibration_visibility == 1).all()
    ):
        raise ValueError(
            "camera_view_v2 calibration frame requires all 14 finite visible points."
        )

    fits = [
        fit_camera(points, size, half_turn)
        for points, half_turn in zip(calibration_points, half_turns, strict=True)
    ]
    views = tuple(
        build_court_view_record(
            camera_id=camera_id,
            camera_center_court_m=fit["camera_center_court_m"],
            contract=contract,
        )
        for camera_id, fit in zip(camera_ids, fits, strict=True)
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(camera_ids)
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=views,
        reference_camera_id=reference_camera,
    )
    document: dict[str, Any] = {
        "camera_ids": list(camera_ids),
        "reference_camera": reference_camera,
        "view_half_turns": list(half_turns),
        "calibration_frame_index": frame_index,
        "court_keypoint_views": [view.to_dict() for view in views],
        "camera_fits": fits,
        "court_reference_provenance": selection.provenance.to_dict(),
    }
    write_model_artifact_court_keypoint_contract(document, contract)
    return CourtReferenceContext(
        keypoints=keypoints,
        visibility=visibility,
        provenance=selection.provenance,
        document=document,
        selection=selection,
    )


def reference_metadata(
    selection: ReferenceViewSelection,
    batch_size: int,
    task: Literal["plcs", "blcs"],
) -> PLCSReferenceMetadata | BLCSReferenceMetadata:
    """Create task-owned typed metadata from one validated view selection."""
    if batch_size <= 0:
        raise ValueError("Reference metadata batch_size must be positive.")
    fields = selection.to_tensor_fields(dtype=torch.float32)

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).repeat((batch_size,) + (1,) * value.ndim)

    forward = expand(fields["reference_from_physical"])
    metadata_type = PLCSReferenceMetadata if task == "plcs" else BLCSReferenceMetadata
    return metadata_type(
        selections=(selection,) * batch_size,
        stable_camera_id_tables=(selection.stable_camera_id_table,) * batch_size,
        reference_view_index=expand(fields["reference_view_index"]),
        view_camera_ids=expand(fields["view_camera_ids"]),
        reference_camera_id=expand(fields["reference_camera_id"]),
        reference_from_physical=forward,
        physical_from_reference=forward.transpose(-1, -2),
    )




__all__ = [
    "CourtReferenceContext",
    "CourtReferenceRuntimeConfig",
    "court_footpoint_polygon_px",
    "fit_camera",
    "prepare_court_reference",
    "reference_metadata",
]
