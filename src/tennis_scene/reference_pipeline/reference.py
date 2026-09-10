"""Court-view alignment and explicit reference metadata for real clips."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar

from src.tasks.base.data.track_query_reference import (
    ReferenceViewSelection,
    StableCameraIdTable,
)
from src.tasks.base.generate_dataset.court_view import (
    build_court_view_record,
    reference_court_keypoint_indices,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.blcs.model_io.contracts import BLCSReferenceMetadata
from src.tasks.plcs.model_io.contracts import PLCSReferenceMetadata
from src.utils.schema.court import CourtConfig, court_keypoints_3d


def fit_camera(
    kp: np.ndarray, size: tuple[int, int], half_turn: bool
) -> dict[str, Any]:
    """Approximate pinhole extrinsics from manual ground points; retain fit error.

    Intrinsics are estimated, not measured calibration. They supply camera-side
    provenance; PLCS/BLCS still consume the original 2D observations directly.
    """
    width, height = size
    xyz = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14].astype(np.float64)
    if half_turn:
        xyz[:, :2] *= -1
    pixels = kp.astype(np.float64) * [width, height]

    def solve(focal: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        k = np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]], np.float64
        )
        ok, r, t = cv2.solvePnP(xyz, pixels, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("Court camera pose fit failed")
        pred, _ = cv2.projectPoints(xyz, r, t, k, None)
        return float(np.square(pred[:, 0] - pixels).mean()), r, t, k

    fit = minimize_scalar(
        lambda f: solve(f)[0], bounds=(width * 0.2, width * 3), method="bounded"
    )
    mse, r, t, k = solve(float(fit.x))
    rotation = cv2.Rodrigues(r)[0]
    center = (-rotation.T @ t).ravel()
    if not np.isfinite(center).all() or center[2] <= 0 or (center[1] > 0) != half_turn:
        raise ValueError(f"Camera fit contradicts explicit court side: {center}")
    return {
        "camera_center_court_m": center.tolist(),
        "K": k.tolist(),
        "R": rotation.tolist(),
        "t": t.ravel().tolist(),
        "rmse_px": float(np.sqrt(mse)),
        "calibration": "approximate single-plane pinhole; no distortion correction",
    }


def build_reference(
    camera_ids: list[str],
    kp: np.ndarray,
    half_turns: list[bool],
    reference_camera: str,
    size: tuple[int, int],
) -> tuple[np.ndarray, ReferenceViewSelection, dict[str, Any]]:
    if len(half_turns) != len(camera_ids) or any(
        type(x) is not bool for x in half_turns
    ):
        raise ValueError("One explicit boolean view_half_turn per camera is required")
    if (
        reference_camera not in camera_ids
        or half_turns[camera_ids.index(reference_camera)]
    ):
        raise ValueError("The reference camera must define the unrotated court gauge")
    contract = resolve_court_keypoint_contract("camera_view_v2")
    fits = [
        fit_camera(view[0], size, turn)
        for view, turn in zip(kp, half_turns, strict=True)
    ]
    views = tuple(
        build_court_view_record(
            camera_id=cam,
            camera_center_court_m=fit["camera_center_court_m"],
            contract=contract,
        )
        for cam, fit in zip(camera_ids, fits, strict=True)
    )
    table = StableCameraIdTable.from_complete_scene_camera_ids(tuple(camera_ids))
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=table,
        selected_views=views,
        reference_camera_id=reference_camera,
    )
    reference = views[selection.reference_view_index]
    aligned = []
    for view, points in zip(views, kp, strict=True):
        indices = reference_court_keypoint_indices(view, reference)[:14]
        if set(indices) != set(range(14)):
            raise ValueError("CourtKP14 is not closed under reference permutation")
        aligned.append(points[:, indices])
    document: dict[str, Any] = {
        "camera_ids": camera_ids,
        "reference_camera": reference_camera,
        "court_keypoint_views": [v.to_dict() for v in views],
        "camera_fits": fits,
        "court_reference_provenance": selection.provenance.to_dict(),
    }
    write_model_artifact_court_keypoint_contract(document, contract)
    return np.asarray(aligned), selection, document


def reference_metadata(
    selection: ReferenceViewSelection, batch_size: int, task: str
) -> PLCSReferenceMetadata | BLCSReferenceMetadata:
    if task not in {"plcs", "blcs"} or batch_size <= 0:
        raise ValueError("Expected plcs/blcs and a positive batch size")
    fields = selection.to_tensor_fields(dtype=torch.float32)

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).repeat((batch_size,) + (1,) * value.ndim)

    forward = expand(fields["reference_from_physical"])
    cls = PLCSReferenceMetadata if task == "plcs" else BLCSReferenceMetadata
    return cls(
        selections=(selection,) * batch_size,
        stable_camera_id_tables=(selection.stable_camera_id_table,) * batch_size,
        reference_view_index=expand(fields["reference_view_index"]),
        view_camera_ids=expand(fields["view_camera_ids"]),
        reference_camera_id=expand(fields["reference_camera_id"]),
        reference_from_physical=forward,
        physical_from_reference=forward.transpose(-1, -2),
    )
