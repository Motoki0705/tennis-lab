"""Reproducible CPU experiment against an existing PLCS SceneResult archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.motion_alignment.diagnostics import (
    agreement,
    ankle_motion,
    direct_plcs_placement,
    reprojection,
    summary,
)
from src.tennis_scene.motion_alignment.similarity import (
    SimilarityConfig,
    fit_similarity,
)
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_player(
    source_keypoints_px: NDArray[np.float64],
    observed: NDArray[np.bool_],
    scene_keypoints: NDArray[np.float64],
    image_size: tuple[int, int],
) -> tuple[int, list[float]]:
    """Explicitly match independent extractor tracks by observed hip locations.

    Never equate independent numeric track IDs. Reject ambiguous or distant
    matches; normalized image-distance thresholds are diagnostic safeguards.
    """
    source = (source_keypoints_px[:, 11:13, :2] / image_size).mean(axis=1)
    target = scene_keypoints[:, :, 11:13].mean(axis=2)
    if not observed.any():
        raise ValueError("Cannot associate a source with no observed frames")
    errors = np.median(
        np.linalg.norm(target[:, observed] - source[observed], axis=-1), axis=1
    )
    order = np.argsort(errors)
    best = int(order[0])
    if errors[best] > 0.05 or (
        len(order) > 1 and errors[order[1]] - errors[best] < 0.03
    ):
        raise ValueError(
            f"Ambiguous source-to-PLCS player association: {errors.tolist()}"
        )
    return best, errors.tolist()


def compare_track(
    *,
    source_root: NDArray[np.float64],
    source_rotation: NDArray[np.float64],
    source_joints: NDArray[np.float64],
    source_confidence: NDArray[np.float64],
    observed: NDArray[np.bool_],
    target_position: NDArray[np.float64],
    target_yaw: NDArray[np.float64],
    observations: NDArray[np.float64],
    visibility: NDArray[np.float64],
    court_visibility: NDArray[np.float64],
    camera_fits: list[dict[str, Any]],
    fps: float,
    image_size: tuple[int, int],
    config: SimilarityConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit full tracks and evaluate temporal holdout, geometry and 2D agreement."""
    heading = np.arctan2(source_rotation[:, 1, 0], source_rotation[:, 0, 0])
    # These are observation-quality proxies, not calibrated PLCS uncertainty.
    vis = np.clip(visibility, 0, 1)
    source_vis = np.clip(source_confidence, 0, 1)
    court = np.clip(court_visibility, 0, 1).max(axis=0).mean(axis=-1)
    target_hips = vis[:, :, 11:13].mean(axis=-1).max(axis=0)
    target_torso = vis[:, :, [5, 6, 11, 12]].min(axis=-1).max(axis=0)
    weight_p = source_vis[:, 11:13].mean(axis=-1) * target_hips * court * observed
    horizontal = np.linalg.norm(source_rotation[:, :2, 0], axis=-1)
    weight_h = (
        source_vis[:, [5, 6, 11, 12]].min(axis=-1)
        * target_torso
        * court
        * observed
        * horizontal**2
    )
    valid = observed & (weight_p > 0)
    direct = direct_plcs_placement(
        source_joints,
        source_root,
        source_rotation,
        target_position,
        target_yaw,
        SMPL_Y_UP_TO_COURT_Z_UP.astype(np.float64),
    )
    arrays: dict[str, Any] = {
        "target_position": target_position,
        "target_yaw": target_yaw,
        "source_position": source_root,
        "source_yaw": heading,
        "source_joints": source_joints,
        "source_rotation": source_rotation,
        "direct_joints": direct,
        "position_weights": weight_p,
        "heading_weights": weight_h,
        "valid": valid,
    }
    metrics: dict[str, Any] = {
        "observed_frames": int(observed.sum()),
        "frames": len(observed),
        "position_weights": summary(weight_p[valid]),
        "heading_weights": summary(weight_h[valid]),
        "direct": {
            "reprojection_px": reprojection(
                direct, observations, visibility, camera_fits, image_size, valid
            ),
            "ankles": ankle_motion(source_joints, direct, valid, source_vis, fps),
        },
    }
    fitted = {}
    for mode in ("fixed", "free"):
        mode_config = replace(config, fixed_scale=1.0 if mode == "fixed" else None)
        fit = fit_similarity(
            source_root,
            target_position,
            heading,
            target_yaw,
            weight_p,
            weight_h,
            config=mode_config,
        )
        transform = fit.transform
        fitted[mode] = transform
        position, yaw = transform.apply(source_root), transform.apply_heading(heading)
        joints = transform.apply(source_joints)
        rotation = transform.rotation() @ source_rotation
        arrays.update(
            {
                f"{mode}_position": position,
                f"{mode}_yaw": yaw,
                f"{mode}_joints": joints,
                f"{mode}_rotation": rotation,
            }
        )
        # Held-out second half never enters the fitting objective.
        train = np.arange(len(valid)) < len(valid) // 2
        half_fit = fit_similarity(
            source_root,
            target_position,
            heading,
            target_yaw,
            weight_p * train,
            weight_h * train,
            config=mode_config,
        )
        heldout = agreement(
            half_fit.transform.apply(source_root),
            half_fit.transform.apply_heading(heading),
            target_position,
            target_yaw,
            valid & ~train,
        )
        rel_source = np.swapaxes(source_rotation[:-1], -1, -2) @ source_rotation[1:]
        rel_out = np.swapaxes(rotation[:-1], -1, -2) @ rotation[1:]
        source_length = np.linalg.norm(
            source_joints[:, 5] - source_joints[:, 6], axis=-1
        )
        output_length = np.linalg.norm(joints[:, 5] - joints[:, 6], axis=-1)
        metrics[mode] = {
            "config": asdict(mode_config),
            "transform": {
                "scale": transform.scale,
                "yaw_deg": float(np.rad2deg(transform.yaw)),
                "translation_m": transform.translation.tolist(),
            },
            "solver": asdict(fit.diagnostics),
            "cost": fit.cost,
            "agreement": agreement(position, yaw, target_position, target_yaw, valid),
            "temporal_quarters": [
                agreement(
                    position,
                    yaw,
                    target_position,
                    target_yaw,
                    valid & np.isin(np.arange(len(valid)), indices),
                )
                for indices in np.array_split(np.arange(len(valid)), 4)
            ],
            "first_half_fit_second_half_agreement": heldout,
            "first_half_transform": {
                "scale": half_fit.transform.scale,
                "yaw_deg": float(np.rad2deg(half_fit.transform.yaw)),
                "translation_m": half_fit.transform.translation.tolist(),
            },
            "reprojection_px": reprojection(
                joints, observations, visibility, camera_fits, image_size, valid
            ),
            "ankles": ankle_motion(source_joints, joints, valid, source_vis, fps),
            "ankle_height_m": summary(joints[valid][:, [15, 16], 2].ravel()),
            "relative_rotation_max_abs_difference": float(
                np.max(np.abs(rel_source - rel_out))
            ),
            "bone_scale_max_abs_difference": float(
                np.max(np.abs(output_length - transform.scale * source_length))
            ),
        }
    return metrics, arrays, fitted


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
