"""Observable diagnostics; PLCS agreement is not ground-truth 3D accuracy."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def wrap_angle(angle: FloatArray) -> FloatArray:
    return cast(FloatArray, np.asarray((angle + np.pi) % (2 * np.pi) - np.pi))


def summary(values: FloatArray) -> dict[str, float | int | None]:
    values = np.asarray(values)[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "rmse": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def agreement(
    position: FloatArray,
    yaw: FloatArray,
    target_position: FloatArray,
    target_yaw: FloatArray,
    valid: NDArray[np.bool_],
) -> dict[str, Any]:
    delta = position - target_position
    return {
        "position_m": summary(np.linalg.norm(delta[valid], axis=-1)),
        "xy_m": summary(np.linalg.norm(delta[valid, :2], axis=-1)),
        "z_m": summary(np.abs(delta[valid, 2])),
        "heading_deg": summary(np.abs(np.rad2deg(wrap_angle(yaw - target_yaw)))[valid]),
    }


def direct_plcs_placement(
    joints: FloatArray,
    root: FloatArray,
    rotation: FloatArray,
    target_position: FloatArray,
    target_yaw: FloatArray,
    basis: FloatArray,
) -> FloatArray:
    """Controlled baseline: same articulation, existing renderer placement rule.

    Remove the full original global rotation, convert the body to Z-up, then
    apply each PLCS yaw/position. This intentionally reproduces the renderer's
    loss of GVHMR global tilt and world translation.
    """
    body = np.einsum("tji,tkj->tki", rotation, joints - root[:, None])
    local = body @ basis.T
    c, s = np.cos(target_yaw), np.sin(target_yaw)
    result = local.copy()
    result[..., 0] = c[:, None] * local[..., 0] - s[:, None] * local[..., 1]
    result[..., 1] = s[:, None] * local[..., 0] + c[:, None] * local[..., 1]
    return cast(FloatArray, result + target_position[:, None])


def reprojection(
    joints: FloatArray,
    observations: FloatArray,
    visibility: FloatArray,
    camera_fits: list[dict[str, Any]],
    image_size: tuple[int, int],
    valid: NDArray[np.bool_],
) -> dict[str, Any]:
    """Compare COCO17 joints to all views using saved approximate calibration."""
    result = {}
    for camera, fit in enumerate(camera_fits):
        xyz = joints @ np.asarray(fit["R"]).T + np.asarray(fit["t"])
        homogeneous = xyz @ np.asarray(fit["K"]).T
        projected = homogeneous[..., :2] / np.maximum(homogeneous[..., 2:], 1e-9)
        usable = (visibility[camera] >= 0.5) & valid[:, None] & (xyz[..., 2] > 0)
        error = np.linalg.norm(projected - observations[camera] * image_size, axis=-1)
        result[f"camera_{camera}"] = summary(error[usable])
    return result


def ankle_motion(
    source_joints: FloatArray,
    output_joints: FloatArray,
    valid: NDArray[np.bool_],
    confidence: FloatArray,
    fps: float,
) -> dict[str, Any]:
    """Low-speed source-ankle proxy, not a measured foot-contact label.

    One source-derived mask is shared by every output method. A static
    similarity must multiply these speeds by exactly its spatial scale.
    """
    source_speed = (
        np.linalg.norm(np.diff(source_joints[:, [15, 16]], axis=0), axis=-1) * fps
    )
    output_speed = (
        np.linalg.norm(np.diff(output_joints[:, [15, 16]], axis=0), axis=-1) * fps
    )
    usable = valid[1:, None] & valid[:-1, None]
    usable = (
        usable & (confidence[1:, [15, 16]] >= 0.5) & (confidence[:-1, [15, 16]] >= 0.5)
    )
    slow = usable & (source_speed < 0.2)
    return {
        "proxy": "source ankle speed <0.2m/s, confidence>=0.5, consecutive observed frames",
        "source_low_speed_mps": summary(source_speed[slow]),
        "output_on_same_mask_mps": summary(output_speed[slow]),
        "output_all_observed_mps": summary(output_speed[usable]),
    }
