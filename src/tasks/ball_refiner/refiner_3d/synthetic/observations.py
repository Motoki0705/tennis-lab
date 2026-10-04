"""Camera-only Meiji geometry and provisional saved-pilot degradation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.geometry.triangulation import PinholeCamera

from .calibration import CalibrationBank


def load_cameras(path: Path, keys: list[str]) -> tuple[tuple[PinholeCamera, ...], NDArray[np.float64]]:
    raw = json.loads(path.read_text())
    values = [raw[key] for key in keys]  # no ball coordinate arrays are opened
    cameras = tuple(PinholeCamera(key, np.asarray(v["K"], dtype=float), np.asarray(v["R"], dtype=float), np.asarray(v["t"], dtype=float)) for key, v in zip(keys, values, strict=True))
    return cameras, np.asarray([[v["w"], v["h"]] for v in values], dtype=float)


def perturb_cameras(
    cameras: tuple[PinholeCamera, ...], settings: dict[str, Any], rng: np.random.Generator,
) -> tuple[PinholeCamera, ...]:
    result = []
    for camera in cameras:
        center = camera.center + rng.normal(0, settings["center_sigma_m"], 3)
        rotation = Rotation.from_rotvec(rng.normal(0, np.deg2rad(settings["axis_angle_sigma_deg"]), 3)).as_matrix() @ camera.rotation
        intrinsic = camera.intrinsic.copy()
        # Log-normal focal perturbation stays positive without retries/clipping.
        intrinsic[:2, :2] *= np.exp(rng.normal(0, settings["focal_relative_sigma"]))
        intrinsic[:2, 2] += rng.normal(0, settings["principal_point_sigma_px"], 2)
        result.append(PinholeCamera(camera.camera_id, intrinsic, rotation, -rotation @ center))
    return tuple(result)


def make_distribution(
    positions: NDArray[np.floating], cameras: tuple[PinholeCamera, ...],
    sizes: NDArray[np.float64], settings: dict[str, Any], rng: np.random.Generator,
    *, rally_index: int, calibration: CalibrationBank,
) -> tuple[BallGMM2D, dict[str, NDArray[Any]], dict[str, Any]]:
    v, t = len(cameras), len(positions)
    gap_lengths = settings["gap_lengths_frames"]
    shared_length = gap_lengths[2 + rally_index % (len(gap_lengths) - 2)]
    if t < shared_length + 8:
        raise ValueError(f"Rally too short for requested shared gap: {t} < {shared_length + 8}")
    occlusion: NDArray[np.bool_] = np.zeros((v, t), dtype=bool)
    start = (t - shared_length) // 2
    occlusion[:, start:start + shared_length] = True
    intervals = [{"cameras": list(range(v)), "start": start, "stop": start + shared_length}]
    for camera in range(v):
        length = gap_lengths[camera % 2]
        begin = int(rng.integers(0, t - length + 1))
        occlusion[camera, begin:begin + length] = True
        intervals.append({"cameras": [camera], "start": begin, "stop": begin + length})
    projected = [camera.project(positions) for camera in cameras]
    truth_px = np.stack([item[0] for item in projected])
    front = np.stack([item[1] for item in projected])
    scale = sizes - 1
    out_of_frame = (~front) | (truth_px < 0).any(-1) | (truth_px > scale[:, None]).any(-1)
    distribution, rows, clipped = calibrated_distribution(truth_px, sizes, settings, rng, calibration=calibration, occlusion=occlusion, out_of_frame=out_of_frame)
    masks = {"occlusion_mask": occlusion, "out_of_frame_mask": out_of_frame, "calibration_rows": rows}
    metadata = {"gap_intervals": intervals, "shared_gap_length": shared_length, "clipped_component_means": clipped, "calibration_status": settings["status"], "calibration_bank_sha256": settings["calibration"]["bank_sha256"]}
    return distribution, masks, metadata


def calibrated_distribution(
    truth_px: NDArray[np.floating], sizes: NDArray[np.float64], settings: dict[str, Any],
    rng: np.random.Generator, *, calibration: CalibrationBank,
    occlusion: NDArray[np.bool_], out_of_frame: NDArray[np.bool_],
) -> tuple[BallGMM2D, NDArray[np.int64], int]:
    """Apply one swappable bank to fixed projections and masks, retaining all K."""
    v, t, _ = truth_px.shape
    if occlusion.shape != (v, t) or out_of_frame.shape != (v, t):
        raise ValueError("Calibration masks must match fixed projections")
    scale = sizes - 1
    rows = np.stack([calibration.draw_rows(camera, occlusion[camera], rng, block_frames=settings["calibration"]["block_frames"]) for camera in range(v)])
    bank = calibration.arrays
    means_uv = truth_px[:, :, None] / scale[:, None, None] + bank["error_uv"][rows]
    clipped = ((means_uv < 0) | (means_uv > 1)).any(-1)
    means_uv = np.clip(means_uv, 0, 1)
    presence_logits = np.where(out_of_frame, settings["calibration"]["out_of_frame_presence_logit"], bank["presence_logits"][rows])
    distribution = BallGMM2D(
        torch.from_numpy(means_uv.astype(np.float32)),
        torch.from_numpy(bank["scale_tril_uv"][rows].copy()),
        torch.from_numpy(bank["mixture_logits"][rows].copy()),
        torch.from_numpy(presence_logits.astype(np.float32)),
    )
    if not bool(((distribution.presence_probability > 0) & (distribution.presence_probability < 1)).all()) or not bool((distribution.weights > 0).all()):
        raise ValueError("Full enumeration requires interior pilot presence and positive component weights")
    return distribution, rows, int(clipped.sum())
