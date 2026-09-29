"""Camera-only Meiji geometry and hypothesized #935-compatible degradation."""

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
    *, rally_index: int,
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
    shift = np.asarray(settings["alternative_shift_m"])
    alternatives = np.stack([positions, positions + shift, positions - shift], axis=1)
    projected = [camera.project(alternatives) for camera in cameras]
    means_px = np.stack([item[0] for item in projected])
    front = np.stack([item[1] for item in projected])
    scale = sizes - 1
    out_of_frame = (~front[:, :, 0]) | (means_px[:, :, 0] < 0).any(-1) | (means_px[:, :, 0] > scale[:, None]).any(-1)
    sigma = rng.uniform(settings["source_pixel_sigma_range"][0], settings["source_pixel_sigma_range"][1], size=(v, 1, 3, 2))
    sigma = np.broadcast_to(sigma, (v, t, 3, 2)).copy()
    sigma[:, :, 2] *= settings["distractor_sigma_multiplier"]
    sigma *= np.where(occlusion, settings["gap_sigma_multiplier"], 1)[:, :, None, None]
    rho = rng.uniform(settings["correlation_range"][0], settings["correlation_range"][1], size=(v, 1, 3))
    chol = np.zeros((v, t, 3, 2, 2))
    chol[..., 0, 0] = sigma[..., 0]
    chol[..., 1, 0] = rho * sigma[..., 1]
    chol[..., 1, 1] = np.sqrt(1 - rho ** 2) * sigma[..., 1]
    innovations = rng.standard_normal((v, t, 3, 2))
    errors = np.empty_like(innovations)
    errors[:, 0] = innovations[:, 0]
    ar = settings["error_ar1"]
    for frame in range(1, t):
        errors[:, frame] = ar * errors[:, frame - 1] + np.sqrt(1 - ar ** 2) * innovations[:, frame]
    means_px += np.einsum("vtkij,vtkj->vtki", chol, errors)
    # This explicit synthetic policy mimics the bounded #935 head, and is counted.
    means_uv = means_px / scale[:, None, None]
    clipped = ((means_uv < 0) | (means_uv > 1)).any(-1)
    means_uv = np.clip(means_uv, 0, 1)
    weights = np.broadcast_to(settings["observed_weights"], (v, t, 3)).copy()
    weights[occlusion] = settings["gap_weights"]
    presence_logits = np.where(out_of_frame, -settings["presence_logit_magnitude"], settings["presence_logit_magnitude"])
    distribution = BallGMM2D(
        torch.from_numpy(means_uv.astype(np.float32)),
        torch.from_numpy((chol / scale[:, None, None, :, None]).astype(np.float32)),
        torch.from_numpy(np.log(weights).astype(np.float32)),
        torch.from_numpy(presence_logits.astype(np.float32)),
    )
    masks = {"occlusion_mask": occlusion, "out_of_frame_mask": out_of_frame}
    metadata = {"gap_intervals": intervals, "shared_gap_length": shared_length, "clipped_component_means": int(clipped.sum())}
    return distribution, masks, metadata
