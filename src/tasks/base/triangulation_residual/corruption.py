"""Separate clean cameras/targets from noisy observations and estimated cameras."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from src.tasks.base.triangulation_residual.configuration import CorruptionConfig
from src.tasks.base.triangulation_residual.contracts import CameraRig
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


@dataclass(frozen=True)
class CorruptedObservations:
    observations_px: np.ndarray
    scores: np.ndarray
    court_px: np.ndarray
    court_scores: np.ndarray
    estimated_rig: CameraRig
    true_rig: CameraRig
    clean_uv: np.ndarray  # V,T,J,2; ideal undistorted target, NEVER model input
    clean_visible: np.ndarray  # V,T,J
    severity: float


def _distort(pixel: np.ndarray, rig: CameraRig, radial: np.ndarray) -> np.ndarray:
    shape = (len(rig.K),) + (1,) * (pixel.ndim - 2) + (2,)
    principal = rig.K[:, :2, 2].reshape(shape)
    focal = rig.K[:, [0, 1], [0, 1]].reshape(shape)
    xy = (pixel - principal) / focal
    k = radial.reshape((len(rig.K),) + (1,) * (pixel.ndim - 1))
    distorted: np.ndarray = np.asarray(
        principal + focal * xy * (1 + k * np.sum(xy**2, axis=-1, keepdims=True))
    )
    return distorted


def corrupt_observations(
    world_m: np.ndarray, rig: CameraRig, rng: np.random.Generator, cfg: CorruptionConfig
) -> CorruptedObservations:
    views, frames, joints = len(rig.K), len(world_m), world_m.shape[1]
    mix = rng.random()
    severity = (
        0.0
        if mix < cfg.clean_probability
        else (
            rng.uniform(2, 4)
            if mix < cfg.clean_probability + cfg.hard_probability
            else 1.0
        )
    )
    # Change true field of view independently of calibration error so training
    # covers wide-angle and narrower views without reading the evaluation clip.
    true_k = rig.K.copy()
    zoom = np.exp(
        rng.uniform(np.log(cfg.focal_scale_min), np.log(cfg.focal_scale_max), views)
    )
    true_k[:, 0, 0] *= zoom
    true_k[:, 1, 1] *= zoom
    true_rig = CameraRig(true_k, rig.R.copy(), rig.t.copy(), rig.image_size.copy())
    clean, depth = project_multiview(world_m.astype(float), true_rig.matrices)
    clean = clean.transpose(2, 0, 1, 3)
    uv = clean / rig.image_size[:, None, None]
    clean_visible = (
        (depth.transpose(2, 0, 1) > 0.05) & (uv >= 0).all(-1) & (uv <= 1).all(-1)
    )
    court_world = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy().astype(float)[:14]
    court, court_depth = project_multiview(court_world, true_rig.matrices)
    court = court.transpose(1, 0, 2)
    radial = rng.normal(0, cfg.radial_std * severity, views)
    obs = _distort(clean, true_rig, radial)
    court_obs = _distort(court, true_rig, radial)
    pixel_scale = rig.image_size[:, 1] / 1080
    joint_scale = np.ones(joints)
    if joints == 17:
        joint_scale[[7, 8, 9, 10, 15, 16]] = 1.6
    sigma = (
        cfg.observation_sigma_px
        * severity
        * pixel_scale[:, None, None, None]
        * joint_scale[None, None, :, None]
    )
    white = rng.normal(size=obs.shape) * sigma
    smooth = rng.normal(size=obs.shape)
    for t in range(1, frames):
        smooth[:, t] = 0.85 * smooth[:, t - 1] + np.sqrt(1 - 0.85**2) * smooth[:, t]
    smooth *= cfg.temporal_sigma_px * severity * pixel_scale[:, None, None, None]
    bias = (
        rng.normal(size=(views, 1, joints, 2))
        * cfg.view_bias_px
        * severity
        * pixel_scale[:, None, None, None]
    )
    obs += white + smooth + bias
    conf = np.clip(
        rng.normal(0.85, cfg.confidence_noise, (views, frames, joints)), 0.35, 1
    )
    if severity:
        outliers = rng.random((views, frames, joints)) < cfg.outlier_probability * min(
            severity, 2
        )
        obs += (
            outliers[..., None]
            * rng.normal(size=obs.shape)
            * cfg.outlier_sigma_px
            * severity
            * pixel_scale[:, None, None, None]
        )
        conf[outliers] = rng.uniform(0.1, 0.8, outliers.sum())
        conf[rng.random(conf.shape) < cfg.dropout_probability] = 0
        for v in range(views):
            if rng.random() < cfg.burst_probability:
                start = int(rng.integers(frames))
                stop = min(
                    frames, start + int(rng.integers(1, cfg.burst_max_frames + 1))
                )
                if rng.random() < 0.4:
                    conf[v, start:stop] = 0
                else:
                    conf[v, start:stop, int(rng.integers(joints))] = 0
            if v and rng.random() < cfg.time_shift_probability:
                shift = int(
                    rng.integers(
                        -cfg.time_shift_max_frames, cfg.time_shift_max_frames + 1
                    )
                )
                indices: np.ndarray = np.arange(frames) + shift
                inside = (indices >= 0) & (indices < frames)
                obs[v] = obs[v, np.clip(indices, 0, frames - 1)]
                conf[v] = conf[v, np.clip(indices, 0, frames - 1)] * inside[:, None]
    observed_uv = obs / rig.image_size[:, None, None]
    conf *= clean_visible & (observed_uv >= 0).all(-1) & (observed_uv <= 1).all(-1)
    obs[conf == 0] = np.nan
    court_obs += (
        rng.normal(size=court.shape) * 1.5 * severity * pixel_scale[:, None, None]
    )
    court_uv = court_obs / rig.image_size[:, None]
    court_scores = (
        (court_depth.T > 0.05) & (court_uv >= 0).all(-1) & (court_uv <= 1).all(-1)
    ).astype(float)
    estimated_k = true_k.copy()
    estimated_k[:, 0, 0] *= np.exp(
        rng.normal(0, cfg.camera_focal_log_std * severity, views)
    )
    estimated_k[:, 1, 1] *= np.exp(
        rng.normal(0, cfg.camera_focal_log_std * severity, views)
    )
    estimated_k[:, :2, 2] += (
        rng.normal(size=(views, 2))
        * cfg.camera_principal_std_px
        * severity
        * pixel_scale[:, None]
    )
    delta_rotation = Rotation.from_rotvec(
        rng.normal(size=(views, 3)) * np.deg2rad(cfg.camera_rotation_std_deg * severity)
    ).as_matrix()
    estimated_r = delta_rotation @ rig.R
    estimated_c = (
        rig.centers + rng.normal(size=(views, 3)) * cfg.camera_center_std_m * severity
    )
    estimated_t = -np.einsum("vij,vj->vi", estimated_r, estimated_c)
    estimated = CameraRig(estimated_k, estimated_r, estimated_t, rig.image_size.copy())
    return CorruptedObservations(
        obs,
        conf,
        court_obs,
        court_scores,
        estimated,
        true_rig,
        np.where(clean_visible[..., None], uv, 0).astype(np.float32),
        clean_visible,
        severity,
    )
