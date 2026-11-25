"""Explicit monocular PLCS (+ HTD-Refine) player 3D skeleton component for tennis_scene.

Consumes calibrated pinhole cameras (from ``court_detection -> court_calibration``
or ``camera_alignment``), real-detector ``boxes_xys`` and ViTPose COCO-17 2D
observations, and reconstructs court-space 3D player skeletons (``PlayerSkeleton``)
and placement (``ReconstructedPlayers`` / ``SceneResult v2``) without any HMR2
image-feature extraction or SMPL-X body-model assets.

Contracts enforced strictly (zero silent fallbacks):
- Uncalibrated or missing cameras raise ``ReconstructionUnavailable`` / ``ValueError``.
- GSI-interpolated frames (``raw.observed == False``) are never treated as real
  observations and remain invalid/zeroed in outputs.
- Every accepted 3D skeleton frame satisfies finite bounds and
  ``SkeletonValues.require_valid`` across bone lengths, joint angles, torsion
  angles, and torso twist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import write_model_artifact_court_keypoint_contract
from src.tasks.plcs.geometry.monocular_camera_bbox import LEFT_HIP_IDX, RIGHT_HIP_IDX
from src.tasks.plcs.inference.htd_refiner import (
    HTDRefineCameraView,
    HTDRefinerConfig,
    PLCSHTDRefiner,
)
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io.contracts import PLCSPhysicalPrediction
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryResult,
    LocalCourtCalibration,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.identity import IDENTITIES_PORT
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    ReconstructedPlayers,
)
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.input_assembly.observations import (
    gather_people,
    identified_people,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
    group_observations,
)
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathResolver
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.geometry.triangulation import PinholeCamera, PointRejection
from src.utils.schema.player import NUM_HUMAN_KP
from src.utils.video import VideoInfo


class MonocularPLCSSequencePredictor(Protocol):
    """Protocol satisfied by ``PLCSPredictor`` for monocular sequence inference."""

    def predict_monocular_sequence(
        self,
        *,
        kp2d_px: NDArray[np.floating[Any]],
        kp_vis: NDArray[np.floating[Any]],
        bbx_xys: NDArray[np.floating[Any]] | None,
        frame_observed: NDArray[np.bool_] | None,
        R_w2c: NDArray[np.floating[Any]],
        C_w: NDArray[np.floating[Any]],
        fx: float | NDArray[np.floating[Any]],
        fy: float | NDArray[np.floating[Any]],
        cx0: float | NDArray[np.floating[Any]],
        cy0: float | NDArray[np.floating[Any]],
        image_w: int | float,
        image_h: int | float,
        fps: float = 60.0,
        vis_threshold: float = 0.0,
        htd_refiner: PLCSHTDRefiner | None = None,
    ) -> PLCSPhysicalPrediction: ...


@dataclass(frozen=True)
class MonocularPLCSConfig:
    """Explicit configuration for monocular PLCS (+ HTD-Refine) reconstruction."""

    checkpoint: Path | None = None
    device: str = "cpu"
    joint_confidence: float = 0.25
    reprojection_px: float = 40.0
    htd_refine: bool = True
    htd_config: HTDRefinerConfig = field(default_factory=HTDRefinerConfig)
    use_multiview_reprojection_refine: bool = True
    height_range_m: tuple[float, float] = (-0.5, 4.0)
    max_abs_xy_m: float = 40.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.joint_confidence) and 0.0 <= self.joint_confidence <= 1.0):
            raise ValueError("joint_confidence must be finite and within [0, 1].")
        if not (math.isfinite(self.reprojection_px) and self.reprojection_px > 0.0):
            raise ValueError("reprojection_px must be finite and positive.")
        if not (math.isfinite(self.max_abs_xy_m) and self.max_abs_xy_m > 0.0):
            raise ValueError("max_abs_xy_m must be finite and positive.")
        z_lo, z_hi = self.height_range_m
        if not (math.isfinite(z_lo) and math.isfinite(z_hi) and z_lo < z_hi):
            raise ValueError("height_range_m must be finite with min < max.")

    def cache_identity(self) -> dict[str, Any]:
        """Return deterministic settings identity for component store caching."""
        return {
            "joint_confidence": self.joint_confidence,
            "reprojection_px": self.reprojection_px,
            "htd_refine": self.htd_refine,
            "use_multiview_reprojection_refine": self.use_multiview_reprojection_refine,
            "height_range_m": list(self.height_range_m),
            "max_abs_xy_m": self.max_abs_xy_m,
            "htd_num_steps": self.htd_config.num_steps,
            "htd_vel_weight": self.htd_config.vel_weight,
            "htd_acc_weight": self.htd_config.acc_weight,
            "htd_reproj_weight": self.htd_config.reproj_weight,
            "htd_bone_var_weight": self.htd_config.bone_var_weight,
            "htd_bone_sym_weight": self.htd_config.bone_sym_weight,
            "htd_contact_weight": self.htd_config.contact_weight,
            "htd_ground_penetration_weight": self.htd_config.ground_penetration_weight,
        }


@dataclass(frozen=True)
class MonocularPLCSResult:
    """Combined 3D skeleton and HMR2/SMPL-free placement output."""

    skeleton: PlayerSkeleton
    players: ReconstructedPlayers


def _validate_calibrated_cameras(
    cameras: Sequence[PinholeCamera],
    expected_camera_ids: tuple[str, ...],
) -> tuple[PinholeCamera, ...]:
    if not cameras:
        raise ReconstructionUnavailable(
            "monocular_plcs_uncalibrated_camera",
            "Monocular PLCS requires at least one calibrated pinhole camera.",
        )
    cam_tuple = tuple(cameras)
    if len(cam_tuple) != len(expected_camera_ids):
        raise ValueError(
            f"Calibrated camera count ({len(cam_tuple)}) differs from observation "
            f"camera count ({len(expected_camera_ids)})."
        )
    actual_ids = tuple(c.camera_id for c in cam_tuple)
    if actual_ids != expected_camera_ids:
        raise ValueError(
            f"Calibrated camera order {actual_ids!r} disagrees with observation "
            f"cameras {expected_camera_ids!r}."
        )
    for cam in cam_tuple:
        if not isinstance(cam, PinholeCamera):
            raise TypeError(f"Expected PinholeCamera, got {type(cam).__name__}.")
        if (
            not np.isfinite(cam.intrinsic).all()
            or not np.isfinite(cam.rotation).all()
            or not np.isfinite(cam.translation).all()
            or cam.intrinsic[0, 0] <= 0.0
            or cam.intrinsic[1, 1] <= 0.0
        ):
            raise ValueError(f"Camera {cam.camera_id!r} has invalid calibration.")
        if not np.allclose(cam.rotation.T @ cam.rotation, np.eye(3), atol=1e-4) or not np.isclose(
            float(np.linalg.det(cam.rotation)), 1.0, atol=1e-4
        ):
            raise ValueError(f"Camera {cam.camera_id!r} rotation is not proper SO(3).")
    return cam_tuple


def _extract_player_view_observations(
    *,
    player_idx: int,
    view_idx: int,
    grouped: GroupedObservations,
    raw: ObjectObservations,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
    """Extract observed-only 2D keypoints, confidences, and ``boxes_xys`` for one (player, view).

    Frames where ``raw.observed`` is ``False`` (including GSI-interpolated track
    frames) are strictly excluded and zero-filled.
    """
    if raw.boxes_xys is None:
        raise ValueError("Monocular PLCS requires raw ObjectObservations.boxes_xys.")
    frames = grouped.uv_px.shape[2]
    kp2d_px = np.zeros((frames, NUM_HUMAN_KP, 2), dtype=np.float32)
    kp_conf = np.zeros((frames, NUM_HUMAN_KP), dtype=np.float32)
    bbx_xys = np.zeros((frames, 3), dtype=np.float32)
    frame_obs = np.zeros((frames,), dtype=np.bool_)

    raw_idx = grouped.raw_indices[player_idx, view_idx]  # (T,)
    for t in range(frames):
        d = int(raw_idx[t])
        if d < 0:
            continue
        # Strictly require real detector support; never treat GSI-interpolated frames as observed.
        if not bool(raw.observed[view_idx, t, d]):
            continue
        vis_t = grouped.visibility[player_idx, view_idx, t]
        if not bool(vis_t.any()):
            continue
        box = raw.boxes_xys[view_idx, t, d]
        if not np.isfinite(box).all() or float(box[2]) <= 0.0:
            raise ValueError(
                f"Invalid boxes_xys on observed frame t={t}, camera={raw.camera_ids[view_idx]!r}: {box.tolist()}."
            )
        frame_obs[t] = True
        bbx_xys[t] = box
        kp2d_px[t, vis_t] = grouped.uv_px[player_idx, view_idx, t, vis_t]
        kp_conf[t, vis_t] = grouped.confidence[player_idx, view_idx, t, vis_t]

    return kp2d_px, kp_conf, bbx_xys, frame_obs


def derive_players_from_skeleton(
    skeleton: PlayerSkeleton,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ReconstructedPlayers:
    """Build HMR2/SMPL-free ``ReconstructedPlayers`` directly from a ``PlayerSkeleton``.

    Root position is the mid-hip pelvis ``0.5 * (left_hip + right_hip)`` in court
    metres and heading yaw is derived from the transverse hip axis
    ``left_hip - right_hip`` (matching PLCS canonical convention). All SMPL fields
    are explicitly zeroed with ``smpl_valid = False`` and ``vertices_local = None``.
    """
    players, frames = skeleton.positions.shape[:2]
    root = np.zeros((players, frames, 3), dtype=np.float32)
    yaw = np.zeros((players, frames), dtype=np.float32)
    root_valid = np.zeros((players, frames), dtype=np.bool_)
    heading_valid = np.zeros((players, frames), dtype=np.bool_)
    root_reasons = np.full(
        (players, frames), int(PointRejection.INSUFFICIENT_VIEWS), dtype=np.uint8
    )

    hips_valid = skeleton.valid[:, :, LEFT_HIP_IDX] & skeleton.valid[:, :, RIGHT_HIP_IDX]
    for p in range(players):
        for t in range(frames):
            if not bool(hips_valid[p, t]):
                if bool(skeleton.valid[p, t].any()):
                    # Skeleton exists on other joints but hips unavailable -> invalid root.
                    root_reasons[p, t] = int(PointRejection.DEGENERATE_RAYS)
                continue
            lh = skeleton.positions[p, t, LEFT_HIP_IDX].astype(np.float64)
            rh = skeleton.positions[p, t, RIGHT_HIP_IDX].astype(np.float64)
            pelvis = 0.5 * (lh + rh)
            trans_xy = lh[:2] - rh[:2]
            if not np.isfinite(pelvis).all() or float(np.linalg.norm(trans_xy)) < 1e-4:
                root_reasons[p, t] = int(PointRejection.DEGENERATE_RAYS)
                continue
            root[p, t] = pelvis.astype(np.float32)
            yaw[p, t] = float(math.atan2(float(trans_xy[1]), float(trans_xy[0])))
            root_valid[p, t] = True
            heading_valid[p, t] = True
            root_reasons[p, t] = int(PointRejection.VALID)

    return ReconstructedPlayers(
        position=root,
        yaw=yaw,
        root_valid=root_valid,
        heading_valid=heading_valid,
        smpl_valid=np.zeros((players, frames), dtype=np.bool_),
        body_pose=np.zeros((players, frames, 63), dtype=np.float32),
        global_orient=np.zeros((players, frames, 3), dtype=np.float32),
        betas=np.zeros((players, 10), dtype=np.float32),
        vertices_local=None,
        root_reasons=root_reasons,
        mesh_reprojection_px=np.zeros((players, frames), dtype=np.float32),
        diagnostics={
            "method": "monocular_plcs_skeleton_only",
            "hmr2_used": False,
            "smplx_used": False,
            **(dict(diagnostics) if diagnostics is not None else {}),
        },
    )


def reconstruct_players_monocular_plcs(
    grouped: GroupedObservations,
    raw: ObjectObservations,
    cameras: Sequence[PinholeCamera],
    *,
    predictor: MonocularPLCSSequencePredictor,
    fps: float,
    config: MonocularPLCSConfig,
    pixel_threshold_scale: float = 1.0,
) -> MonocularPLCSResult:
    """Reconstruct court-space 3D player skeletons from 1..V calibrated cameras via monocular PLCS (+ HTD-Refine)."""
    if not (math.isfinite(fps) and fps > 0.0):
        raise ValueError(f"fps must be finite and positive, got {fps}.")
    if not (math.isfinite(pixel_threshold_scale) and pixel_threshold_scale > 0.0):
        raise ValueError("pixel_threshold_scale must be finite and positive.")

    calibrated = _validate_calibrated_cameras(cameras, raw.camera_ids)
    players, views, frames, joints, _ = grouped.uv_px.shape
    if joints != NUM_HUMAN_KP:
        raise ValueError(f"Monocular PLCS requires COCO-17 keypoints, got {joints}.")
    if views != len(calibrated) or raw.num_frames != frames:
        raise ValueError("Grouped observations disagree with camera count or timeline.")

    width, height = raw.size
    reproj_thresh = float(config.reprojection_px * pixel_threshold_scale)
    refiner = PLCSHTDRefiner(config.htd_config) if config.htd_refine else None

    out_pos = np.zeros((players, frames, NUM_HUMAN_KP, 3), dtype=np.float32)
    out_valid = np.zeros((players, frames, NUM_HUMAN_KP), dtype=np.bool_)
    out_reasons = np.full(
        (players, frames, NUM_HUMAN_KP),
        int(PointRejection.INSUFFICIENT_VIEWS),
        dtype=np.uint8,
    )
    out_inliers = np.zeros((players, views, frames, NUM_HUMAN_KP), dtype=np.bool_)
    out_reproj = np.zeros((players, views, frames, NUM_HUMAN_KP), dtype=np.float32)

    player_diagnostics: list[dict[str, Any]] = []
    z_min, z_max = config.height_range_m

    for p in range(players):
        view_obs_data: list[
            tuple[int, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]
        ] = []
        any_obs = np.zeros((frames,), dtype=np.bool_)

        for v in range(views):
            kp2d_px, kp_conf, bbx_xys, frame_obs = _extract_player_view_observations(
                player_idx=p,
                view_idx=v,
                grouped=grouped,
                raw=raw,
            )
            if bool(frame_obs.any()):
                view_obs_data.append((v, kp2d_px, kp_conf, bbx_xys, frame_obs))
                any_obs |= frame_obs

        if not view_obs_data or not bool(any_obs.any()):
            player_diagnostics.append(
                {
                    "player_id": int(grouped.identities[p]),
                    "observed_frames": 0,
                    "reconstructed_frames": 0,
                    "views_used": [],
                    "htd_refined": False,
                }
            )
            continue

        # Run monocular PLCS per observing calibrated view, then fuse per-frame
        # by mean visible joint confidence so single-view frames (occlusion in other
        # views) and multi-view frames are both reconstructed cleanly.
        fused_kp_sum = np.zeros((frames, NUM_HUMAN_KP, 3), dtype=np.float64)
        fused_vel_sum = np.zeros((frames, NUM_HUMAN_KP, 3), dtype=np.float64)
        fused_acc_sum = np.zeros((frames, NUM_HUMAN_KP, 3), dtype=np.float64)
        fused_contact_sum = np.zeros((frames, 4), dtype=np.float64)
        weight_sum = np.zeros((frames,), dtype=np.float64)
        vel_weight_sum = np.zeros((frames,), dtype=np.float64)
        acc_weight_sum = np.zeros((frames,), dtype=np.float64)
        contact_weight_sum = np.zeros((frames,), dtype=np.float64)

        refine_views: list[HTDRefineCameraView] = []
        view_scores: list[tuple[int, float, int]] = []

        for v, kp2d_px, kp_conf, bbx_xys, frame_obs in view_obs_data:
            cam = calibrated[v]
            fx = float(cam.intrinsic[0, 0])
            fy = float(cam.intrinsic[1, 1])
            cx0 = float(cam.intrinsic[0, 2])
            cy0 = float(cam.intrinsic[1, 2])
            R_np = cam.rotation.astype(np.float32)
            C_np = cam.center.astype(np.float32)
            R_t = torch.from_numpy(R_np)
            C_t = torch.from_numpy(C_np)
            kp2d_t = torch.from_numpy(kp2d_px)
            conf_t = torch.from_numpy(kp_conf)
            obs_t = torch.from_numpy(frame_obs)

            pred = predictor.predict_monocular_sequence(
                kp2d_px=kp2d_px,
                kp_vis=kp_conf,
                bbx_xys=bbx_xys,
                frame_observed=frame_obs,
                R_w2c=R_np,
                C_w=C_np,
                fx=fx,
                fy=fy,
                cx0=cx0,
                cy0=cy0,
                image_w=width,
                image_h=height,
                fps=float(fps),
                vis_threshold=float(config.joint_confidence),
                htd_refiner=None,
            )
            if pred.world_kp_3d_meters is None:
                raise ValueError(
                    "Monocular PLCS predictor must return world_kp_3d_meters for tennis_scene."
                )
            world_kp_raw = np.asarray(pred.world_kp_3d_meters, dtype=np.float64)
            if world_kp_raw.ndim == 4 and world_kp_raw.shape[0] == 1:
                world_kp_np = world_kp_raw[0]
            else:
                world_kp_np = world_kp_raw
            if world_kp_np.shape != (frames, NUM_HUMAN_KP, 3):
                raise ValueError(
                    f"Monocular PLCS world_kp_3d_meters shape mismatch: expected "
                    f"{(frames, NUM_HUMAN_KP, 3)}, got {world_kp_np.shape}."
                )

            obs_conf = np.where(kp_conf >= config.joint_confidence, kp_conf, 0.0)
            frame_w = np.where(frame_obs, np.maximum(obs_conf.mean(axis=-1), 1e-3), 0.0)
            fused_kp_sum += world_kp_np * frame_w[:, None, None]
            weight_sum += frame_w

            if pred.pred_vel_3d_mps is not None:
                vel_raw = np.asarray(pred.pred_vel_3d_mps, dtype=np.float64)
                vel_np_v = vel_raw[0] if vel_raw.ndim == 4 and vel_raw.shape[0] == 1 else vel_raw
                fused_vel_sum += vel_np_v * frame_w[:, None, None]
                vel_weight_sum += frame_w
            if pred.pred_acc_3d_mps2 is not None:
                acc_raw = np.asarray(pred.pred_acc_3d_mps2, dtype=np.float64)
                acc_np_v = acc_raw[0] if acc_raw.ndim == 4 and acc_raw.shape[0] == 1 else acc_raw
                fused_acc_sum += acc_np_v * frame_w[:, None, None]
                acc_weight_sum += frame_w
            if pred.static_contact_logits is not None:
                cnt_raw = np.asarray(pred.static_contact_logits, dtype=np.float64)
                cnt_np_v = cnt_raw[0] if cnt_raw.ndim == 3 and cnt_raw.shape[0] == 1 else cnt_raw
                fused_contact_sum += cnt_np_v * frame_w[:, None]
                contact_weight_sum += frame_w

            refine_views.append(
                HTDRefineCameraView(
                    R_w2c=R_t,
                    C_w=C_t,
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                    image_w=float(width),
                    image_h=float(height),
                    kp2d_px=kp2d_t,
                    kp_conf=conf_t,
                    frame_observed=obs_t,
                )
            )
            obs_count = int(frame_obs.sum())
            mean_c = float(obs_conf[frame_obs].mean()) if obs_count > 0 else 0.0
            view_scores.append((obs_count, mean_c, -v))

        safe_w = np.where(any_obs, np.maximum(weight_sum, 1e-6), 1.0)
        world_kp = np.where(
            any_obs[:, None, None],
            fused_kp_sum / safe_w[:, None, None],
            0.0,
        ).astype(np.float32)

        pred_vel_t: Tensor | None = None
        if bool((vel_weight_sum > 0.0).any()):
            vel_np = np.where(
                (vel_weight_sum > 0.0)[:, None, None],
                fused_vel_sum / np.maximum(vel_weight_sum, 1e-6)[:, None, None],
                0.0,
            ).astype(np.float32)
            pred_vel_t = torch.from_numpy(vel_np)

        pred_acc_t: Tensor | None = None
        if bool((acc_weight_sum > 0.0).any()):
            acc_np = np.where(
                (acc_weight_sum > 0.0)[:, None, None],
                fused_acc_sum / np.maximum(acc_weight_sum, 1e-6)[:, None, None],
                0.0,
            ).astype(np.float32)
            pred_acc_t = torch.from_numpy(acc_np)

        pred_contact_t: Tensor | None = None
        if bool((contact_weight_sum > 0.0).any()):
            contact_np = np.where(
                (contact_weight_sum > 0.0)[:, None],
                fused_contact_sum / np.maximum(contact_weight_sum, 1e-6)[:, None],
                0.0,
            ).astype(np.float32)
            pred_contact_t = torch.from_numpy(contact_np)

        if refiner is not None:
            if config.use_multiview_reprojection_refine or len(refine_views) == 1:
                selected_refine_views = refine_views
            else:
                best_idx = max(range(len(view_scores)), key=lambda i: view_scores[i])
                selected_refine_views = [refine_views[best_idx]]
            refined = refiner.refine(
                world_kp_init=torch.from_numpy(world_kp),
                pred_vel_3d=pred_vel_t,
                pred_acc_3d=pred_acc_t,
                static_contact_logits=pred_contact_t,
                views=selected_refine_views,
                fps=float(fps),
                valid_frame_mask=torch.from_numpy(any_obs),
            )
            world_kp = refined.world_kp_3d.detach().cpu().numpy().astype(np.float32)

        # Validate bounds & skeleton geometry strictly; zero invalid frames explicitly.
        finite_frame = np.isfinite(world_kp).all(axis=(1, 2))
        in_xy = (np.abs(world_kp[..., :2]) <= config.max_abs_xy_m).all(axis=(1, 2))
        in_z = ((world_kp[..., 2] >= z_min) & (world_kp[..., 2] <= z_max)).all(axis=1)
        frame_ok = any_obs & finite_frame & in_xy & in_z

        if bool(frame_ok.any()):
            valid_kp_t = torch.from_numpy(world_kp[frame_ok])
            for check_fn in (
                compute_bone_lengths,
                compute_joint_angles,
                compute_torsion_angles,
                compute_torso_twist,
            ):
                check_fn(valid_kp_t).require_valid("MonocularPLCS 3D skeleton")

        for t in range(frames):
            if not bool(any_obs[t]):
                out_reasons[p, t, :] = int(PointRejection.INSUFFICIENT_VIEWS)
                continue
            if not bool(frame_ok[t]):
                out_reasons[p, t, :] = int(PointRejection.OUTSIDE_BOUNDS)
                continue
            out_pos[p, t] = world_kp[t]
            out_valid[p, t, :] = True
            out_reasons[p, t, :] = int(PointRejection.VALID)

        # Compute per-view reprojection residuals & inliers against observed 2D joints.
        for v, kp2d_px, kp_conf, _bbx_xys, frame_obs in view_obs_data:
            cam = calibrated[v]
            proj_uv, in_front = cam.project(out_pos[p])
            err = np.linalg.norm(proj_uv - kp2d_px.astype(np.float64), axis=-1)
            obs_joint = frame_obs[:, None] & (kp_conf >= config.joint_confidence) & out_valid[p]
            good_proj = obs_joint & in_front & np.isfinite(err)
            out_reproj[p, v] = np.where(good_proj, err, 0.0).astype(np.float32)
            out_inliers[p, v] = good_proj & (err <= reproj_thresh)

        player_diagnostics.append(
            {
                "player_id": int(grouped.identities[p]),
                "observed_frames": int(any_obs.sum()),
                "reconstructed_frames": int(frame_ok.sum()),
                "views_used": [calibrated[v].camera_id for v, *_ in view_obs_data],
                "htd_refined": refiner is not None,
            }
        )

    skeleton = PlayerSkeleton(
        positions=out_pos,
        valid=out_valid,
        reasons=out_reasons,
        inliers=out_inliers,
        reprojection_px=out_reproj,
    )
    players_out = derive_players_from_skeleton(
        skeleton,
        diagnostics={
            "players": player_diagnostics,
            "htd_refine": config.htd_refine,
            "use_multiview_reprojection_refine": config.use_multiview_reprojection_refine,
        },
    )
    return MonocularPLCSResult(skeleton=skeleton, players=players_out)


def build_single_view_geometry(
    calibration: CalibrationSet,
    camera_id: str,
) -> tuple[CameraGeometryResult, LocalCourtCalibration]:
    """Build a single-camera ``CameraGeometryResult`` from ``LocalCourtCalibration``.

    Raises ``ReconstructionUnavailable`` explicitly if ``camera_id`` is uncalibrated.
    """
    by_id = {view.camera.camera_id: view for view in calibration.views}
    if camera_id not in by_id:
        reason = calibration.excluded.get(camera_id, "not_calibrated")
        raise ReconstructionUnavailable(
            "monocular_plcs_uncalibrated_camera",
            f"Camera {camera_id!r} has no accepted court calibration ({reason}).",
            diagnostics={"camera_id": camera_id, "excluded": dict(calibration.excluded)},
        )
    local = by_id[camera_id]
    cam = local.camera
    ids = (camera_id,)
    sides = (False,)
    contract = resolve_court_keypoint_contract("camera_view_v2")
    record = build_court_view_record(
        camera_id=cam.camera_id,
        camera_center_court_m=cam.center.tolist(),
        contract=contract,
    )
    selection = ReferenceViewSelection.create(
        stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(ids),
        selected_views=(record,),
        reference_camera_id=camera_id,
    )
    document: dict[str, Any] = {
        "camera_ids": [camera_id],
        "reference_camera": camera_id,
        "view_half_turns": [False],
        "side_validation": {"mode": "single_view_local_court"},
        "court_keypoint_views": [record.to_dict()],
        "court_reference_provenance": selection.provenance.to_dict(),
        "camera_fits": [
            {
                "K": cam.intrinsic.tolist(),
                "R": cam.rotation.tolist(),
                "t": cam.translation.tolist(),
                "camera_center_court_m": cam.center.tolist(),
                "rmse_px": local.rmse_px,
                "calibration_frame_index": local.frame_index,
                "calibration": "approximate single-plane pinhole; no distortion correction",
            }
        ],
        "excluded_cameras": dict(calibration.excluded),
    }
    write_model_artifact_court_keypoint_contract(document, contract)
    return CameraGeometryResult(ids, camera_id, sides, (cam,), selection, document), local


def reconstruct_single_view_monocular_scene(
    *,
    source: ClipSource,
    court: CourtKPResult,
    calibration: CalibrationSet,
    camera_id: str,
    observations: ObjectObservations,
    predictor: MonocularPLCSSequencePredictor,
    config: MonocularPLCSConfig,
    visibility_threshold: float = 0.15,
) -> SceneResult:
    """End-to-end single-camera monocular PLCS (+ HTD-Refine) reconstruction to ``SceneResult v2``.

    Requires zero multi-camera alignment, zero HMR2 features, and zero SMPL-X assets.
    """
    geometry, local = build_single_view_geometry(calibration, camera_id)
    if observations.camera_ids != (camera_id,):
        raise ValueError(
            f"Single-view observations must match camera {camera_id!r}, got {observations.camera_ids!r}."
        )
    if observations.num_frames != source.num_frames or observations.size != source.size:
        raise ValueError("Single-view observations disagree with ClipSource timeline/size.")

    # Assign each local track to its own player identity ONLY on real observed frames.
    v, t, d = observations.observed.shape
    object_ids = np.full((v, t, d), -1, dtype=np.int64)
    vis_any = observations.visibility(visibility_threshold).any(axis=-1)
    for track_col in range(d):
        track_id = int(observations.local_track_ids[0, track_col])
        if track_id < 0:
            continue
        active_frames = observations.observed[0, :, track_col] & vis_any[0, :, track_col]
        object_ids[0, active_frames, track_col] = track_id

    grouped = group_observations(observations, object_ids, threshold=visibility_threshold)
    result = reconstruct_players_monocular_plcs(
        grouped,
        observations,
        geometry.cameras,
        predictor=predictor,
        fps=source.fps,
        config=config,
        pixel_threshold_scale=source.pixel_threshold_scale,
    )
    first = source.videos[0]
    has_people = bool(result.skeleton.valid.any())
    return assemble_automatic_scene(
        video_paths=source.paths,
        camera_ids=source.camera_ids,
        info=VideoInfo(first.fps, first.width, first.height, first.num_frames),
        court=court,
        active_indices=(local.source_index,),
        geometry=geometry,
        grouped_people=grouped,
        skeleton=result.skeleton,
        players=result.players,
        ball=None,
        metadata={
            "status": "ok" if has_people else "partial",
            "dataset_clip_id": source.clip_id,
            "pipeline_contract": "monocular_plcs_v1",
            "player_reconstruction_method": "monocular_plcs_gvhmr_htd",
        },
    )


# ---------------------------------------------------------------------------
# Declared pipeline components (compatible with standard_definition overrides)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonocularPLCSPlayerInput:
    """Assembled multi-view inputs for ``MonocularPLCSReconstructionModule``."""

    source: ClipSource
    alignment: CameraAlignmentOutput
    raw: ObjectObservations
    grouped: GroupedObservations


@dataclass(frozen=True)
class MonocularPLCSInputAssembler:
    """Assemble calibrated observations + source ``boxes_xys`` for monocular PLCS."""

    human_threshold: float
    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> MonocularPLCSPlayerInput:
        calibration: CourtCalibrationOutput = artifacts["calibration"]
        active = tuple(v.source_index for v in calibration.calibration.views)
        raw = gather_people(context.source, artifacts).select_views(active)
        grouped = identified_people(raw, artifacts["identities"], self.human_threshold)
        return MonocularPLCSPlayerInput(
            source=context.source,
            alignment=artifacts["alignment"],
            raw=raw,
            grouped=grouped,
        )


class MonocularPLCSReconstructionModule(
    InferenceComponent[MonocularPLCSPlayerInput, PlayerTriangulationOutput]
):
    """Declared ``player_triangulation`` replacement using monocular PLCS (+ HTD-Refine).

    Requires zero HMR2 or SMPL-X assets; declares only its PLCS checkpoint in
    ``model_specs()`` when loaded from disk.
    """

    def __init__(
        self,
        camera_ids: tuple[str, ...],
        config: MonocularPLCSConfig,
        *,
        enabled: bool = True,
        path_resolver: PathResolver | None = None,
        predictor: MonocularPLCSSequencePredictor | None = None,
    ) -> None:
        if enabled and predictor is None and config.checkpoint is None:
            raise ValueError(
                "MonocularPLCSReconstructionModule requires config.checkpoint or an explicit predictor."
            )
        self.config = config
        self.enabled = enabled
        self.path_resolver = path_resolver
        self._injected_predictor = predictor
        self.io = ComponentIO(
            "player_triangulation",
            MonocularPLCSPlayerInput,
            PlayerTriangulationOutput,
            {
                "alignment": InputPort("aligned_cameras"),
                "calibration": InputPort("local_court_calibration"),
                "identities": IDENTITIES_PORT,
                **{f"pose_{c}": InputPort("person_poses", 2) for c in camera_ids},
            },
            "player_skeletons",
        )

    def cache_identity(self) -> dict[str, Any]:
        return {
            "method": "monocular_plcs",
            "enabled": self.enabled,
            **self.config.cache_identity(),
        }

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        if not self.enabled or self._injected_predictor is not None:
            return {}
        assert self.config.checkpoint is not None
        checkpoint = self.config.checkpoint
        device = self.config.device
        resolver = self.path_resolver
        return {
            "plcs_monocular": ModelSpec(
                "plcs_monocular_gvhmr",
                {"checkpoint": checkpoint},
                "plcs_monocular_gvhmr",
                device,
                "float32",
                self.config.cache_identity(),
                lambda: PLCSPredictor.load_from_checkpoint(
                    checkpoint,
                    device=device,
                    path_resolver=resolver,
                ),
            )
        }

    def process_with_models(
        self, inputs: MonocularPLCSPlayerInput, models: ModelBorrower
    ) -> PlayerTriangulationOutput:
        if not self.enabled:
            return PlayerTriangulationOutput(None)
        geometry = inputs.alignment.geometry
        if geometry is None:
            raise ReconstructionUnavailable(
                "monocular_plcs_uncalibrated_camera",
                "Monocular PLCS requires calibrated aligned cameras.",
            )
        predictor: MonocularPLCSSequencePredictor = (
            self._injected_predictor
            if self._injected_predictor is not None
            else models.borrow("plcs_monocular")
        )
        result = reconstruct_players_monocular_plcs(
            inputs.grouped,
            inputs.raw,
            geometry.cameras,
            predictor=predictor,
            fps=inputs.source.fps,
            config=self.config,
            pixel_threshold_scale=inputs.source.pixel_threshold_scale,
        )
        return PlayerTriangulationOutput(result.skeleton)


@dataclass(frozen=True)
class SkeletonOnlyPlacementInput:
    """Minimal input for ``SkeletonOnlyPlacementModule`` (no HMR2 / GVHMR / SMPL-X)."""

    skeleton: PlayerTriangulationOutput


@dataclass(frozen=True)
class SkeletonOnlyPlacementInputAssembler:
    """Pass ``PlayerTriangulationOutput`` straight to ``SkeletonOnlyPlacementModule``."""

    version: int = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> SkeletonOnlyPlacementInput:
        del context
        return SkeletonOnlyPlacementInput(skeleton=artifacts["skeleton"])


class SkeletonOnlyPlacementModule:
    """HMR2/SMPL-free ``body_placement`` component deriving placement from ``PlayerSkeleton``."""

    io = ComponentIO(
        "body_placement",
        SkeletonOnlyPlacementInput,
        BodyPlacementOutput,
        {"skeleton": InputPort("player_skeletons")},
        "placed_bodies",
    )

    def cache_identity(self) -> dict[str, Any]:
        return {"method": "skeleton_only_hip_placement", "smplx_used": False}

    def process(self, inputs: SkeletonOnlyPlacementInput) -> BodyPlacementOutput:
        if inputs.skeleton.skeleton is None:
            return BodyPlacementOutput(None)
        return BodyPlacementOutput(
            derive_players_from_skeleton(inputs.skeleton.skeleton)
        )


__all__ = [
    "MonocularPLCSConfig",
    "MonocularPLCSInputAssembler",
    "MonocularPLCSPlayerInput",
    "MonocularPLCSReconstructionModule",
    "MonocularPLCSResult",
    "MonocularPLCSSequencePredictor",
    "SkeletonOnlyPlacementInput",
    "SkeletonOnlyPlacementInputAssembler",
    "SkeletonOnlyPlacementModule",
    "build_single_view_geometry",
    "derive_players_from_skeleton",
    "reconstruct_players_monocular_plcs",
    "reconstruct_single_view_monocular_scene",
]
