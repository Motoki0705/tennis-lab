"""Scene generator for PLCS training data.

This module generates training scenes by combining motion sequences with
virtual camera configurations and projecting to 2D.

All source-specific representations are adapted to the canonical COCO-17
motion contract before entering this module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    CourtViewRecord,
    apply_court_view_record,
    build_court_view_record,
)
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    MotionSampler,
    MotionSequence,
)
from src.tasks.plcs.motion import place_motion_on_court
from src.utils.data.camera_sampling import camera_candidate_indices
from src.utils.projection.camera_projector import (
    CameraConfig,
    CameraProjector,
)
from src.utils.schema.court import (
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    STANDARD_COURT_CONFIG,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class CameraData:
    """Data for a single camera view."""

    camera_params: dict  # Camera intrinsics/extrinsics
    human_kp_uv: np.ndarray  # (T, 17, 2)
    court_kp_uv: np.ndarray  # (T, 20, 2)
    human_kp_vis: np.ndarray  # (T, 17)
    court_kp_vis: np.ndarray  # (T, 20)

    # Visibility metrics recorded for analysis/debugging.
    human_visibility_ratio: (
        float  # Fraction of frames with at least one visible human keypoint
    )
    court_visibility_count: float  # Average visible court keypoints
    court_view: CourtViewRecord | None = None


@dataclass
class SceneData:
    """Complete scene data container."""

    # Metadata
    meta: dict

    # Per-frame 3D data (T frames)
    position: np.ndarray  # (T, 3) normalized court coordinates
    rotation: np.ndarray  # (T, 2) sin/cos yaw
    canonical_pose_3d: np.ndarray  # (T, J, 3) yaw-canonical local coordinate pose

    # Per-camera data
    cameras: list[CameraData]
    num_persons: int

    # Optional: pre-computed COCO17 world-coordinate joints (bypasses SMPL-H mapping)
    human_kp_3d: np.ndarray | None = None  # (T, 17, 3) if provided

    # Present for multi-object scenes. Object arrays then use shape (T, O, ...).
    person_present: np.ndarray | None = None
    track_instances: list[dict] = field(default_factory=list)
    court_keypoint_contract: CourtKeypointContract | None = None


class SceneGenerator:
    """Generate PLCS training scenes.

    This class:
    - Samples source-independent COCO-17 motion sequences
    - Places players on the court with random initial pose
    - Generates multiple camera views
    - Projects 3D data to 2D UV coordinates
    - Records per-camera visibility metrics
    """

    def __init__(
        self,
        config: DictConfig,
        motion_sampler: MotionSampler | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the scene generator.

        Args:
            config: Configuration dictionary.
            motion_sampler: Pre-configured motion sampler.
            device: Device for computation.

        """
        self.config = config
        self.device = torch.device(device)
        self.court_keypoint_contract = PLCSCourtKeypointRuntimeConfig.from_config(
            config
        ).contract

        # Initialize motion sampler if not provided
        if motion_sampler is None:
            motion_sampler = MotionSampler(
                config=config,
                smplh_model_path=str(config.external_assets.smplh_model_path),
                coco17_regressor_path=str(config.external_assets.coco17_regressor_path),
                device=device,
            )
        self.motion_sampler = motion_sampler

        # Get court keypoints (convert to numpy)
        self.court_kp_3d: np.ndarray | None = None

        # Camera config
        cam_cfg = self.config.camera
        camera_config = CameraConfig(
            z_min=float(cam_cfg.z_min),
            z_max=float(cam_cfg.z_max),
            hfov_deg=float(cam_cfg.hfov_deg),
            image_size=(int(cam_cfg.image_size[0]), int(cam_cfg.image_size[1])),
            fixed_look_at=cast(
                "tuple[float, float, float]",
                tuple(float(v) for v in cam_cfg.fixed_look_at),
            ),
            fixed_camera_indices=camera_candidate_indices(
                cam_cfg.get("fixed_camera_indices"), capacity=6
            ),
            fixed_baseline_clear_extra=float(cam_cfg.fixed_baseline_clear_extra),
            fixed_position_noise_radius=float(cam_cfg.fixed_position_noise_radius),
            fixed_look_at_xy_radius=float(cam_cfg.fixed_look_at_xy_radius),
            layout=str(cam_cfg.layout),
            broadcast_setback=float(cam_cfg.broadcast_setback),
            broadcast_height=float(cam_cfg.broadcast_height),
            broadcast_hfov_deg=float(cam_cfg.broadcast_hfov_deg),
            broadcast_look_at_y=float(cam_cfg.broadcast_look_at_y),
            broadcast_look_at_height=float(cam_cfg.broadcast_look_at_height),
            broadcast_position_noise_radius=float(
                cam_cfg.broadcast_position_noise_radius
            ),
            broadcast_look_at_xy_radius=float(cam_cfg.broadcast_look_at_xy_radius),
            broadcast_hfov_jitter_deg=float(cam_cfg.broadcast_hfov_jitter_deg),
            broadcast_setback_range=(
                cast(
                    "tuple[float, float]",
                    tuple(float(v) for v in cam_cfg.broadcast_setback_range),
                )
                if cam_cfg.broadcast_setback_range is not None
                else None
            ),
            broadcast_height_range=(
                cast(
                    "tuple[float, float]",
                    tuple(float(v) for v in cam_cfg.broadcast_height_range),
                )
                if cam_cfg.broadcast_height_range is not None
                else None
            ),
            broadcast_court_width_frac_range=(
                cast(
                    "tuple[float, float]",
                    tuple(float(v) for v in cam_cfg.broadcast_court_width_frac_range),
                )
                if cam_cfg.broadcast_court_width_frac_range is not None
                else None
            ),
        )
        self.camera_projector = CameraProjector(
            camera_config,
            court_config=STANDARD_COURT_CONFIG,
        )
        self.image_size = self.camera_projector.config.image_size

    def _sample_initial_pose(self) -> tuple[float, float, float]:
        """Sample initial player position on court.

        Returns:
            (x, y, yaw) where x, y are in court coordinates, yaw is in radians.

        """
        # Sample position within singles court with some margin
        margin = 0.5
        x = random.uniform(-HALF_SINGLES_WIDTH + margin, HALF_SINGLES_WIDTH - margin)
        y = random.uniform(-HALF_LENGTH + margin, HALF_LENGTH - margin)
        yaw = random.uniform(-math.pi, math.pi)

        return x, y, yaw

    def _transform_motion_to_court(
        self,
        motion: MotionSequence,
        init_x: float,
        init_y: float,
        init_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform motion sequence to court coordinates.

        Args:
            motion: Source-independent COCO-17 motion.
            init_x: Initial X position on court.
            init_y: Initial Y position on court.
            init_yaw: Initial yaw rotation.

        Returns:
            Tuple of (positions, rotations, canonical_poses):
            - positions: (T, 3) normalized court coordinates
            - rotations: (T, 2) cos/sin yaw
            - canonical_poses: (T, 17, 3) yaw-canonical local coordinate poses

        """
        placed = place_motion_on_court(
            motion,
            initial_x_m=init_x,
            initial_y_m=init_y,
            initial_yaw_rad=init_yaw,
        )
        return placed.position, placed.rotation, placed.canonical_pose_3d

    def _evaluate_camera(
        self,
        human_visible: np.ndarray,
        court_visible: np.ndarray,
    ) -> tuple[float, float]:
        """Evaluate camera quality based on visibility.

        Args:
            human_visible: Human keypoint visibility, (T, 17).
            court_visible: Court keypoint visibility, (T, 20).

        Returns:
            Tuple of (human_visibility_ratio, avg_court_visible).

        """
        # Human: fraction of frames with at least one visible keypoint
        human_per_frame = human_visible.any(axis=1)  # (T,)
        human_ratio = human_per_frame.mean()

        # Court: average number of visible keypoints
        court_per_frame = court_visible.sum(axis=1)  # (T,)
        avg_court = court_per_frame.mean()

        return float(human_ratio), float(avg_court)

    def generate_scene(
        self,
        scene_id: str | None = None,
        *,
        required_fps: float | None = None,
    ) -> SceneData:
        """Generate a complete scene.

        Args:
            scene_id: Optional scene identifier.
            required_fps: Optional native rate required by a shared scene timeline.

        Returns:
            SceneData with all generated data.

        """
        # Sample motion
        motion = self.motion_sampler.sample_motion(required_fps=required_fps)

        # Sample initial pose
        init_x, init_y, init_yaw = self._sample_initial_pose()

        # Transform to court coordinates
        placed = place_motion_on_court(
            motion,
            initial_x_m=init_x,
            initial_y_m=init_y,
            initial_yaw_rad=init_yaw,
        )
        positions = placed.position
        rotations = placed.rotation
        canonical_poses = placed.canonical_pose_3d
        coco17_joints = placed.world_joints_3d
        T = motion.frame_count

        # Get court keypoints (static)
        if self.court_kp_3d is None:
            self.court_kp_3d = self.camera_projector.court_kp_3d.numpy()
        court_3d = self.court_kp_3d  # (20, 3)

        # Generate multiple cameras
        cameras_data = []
        for camera_index, camera in enumerate(self.camera_projector.cameras()):
            # Project human keypoints
            human_uv: np.ndarray = np.zeros((T, 17, 2), dtype=np.float32)
            human_vis: np.ndarray = np.zeros((T, 17), dtype=bool)
            for t in range(T):
                points_t = torch.from_numpy(coco17_joints[t]).float()
                uv_t, vis_t = self.camera_projector.project_points_to_uv(
                    points_t, camera
                )
                human_uv[t] = uv_t.numpy()
                human_vis[t] = vis_t.numpy()

            # Project court keypoints (same for all frames)
            court_points_t = torch.from_numpy(court_3d).float()
            court_uv_t, court_vis_t = self.camera_projector.project_points_to_uv(
                court_points_t, camera
            )
            court_uv_single = court_uv_t.numpy()
            court_vis_single = court_vis_t.numpy()
            court_view = build_court_view_record(
                camera_id=f"camera_{camera_index}",
                camera_center_court_m=camera.C.numpy(),
                contract=self.court_keypoint_contract,
            )
            court_uv_single = apply_court_view_record(
                court_uv_single,
                court_view,
                keypoint_axis=0,
            )
            court_vis_single = apply_court_view_record(
                court_vis_single,
                court_view,
                keypoint_axis=0,
            )
            court_uv = np.tile(court_uv_single[None, ...], (T, 1, 1))
            court_vis = np.tile(court_vis_single[None, ...], (T, 1))

            # Evaluate camera
            human_ratio, avg_court = self._evaluate_camera(human_vis, court_vis)

            # Store camera data
            cam_data = CameraData(
                camera_params={
                    "C": camera.C.tolist(),
                    "R": camera.R.tolist(),
                    "f": camera.f,
                    "cx": camera.cx,
                    "cy": camera.cy,
                    "w": camera.w,
                    "h": camera.h,
                    "image_size": self.image_size,
                },
                human_kp_uv=human_uv,
                court_kp_uv=court_uv,
                human_kp_vis=human_vis,
                court_kp_vis=court_vis,
                human_visibility_ratio=human_ratio,
                court_visibility_count=avg_court,
                court_view=court_view,
            )
            cameras_data.append(cam_data)

        # Build metadata
        meta = {
            "scene_id": scene_id or f"scene_{random.randint(0, 999999):06d}",
            "motion_source": motion.source_path,
            "motion_category": motion.category,
            "gender": motion.gender,
            "fps": motion.fps,
            "num_frames": T,
            "initial_position": (init_x, init_y),
            "initial_yaw": init_yaw,
            "num_cameras_sampled": len(cameras_data),
            "motion_source_kind": motion.source_kind.value,
            "motion_source_id": motion.source_id,
        }

        return SceneData(
            meta=meta,
            position=positions,
            rotation=rotations,
            canonical_pose_3d=canonical_poses,
            cameras=cameras_data,
            num_persons=1,
            human_kp_3d=coco17_joints,
            court_keypoint_contract=self.court_keypoint_contract,
        )
