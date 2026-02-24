"""Scene generator for PLCS training data.

This module generates training scenes by combining motion sequences with
virtual camera configurations and projecting to 2D.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    MotionSampler,
    MotionSequence,
)
from src.utils.schema.player import (
    FACE_KEYPOINT_OFFSETS,
    SMPLH_TO_COCO17_MAPPING,
)
from src.utils.schema.court import (
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
)
from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)
from src.utils.projection.camera_projector import (
    CameraConfig,
    CameraProjector,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class CameraData:
    """Data for a single camera view."""

    camera_params: dict  # Camera intrinsics/extrinsics
    human_kp_uv: np.ndarray  # (T, 17, 2)
    court_kp_uv: np.ndarray  # (T, 20, 2)
    human_kp_visible: np.ndarray  # (T, 17)
    court_kp_visible: np.ndarray  # (T, 20)

    # Filtering metrics
    human_visibility_ratio: float  # Fraction of frames with sufficient human visibility
    court_visibility_count: float  # Average visible court keypoints


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


class SceneGenerator:
    """Generate PLCS training scenes.

    This class:
    - Samples motion sequences from AMASS
    - Places players on the court with random initial pose
    - Generates multiple camera views
    - Projects 3D data to 2D UV coordinates
    - Filters cameras based on visibility criteria
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        motion_sampler: MotionSampler | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the scene generator.

        Args:
            config: Configuration dictionary.
            motion_sampler: Pre-configured motion sampler.
            device: Device for computation.

        """
        self.config = config or {}
        self.device = torch.device(device)

        # Initialize motion sampler if not provided
        if motion_sampler is None:
            smplh_path = self.config.get("smplh_model_path", "data/smplx/smplh")
            motion_sampler = MotionSampler(
                config=config,
                smplh_model_path=smplh_path,
                device=device,
            )
        self.motion_sampler = motion_sampler

        # Get court keypoints (convert to numpy)
        self.court_kp_3d = None

        # Parse config
        sim_cfg = self.config.get("simulation", {})
        self.num_cameras = sim_cfg.get("num_cameras", 15)
        self.human_visibility_threshold = sim_cfg.get("human_visibility_threshold", 0.8)
        self.court_visibility_threshold = sim_cfg.get("court_visibility_threshold", 15)

        # Camera config
        cam_cfg = self.config.get("camera", {})
        camera_config = CameraConfig(
            z_min=cam_cfg.get("z_min", 3.0),
            z_max=cam_cfg.get("z_max", 5.0),
            hfov_deg=cam_cfg.get("hfov_deg", 60.0),
            image_size=tuple(cam_cfg.get("image_size", [1280, 720])),
            target_x_range=tuple(cam_cfg.get("target_x_range", [-2.0, 2.0])),
            target_y_range=tuple(cam_cfg.get("target_y_range", [-2.0, 2.0])),
            target_z_range=tuple(cam_cfg.get("target_z_range", [0.5, 1.5])),
        )
        self.camera_projector = CameraProjector(camera_config)
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
            motion: Motion sequence with joints_3d computed.
            init_x: Initial X position on court.
            init_y: Initial Y position on court.
            init_yaw: Initial yaw rotation.

        Returns:
            Tuple of (positions, rotations, canonical_poses):
            - positions: (T, 3) normalized court coordinates
            - rotations: (T, 2) cos/sin yaw
            - canonical_poses: (T, J, 3) yaw-canonical local coordinate poses

        """
        if motion.joints_3d is None:
            raise ValueError(
                "Motion joints_3d not computed. Call compute_joints_3d first."
            )

        T = motion.num_frames
        joints_3d = motion.joints_3d  # (T, J, 3)

        # Get pelvis (root) position from original motion
        original_trans = motion.trans  # (T, 3)

        # Compute initial offset (first frame XY only, keep Z from trans)
        init_offset_xy = original_trans[0, :2].copy()

        # Center motion at origin (XY only)
        centered_trans = original_trans.copy()
        centered_trans[:, 0] -= init_offset_xy[0]
        centered_trans[:, 1] -= init_offset_xy[1]
        # Keep original Z (pelvis height from ground)

        # Compute motion yaw first so translation and body rotation share the same offset.
        motion_yaw = self._extract_global_yaw_from_motion(motion)  # (T,)
        yaw_offset = self._wrap_angle(init_yaw - motion_yaw[0])

        # Rotation matrix for yaw offset (not raw init_yaw)
        cos_yaw = math.cos(yaw_offset)
        sin_yaw = math.sin(yaw_offset)
        rot_mat = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Transform to court coordinates
        court_trans = centered_trans @ rot_mat.T
        court_trans[:, 0] += init_x
        court_trans[:, 1] += init_y

        # Normalize positions
        positions = np.zeros((T, 3), dtype=np.float32)
        positions[:, 0] = court_trans[:, 0] / COURT_COORD_SCALE_X
        positions[:, 1] = court_trans[:, 1] / COURT_COORD_SCALE_Y
        positions[:, 2] = court_trans[:, 2] / COURT_COORD_SCALE_Z

        # Compute rotations (yaw): motion-relative yaw with randomized initial yaw.
        relative_yaw = self._wrap_angle(motion_yaw - motion_yaw[0])  # t=0 -> 0
        world_yaw = self._wrap_angle(relative_yaw + init_yaw)  # add random initial yaw

        rotations = np.zeros((T, 2), dtype=np.float32)
        rotations[:, 0] = np.cos(world_yaw).astype(np.float32)  # cos(yaw)
        rotations[:, 1] = np.sin(world_yaw).astype(np.float32)  # sin(yaw)

        # Canonical poses: joints relative to pelvis, then remove per-frame global yaw
        pelvis = joints_3d[:, 0:1, :]  # (T, 1, 3)
        root_relative = joints_3d - pelvis  # (T, J, 3)
        cos_m = np.cos(motion_yaw).astype(np.float32)
        sin_m = np.sin(motion_yaw).astype(np.float32)
        canonical_poses = np.empty_like(root_relative, dtype=np.float32)
        canonical_poses[..., 0] = (
            root_relative[..., 0] * cos_m[:, None]
            + root_relative[..., 1] * sin_m[:, None]
        )
        canonical_poses[..., 1] = (
            -root_relative[..., 0] * sin_m[:, None]
            + root_relative[..., 1] * cos_m[:, None]
        )
        canonical_poses[..., 2] = root_relative[..., 2]

        return positions, rotations, canonical_poses

    def _extract_global_yaw_from_motion(self, motion: MotionSequence) -> np.ndarray:
        """Extract per-frame global yaw (Z axis) from AMASS global_orient."""
        aa = motion.poses.reshape(motion.num_frames, 52, 3)[:, 0, :]  # (T, 3)
        theta = np.linalg.norm(aa, axis=1)  # (T,)

        axis = np.zeros_like(aa, dtype=np.float32)
        valid = theta > 1e-8
        axis[valid] = aa[valid] / theta[valid, None]

        x = axis[:, 0]
        y = axis[:, 1]
        z = axis[:, 2]
        c = np.cos(theta)
        s = np.sin(theta)
        one_minus_c = 1.0 - c

        # Rodrigues (need R[0,0], R[1,0] only for yaw = atan2(R10, R00))
        r00 = c + x * x * one_minus_c
        r10 = y * x * one_minus_c + z * s

        return np.arctan2(r10, r00).astype(np.float32)

    def _wrap_angle(self, angle: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)

    def _smplh_to_coco17(
        self,
        joints_3d: np.ndarray,
        yaw: float | np.ndarray,
    ) -> np.ndarray:
        """Convert SMPL-H joints to COCO 17 format.

        Args:
            joints_3d: SMPL-H joints, shape (T, J, 3) or (J, 3).
            yaw: Yaw angle for face keypoint orientation, scalar or (T,).

        Returns:
            COCO 17 keypoints, shape (T, 17, 3) or (17, 3).

        """
        squeeze = joints_3d.ndim == 2
        if squeeze:
            joints_3d = joints_3d[None, ...]  # (1, J, 3)

        T = joints_3d.shape[0]
        coco17 = np.zeros((T, 17, 3), dtype=np.float32)

        # Map body keypoints
        for coco_idx, smplh_idx in SMPLH_TO_COCO17_MAPPING.items():
            if smplh_idx >= 0:
                coco17[:, coco_idx, :] = joints_3d[:, smplh_idx, :]

        # Compute face keypoints from head
        head_pos = joints_3d[:, 15, :]  # head joint

        yaw_arr = np.asarray(yaw, dtype=np.float32)
        if yaw_arr.ndim == 0:
            yaw_arr = np.full((T,), float(yaw_arr), dtype=np.float32)
        elif yaw_arr.shape != (T,):
            raise ValueError(f"yaw must be scalar or shape ({T},), got {yaw_arr.shape}")

        cos_yaw = np.cos(yaw_arr).astype(np.float32)
        sin_yaw = np.sin(yaw_arr).astype(np.float32)

        for coco_idx, offset in FACE_KEYPOINT_OFFSETS.items():
            offset_arr = np.array(offset, dtype=np.float32)
            rotated_offset = np.stack(
                [
                    offset_arr[0] * cos_yaw - offset_arr[1] * sin_yaw,
                    offset_arr[0] * sin_yaw + offset_arr[1] * cos_yaw,
                    np.full((T,), offset_arr[2], dtype=np.float32),
                ],
                axis=1,
            )
            coco17[:, coco_idx, :] = head_pos + rotated_offset

        if squeeze:
            coco17 = coco17[0]

        return coco17

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
        # Human: fraction of frames where configured keypoint visibility is satisfied
        human_per_frame = human_visible.mean(axis=1)  # (T,)
        human_ratio = (human_per_frame >= self.human_visibility_threshold).mean()

        # Court: average number of visible keypoints
        court_per_frame = court_visible.sum(axis=1)  # (T,)
        avg_court = court_per_frame.mean()

        return float(human_ratio), float(avg_court)

    def generate_scene(
        self,
        scene_id: str | None = None,
        category: str | None = None,
    ) -> SceneData:
        """Generate a complete scene.

        Args:
            scene_id: Optional scene identifier.
            category: Optional motion category to sample from.

        Returns:
            SceneData with all generated data.

        """
        # Sample motion
        motion = self.motion_sampler.sample_motion(category=category)
        self.motion_sampler.compute_joints_3d(motion)

        # Sample initial pose
        init_x, init_y, init_yaw = self._sample_initial_pose()

        # Transform to court coordinates
        positions, rotations, canonical_poses = self._transform_motion_to_court(
            motion, init_x, init_y, init_yaw
        )

        # Get world-space joints for projection
        T = motion.num_frames
        pelvis_world = np.zeros((T, 3), dtype=np.float32)
        pelvis_world[:, 0] = positions[:, 0] * COURT_COORD_SCALE_X
        pelvis_world[:, 1] = positions[:, 1] * COURT_COORD_SCALE_Y
        pelvis_world[:, 2] = positions[:, 2] * COURT_COORD_SCALE_Z

        cos_yaw = rotations[:, 0].astype(np.float32)
        sin_yaw = rotations[:, 1].astype(np.float32)
        world_joints = np.empty_like(canonical_poses, dtype=np.float32)
        world_joints[..., 0] = (
            canonical_poses[..., 0] * cos_yaw[:, None]
            - canonical_poses[..., 1] * sin_yaw[:, None]
            + pelvis_world[:, None, 0]
        )
        world_joints[..., 1] = (
            canonical_poses[..., 0] * sin_yaw[:, None]
            + canonical_poses[..., 1] * cos_yaw[:, None]
            + pelvis_world[:, None, 1]
        )
        world_joints[..., 2] = canonical_poses[..., 2] + pelvis_world[:, None, 2]

        # Convert to COCO 17 (face keypoints use per-frame yaw)
        yaw_world = np.arctan2(rotations[:, 1], rotations[:, 0]).astype(np.float32)
        coco17_joints = self._smplh_to_coco17(world_joints, yaw_world)  # (T, 17, 3)

        # Get court keypoints (static)
        if self.court_kp_3d is None:
            self.court_kp_3d = self.camera_projector.court_kp_3d.numpy()
        court_3d = self.court_kp_3d  # (20, 3)

        # Generate multiple cameras
        cameras_data = []
        for _ in range(self.num_cameras):
            camera = self.camera_projector.sample_camera()

            # Project human keypoints
            human_uv = np.zeros((T, 17, 2), dtype=np.float32)
            human_vis = np.zeros((T, 17), dtype=bool)
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
            court_uv = np.tile(court_uv_single[None, ...], (T, 1, 1))
            court_vis = np.tile(court_vis_single[None, ...], (T, 1))

            # Evaluate camera
            human_ratio, avg_court = self._evaluate_camera(human_vis, court_vis)

            # Store camera data
            cam_data = CameraData(
                camera_params={
                    "center": camera.C.tolist(),
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
                human_kp_visible=human_vis,
                court_kp_visible=court_vis,
                human_visibility_ratio=human_ratio,
                court_visibility_count=avg_court,
            )
            cameras_data.append(cam_data)

        # Filter cameras based on visibility criteria
        filtered_cameras = [
            cam
            for cam in cameras_data
            if cam.human_visibility_ratio >= self.human_visibility_threshold
            and cam.court_visibility_count >= self.court_visibility_threshold
        ]

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
            "num_cameras_filtered": len(filtered_cameras),
        }

        return SceneData(
            meta=meta,
            position=positions,
            rotation=rotations,
            canonical_pose_3d=canonical_poses,
            cameras=filtered_cameras,
        )
