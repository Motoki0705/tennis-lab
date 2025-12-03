"""Scene generator for PLCS training data.

This module generates training scenes by combining motion sequences with
virtual camera configurations and projecting to 2D.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.plcs.data.motion_sampler import MotionSampler, MotionSequence
from src.utils.geometry import (
    BASELINE_CLEAR,
    FACE_KEYPOINT_OFFSETS,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_POST,
    SIDELINE_CLEAR,
    SMPLH_TO_COCO17_MAPPING,
    Camera,
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
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
    canonical_pose_3d: np.ndarray  # (T, J, 3) local coordinate pose

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
        court_kp_tensor = court_keypoints_3d()  # (20, 3)
        self.court_kp_3d = (
            court_kp_tensor.numpy()
            if hasattr(court_kp_tensor, "numpy")
            else court_kp_tensor
        )

        # Parse config
        sim_cfg = self.config.get("simulation", {})
        self.num_cameras = sim_cfg.get("num_cameras", 15)
        self.human_visibility_threshold = sim_cfg.get("human_visibility_threshold", 0.8)
        self.court_visibility_threshold = sim_cfg.get("court_visibility_threshold", 15)

        # Camera config
        cam_cfg = self.config.get("camera", {})
        self.cam_z_min = cam_cfg.get("z_min", 3.0)
        self.cam_z_max = cam_cfg.get("z_max", 5.0)
        self.cam_hfov_deg = cam_cfg.get("hfov_deg", 60.0)
        self.image_size = tuple(cam_cfg.get("image_size", [1280, 720]))

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

    def _sample_camera(self) -> Camera:
        """Sample a camera position around the court.

        Returns:
            Camera instance.

        """
        # Sample camera on fence perimeter
        fence_x = HALF_DOUBLES_WIDTH + SIDELINE_CLEAR
        fence_y = HALF_LENGTH + BASELINE_CLEAR

        # Random position on fence rectangle
        perimeter = 2 * (fence_x + fence_y)
        t = random.uniform(0, perimeter)

        if t < fence_x:
            # Top side
            cam_x = -fence_x + t * 2
            cam_y = fence_y
        elif t < fence_x + fence_y:
            # Right side
            cam_x = fence_x
            cam_y = fence_y - (t - fence_x) * 2
        elif t < 2 * fence_x + fence_y:
            # Bottom side
            cam_x = fence_x - (t - fence_x - fence_y) * 2
            cam_y = -fence_y
        else:
            # Left side
            cam_x = -fence_x
            cam_y = -fence_y + (t - 2 * fence_x - fence_y) * 2

        # Random height
        cam_z = random.uniform(self.cam_z_min, self.cam_z_max)

        # Look at court center with some random offset
        target_x = random.uniform(-2, 2)
        target_y = random.uniform(-2, 2)
        target_z = random.uniform(0.5, 1.5)

        camera = make_look_at_camera(
            center=(cam_x, cam_y, cam_z),
            look_at=(target_x, target_y, target_z),
            hfov_deg=self.cam_hfov_deg,
            image_size=self.image_size,
        )

        return camera

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
            - rotations: (T, 2) sin/cos yaw
            - canonical_poses: (T, J, 3) local coordinate poses

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

        # Rotation matrix for init_yaw
        cos_yaw = math.cos(init_yaw)
        sin_yaw = math.sin(init_yaw)
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
        positions[:, 0] = court_trans[:, 0] / HALF_DOUBLES_WIDTH
        positions[:, 1] = court_trans[:, 1] / HALF_LENGTH
        positions[:, 2] = court_trans[:, 2] / NET_HEIGHT_POST

        # Compute rotations (yaw)
        # Extract yaw from motion (simplified: assume forward is +Y in local frame)
        rotations = np.zeros((T, 2), dtype=np.float32)
        rotations[:, 0] = sin_yaw  # sin(yaw)
        rotations[:, 1] = cos_yaw  # cos(yaw)

        # Canonical poses: joints relative to pelvis, in local frame
        pelvis = joints_3d[:, 0:1, :]  # (T, 1, 3)
        canonical_poses = joints_3d - pelvis  # (T, J, 3)

        return positions, rotations, canonical_poses

    def _smplh_to_coco17(
        self,
        joints_3d: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        """Convert SMPL-H joints to COCO 17 format.

        Args:
            joints_3d: SMPL-H joints, shape (T, J, 3) or (J, 3).
            yaw: Yaw angle for face keypoint orientation.

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

        # Rotation for face direction
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        rot = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        for coco_idx, offset in FACE_KEYPOINT_OFFSETS.items():
            offset_arr = np.array(offset, dtype=np.float32)
            rotated_offset = offset_arr @ rot.T
            coco17[:, coco_idx, :] = head_pos + rotated_offset

        if squeeze:
            coco17 = coco17[0]

        return coco17

    def _project_to_uv(
        self,
        points_3d: np.ndarray,
        camera: Camera,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D points to UV coordinates.

        Args:
            points_3d: 3D points, shape (..., 3).
            camera: Camera for projection.

        Returns:
            Tuple of (uv, visible):
            - uv: UV coordinates, shape (..., 2), normalized [0, 1]
            - visible: Visibility mask, shape (...)

        """
        original_shape = points_3d.shape[:-1]
        points_flat = points_3d.reshape(-1, 3)

        # Convert to torch for projection
        points_t = torch.from_numpy(points_flat).float()
        uv_t, in_front = project_points(camera, points_t)
        uv = uv_t.numpy()
        in_front = in_front.numpy()

        # Normalize UV
        uv[:, 0] /= self.image_size[0]
        uv[:, 1] /= self.image_size[1]

        # Check visibility (within [0, 1] range and in front of camera)
        in_bounds = (
            (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
        )

        visible = in_bounds & in_front

        # Reshape
        uv = uv.reshape(*original_shape, 2)
        visible = visible.reshape(original_shape)

        return uv, visible

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
        # Human: fraction of frames where >= 80% of keypoints visible
        human_per_frame = human_visible.mean(axis=1)  # (T,)
        human_ratio = (human_per_frame >= 0.8).mean()

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
        joints_3d = motion.joints_3d  # (T, J, 3)

        # Transform joints to world (court) coordinates
        cos_yaw = math.cos(init_yaw)
        sin_yaw = math.sin(init_yaw)
        rot_mat = np.array(
            [
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Center and rotate joints
        init_offset = motion.trans[0]
        centered_joints = joints_3d - init_offset
        # Apply rotation to all joints
        world_joints = np.einsum("tji,ki->tjk", centered_joints, rot_mat)
        world_joints[..., 0] += init_x
        world_joints[..., 1] += init_y

        # Convert to COCO 17
        coco17_joints = self._smplh_to_coco17(world_joints, init_yaw)  # (T, 17, 3)

        # Get court keypoints (static)
        court_3d = self.court_kp_3d  # (20, 3)

        # Generate multiple cameras
        cameras_data = []
        for _ in range(self.num_cameras):
            camera = self._sample_camera()

            # Project human keypoints
            human_uv = np.zeros((T, 17, 2), dtype=np.float32)
            human_vis = np.zeros((T, 17), dtype=bool)
            for t in range(T):
                uv, vis = self._project_to_uv(coco17_joints[t], camera)
                human_uv[t] = uv
                human_vis[t] = vis

            # Project court keypoints (same for all frames)
            court_uv_single, court_vis_single = self._project_to_uv(court_3d, camera)
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

    def save_scene(
        self,
        scene: SceneData,
        output_path: Path | str,
    ) -> None:
        """Save scene to NPZ file.

        Args:
            scene: Scene data to save.
            output_path: Path for output NPZ file.

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data dict
        data = {
            # Metadata (stored as object array for dict)
            "meta": np.array(scene.meta, dtype=object),
            # 3D data
            "position": scene.position,
            "rotation": scene.rotation,
            "canonical_pose_3d": scene.canonical_pose_3d,
            # Number of cameras
            "num_cameras": np.array([len(scene.cameras)]),
        }

        # Camera data
        for i, cam in enumerate(scene.cameras):
            prefix = f"cam_{i}_"
            data[f"{prefix}params"] = np.array(cam.camera_params, dtype=object)
            data[f"{prefix}human_kp_uv"] = cam.human_kp_uv
            data[f"{prefix}court_kp_uv"] = cam.court_kp_uv
            data[f"{prefix}human_kp_visible"] = cam.human_kp_visible
            data[f"{prefix}court_kp_visible"] = cam.court_kp_visible
            data[f"{prefix}human_visibility_ratio"] = np.array(
                [cam.human_visibility_ratio]
            )
            data[f"{prefix}court_visibility_count"] = np.array(
                [cam.court_visibility_count]
            )

        np.savez_compressed(output_path, **data)

    def build_scene_meta(self, scene: SceneData) -> dict:
        """Build JSON-serializable metadata dictionary.

        Args:
            scene: Scene data.

        Returns:
            Dictionary suitable for JSON serialization.

        """
        meta = scene.meta
        return {
            "scene_id": meta["scene_id"],
            "motion_source": meta["motion_source"],
            "motion_category": meta["motion_category"],
            "gender": meta["gender"],
            "fps": meta["fps"],
            "num_frames": meta["num_frames"],
            "duration_sec": meta["num_frames"] / meta["fps"],
            "initial_position": list(meta["initial_position"]),
            "initial_yaw_deg": float(np.degrees(meta["initial_yaw"])),
            "num_cameras_sampled": meta["num_cameras_sampled"],
            "num_cameras": meta["num_cameras_filtered"],
            "cameras": [
                {
                    "human_visibility_ratio": cam.human_visibility_ratio,
                    "court_visibility_count": cam.court_visibility_count,
                }
                for cam in scene.cameras
            ],
            # Data statistics
            "position_range": {
                "x": [
                    float(scene.position[:, 0].min()),
                    float(scene.position[:, 0].max()),
                ],
                "y": [
                    float(scene.position[:, 1].min()),
                    float(scene.position[:, 1].max()),
                ],
                "z": [
                    float(scene.position[:, 2].min()),
                    float(scene.position[:, 2].max()),
                ],
            },
        }

    @staticmethod
    def load_scene(path: Path | str) -> SceneData:
        """Load scene from NPZ file.

        Args:
            path: Path to NPZ file.

        Returns:
            Loaded SceneData.

        """
        data = np.load(path, allow_pickle=True)

        meta = data["meta"].item()
        position = data["position"]
        rotation = data["rotation"]
        canonical_pose_3d = data["canonical_pose_3d"]

        num_cameras = int(data["num_cameras"][0])
        cameras = []
        for i in range(num_cameras):
            prefix = f"cam_{i}_"
            cam = CameraData(
                camera_params=data[f"{prefix}params"].item(),
                human_kp_uv=data[f"{prefix}human_kp_uv"],
                court_kp_uv=data[f"{prefix}court_kp_uv"],
                human_kp_visible=data[f"{prefix}human_kp_visible"],
                court_kp_visible=data[f"{prefix}court_kp_visible"],
                human_visibility_ratio=float(
                    data[f"{prefix}human_visibility_ratio"][0]
                ),
                court_visibility_count=float(
                    data[f"{prefix}court_visibility_count"][0]
                ),
            )
            cameras.append(cam)

        return SceneData(
            meta=meta,
            position=position,
            rotation=rotation,
            canonical_pose_3d=canonical_pose_3d,
            cameras=cameras,
        )
