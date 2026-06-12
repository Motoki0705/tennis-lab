"""Type definitions for BLCS data structures.
This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the BLCS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch


class BLCSSample(TypedDict):
    """Schema for single BLCS dataset sample.

    Used by BallTrajectoryDataset.__getitem__(). Represents a single
    ball trajectory sequence from one camera view.
    """

    ball_uv: torch.Tensor  # (T, 2) ball 2D trajectory in normalized UV
    ball_vis: torch.Tensor  # (T,) ball visibility flags
    court_kp: torch.Tensor  # (20, 2) court 2D keypoints in normalized UV
    court_vis: torch.Tensor  # (20,) court keypoint visibility flags
    position_3d: torch.Tensor  # (T, 3) ground truth 3D trajectory (normalized)
    velocity_3d: torch.Tensor  # (T, 3) 3D velocity vectors
    seq_len: torch.Tensor  # scalar, actual sequence length


class BLCSBatch(TypedDict):
    """Schema for batched BLCS dataset samples.

    Used by single-profile batch adaptation. Sequences are padded to max length in batch.
    """

    ball_uv: torch.Tensor  # (B, T_max, 2) padded ball trajectories
    ball_vis: torch.Tensor  # (B, T_max) padded visibility flags
    ball_uv_target: NotRequired[torch.Tensor]  # (B, 1, T_max, 2) clean 2D loss target
    ball_vis_target: NotRequired[torch.Tensor]  # (B, 1, T_max) clean visibility target
    ball_mask: torch.Tensor  # (B, T_max) padding mask (1=valid)
    court_kp: torch.Tensor  # (B, 20, 2) court keypoints
    court_vis: torch.Tensor  # (B, 20) court keypoint visibility
    position_3d: torch.Tensor  # (B, T_max, 3) padded ground truth trajectories
    velocity_3d: torch.Tensor  # (B, T_max, 3) padded velocities
    seq_len: torch.Tensor  # (B,) actual sequence lengths


class BLCSMultiViewSample(TypedDict):
    """Schema for multi-view BLCS dataset sample.

    Used by BallTrajectoryDataset.__getitem__() in multiview mode. Contains observations
    from multiple cameras for the same ball trajectory.

    Note: court_kp is expanded to match the temporal dimension (T) for the
    alternating attention architecture. This allows per-frame court context
    without temporal aggregation.
    """

    ball_uv: torch.Tensor  # (N_cam, T, 2) ball 2D trajectories from each camera
    ball_vis: torch.Tensor  # (N_cam, T) ball visibility masks (1=visible)
    ball_uv_target: NotRequired[torch.Tensor]  # (N_cam, T, 2) clean pre-augmentation UV
    ball_vis_target: NotRequired[torch.Tensor]  # (N_cam, T) clean pre-augmentation vis
    ball_mask: torch.Tensor  # (N_cam, T) sequence padding masks (1=valid token)
    court_kp: torch.Tensor  # (N_cam, T, 20, 2) court keypoints expanded to T
    court_vis: torch.Tensor  # (N_cam, T, 20) court visibility expanded to T
    position_3d: torch.Tensor  # (T, 3) ground truth 3D trajectory (shared)
    velocity_3d: torch.Tensor  # (T, 3) 3D velocity vectors (shared)
    seq_len: torch.Tensor  # scalar, actual sequence length
    # Camera parameters (per-camera, for reprojection loss)
    camera_R: torch.Tensor  # (N_cam, 3, 3) rotation matrices (world → camera)
    camera_C: torch.Tensor  # (N_cam, 3) camera centres in world coordinates
    camera_f: torch.Tensor  # (N_cam,) focal lengths in pixels
    camera_cx: torch.Tensor  # (N_cam,) principal-point x
    camera_cy: torch.Tensor  # (N_cam,) principal-point y
    camera_w: torch.Tensor  # (N_cam,) image width
    camera_h: torch.Tensor  # (N_cam,) image height


class BLCSMultiViewBatch(TypedDict):
    """Schema for batched multi-view BLCS dataset samples."""

    ball_uv: torch.Tensor  # (B, N_max, T_max, 2)
    ball_vis: torch.Tensor  # (B, N_max, T_max)
    ball_uv_target: NotRequired[torch.Tensor]  # (B, N_max, T_max, 2)
    ball_vis_target: NotRequired[torch.Tensor]  # (B, N_max, T_max)
    ball_mask: torch.Tensor  # (B, N_max, T_max) padding mask
    court_kp: torch.Tensor  # (B, N_max, T_max, 20, 2)
    court_vis: torch.Tensor  # (B, N_max, T_max, 20)
    position_3d: torch.Tensor  # (B, T_max, 3)
    velocity_3d: torch.Tensor  # (B, T_max, 3)
    seq_len: torch.Tensor  # (B,)
    # Camera parameters (padded to N_max cameras)
    camera_R: torch.Tensor  # (B, N_max, 3, 3)
    camera_C: torch.Tensor  # (B, N_max, 3)
    camera_f: torch.Tensor  # (B, N_max)
    camera_cx: torch.Tensor  # (B, N_max)
    camera_cy: torch.Tensor  # (B, N_max)
    camera_w: torch.Tensor  # (B, N_max)
    camera_h: torch.Tensor  # (B, N_max)


@dataclass(frozen=True)
class BLCSSceneMeta:
    """Metadata schema for BLCS NPZ scene files (rally-only scene format)."""

    scene_id: str
    initial_from_cell: int  # Starting cell for first shot (0-8)
    initial_from_side: str  # "near" or "far"

    rally_length: int  # Number of shots in rally
    end_reason: str  # Rally termination reason
    winner_side: str | None  # "near", "far", or None

    shots: list[dict]  # List of BLCSShotEventMeta.to_dict() results

    fps_out: int
    sim_fps: int
    num_frames: int
    num_cameras_sampled: int
    num_cameras: int

    # Per-scene variation metadata (may be absent in older files)
    physics_config: dict | None = None
    court_config: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "initial_from_cell": self.initial_from_cell,
            "initial_from_side": self.initial_from_side,
            "rally_length": self.rally_length,
            "end_reason": self.end_reason,
            "winner_side": self.winner_side,
            "shots": self.shots,
            "fps_out": self.fps_out,
            "sim_fps": self.sim_fps,
            "num_frames": self.num_frames,
            "num_cameras_sampled": self.num_cameras_sampled,
            "num_cameras": self.num_cameras,
            "physics_config": self.physics_config,
            "court_config": self.court_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BLCSSceneMeta:
        """Create instance from dictionary loaded from JSON/NPZ."""
        return cls(
            scene_id=data["scene_id"],
            initial_from_cell=data["initial_from_cell"],
            initial_from_side=data["initial_from_side"],
            rally_length=data["rally_length"],
            end_reason=data["end_reason"],
            winner_side=data.get("winner_side"),
            shots=data["shots"],
            fps_out=data["fps_out"],
            sim_fps=data["sim_fps"],
            num_frames=data["num_frames"],
            num_cameras_sampled=data["num_cameras_sampled"],
            num_cameras=data["num_cameras"],
            physics_config=data["physics_config"],
            court_config=data["court_config"],
        )
