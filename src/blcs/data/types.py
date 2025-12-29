"""Type definitions for BLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the BLCS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class BLCSSample(TypedDict):
    """Schema for single BLCS dataset sample.

    Used by BallTrajectoryDataset.__getitem__(). Represents a single
    ball trajectory sequence from one camera view.
    """

    ball_uv: torch.Tensor  # (T, 2) ball 2D trajectory in normalized UV
    ball_mask: torch.Tensor  # (T,) ball visibility mask
    court_kp: torch.Tensor  # (20, 2) court 2D keypoints in normalized UV
    court_vis: torch.Tensor  # (20,) court keypoint visibility flags
    position_3d: torch.Tensor  # (T, 3) ground truth 3D trajectory (normalized)
    velocity_3d: torch.Tensor  # (T, 3) 3D velocity vectors
    seq_len: torch.Tensor  # scalar, actual sequence length


class BLCSBatch(TypedDict):
    """Schema for batched BLCS dataset samples.

    Used by collate_trajectories(). Sequences are padded to max length in batch.
    """

    ball_uv: torch.Tensor  # (B, T_max, 2) padded ball trajectories
    ball_mask: torch.Tensor  # (B, T_max) padded visibility masks
    court_kp: torch.Tensor  # (B, 20, 2) court keypoints
    court_vis: torch.Tensor  # (B, 20) court keypoint visibility
    position_3d: torch.Tensor  # (B, T_max, 3) padded ground truth trajectories
    velocity_3d: torch.Tensor  # (B, T_max, 3) padded velocities
    seq_len: torch.Tensor  # (B,) actual sequence lengths


@dataclass(frozen=True)
class BLCSSceneMeta:
    """Metadata schema for BLCS NPZ scene files.

    This defines the structure of the 'meta' field stored as JSON in NPZ files.
    All generated ball trajectory scenes should conform to this schema.
    """

    scene_id: str
    from_cell: int  # starting court cell (0-11)
    from_side: str  # "near" or "far"
    category: str  # shot category (e.g., "serve", "groundstroke")
    to_cell: int  # ending court cell (-1 if out of bounds)
    t_net: int  # frame index when ball crosses net
    t_fence: int  # frame index when ball hits fence (or -1)
    t_bounce1: int  # first bounce frame index (or -1)
    t_bounce2: int  # second bounce frame index (or -1)
    fps_out: int  # output frames per second
    sim_fps: int  # simulation frames per second
    num_frames: int  # total number of frames in trajectory
    num_cameras_sampled: int  # number of cameras generated
    num_cameras: int  # number of valid cameras (after filtering)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "from_cell": self.from_cell,
            "from_side": self.from_side,
            "category": self.category,
            "to_cell": self.to_cell,
            "t_net": self.t_net,
            "t_fence": self.t_fence,
            "t_bounce1": self.t_bounce1,
            "t_bounce2": self.t_bounce2,
            "fps_out": self.fps_out,
            "sim_fps": self.sim_fps,
            "num_frames": self.num_frames,
            "num_cameras_sampled": self.num_cameras_sampled,
            "num_cameras": self.num_cameras,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BLCSSceneMeta:
        """Create instance from dictionary loaded from JSON/NPZ."""
        return cls(
            scene_id=data["scene_id"],
            from_cell=data["from_cell"],
            from_side=data["from_side"],
            category=data["category"],
            to_cell=data["to_cell"],
            t_net=data["t_net"],
            t_fence=data["t_fence"],
            t_bounce1=data["t_bounce1"],
            t_bounce2=data["t_bounce2"],
            fps_out=data["fps_out"],
            sim_fps=data["sim_fps"],
            num_frames=data["num_frames"],
            num_cameras_sampled=data["num_cameras_sampled"],
            num_cameras=data["num_cameras"],
        )


@dataclass(frozen=True)
class BLCSCameraParams:
    """Camera parameters schema for BLCS scenes.

    Stored as JSON string in NPZ files under 'cam_{i}_params' keys.
    Identical structure to PLCS for consistency.
    """

    center: list[float]  # [x, y, z] camera center in world coordinates
    R: list[list[float]]  # 3x3 rotation matrix (world to camera)
    f: float  # focal length in pixels
    cx: float  # principal point x-coordinate
    cy: float  # principal point y-coordinate
    w: int  # image width in pixels
    h: int  # image height in pixels

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "center": self.center,
            "R": self.R,
            "f": self.f,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BLCSCameraParams:
        """Create instance from dictionary loaded from JSON/NPZ."""
        return cls(
            center=data["center"],
            R=data["R"],
            f=data["f"],
            cx=data["cx"],
            cy=data["cy"],
            w=data["w"],
            h=data["h"],
        )
