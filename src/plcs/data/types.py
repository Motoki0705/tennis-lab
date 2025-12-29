"""Type definitions for PLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the PLCS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class PLCSFrameBatch(TypedDict):
    """Schema for frame-level PLCS dataset batch.

    Used by SceneDataset.__getitem__(). All tensors are for a single frame.
    """

    human_kp: torch.Tensor  # (34,) flattened human keypoints, normalized UV
    court_kp: torch.Tensor  # (40,) flattened court keypoints, normalized UV
    human_vis: torch.Tensor  # (17,) visibility flags for human keypoints
    court_vis: torch.Tensor  # (20,) visibility flags for court keypoints
    position: torch.Tensor  # (3,) normalized court position [x_norm, y_norm, z_norm]
    rotation: torch.Tensor  # (2,) player orientation [sin(yaw), cos(yaw)]


class PLCSSequenceBatch(TypedDict):
    """Schema for sequence-level PLCS dataset batch.

    Used by SceneSequenceDataset.__getitem__(). Contains temporal sequences.
    """

    human_kp: torch.Tensor  # (T, 17, 2) human keypoints over time
    court_kp: torch.Tensor  # (1, 20, 2) aggregated court keypoints (time-invariant)
    human_vis: torch.Tensor  # (T, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (1, 20) aggregated visibility flags for court
    position: torch.Tensor  # (T, 3) normalized court positions over time
    rotation: torch.Tensor  # (T, 2) player orientations over time


@dataclass(frozen=True)
class PLCSSceneMeta:
    """Metadata schema for PLCS NPZ scene files.

    This defines the structure of the 'meta' field stored as JSON in NPZ files.
    All generated scenes should conform to this schema.
    """

    scene_id: str
    motion_source: str  # e.g., "amass", "custom"
    motion_category: str  # e.g., "walk", "run", "tennis_serve"
    gender: str  # "male", "female", or "neutral"
    fps: int  # frames per second
    num_frames: int  # total number of frames in the scene
    initial_position: list[float]  # [x, y, z] starting position
    initial_yaw: float  # initial yaw angle in radians
    num_cameras_sampled: int  # number of cameras generated for this scene
    num_cameras: int  # number of valid cameras (after filtering)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "motion_source": self.motion_source,
            "motion_category": self.motion_category,
            "gender": self.gender,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "initial_position": self.initial_position,
            "initial_yaw": self.initial_yaw,
            "num_cameras_sampled": self.num_cameras_sampled,
            "num_cameras": self.num_cameras,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PLCSSceneMeta:
        """Create instance from dictionary loaded from JSON/NPZ."""
        return cls(
            scene_id=data["scene_id"],
            motion_source=data["motion_source"],
            motion_category=data["motion_category"],
            gender=data["gender"],
            fps=data["fps"],
            num_frames=data["num_frames"],
            initial_position=data["initial_position"],
            initial_yaw=data["initial_yaw"],
            num_cameras_sampled=data["num_cameras_sampled"],
            num_cameras=data["num_cameras"],
        )


@dataclass(frozen=True)
class PLCSCameraParams:
    """Camera parameters schema for PLCS scenes.

    Stored as JSON string in NPZ files under 'cam_{i}_params' keys.
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
    def from_dict(cls, data: dict) -> PLCSCameraParams:
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
