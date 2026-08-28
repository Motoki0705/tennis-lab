"""Type definitions for PLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the PLCS pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch


class PLCSBatch(TypedDict):
    """Unified PLCS batch schema for frame/sequence/single/multiview modes.

    Shapes use camera-time ordering:
    - ``human_kp``: (B, N, T, 17, 2)
    - ``court_kp``: (B, N, T, 20, 2)
    - ``human_vis``: (B, N, T, 17)
    - ``padding_mask``: (B, N, T), ``True`` marks padding
    - ``court_vis``: (B, N, T, 20)
    - ``position``: (B, T, 3)
    - ``rotation``: (B, T, 2)
    """

    human_kp: torch.Tensor
    court_kp: torch.Tensor
    human_vis: torch.Tensor
    padding_mask: torch.Tensor
    court_vis: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor
    human_kp_3d: torch.Tensor
    human_kp_target: torch.Tensor
    human_vis_target: torch.Tensor
    camera_R: torch.Tensor
    camera_C: torch.Tensor
    camera_f: torch.Tensor
    camera_cx: torch.Tensor
    camera_cy: torch.Tensor
    camera_w: torch.Tensor
    camera_h: torch.Tensor


@dataclass(frozen=True)
class PLCSSceneMeta:
    """Metadata schema for PLCS scene files.

    This defines the structure of the 'meta' field stored as meta.json in scene directories.
    All generated scenes should conform to this schema.
    """

    scene_id: str
    motion_source: str  # e.g., "amass", "custom"
    motion_category: str  # e.g., "walk", "run", "tennis_serve"
    gender: str  # "male", "female", or "neutral"
    fps: int  # frames per second
    num_frames: int  # total number of frames in the scene
    initial_position: list[float]  # [x, y] starting position on court
    initial_yaw: float  # initial yaw angle in radians
    num_cameras_sampled: int  # number of cameras generated for this scene
    num_cameras: int  # number of cameras stored for this scene
    court_coordinate_normalization: dict[str, Any]
    track_instances: list[dict]

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
            "court_coordinate_normalization": self.court_coordinate_normalization,
            "track_instances": self.track_instances,
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
            court_coordinate_normalization=data["court_coordinate_normalization"],
            track_instances=data["track_instances"],
        )
