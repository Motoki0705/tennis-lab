"""Type definitions for PLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the PLCS pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import torch


class PLCSBatch(TypedDict):
    """Unified PLCS batch schema for frame/sequence/single/multiview modes.

    Shapes use camera-time ordering:
    - ``human_kp``: (B, N, T, 17, 2)
    - ``court_kp``: (B, N, T, 20, 2)
    - ``human_vis``: (B, N, T, 17)
    - ``human_mask``: (B, N, T), padding mask (True/1=valid token)
    - ``court_vis``: (B, N, T, 20)
    - ``position``: (B, T, 3)
    - ``rotation``: (B, T, 2)
    """

    human_kp: torch.Tensor
    court_kp: torch.Tensor
    human_vis: torch.Tensor
    human_mask: torch.Tensor
    court_vis: torch.Tensor
    position: torch.Tensor
    rotation: torch.Tensor


class PLCSFrameBatch(TypedDict):
    """Schema for frame-level PLCS dataset batch.

    Used by SceneDataset.__getitem__(). All tensors are for a single frame.
    """

    human_kp: torch.Tensor  # (34,) flattened human keypoints, normalized UV
    court_kp: torch.Tensor  # (40,) flattened court keypoints, normalized UV
    human_vis: torch.Tensor  # (17,) visibility flags for human keypoints
    court_vis: torch.Tensor  # (20,) visibility flags for court keypoints
    position: torch.Tensor  # (3,) normalized court position [x_norm, y_norm, z_norm]
    rotation: torch.Tensor  # (2,) player orientation [cos(yaw), sin(yaw)]
    human_kp_3d: torch.Tensor  # (17, 3) COCO17 world/court-coordinate keypoints


class PLCSSequenceBatch(TypedDict):
    """Schema for sequence-level PLCS dataset batch.

    Used by SceneSequenceDataset.__getitem__(). Contains temporal sequences.
    """

    human_kp: torch.Tensor  # (T, 17, 2) human keypoints over time
    court_kp: torch.Tensor  # (T, 20, 2) court keypoints over time (not aggregated)
    human_vis: torch.Tensor  # (T, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (T, 20) visibility flags for court keypoints
    position: torch.Tensor  # (T, 3) normalized court positions over time
    rotation: torch.Tensor  # (T, 2) player orientations over time


class PLCSMultiViewBatch(TypedDict):
    """Schema for multi-view PLCS dataset batch.

    Used by MultiViewSceneDataset.__getitem__(). Contains observations from
    multiple cameras for the same frame, enabling multi-camera fusion models.
    """

    human_kp: torch.Tensor  # (N_cam, 17, 2) human keypoints from each camera
    court_kp: torch.Tensor  # (N_cam, 20, 2) court keypoints from each camera
    human_vis: torch.Tensor  # (N_cam, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (N_cam, 20) visibility flags for court keypoints
    camera_params: list[dict]  # List of camera parameter dicts
    num_views: torch.Tensor  # scalar, number of views in this sample
    position: torch.Tensor  # (3,) normalized court position (shared GT)
    rotation: torch.Tensor  # (2,) player orientation (shared GT)


class PLCSMultiViewSequenceBatch(TypedDict):
    """Schema for multi-view sequential PLCS dataset batch.

    Used by MultiViewSequenceDataset.__getitem__(). Contains observations from
    multiple cameras over a temporal sequence for multi-camera sequential models.

    Uses camera-time ordering: (N_cam, T, ...) where N_cam=cameras, T=time.
    """

    human_kp: torch.Tensor  # (N_cam, T, 17, 2) human keypoints from each camera
    court_kp: torch.Tensor  # (N_cam, T, 20, 2) court keypoints from each camera
    human_vis: torch.Tensor  # (N_cam, T, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (N_cam, T, 20) visibility flags for court keypoints
    camera_params: list[dict]  # List of camera parameter dicts
    num_views: torch.Tensor  # scalar, number of views in this sample
    seq_len: torch.Tensor  # scalar, actual sequence length in this sample
    view_mask: torch.Tensor  # (N_cam,) True for valid camera views
    seq_mask: torch.Tensor  # (T,) True for valid frames
    position: torch.Tensor  # (T, 3) normalized court positions over time
    rotation: torch.Tensor  # (T, 2) player orientations over time


class PLCSMultiViewBatchCollated(TypedDict):
    """Schema for collated multi-view PLCS dataset batches."""

    human_kp: torch.Tensor  # (B, N_cam, 17, 2) human keypoints from each camera
    court_kp: torch.Tensor  # (B, N_cam, 20, 2) court keypoints from each camera
    human_vis: torch.Tensor  # (B, N_cam, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (B, N_cam, 20) visibility flags for court keypoints
    camera_params: list[list[dict]]  # Per-sample list of camera parameter dicts
    num_views: torch.Tensor  # (B,) number of views in each sample
    view_mask: torch.Tensor  # (B, N_cam) True for valid camera views
    position: torch.Tensor  # (B, 3) normalized court position (shared GT)
    rotation: torch.Tensor  # (B, 2) player orientation (shared GT)


class PLCSMultiViewSequenceBatchCollated(TypedDict):
    """Schema for collated multi-view sequential PLCS dataset batches.

    Uses camera-time ordering: (B, N_cam, T, ...) where N_cam=cameras, T=time.
    """

    human_kp: torch.Tensor  # (B, N_cam, T, 17, 2) human keypoints from each camera
    court_kp: torch.Tensor  # (B, N_cam, T, 20, 2) court keypoints from each camera
    human_vis: torch.Tensor  # (B, N_cam, T, 17) visibility flags for human keypoints
    court_vis: torch.Tensor  # (B, N_cam, T, 20) visibility flags for court keypoints
    camera_params: list[list[dict]]  # Per-sample list of camera parameter dicts
    num_views: torch.Tensor  # (B,) number of views in each sample
    seq_len: torch.Tensor  # (B,) sequence length per sample
    view_mask: torch.Tensor  # (B, N_cam) True for valid camera views
    seq_mask: torch.Tensor  # (B, T) True for valid frames
    position: torch.Tensor  # (B, T, 3) normalized court positions over time
    rotation: torch.Tensor  # (B, T, 2) player orientations over time


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
