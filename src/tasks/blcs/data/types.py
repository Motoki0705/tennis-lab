"""Type definitions for BLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the BLCS pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, TypeVar

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
    ball_mask: torch.Tensor  # (N_cam, T) sequence padding masks (1=valid token)
    court_kp: torch.Tensor  # (N_cam, T, 20, 2) court keypoints expanded to T
    court_vis: torch.Tensor  # (N_cam, T, 20) court visibility expanded to T
    position_3d: torch.Tensor  # (T, 3) ground truth 3D trajectory (shared)
    velocity_3d: torch.Tensor  # (T, 3) 3D velocity vectors (shared)
    seq_len: torch.Tensor  # scalar, actual sequence length


class BLCSMultiViewBatch(TypedDict):
    """Schema for batched multi-view BLCS dataset samples."""

    ball_uv: torch.Tensor  # (B, N_max, T_max, 2)
    ball_vis: torch.Tensor  # (B, N_max, T_max)
    ball_mask: torch.Tensor  # (B, N_max, T_max) padding mask
    court_kp: torch.Tensor  # (B, N_max, T_max, 20, 2)
    court_vis: torch.Tensor  # (B, N_max, T_max, 20)
    position_3d: torch.Tensor  # (B, T_max, 3)
    velocity_3d: torch.Tensor  # (B, T_max, 3)
    seq_len: torch.Tensor  # (B,)


@dataclass(frozen=True)
class BLCSSceneMeta:
    """Metadata schema for BLCS NPZ scene files (rally-only scene format)."""

    scene_id: str
    initial_from_cell: int  # Starting cell for first shot (0-19)
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
        )


class BLCSShotEventMeta:
    """Metadata for a single shot within a rally.

    Records timing and event information for each shot in the rally sequence.
    """

    shot_index: int  # 0-indexed shot number in rally
    from_side: str  # "near" or "far"
    from_cell: int  # Starting cell ID (0-19)
    category: str  # Shot category

    # Frame indices (relative to rally start, at output_fps)
    t_start: int  # Frame when this shot starts
    t_net: int  # Frame when ball crosses net (-1 if not crossed)
    t_bounce1: int  # First bounce frame (-1 if not bounced)
    t_bounce2: int  # Second bounce frame (-1 if not bounced)
    t_return: int  # Frame when return hit occurs (-1 if rally ended)

    # Landing info
    to_cell: int  # Target cell (-1 if out/net)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "shot_index": self.shot_index,
            "from_side": self.from_side,
            "from_cell": self.from_cell,
            "category": self.category,
            "t_start": self.t_start,
            "t_net": self.t_net,
            "t_bounce1": self.t_bounce1,
            "t_bounce2": self.t_bounce2,
            "t_return": self.t_return,
            "to_cell": self.to_cell,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BLCSShotEventMeta:
        """Create instance from dictionary loaded from JSON/NPZ."""
        return cls(
            shot_index=data["shot_index"],
            from_side=data["from_side"],
            from_cell=data["from_cell"],
            category=data["category"],
            t_start=data["t_start"],
            t_net=data["t_net"],
            t_bounce1=data["t_bounce1"],
            t_bounce2=data["t_bounce2"],
            t_return=data["t_return"],
            to_cell=data["to_cell"],
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

# Pydantic schemas for runtime validation

_T = TypeVar("_T")
PYDANTIC_AVAILABLE: bool = False

if TYPE_CHECKING:
    class BaseModel:  # pragma: no cover - typing-only stub
        pass

    def Field(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - typing-only stub
        return None

    def field_validator(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., _T]], Callable[..., _T]]:  # pragma: no cover
        def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
            return func

        return decorator

    PYDANTIC_AVAILABLE = True
else:
    try:
        from pydantic import BaseModel, Field, field_validator
    except ImportError:
        BaseModel = object  # type: ignore[assignment]

        def Field(*args: Any, **kwargs: Any) -> Any:
            return None

        def field_validator(
            *args: Any, **kwargs: Any
        ) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
            def decorator(func: Callable[..., _T]) -> Callable[..., _T]:
                return func

            return decorator
    else:
        PYDANTIC_AVAILABLE = True


class BLCSSceneMetaModel(BaseModel):
    """Pydantic model for rally-scene metadata with runtime validation."""

    scene_id: str = Field(..., min_length=1)
    initial_from_cell: int = Field(..., ge=0, le=19)
    initial_from_side: str = Field(..., pattern="^(near|far)$")
    rally_length: int = Field(..., gt=0)
    end_reason: str = Field(..., min_length=1)
    winner_side: str | None = Field(default=None, pattern="^(near|far)$")
    shots: list[dict] = Field(default_factory=list)
    fps_out: int = Field(..., gt=0)
    sim_fps: int = Field(..., gt=0)
    num_frames: int = Field(..., gt=0)
    num_cameras_sampled: int = Field(..., ge=0)
    num_cameras: int = Field(..., ge=0)

    model_config = {"frozen": True}


class BLCSCameraParamsModel(BaseModel):
    """Pydantic model for camera parameters (same as PLCS)."""

    center: list[float] = Field(..., min_length=3, max_length=3)
    R: list[list[float]] = Field(..., description="3x3 rotation matrix")
    f: float = Field(..., gt=0)
    cx: float
    cy: float
    w: int = Field(..., gt=0)
    h: int = Field(..., gt=0)

    @field_validator("R")
    @classmethod
    def validate_rotation_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("R must be 3x3 matrix")
        return v

    @field_validator("center")
    @classmethod
    def validate_center(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("Center must have 3 coordinates")
        return v

    model_config = {"frozen": True}
