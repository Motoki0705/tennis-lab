"""Type definitions for BLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the BLCS pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

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


class BLCSMultiViewSample(TypedDict):
    """Schema for multi-view BLCS dataset sample.

    Used by MultiViewBallTrajectoryDataset.__getitem__(). Contains observations
    from multiple cameras for the same ball trajectory.

    Note: court_kp is expanded to match the temporal dimension (T) for the
    alternating attention architecture. This allows per-frame court context
    without temporal aggregation.
    """

    ball_uv: torch.Tensor  # (N_cam, T, 2) ball 2D trajectories from each camera
    ball_mask: torch.Tensor  # (N_cam, T) ball visibility masks
    court_kp: torch.Tensor  # (N_cam, T, 20, 2) court keypoints expanded to T
    court_vis: torch.Tensor  # (N_cam, T, 20) court visibility expanded to T
    camera_params: list[dict]  # List of camera parameter dicts
    num_views: torch.Tensor  # scalar, number of views in this sample
    position_3d: torch.Tensor  # (T, 3) ground truth 3D trajectory (shared)
    velocity_3d: torch.Tensor  # (T, 3) 3D velocity vectors (shared)
    seq_len: torch.Tensor  # scalar, actual sequence length


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
    """Pydantic model for BLCS scene metadata with runtime validation.

    Provides stronger validation than dataclass:
    - Runtime type checking
    - Value constraints (e.g., fps_out > 0)
    - Coordinate range validation
    """

    scene_id: str = Field(..., min_length=1)
    from_cell: int = Field(..., ge=0, le=11, description="Starting court cell (0-11)")
    from_side: str = Field(..., pattern="^(near|far)$")
    category: str = Field(..., description="Shot category")
    to_cell: int = Field(..., ge=-1, le=11, description="-1 for out, 0-11 for cell")
    t_net: int = Field(..., ge=-1, description="Frame when ball crosses net")
    t_fence: int = Field(..., ge=-1)
    t_bounce1: int = Field(..., ge=-1)
    t_bounce2: int = Field(..., ge=-1)
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
