"""Type definitions for PLCS data structures.

This module provides TypedDict schemas for dataset batches and dataclasses
for metadata, ensuring type safety throughout the PLCS pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

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

# Pydantic schemas for runtime validation
# These provide stronger validation than dataclasses

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


class PLCSSceneMetaModel(BaseModel):
    """Pydantic model for PLCS scene metadata with runtime validation.

    This provides stronger validation than the dataclass version:
    - Type validation at runtime
    - Value constraints (e.g., fps > 0)
    - Automatic JSON serialization/deserialization
    """

    scene_id: str = Field(..., min_length=1, description="Unique scene identifier")
    motion_source: str = Field(..., description="Motion data source (e.g., 'amass')")
    motion_category: str = Field(..., description="Motion category (e.g., 'walk', 'run')")
    gender: str = Field(..., pattern="^(male|female|neutral)$", description="Gender")
    fps: int = Field(..., gt=0, description="Frames per second")
    num_frames: int = Field(..., gt=0, description="Total number of frames")
    initial_position: list[float] = Field(..., min_length=3, max_length=3)
    initial_yaw: float
    num_cameras_sampled: int = Field(..., ge=0)
    num_cameras: int = Field(..., ge=0)

    @field_validator("initial_position")
    @classmethod
    def validate_position(cls, v: list[float]) -> list[float]:
        """Validate that position has exactly 3 coordinates."""
        if len(v) != 3:
            raise ValueError(f"Position must have 3 coordinates, got {len(v)}")
        return v

    model_config = {"frozen": True}  # Immutable like dataclass(frozen=True)


class PLCSCameraParamsModel(BaseModel):
    """Pydantic model for camera parameters with validation."""

    center: list[float] = Field(..., min_length=3, max_length=3)
    R: list[list[float]] = Field(..., description="3x3 rotation matrix")
    f: float = Field(..., gt=0, description="Focal length in pixels")
    cx: float = Field(..., description="Principal point x")
    cy: float = Field(..., description="Principal point y")
    w: int = Field(..., gt=0, description="Image width")
    h: int = Field(..., gt=0, description="Image height")

    @field_validator("R")
    @classmethod
    def validate_rotation_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate 3x3 rotation matrix shape."""
        if len(v) != 3:
            raise ValueError(f"R must have 3 rows, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 3:
                raise ValueError(f"R row {i} must have 3 columns, got {len(row)}")
        return v

    @field_validator("center")
    @classmethod
    def validate_center(cls, v: list[float]) -> list[float]:
        """Validate center has 3 coordinates."""
        if len(v) != 3:
            raise ValueError(f"Center must have 3 coordinates, got {len(v)}")
        return v

    model_config = {"frozen": True}
