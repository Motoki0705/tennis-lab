"""Pydantic models for the annotation backend API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    """Metadata about the loaded video."""

    fps: float
    frame_count: int
    width: int
    height: int


class BallClipConfig(BaseModel):
    """Defines the sequential clip window used for ball annotation."""

    start_frame: int = Field(ge=0)
    clip_length: int = Field(ge=1)


class Point2D(BaseModel):
    """A 2D point in pixel coordinates."""

    x_px: float
    y_px: float


class BallFrameAnnotation(BaseModel):
    """Ball annotation for a single frame."""

    visibility: Literal[0, 1, 2] = 0
    x_px: float = 0.0
    y_px: float = 0.0
    score: float = 0.0
    source: Literal["manual", "assist", "unknown"] = "manual"


class CourtKeypoint(BaseModel):
    """Court keypoint annotation in pixel coordinates."""

    x_px: float = 0.0
    y_px: float = 0.0
    visibility: Literal[0, 1] = 0
    source: Literal["manual", "assist", "homography", "unknown"] = "manual"


class CourtFrameAnnotation(BaseModel):
    """CourtKP20 annotations for a single frame."""

    frame_idx: int = Field(ge=0)
    keypoints: list[CourtKeypoint]


class ExportResult(BaseModel):
    """Result from an export action."""

    output_dir: str

