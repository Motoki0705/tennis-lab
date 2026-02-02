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


class BallAssistMeta(BaseModel):
    """Metadata for a cached ball assist run."""

    checkpoint_path: str | None = None
    model_type: Literal["wasb", "hrcnet"] = "wasb"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(default=64, ge=1)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_disp: int = Field(default=300, ge=1)
    created_at: str = ""


class BallAssistState(BaseModel):
    """Cached ball assist annotations for a clip."""

    clip: BallClipConfig
    meta: BallAssistMeta
    annotations: dict[int, BallFrameAnnotation] = Field(default_factory=dict)


class BallAssistSummary(BaseModel):
    """Summary of cached ball assist availability."""

    available: bool
    clip_matches_current: bool
    clip: BallClipConfig | None = None
    meta: BallAssistMeta | None = None
    count: int = 0


class BallAssistRunRequest(BaseModel):
    """Optional overrides when running ball assist."""

    checkpoint_path: str | None = None
    model_type: Literal["wasb", "hrcnet"] | None = None
    device: Literal["cpu", "cuda"] | None = None
    batch_size: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_disp: int | None = Field(default=None, ge=1)


class BallAssistRunResult(BaseModel):
    """Result of a ball assist run."""

    clip: BallClipConfig
    meta: BallAssistMeta
    count: int


class BallAssistAll(BaseModel):
    """Container for all cached assist annotations."""

    annotations: dict[int, BallFrameAnnotation]


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

