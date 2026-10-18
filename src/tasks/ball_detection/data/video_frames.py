"""Explicit metadata for native video-backed frame payloads in ball store v3."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

VIDEO_FRAME_SCHEMA = "ball_detection_frames.v3"
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class VideoFrameStorage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    media_sha256: Digest
    media_bytes: int = Field(gt=0)
    jpeg_quality: int = Field(ge=1, le=100)
    runtime: Digest
    frame_sha256: tuple[Digest, ...] = Field(min_length=1)


def video_name(clip_index: int) -> str:
    return f"clip-{clip_index:05d}.mp4"
