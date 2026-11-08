"""Tool contracts contain no exploration order or prediction-review decisions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    Identifier,
)
from src.tasks.base.annotation.images.contracts import Crop
from src.tasks.base.annotation.workers.contracts import WorkRequest

from ..contracts import VideoBallFrame


class AnnotationPolicy(Document):
    policy_id: Identifier
    version: Identifier
    instructions: str = Field(min_length=1)


DEFAULT_POLICY = AnnotationPolicy(
    policy_id="adaptive",
    version="1",
    instructions="Choose your own exploration: temporal overviews, selected native crops, sparse anchors and interpolation, or adaptive revisits. Revise your judgments when context improves them.",
)


class AnnotationRun(Document):
    schema_version: Literal["ball.blank-run.v1"] = "ball.blank-run.v1"
    context: ArtifactRef
    policy: AnnotationPolicy
    # A serialized, pinned ExecutionPlan, not environment-dependent defaults.
    plan: ArtifactRef
    runtime: ArtifactRef
    timeout_seconds: float = Field(gt=0)
    worker_version: Literal["3"] = "3"
    tools_version: Literal["camera-tools.v1"] = "camera-tools.v1"


class AgentTask(Document):
    work: WorkRequest
    run: ArtifactRef
    initial_revision: ArtifactRef


class FrameQuery(Document):
    camera_id: Identifier
    frames: tuple[int, ...] = Field(min_length=1, max_length=128)


class ImageTile(Document):
    frame_index: int = Field(ge=0)
    crop: Crop


class ImageQuery(Document):
    camera_id: Identifier
    tiles: tuple[ImageTile, ...] = Field(min_length=1, max_length=16)
    columns: int = Field(ge=1, le=4)
    tile_width: int = Field(ge=32, le=1024)
    tile_height: int = Field(ge=32, le=1024)
    reason: str = Field(min_length=1, max_length=500)


class EditQuery(Document):
    camera_id: Identifier
    expected_revision: ArtifactRef
    rows: tuple[VideoBallFrame, ...] = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=500)


class HandoffQuery(Document):
    camera_id: Identifier
    after_frame: int | None = Field(default=None, ge=0)
    limit: int = Field(default=128, ge=1, le=128)


class AgentDecision(Document):
    status: Literal["complete", "deferred"]
    revision: ArtifactRef
    reason: str = Field(min_length=1, max_length=500)


class AgentView(Document):
    camera_id: Identifier
    revision: ArtifactRef
    summary: dict[str, int | bool]
    rows: tuple[VideoBallFrame, ...] = ()
    remaining: tuple[int, ...] = ()
    next_after_frame: int | None = None
    image_path: str | None = None
    image: ArtifactRef | None = None
    metadata: ArtifactRef | None = None


class VideoCompletion(Document):
    schema_version: Literal["ball.video-completion.v1"] = "ball.video-completion.v1"
    run: ArtifactRef
    context: ArtifactRef
    revision: ArtifactRef
    annotation: ArtifactRef
    actor: ArtifactRef
    reason: str = Field(min_length=1, max_length=500)


class ServiceEvent(Document):
    task: ArtifactRef
    request: ArtifactRef
    operation: Identifier
    result: ArtifactRef
    # A monotonically assigned position in this attempt, not file enumeration order.
    sequence: int = Field(ge=1)
