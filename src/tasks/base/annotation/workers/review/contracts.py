"""Only parent-owned references and bounded, explicit worker requests cross here."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...contracts import ArtifactRef, Document, FrameRange
from ...images.contracts import ImageTile as ImageTile
from ...images.contracts import PageView
from ..contracts import WorkRequest

MAX_EDIT_BYTES = 1_048_576
MAX_PARAMETERS_BYTES = 32_768
MAX_ROWS_BYTES = 65_536
OPERATIONS = ("review_handoff", "review_rows", "review_image", "review_edit")


class ReviewWorkerInput(Document):
    work: WorkRequest
    attempt_path: str
    initial_handoff: ArtifactRef


class HandoffRequest(Document):
    scope: FrameRange
    previous_revision: ArtifactRef | None = None
    after_row: str | None = Field(default=None, max_length=128)
    expected_revision: ArtifactRef | None = None


class RowsRequest(Document):
    expected_revision: ArtifactRef
    scope: FrameRange


class ImageRequest(PageView):
    expected_revision: ArtifactRef
    drawing: Literal["plain", "prediction", "revision"]


class EditRequest(Document):
    expected_revision: ArtifactRef
    changes: ArtifactRef
    reason: str = Field(min_length=1, max_length=500)


class ReviewWorkerResult(Document):
    status: Literal["complete", "deferred"]
    revision: ArtifactRef
    reason: str = Field(min_length=1, max_length=500)
