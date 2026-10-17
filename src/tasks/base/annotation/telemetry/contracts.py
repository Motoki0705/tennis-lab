"""Operational evidence is independent of the task's interpretation of a label."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import ArtifactRef, Digest, Document, PositiveInt
from ..contracts import after_validator as _after_validator

_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


after_validator = cast(_AfterValidator, _after_validator)


class Usage(Document):
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class RenderedImage(Document):
    image: ArtifactRef
    # Frame/PTS, crop, scale, tiles, drawing, annotation revision: renderer-owned.
    metadata: ArtifactRef


class ImageInput(Document):
    image: ArtifactRef
    width: PositiveInt
    height: PositiveInt
    detail: str | None
    rollout_line: PositiveInt
    block_index: int = Field(ge=0)
    source: RenderedImage | None
    binding: Literal["exact_bytes", "exact_bytes_and_view_path", "unbound"]


class ToolEvidence(Document):
    artifacts: tuple[ArtifactRef, ...] = ()
    dependencies: tuple[ArtifactRef, ...] = ()
    unavailable_reason: str | None = None

    @after_validator
    def explicit_outcome(self) -> Self:
        if self.artifacts:
            if self.unavailable_reason is not None:
                raise ValueError("linked tool evidence cannot be unavailable")
        elif (
            self.dependencies
            or not self.unavailable_reason
            or not self.unavailable_reason.strip()
        ):
            raise ValueError("unlinked tool evidence requires an explicit reason")
        return self


class ToolOutput(Document):
    call_id: str
    name: str | None
    input_digest: Digest | None
    input_chars: int = Field(ge=0)
    input_timestamp: str | None
    output_digest: Digest
    text_chars: int = Field(ge=0)
    output_line: PositiveInt
    timestamp: str | None
    images: tuple[ImageInput, ...]
    observed_view_paths: tuple[str, ...]
    category: str
    evidence: ToolEvidence | None = None


class ResponseRecord(Document):
    schema_version: Literal["annotation.response-evidence.v1"] = (
        "annotation.response-evidence.v1"
    )
    index: PositiveInt
    response_id: str
    thread_id: str | None
    session_id: str | None
    turn_id: str | None
    root_turn_id: str | None
    rollout_line: PositiveInt
    timestamp: str | None
    usage: Usage | None
    logged_thread_usage: Usage | None
    logged_turn_usage: Usage | None
    context_window: PositiveInt | None
    context_proxy_tokens: int | None = Field(ge=0)
    tool_outputs: tuple[ToolOutput, ...]


class ResponseEntry(Document):
    index: PositiveInt
    response_id: str
    record: ArtifactRef
    page: ArtifactRef
    images: int = Field(ge=0)
    text_chars: int = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)


class ResponsePage(Document):
    entries: tuple[ResponseEntry, ...] = Field(min_length=1, max_length=100)


class TelemetryReport(Document):
    schema_version: Literal["annotation.telemetry.v1"] = "annotation.telemetry.v1"
    launch: ArtifactRef
    execution: ArtifactRef
    thread_id: str
    rollout_path: str
    rollout_digest: Digest
    rollout_bytes: int = Field(ge=0)
    cli_version: str | None
    model_provider: str | None
    model_contexts: tuple[str, ...]
    responses: int = Field(ge=0)
    duplicate_usage_records: int = Field(ge=0)
    conflicting_usage_records: int = Field(ge=0)
    missing_response_ids: int = Field(ge=0)
    missing_usage: int = Field(ge=0)
    token_notifications: int = Field(ge=0)
    compactions: int = Field(ge=0)
    orphan_outputs: ArtifactRef
    orphan_output_count: int = Field(ge=0)
    pages: tuple[ArtifactRef, ...]
    index_page: ArtifactRef
    # Null means incomplete totals, not zero. Known subtotals remain available.
    totals: dict[str, int | None]
    known_subtotals: dict[str, int]
    missing_metric_counts: dict[str, int]
    image_inputs: int = Field(ge=0)
    unbound_image_inputs: int = Field(ge=0)
    catalog_digest: Digest
    classifier_id: str
    prompt_chars: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
