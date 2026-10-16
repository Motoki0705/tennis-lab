"""The common runner treats adapter inputs, labels, and receipts as opaque artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    after_validator,
    content_digest,
    relative_path,
)
from ..paths import AnnotationLayout
from ..plan import CampaignPlan
from ..reproducibility import CodeBoundDocument
from ..resources.contracts import ResourceRequest
from ..telemetry.contracts import RenderedImage

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class WorkRequest(Document):
    adapter: Identifier
    adapter_version: Identifier
    stage: Identifier
    subject_id: Digest
    payload: ArtifactRef
    inputs: tuple[ArtifactRef, ...]

    @property
    def work_id(self) -> str:
        # A new campaign/version never makes a second annotation of the same subject.
        result: str = content_digest([self.adapter, self.subject_id])
        return result


class PreparedWorker(Document):
    prompt: ArtifactRef
    inputs: tuple[ArtifactRef, ...] = ()
    result_path: str = "work/result.json"
    writable_directories: tuple[str, ...] = ("work",)

    @_after
    def inside_attempt(self) -> Self:
        relative_path(self.result_path)
        for directory in self.writable_directories:
            relative_path(directory)
        if "work" not in self.writable_directories:
            raise ValueError("worker work directory must be writable")
        result = Path(self.result_path)
        if not result.is_relative_to("work"):
            raise ValueError("worker result must be in attempt/work")
        if any(
            Path(part).parts[0]
            in {"owner.json", "exit.json", "launch.json", "events.jsonl", "stderr.log"}
            for part in self.writable_directories
        ):
            raise ValueError("workers cannot write supervisor receipts")
        return self


class WorkOutcome(Document):
    status: Literal["complete", "deferred", "not_applicable"]
    receipt: ArtifactRef
    adoption: ArtifactRef | None = None
    reason: str | None = None

    @_after
    def explicit_reason(self) -> Self:
        if self.status != "complete" and not (self.reason or "").strip():
            raise ValueError("deferred/excluded work requires a reason")
        if self.status != "complete" and self.adoption is not None:
            raise ValueError("only complete work can adopt teachers")
        return self


class Launch(CodeBoundDocument):
    schema_version: Literal["annotation.worker-launch.v1"] = (
        "annotation.worker-launch.v1"
    )
    work_id: Digest
    attempt: Identifier
    plan: ArtifactRef
    request: WorkRequest
    prepared: PreparedWorker
    directory: str
    code_root: str
    codex_binary: str
    codex_home: str
    timeout_seconds: float = Field(gt=0)
    test_command: tuple[str, ...] | None = None
    test_root: str | None = None
    resource_scope: ArtifactRef | None = None
    resource_request: ResourceRequest | None = None

    @_after
    def paired_resources(self) -> Self:
        if (self.resource_scope is None) != (self.resource_request is None):
            raise ValueError("resource scope and request must be supplied together")
        return self


class ExitReceipt(Document):
    launch_digest: Digest
    exit_code: int
    status: Literal["succeeded", "failed", "quota", "timed_out", "cancelled"]
    started_at: str
    ended_at: str
    elapsed_seconds: float = Field(ge=0)
    events: ArtifactRef
    errors: ArtifactRef
    thread_id: str | None
    reason: str | None


class ChildIdentity(Document):
    pid: int = Field(gt=0)
    start_ticks: int = Field(gt=0)
    group: int = Field(gt=0)
    session: int = Field(gt=0)
    boot_id: str = Field(min_length=1)


class ChildReceipt(Document):
    """The actual CLI child, separate from its supervisor and reserved capacity."""

    launch_digest: Digest
    captured_at: str
    process: ChildIdentity | None
    unavailable_reason: str | None

    @_after
    def explicit_missing_identity(self) -> Self:
        if (self.process is None) != bool(self.unavailable_reason):
            raise ValueError(
                "an unavailable child identity requires an explicit reason"
            )
        return self


class AttemptResult(Document):
    launch: ArtifactRef
    exit: ArtifactRef
    result: ArtifactRef
    outcome: WorkOutcome
    telemetry: ArtifactRef | None


class AttemptEvidence(Document):
    """Verified on-disk evidence survives retries and mutable task-state changes."""

    number: int = Field(ge=1)
    state: str
    reason: str | None
    launch: ArtifactRef
    execution: ArtifactRef | None
    result: ArtifactRef | None
    telemetry: ArtifactRef | None


class TaskAdapter(Protocol):
    name: str
    version: str

    def input_references(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
    ) -> tuple[ArtifactRef, ...]:
        """Enumerate dependencies of the opaque payload, without changing it."""
        ...

    def prepare(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> PreparedWorker: ...

    def telemetry_images(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
    ) -> tuple[RenderedImage, ...]: ...

    def classify_tool(self, name: str | None, raw_input: str) -> str: ...

    def validate(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
        result: ArtifactRef,
        execution: ExitReceipt,
        telemetry: ArtifactRef | None,
    ) -> WorkOutcome: ...
