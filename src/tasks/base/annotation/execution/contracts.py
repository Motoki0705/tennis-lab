"""Frozen local commands; task operations remain outside the execution layer."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import ArtifactRef, Digest, Document
from ..resources.contracts import ResourceRequest


class QueueConfiguration(Document):
    script: str
    script_digest: Digest
    directory: str
    provider: Literal["codex"] = "codex"
    session: str = Field(min_length=1)
    issue: int = Field(gt=0)


class StageLaunch(Document):
    scope: ArtifactRef
    resource: ResourceRequest
    command: tuple[str, ...] = Field(min_length=1)
    code_root: str
    timeout_seconds: float = Field(gt=0)
    input: ArtifactRef | None = None
    dependencies: tuple[ArtifactRef, ...] = ()
    queue: QueueConfiguration | None = None
    test_root: str | None = None


class StageExit(Document):
    launch: ArtifactRef
    status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    exit_code: int
    started_at: str
    ended_at: str
    elapsed_seconds: float = Field(ge=0)
    stdout: ArtifactRef
    stderr: ArtifactRef
    reason: str | None
