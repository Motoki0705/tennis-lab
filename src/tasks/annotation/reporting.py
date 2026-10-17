"""Persist a detailed workflow observation beneath its fixed campaign root."""

from __future__ import annotations

from typing import Any, Literal

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import ArtifactRef, Document

from .contracts import Runtime
from .controller import Controller


class WorkflowReport(Document):
    schema_version: Literal["annotation.workflow-report.v1"] = (
        "annotation.workflow-report.v1"
    )
    runtime: ArtifactRef
    progress: dict[str, Any]


def report(controller: Controller, runtime: ArtifactRef) -> ArtifactRef:
    if read_document(controller.layout, runtime, Runtime) != controller.runtime:
        raise ValueError("report runtime differs from the controller")
    document = WorkflowReport(runtime=runtime, progress=controller.status(details=True))
    return write_document(
        controller.layout,
        controller.layout.path(
            "campaigns",
            controller.plan.campaign_id,
            "reports",
            "workflow",
            document.sha256 + ".json",
        ),
        document,
    )
