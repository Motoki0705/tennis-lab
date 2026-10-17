"""Workers request bounded parent-owned operations through their own work inbox."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from ..artifacts import annotation_reference, read_document, verify_reference
from ..contracts import ArtifactRef, Digest, Document, Identifier
from ..paths import AnnotationLayout


class ServiceRequest(Document):
    task_request: ArtifactRef
    operation: Identifier
    parameters: dict[str, Any]


class ServiceReply(Document):
    request_digest: Digest
    status: Literal["succeeded", "failed"]
    result: ArtifactRef | None
    reason: str | None


def request_service(
    layout: AnnotationLayout,
    directory: Path,
    request: ServiceRequest,
    *,
    timeout_seconds: float = 600,
) -> ArtifactRef:
    # Only the worker's own inbox is writable. Replies and render outputs are parent-owned.
    inbox = directory / "work" / "service-requests"
    layout.artifact_path(
        ArtifactRef(
            role="annotation",
            path=inbox.relative_to(layout.root).as_posix(),
            digest="0" * 64,
            size_bytes=0,
        )
    )
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / (request.sha256 + ".json")
    payload = request.canonical_bytes() + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                "service request identity already contains different input"
            )
    else:
        with tempfile.NamedTemporaryFile(dir=inbox, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Same content-addressed requests may race, with identical payloads only.
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    reply_path = directory / "services" / request.sha256 / "reply.json"
    deadline = time.monotonic() + timeout_seconds
    while not reply_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "parent service has not replied; this request was not retried or replaced"
            )
        time.sleep(0.2)
    reply = read_document(
        layout, annotation_reference(layout, reply_path), ServiceReply
    )
    if reply.request_digest != request.sha256:
        raise ValueError("service response belongs to another request")
    if reply.status != "succeeded" or reply.result is None:
        raise RuntimeError(f"parent service failed: {reply.reason}")
    verify_reference(layout, reply.result)
    result: ArtifactRef = reply.result
    return result
