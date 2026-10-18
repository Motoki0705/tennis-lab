"""Bind observed review CLI receipts to the owning attempt, without label logic."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ...artifacts import read_document
from ...contracts import ArtifactRef
from ...editing.contracts import EditCommand, EditReceipt, Revision
from ...handoff import read_handoff
from ...paths import AnnotationLayout
from ...plan import references
from ...registry import Registry
from ...storage.references import distinct
from ...telemetry.contracts import ToolEvidence
from ..contracts import Launch


def _envelopes(text: str, depth: int = 0) -> Iterator[dict[str, Any]]:
    """Accept CLI JSON lines, including an exec tool's structured stdout wrapper."""
    for line in text.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if (
            value.get("operation") in {"review_edit", "review_handoff"}
            and "receipt" in value
            and "result" in value
        ):
            yield value
        elif depth < 2:
            for key in ("output", "stdout"):
                if isinstance(value.get(key), str):
                    yield from _envelopes(value[key], depth + 1)


def review_tool_evidence(
    layout: AnnotationLayout, launch_ref: ArtifactRef, category: str, text: str
) -> ToolEvidence | None:
    envelopes = list(_envelopes(text))
    if not envelopes:
        return (
            ToolEvidence(
                unavailable_reason="no complete review receipt in observed tool output; failed or truncated output is not an edit or handoff success"
            )
            if category in {"review_edit", "review_handoff"}
            else None
        )
    launch = read_document(layout, launch_ref, Launch)
    artifacts: list[ArtifactRef] = []
    dependencies: list[ArtifactRef] = []
    for envelope in envelopes:
        ref = ArtifactRef.model_validate(envelope["receipt"])
        if envelope["operation"] == "review_edit":
            receipt = read_document(layout, ref, EditReceipt)
            if receipt.model_dump(mode="json") != envelope["result"]:
                raise ValueError("observed edit summary differs from its receipt")
            command = read_document(layout, receipt.command, EditCommand)
            previous = read_document(layout, receipt.previous, Revision)
            revision = read_document(layout, receipt.revision, Revision)
            if (
                command.actor != launch_ref
                or command.context != launch.request.payload
                or command.expected_revision != receipt.previous
                or previous.context != command.context
                or revision.context != command.context
                or revision.previous != receipt.previous
                or revision.command != receipt.command
                or revision.sequence != previous.sequence + 1
            ):
                raise ValueError("edit evidence belongs to another attempt or revision")
            with Registry(layout).transaction() as db:
                row = db.execute(
                    "SELECT receipt FROM annotation_edits WHERE id=?", (command.sha256,)
                ).fetchone()
            if row is None or EditReceipt.model_validate_json(row[0]) != receipt:
                raise ValueError("edit evidence has no committed parent receipt")
            dependencies.extend(
                ref
                for document in (receipt, command, previous, revision)
                for ref in references(document)
            )
        else:
            handoff = read_handoff(layout, ref)
            if (
                handoff.context != launch.request.payload
                or handoff.model_dump(mode="json") != envelope["result"]
            ):
                raise ValueError("handoff evidence belongs to another review context")
            dependencies.extend(references(handoff))
            # A revision reference alone does not keep its annotation payload alive.
            for revision_ref in (handoff.revision, handoff.previous_revision):
                if revision_ref is not None:
                    dependencies.append(
                        read_document(layout, revision_ref, Revision).payload
                    )
            for command_ref in handoff.commands:
                dependencies.extend(
                    references(read_document(layout, command_ref, EditCommand))
                )
        artifacts.append(ref)
    return ToolEvidence(
        artifacts=distinct(layout, artifacts),
        dependencies=distinct(layout, dependencies),
    )
