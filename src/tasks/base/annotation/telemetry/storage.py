"""Enumerate persisted worker evidence; external Codex state is not a GC scope."""

from __future__ import annotations

import json

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import references
from ..storage.references import distinct
from ..workers.contracts import ExitReceipt, Launch
from .capture import iter_responses, verify_report
from .contracts import ResponsePage, ToolOutput


def retained_report(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ArtifactRef, ...]:
    report = verify_report(layout, reference)
    launch = read_document(layout, report.launch, Launch)
    execution = read_document(layout, report.execution, ExitReceipt)
    retained = [
        reference,
        *references(report),
        *references(launch),
        *references(execution),
    ]
    for page_ref in report.pages:
        retained.extend(references(read_document(layout, page_ref, ResponsePage)))
    for response in iter_responses(layout, reference):
        retained.extend(references(response))
    orphans = json.loads(verify_reference(layout, report.orphan_outputs).read_bytes())
    for value in orphans:
        retained.extend(references(ToolOutput.model_validate_json(json.dumps(value))))
    return distinct(layout, retained)
