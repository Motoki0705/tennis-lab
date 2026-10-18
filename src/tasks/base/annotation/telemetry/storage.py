"""Enumerate persisted worker evidence; external Codex state is not a GC scope."""

from __future__ import annotations

import json

from ..artifacts import read_document, verify_reference
from ..contracts import ArtifactRef
from ..images import read_rendered
from ..paths import AnnotationLayout
from ..plan import references
from ..storage.references import distinct
from ..workers.contracts import ExitReceipt, Launch
from .capture import iter_responses, verify_report
from .contracts import ResponsePage, ToolOutput


def _retained_output(layout: AnnotationLayout, output: ToolOutput) -> list[ArtifactRef]:
    retained = list(references(output))
    for image in output.images:
        if image.source is not None:
            request, receipt = read_rendered(layout, image.source)
            retained.extend((*references(receipt), *references(request)))
    return retained


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
        for output in response.tool_outputs:
            retained.extend(_retained_output(layout, output))
    orphans = json.loads(verify_reference(layout, report.orphan_outputs).read_bytes())
    for value in orphans:
        output = ToolOutput.model_validate_json(json.dumps(value))
        retained.extend(_retained_output(layout, output))
    result: tuple[ArtifactRef, ...] = distinct(layout, retained)
    return result
