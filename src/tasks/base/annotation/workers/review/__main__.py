"""Bounded requests only; the worker cannot directly mutate annotations or render media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

from ...artifacts import annotation_reference, read_document
from ...editing.contracts import EditReceipt
from ...handoff import read_handoff
from ...paths import AnnotationLayout
from ...telemetry.contracts import RenderedImage
from ..services import ServiceRequest, request_service
from .contracts import MAX_PARAMETERS_BYTES, MAX_ROWS_BYTES, OPERATIONS
from .service import resolve_attempt

PATH_BOUNDARY = NonHydraPathBoundary(
    name="base.annotation.review_worker.cli",
    fields=(
        BoundaryPathField(
            "request",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
        BoundaryPathField(
            "parameters",
            PathRole.OUTPUT,
            PathDirection.INPUT,
            PathKind.FILE,
            must_exist=True,
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--operation", required=True, choices=OPERATIONS)
    parser.add_argument("--parameters", required=True, type=Path)
    args = parser.parse_args()
    layout = AnnotationLayout.production()
    paths = PATH_BOUNDARY.validate(
        {"request": args.request, "parameters": args.parameters},
        resolver=layout.resolver,
    )
    ref = annotation_reference(layout, paths.declared("request").path)
    if ref.digest != args.request_sha256:
        raise ValueError("review request hash mismatch")
    attempt = resolve_attempt(layout, ref)
    path = paths.declared("parameters").path
    if (
        not path.is_relative_to(attempt.directory / "work")
        or path.stat().st_size > MAX_PARAMETERS_BYTES
    ):
        raise ValueError(
            "parameters must fit the byte limit and belong to attempt/work"
        )
    # Bound the actual read too, even if a worker rewrites the file after stat().
    with path.open("rb") as stream:
        raw = stream.read(MAX_PARAMETERS_BYTES + 1)
    if len(raw) > MAX_PARAMETERS_BYTES:
        raise ValueError("parameters exceed their byte limit")
    service = ServiceRequest.model_validate_json(
        json.dumps(
            {
                "task_request": ref.model_dump(mode="json"),
                "operation": args.operation,
                "parameters": json.loads(raw),
            }
        )
    )
    if len(service.canonical_bytes()) + 1 > MAX_PARAMETERS_BYTES:
        raise ValueError("service envelope exceeds its byte limit")
    result = request_service(layout, attempt.directory, service)
    if args.operation == "review_handoff":
        value = read_handoff(layout, result).model_dump(mode="json")
    elif args.operation == "review_edit":
        value = read_document(layout, result, EditReceipt).model_dump(mode="json")
    elif args.operation == "review_rows":
        if result.size_bytes > MAX_ROWS_BYTES:
            raise ValueError("row response exceeds its byte limit")
        value = json.loads(layout.artifact_path(result).read_bytes())
    else:
        image = read_document(layout, result, RenderedImage)
        value = {
            "image": image.model_dump(mode="json"),
            "evidence_id": image.sha256,
            "view_path": str(layout.artifact_path(image.image)),
            "requires_view_image": True,
        }
    print(
        json.dumps(
            {"receipt": result.model_dump(mode="json"), "result": value},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
