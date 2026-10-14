"""Small CPU-only tools available to an isolated screening worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..artifacts import annotation_reference, read_document
from ..paths import AnnotationLayout
from .contracts import PreviewReceipt, ScreeningPolicy, ScreeningRequest, SourceDecision
from .service import attempt_directory, check_decision, request_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images")
    images.add_argument("--start", required=True, type=float)
    images.add_argument("--stop", required=True, type=float)
    images.add_argument("--step", required=True, type=float)
    check = commands.add_parser("check")
    check.add_argument("--decision", required=True, type=Path)
    args = parser.parse_args()
    layout = AnnotationLayout.production()
    ref = annotation_reference(layout, args.request)
    if ref.digest != args.request_sha256:
        raise ValueError("screening request hash mismatch")
    request = read_document(layout, ref, ScreeningRequest)
    if args.command == "images":
        preview_ref = request_preview(
            layout, request, start=args.start, stop=args.stop, step=args.step
        )
        preview = read_document(layout, preview_ref, PreviewReceipt)
        print(
            json.dumps(
                {
                    "receipt": str(layout.artifact_path(preview_ref)),
                    "pages": [
                        str(layout.artifact_path(page.image)) for page in preview.pages
                    ],
                    "images_are_generated_not_yet_presented": True,
                }
            )
        )
    else:
        path = args.decision.absolute()
        if not path.is_relative_to(attempt_directory(layout, request) / "work"):
            raise ValueError("worker decisions must stay in attempt/work")
        decision = read_document(
            layout, annotation_reference(layout, path), SourceDecision
        )
        policy = read_document(layout, request.policy, ScreeningPolicy)
        check_decision(request, policy, decision)
        print(
            json.dumps(
                {
                    "selected": len(decision.clips),
                    "status": decision.status,
                    "structural_check_only": True,
                }
            )
        )


if __name__ == "__main__":
    main()
