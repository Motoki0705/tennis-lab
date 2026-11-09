"""Small, asymmetric document fixture shared by contract and real-store tests."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
schema: Any = importlib.import_module("explain_html.schema")
document_api: Any = importlib.import_module("explain_html.document")
store_api: Any = importlib.import_module("explain_html.store")
paths_api: Any = importlib.import_module("explain_html.paths")

HTML = """<!doctype html><html><head><title>Ray geometry</title></head><body>
<header id="overview" data-content-id="overview"><link rel="stylesheet" href="assets/theme.css">Two views locate a ball.</header>
<main class="grid"><section id="geometry" data-content-id="geometry"><h2>Geometry</h2>
<p>😀 Intersect two rays. Then measure 7 m.</p>
<figure data-figure-id="court-diagram"><img data-asset-id="court" src="assets/court.svg" alt="Two rays meet"><figcaption>Ray intersection.</figcaption></figure></section>
<aside data-content-id="limits"><details open><summary>Assumptions</summary>
<p data-evidence-id="baseline">Two synchronized cameras; baseline 7 m.</p>
<a href="#geometry">Return to the derivation</a></details>
<div data-figure-id="error-plot">At 0 px error, the example depth is 7 m.</div></aside>
</main></body></html>"""

PAYLOADS = {
    "court": b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 7 3"><path d="M0 0L3 2L7 0"/></svg>',
    "theme": b".grid{display:grid;grid-template-columns:2fr 1fr}",
    "plot_renderer": b'export const version = "1";\n',
}


def make_document() -> Any:
    provenance = {
        "kind": "hypothetical",
        "source": "Explicit toy geometry, not measured data.",
    }
    assets = []
    for asset_id, kind, extension, media_type in [
        ("court", "svg", "svg", "image/svg+xml"),
        ("theme", "stylesheet", "css", "text/css"),
        ("plot_renderer", "renderer", "mjs", "text/javascript"),
    ]:
        assets.append(
            {
                "asset_id": asset_id,
                "path": f"assets/{asset_id}.{extension}",
                "media_type": media_type,
                "sha256": hashlib.sha256(PAYLOADS[asset_id]).hexdigest(),
                "kind": kind,
                "caption": "Toy geometry explanation.",
                "alt": "Two camera rays and their intersection.",
                "provenance": provenance,
            }
        )
    return schema.parse(
        schema.Document,
        json.dumps(
            {
                "document_id": "ray-geometry",
                "purpose": "Explain triangulation and its assumptions.",
                "audience": "Readers who know image coordinates.",
                "html": HTML,
                "assets": assets,
                "evidence": [
                    {
                        "evidence_id": "baseline",
                        "statement": "The hypothetical baseline is 7 m.",
                        "provenance": provenance,
                        "values": {"baseline_m": 7},
                        "conditions": ["Synchronized views."],
                    },
                    {
                        "evidence_id": "cameras",
                        "statement": "Two cameras observe the same ball.",
                        "provenance": provenance,
                        "values": {"count": 2},
                        "conditions": ["Known calibration."],
                    },
                ],
                "figures": [
                    {
                        "figure_id": "court-diagram",
                        "content_id": "geometry",
                        "kind": "svg",
                        "asset_ids": ["court"],
                        "caption": "Intersect two calibrated rays.",
                        "alt": "Rays meet at a point.",
                        "provenance": provenance,
                    },
                    {
                        "figure_id": "error-plot",
                        "content_id": "limits",
                        "kind": "graph",
                        "asset_ids": ["plot_renderer"],
                        "caption": "Toy sensitivity to pixel error.",
                        "alt": "Depth varies with pixel error.",
                        "provenance": provenance,
                        "renderer": {"asset_id": "plot_renderer", "version": "1"},
                        "definition": {
                            "data": [[0, 7], [1, 9], [3, 18]],
                            "x_unit": "px",
                            "y_unit": "m",
                        },
                    },
                ],
            }
        ),
    )


def changed_document(document: Any) -> Any:
    body = document.model_dump(mode="json")
    body["html"] = body["html"].replace(
        "Then measure 7 m.", "Then measure 7 m. An example uses a 2 px error."
    )
    return schema.parse(schema.Document, json.dumps(body))


def candidate_for(
    document: Any,
    *,
    before: Any = None,
    request_id: str = "initial",
    changes: list[dict[str, str]] | None = None,
    evidence_changes: list[dict[str, str]] | None = None,
    target: dict[str, Any] | None = None,
) -> Any:
    return schema.parse(
        schema.Candidate,
        json.dumps(
            {
                "request": {
                    "document_id": "ray-geometry",
                    "base_revision": before.revision_id if before else None,
                    "request_id": request_id,
                    "target": target or {"kind": "document"},
                    "instruction": "Add a numerical example; retain facts and references.",
                },
                "document": document.model_dump(mode="json"),
                "changes": changes
                if changes is not None
                else [
                    {"content_id": "geometry", "kind": "added"},
                    {"content_id": "limits", "kind": "added"},
                    {"content_id": "overview", "kind": "added"},
                ],
                "evidence_changes": evidence_changes
                if evidence_changes is not None
                else [
                    {
                        "evidence_id": "baseline",
                        "kind": "added",
                        "reason": "Initial evidence.",
                    },
                    {
                        "evidence_id": "cameras",
                        "kind": "added",
                        "reason": "Initial evidence.",
                    },
                ],
                "change_summary": "Explain the two-view geometry.",
            }
        ),
    )


def validation_for(candidate: Any, **overrides: Any) -> Any:
    data = {
        "candidate_digest": schema.candidate_digest(candidate),
        "structural": {"status": "passed"},
        "semantic": {"status": "passed"},
        "rendering": {"status": "passed"},
        **overrides,
    }
    return schema.parse(schema.ValidationResult, json.dumps(data))


def revision_for(document: Any) -> Any:
    return schema.Revision(
        revision_id="1" * 32,
        parent_revision=None,
        restored_from=None,
        request_id="initial",
        change_summary="Initial explanation.",
        changes=[],
        evidence_changes=[],
        document=document,
    )
