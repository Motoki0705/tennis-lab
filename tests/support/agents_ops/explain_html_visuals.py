"""Real visual assets layered onto the shared revision-store document fixture."""

from __future__ import annotations

import hashlib
import importlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

from tests.support.agents_ops.explain_html import (
    PAYLOADS,
    candidate_for,
    document_api,
    make_document,
    schema,
    store_api,
    validation_for,
)

visuals: Any = importlib.import_module("explain_html.visuals")
images: Any = importlib.import_module("explain_html.images")


def rejected_html(document: Any) -> dict[str, str]:
    """Literal regression inputs, independent of parser/renderer internals."""
    forms = {
        "abrupt-comment": "<!--><script>globalThis.PWNED=1</script>-->",
        "abrupt-dash-comment": "<!---><script>globalThis.PWNED=1</script>-->",
        "bang-comment-close": "<!-- x --!><script>globalThis.PWNED=1</script>-->",
        "spaced-comment-close": "<!-- x -- >",
        "bogus-declaration": "<!bogus>",
        "cdata-declaration": "<![CDATA[not HTML]]>",
        "invalid-doctype": "<!DOCTYPE html [bad]>",
    }
    result = {
        name: document.html.replace("<head><meta", text + "<head><meta", 1)
        for name, text in forms.items()
    }
    result["unclosed-comment"] = document.html + "<!-- unfinished"
    result["unclosed-declaration"] = document.html + "<!"
    for attribute, value in [
        ("data-managed-graph", "1"),
        ("data-definition", "{}"),
        ("data-plot", "1"),
        ("data-values", "1"),
        ("data-managed-slider", "1"),
        ("data-slider-value", "1"),
        ("data-graph-targets", "[]"),
    ]:
        result[attribute] = document.html.replace(
            "</header>", f"<div {attribute}='{value}'></div></header>", 1
        )
    return result


ERROR_GRAPH: dict[str, Any] = {
    "type": "observation_error",
    "x": {"label": "Observation error", "unit": "px", "domain": [-10, 10]},
    "y": {"label": "Depth", "unit": "m", "domain": [0, 26]},
    "data": {
        "focal_px": 600,
        "baseline_m": 0.8,
        "disparity_px": 30,
        "sample_errors_px": [-10, 0, 10],
    },
    "slider": {
        "label": "Observation error",
        "unit": "px",
        "min": -10,
        "max": 10,
        "step": 1,
        "initial": 0,
    },
}
COMPARISON: dict[str, Any] = {
    "type": "comparison",
    "x": {"label": "Frame", "unit": "frame", "domain": [0, 9]},
    "y": {"label": "Position", "unit": "m", "domain": [0, 12]},
    "series": [
        {"label": "Camera A", "points": [{"x": 1, "y": 7}, {"x": 8, "y": 3}]},
        {"label": "Camera B", "points": [{"x": 2, "y": 4}, {"x": 7, "y": 11}]},
    ],
}


def png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), (25, 80, 160)).save(stream, format="PNG")
    return stream.getvalue()


def make_visual_bundle(renderer_version: str = "2") -> tuple[Any, dict[str, bytes]]:
    base = make_document()
    provenance = base.figures[0].provenance
    renderer = visuals.managed_renderer(renderer_version)
    theme_content = b"body{font-family:system-ui;margin:1rem;color:#182533}figure{margin:1rem 0}td,th{padding:.25rem;text-align:left}table{border-collapse:collapse}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:2rem}@media(max-width:700px){.grid{grid-template-columns:minmax(0,1fr)}}"
    theme = base.assets[1].model_copy(
        update={"sha256": hashlib.sha256(theme_content).hexdigest()}
    )
    image = images.image_from_bytes(
        png(),
        asset_id="concept",
        caption="Raster import fixture.",
        alt="A small blue rectangle.",
        provenance=provenance,
    )
    html = base.html.replace(
        '<div data-figure-id="error-plot">At 0 px error, the example depth is 7 m.</div>',
        '<figure data-figure-id="error-plot"></figure><figure data-figure-id="comparison"></figure><figure data-figure-id="concept-image"></figure>',
    )
    html = html.replace(
        "</head>",
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head>',
    )
    html = html.replace(
        "<head>",
        "<!-- harmless literal <head> in a comment -->\n<head><style>p{line-height:1.4}</style>",
        1,
    )
    figures = [
        base.figures[0],
        schema.Figure(
            figure_id="error-plot",
            content_id="limits",
            kind="graph",
            asset_ids=[renderer.asset.asset_id],
            caption="Signed disparity error changes estimated depth; read the units and source table.",
            alt="Depth decreases from 24 to 12 m as disparity error increases.",
            provenance=provenance,
            renderer=schema.Renderer(
                asset_id=renderer.asset.asset_id, version=renderer_version
            ),
            definition=ERROR_GRAPH,
        ),
        schema.Figure(
            figure_id="comparison",
            content_id="limits",
            kind="graph",
            asset_ids=[renderer.asset.asset_id],
            caption="Compare two asymmetric hypothetical tracks.",
            alt="Two series with different frame and position coordinates.",
            provenance=provenance,
            renderer=schema.Renderer(
                asset_id=renderer.asset.asset_id, version=renderer_version
            ),
            definition=COMPARISON,
        ),
        schema.Figure(
            figure_id="concept-image",
            content_id="limits",
            kind="image",
            asset_ids=[image.asset.asset_id],
            caption=image.asset.caption,
            alt=image.asset.alt,
            provenance=provenance,
        ),
    ]
    document = base.model_copy(
        update={
            "html": html,
            "assets": [base.assets[0], theme, renderer.asset, image.asset],
            "figures": figures,
        }
    )
    payloads = {
        "court": PAYLOADS["court"],
        "theme": theme_content,
        renderer.asset.asset_id: renderer.content,
        image.asset.asset_id: image.content,
    }
    document = visuals.render_document(document)
    visuals.validate_visuals(document, payloads)
    return document, payloads


def make_renderer_history(root: Path) -> tuple[Any, Any, Any, Any]:
    """Real store with an old v1 revision and a newer v2; verdicts are test fakes."""
    store = store_api.RevisionStore(root)
    old, old_payloads = make_visual_bundle("1")
    initial = candidate_for(old)
    first = store.import_initial(initial, old_payloads, validation_for(initial))
    current, payloads = make_visual_bundle("2")
    edit = candidate_for(
        current,
        before=first,
        request_id="upgrade-renderer",
        changes=[
            change.model_dump(mode="json")
            for change in document_api.content_changes(old, current)
        ],
        evidence_changes=[],
    )
    second = store.commit(store.stage(edit, payloads), validation_for(edit))
    snapshot = store.snapshot(old.document_id, first.revision_id)
    return store, first, second, snapshot
