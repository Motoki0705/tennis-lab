"""Visual safety and metadata contracts; no browser or external generation."""

from __future__ import annotations

import copy
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.explain_html import schema
from tests.support.agents_ops.explain_html_visuals import COMPARISON, ERROR_GRAPH, png

visuals: Any = importlib.import_module("explain_html.visuals")
images: Any = importlib.import_module("explain_html.images")
pytestmark = pytest.mark.unit


PROVENANCE = schema.Provenance(kind="hypothetical", source="A labeled toy example.")
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 4"><path d="M0 0L3 2L9 0" stroke="black" fill="none"/></svg>'


def bundle(definition: dict[str, Any] | None = None) -> tuple[Any, dict[str, bytes]]:
    renderer = visuals.managed_renderer()
    figure = schema.Figure(
        figure_id="depth",
        content_id="geometry",
        kind="graph",
        asset_ids=[renderer.asset.asset_id],
        caption="Observe the depth change.",
        alt="Depth against observation error.",
        provenance=PROVENANCE,
        renderer=schema.Renderer(asset_id=renderer.asset.asset_id, version="1"),
        definition=definition or ERROR_GRAPH,
    )
    document = schema.Document(
        document_id="ray-geometry",
        purpose="Explain stereo geometry.",
        audience="Readers.",
        evidence=[],
        assets=[renderer.asset],
        figures=[figure],
        html='<html><head><title>Depth</title></head><body><section data-content-id="geometry"><figure data-figure-id="depth"></figure></section></body></html>',
    )
    return visuals.render_document(document), {
        renderer.asset.asset_id: renderer.content
    }


def test_raster_import_preserves_bytes_and_public_metadata(tmp_path: Path) -> None:
    path = tmp_path / "initial-imagegen.png"
    path.write_bytes(png())
    imported = images.import_image(
        path,
        asset_id="concept",
        caption="Two cameras.",
        alt="Two rays intersect.",
        provenance=schema.Provenance(
            kind="generated_concept", source="Codex imagegen, initial session."
        ),
    )
    assert imported.content == path.read_bytes()
    assert imported.asset.path == "assets/concept.png"
    assert imported.asset.provenance.kind == "generated_concept"
    assert imported.asset.media_type == "image/png"


def test_import_rejects_mime_spoofing_missing_source_and_invalid_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"<svg onload='alert(1)'></svg>")
    with pytest.raises(schema.ContractError, match="invalid_document"):
        images.import_image(
            path, asset_id="photo", caption="A.", alt="B.", provenance=PROVENANCE
        )
    corrupt = bytearray(png())
    corrupt[45] ^= 255  # Real PNG IDAT CRC failure; Pillow raises SyntaxError.
    path.write_bytes(corrupt)
    with pytest.raises(schema.ContractError, match="invalid_document"):
        images.import_image(
            path, asset_id="photo", caption="A.", alt="B.", provenance=PROVENANCE
        )
    with pytest.raises(ValueError):
        schema.Provenance(kind="measured", source=" ")
    with pytest.raises(ValueError):
        schema.Asset(
            asset_id="../escape",
            path="assets/a.png",
            media_type="image/png",
            sha256="0" * 64,
            kind="image",
            caption="A",
            alt="B",
            provenance=PROVENANCE,
        )


def test_svg_allows_geometry_and_refuses_executable_or_external_content() -> None:
    visuals.validate_svg(SVG)
    for attack in [
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 4"><script>alert(1)</script></svg>',
        SVG.replace(b"<path ", b'<path onload="alert(1)" '),
        SVG.replace(b"<path ", b'<image href="https://example.com/a.png" '),
        SVG.replace(b'fill="none"', b'fill="url(https://example.com/a)"'),
        b'<!DOCTYPE svg [<!ENTITY x "test">]>' + SVG,
        SVG.replace(b"<path ", b"<foreignObject "),
    ]:
        with pytest.raises(schema.ContractError, match="invalid_document"):
            visuals.validate_svg(attack)


def test_graph_definitions_reject_missing_units_singularities_and_arbitrary_expressions() -> (
    None
):
    import copy

    for path, value in [
        (("x", "unit"), ""),
        (("slider", "min"), -30),
        (("data", "focal_px"), "600"),
        (("data", "formula"), 'fetch("/secret")'),
        (("y", "domain"), [0, 2]),
    ]:
        bad = copy.deepcopy(ERROR_GRAPH)
        bad[path[0]][path[1]] = value
        with pytest.raises(schema.ContractError, match="invalid_document"):
            bundle(bad)
    comparison, payloads = bundle(COMPARISON)
    visuals.validate_visuals(comparison, payloads)
    assert "Camera B" in comparison.html and "11" in comparison.html


def test_rendered_figures_expose_data_units_provenance_and_pinned_renderer() -> None:
    document, payloads = bundle()
    visuals.validate_visuals(document, payloads)
    assert "24" in document.html and "16" in document.html and "12" in document.html
    assert "Observation error (px)" in document.html and "Depth (m)" in document.html
    assert (
        "A labeled toy example." in document.html
        and "Hypothetical example" in document.html
    )
    assert "script-src &#x27;sha256-" in document.html
    assert visuals.render_document(document) == document
    tampered = {key: value + b"alert(1)" for key, value in payloads.items()}
    with pytest.raises(schema.ContractError, match="invalid_document"):
        visuals.validate_visuals(document, tampered)
    forged_asset = document.assets[0].model_copy(
        update={"sha256": hashlib.sha256(next(iter(tampered.values()))).hexdigest()}
    )
    with pytest.raises(schema.ContractError, match="invalid_document"):
        visuals.validate_visuals(
            document.model_copy(update={"assets": [forged_asset]}), tampered
        )
    missing = document.model_copy(update={"assets": []})
    with pytest.raises(schema.ContractError):
        visuals.validate_visuals(missing, {})
    # Rounded display labels must not round the actual slider input boundaries.
    precise = copy.deepcopy(ERROR_GRAPH)
    bound = 0.123456789123
    precise["x"]["domain"] = [-bound, bound]
    precise["data"]["sample_errors_px"] = [-bound, 0, bound]
    precise["slider"].update(min=-bound, max=bound, step=bound)
    exact, _ = bundle(precise)
    attributes = re.search(
        r'<input[^>]+min="([^"]+)" max="([^"]+)" step="([^"]+)"', exact.html
    )
    assert attributes is not None
    assert [float(value) for value in attributes.groups()] == [-bound, bound, bound]


def test_generated_html_cannot_introduce_scripts_events_or_false_graph_labels() -> None:
    document, payloads = bundle()
    for attack in [
        document.html.replace("</body>", "<script>alert(1)</script></body>"),
        document.html.replace("<section ", '<section onclick="alert(1)" '),
        document.html.replace("</body>", '<iframe srcdoc="bad"></iframe></body>'),
        document.html.replace("Depth (m)", "Depth (px)"),
        document.html.replace(
            "</head>", '<style>@import "https://example.com/a";</style></head>'
        ),
    ]:
        with pytest.raises(schema.ContractError):
            visuals.validate_visuals(
                document.model_copy(update={"html": attack}), payloads
            )


@pytest.mark.parametrize("mode", ["success", "unavailable", "failure"])
def test_image_adapter_reports_real_outcome_without_inventing_an_asset(
    mode: str,
) -> None:
    class Fake:
        calls = 0

        def capability(self) -> Any:
            return images.ImageCapability(
                available=mode != "unavailable", reason="Test provider."
            )

        def generate(self, prompt: str) -> Any:
            self.calls += 1
            if mode == "failure":
                raise RuntimeError("PRIVATE provider context")
            return images.GeneratedImage(
                content=png(), source="Fake image provider for contract tests."
            )

    adapter = Fake()
    if mode == "success":
        result = images.generate_image(
            "Two cameras.",
            asset_id="concept",
            caption="Concept.",
            alt="Two rays.",
            adapter=adapter,
        )
        assert (
            result.content == png()
            and result.asset.provenance.kind == "generated_concept"
        )
    else:
        code = (
            "image_generation_unavailable"
            if mode == "unavailable"
            else "asset_generation_failed"
        )
        with pytest.raises(schema.ContractError, match=code) as caught:
            images.generate_image(
                "Two cameras.",
                asset_id="concept",
                caption="Concept.",
                alt="Two rays.",
                adapter=adapter,
            )
        assert "PRIVATE" not in caught.value.error.message
    assert adapter.calls == (mode != "unavailable")
