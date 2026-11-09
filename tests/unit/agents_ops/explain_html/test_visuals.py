"""Visual safety and metadata contracts; no browser or external generation."""

from __future__ import annotations

import copy
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.explain_html import document_api, schema
from tests.support.agents_ops.explain_html_visuals import (
    COMPARISON,
    ERROR_GRAPH,
    png,
)

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
        renderer=schema.Renderer(
            asset_id=renderer.asset.asset_id, version=visuals.RENDERER_VERSION
        ),
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
    # HTML only recognizes ASCII whitespace here. NBSP creates an implicit body
    # in Chromium, so an apparent subsequent head must never receive our CSP.
    for prefix in ["\u00a0", "&#160;"]:
        bad = document.model_copy(
            update={"html": document.html.replace("<head>", prefix + "<head>", 1)}
        )
        with pytest.raises(schema.ContractError, match="invalid_document"):
            visuals.validate_visuals(bad, payloads)
    for markup in [
        "<!--><script>globalThis.PWNED=1</script>-->",
        "<!--->",
        "<!-- x --!>",
        "<!-- unclosed",
        "<!bogus>",
        "<![CDATA[x]]>",
        "<!",
    ]:
        bad = document.model_copy(update={"html": document.html + markup})
        with pytest.raises(schema.ContractError, match="invalid_document"):
            document_api.inspect_document(bad)
        with pytest.raises(schema.ContractError):
            visuals.validate_visuals(bad, payloads)
    # The header-less unit fixture still exercises every reserved attribute.
    for attribute in [
        "data-managed-graph",
        "data-definition",
        "data-plot",
        "data-values",
        "data-managed-slider",
        "data-slider-value",
        "data-graph-targets",
        "data-render-error",
        "data-managed-extra",
    ]:
        bad = document.model_copy(
            update={
                "html": document.html.replace(
                    "</body>", f'<div {attribute}="1"></div></body>'
                )
            }
        )
        with pytest.raises(schema.ContractError):
            visuals.validate_visuals(bad, payloads)
    for function in ["image-set", "-webkit-image-set", "image", "src"]:
        css = f'background-image:{function}("assets/undeclared.png" 1x)'
        for html in [
            document.html.replace("<section ", f"<section style='{css}' "),
            document.html.replace("</head>", f"<style>p{{{css}}}</style></head>"),
        ]:
            with pytest.raises(schema.ContractError):
                visuals.validate_visuals(
                    document.model_copy(update={"html": html}), payloads
                )
        content = f"p{{{css}}}".encode()
        asset = document.assets[0].model_copy(
            update={
                "asset_id": "untrusted_css",
                "path": "assets/untrusted_css.css",
                "kind": "stylesheet",
                "media_type": "text/css",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        with pytest.raises(schema.ContractError, match="invalid_document"):
            visuals.validate_visuals(
                document.model_copy(update={"assets": [*document.assets, asset]}),
                {**payloads, "untrusted_css": content},
            )
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


def test_css_properties_distinguish_scroll_behavior_from_executable_properties() -> (
    None
):
    document, payloads = bundle()
    for css in [
        "html{scroll-behavior:smooth}",
        "@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}",
    ]:
        compiled = visuals.render_document(
            document.model_copy(
                update={
                    "html": document.html.replace(
                        "</head>", f"<style>{css}</style></head>"
                    )
                }
            )
        )
        visuals.validate_visuals(compiled, payloads)
    for css in [
        "p{behavior:evil}",
        "p{BEHAVIOR : evil}",
        "p{-moz-binding:evil}",
        "p{width:expression(alert(1))}",
        '@import "evil";',
        "p{background:url(evil)}",
        "p{beha/**/vior:evil}",
    ]:
        with pytest.raises(schema.ContractError):
            visuals.render_document(
                document.model_copy(
                    update={
                        "html": document.html.replace(
                            "</head>", f"<style>{css}</style></head>"
                        )
                    }
                )
            )


def test_compiler_replaces_equivalent_policy_and_rejects_policy_or_code_changes() -> (
    None
):
    document, payloads = bundle()
    policy = visuals.content_security_policy(document)
    equivalent = " ;  ".join(reversed(policy.split("; "))) + ";"
    from html import escape

    raw = re.sub(
        r'<meta data-visual-policy="1"[^>]+>',
        "<meta content='"
        + escape(equivalent, quote=True)
        + "' HTTP-EQUIV='content-security-policy' />",
        document.html,
    )
    assert raw != document.html
    compiled = visuals.render_document(document.model_copy(update={"html": raw}))
    assert compiled == document
    visuals.validate_visuals(compiled, payloads)
    for html in [
        raw.replace("default-src", "unsafe-default-src"),
        raw.replace("connect-src &#x27;none&#x27;", "connect-src https:"),
        raw.replace("script-src", "script-src &#x27;unsafe-inline&#x27;"),
        document.html.replace("<script ", '<script src="assets/visuals_v2.js" '),
        document.html.replace("</script>", "alert(1)</script>"),
    ]:
        with pytest.raises(schema.ContractError, match="invalid_document"):
            visuals.render_document(document.model_copy(update={"html": html}))
    # Omitting all system components is the writer contract. Recompilation must
    # also work when there is no previous CSP/script/style to copy.
    raw = re.sub(r'<meta data-visual-policy="1"[^>]+>', "", document.html)
    raw = re.sub(r"<script\b.*?</script>", "", raw, flags=re.S)
    raw = re.sub(r'<style data-visual-style="1">.*?</style>', "", raw, flags=re.S)
    assert (
        visuals.render_document(document.model_copy(update={"html": raw})) == document
    )


def test_native_mathml_is_allowed_but_active_math_and_unrendered_tex_are_refused() -> (
    None
):
    document, payloads = bundle()
    math = '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" aria-label="Depth equals focal length times baseline over disparity"><mi>Z</mi><mo>=</mo><mfrac><mrow><mi>f</mi><mi>B</mi></mrow><mi>d</mi></mfrac></math>'
    compiled = visuals.render_document(
        document.model_copy(
            update={"html": document.html.replace("</section>", math + "</section>")}
        )
    )
    visuals.validate_visuals(compiled, payloads)
    for markup in [
        math.replace("<mi>Z", '<mi onclick="evil()">Z'),
        math.replace("<mi>Z", '<mi href="https://example.com">Z'),
        '<math><annotation-xml encoding="text/html"><script>evil()</script></annotation-xml></math>',
        '<math><mglyph src="evil"></mglyph></math>',
        "<mi>Z</mi>",
        r"\(Z=\frac{fB}{d}\)",
        r"$$Z=20$$",
    ]:
        with pytest.raises(schema.ContractError):
            visuals.render_document(
                document.model_copy(
                    update={
                        "html": document.html.replace(
                            "</section>", markup + "</section>"
                        )
                    }
                )
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
