"""Managed visual markup, asset safety and revision-pinned offline rendering.

Call render_document() before assembling a Candidate; then validate_visuals() on
the immutable staged document and *all* staged bytes. This is a deterministic
safety check, not a semantic or browser-rendering verdict.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

from .document import inspect_document
from .images import MAX_ASSET_BYTES, ImportedAsset, raster_type
from .parsing import StrictHTMLParser, reject_css_resources
from .schema import Asset, ContractError, Document, Figure, Provenance, parse
from .visual_definitions import Graph, ObservationError, graph_series, parse_graph

RENDERER_ID = "visuals_v2"
RENDERER_VERSION = "2"
_RENDERERS = {
    version: Path(__file__).parent / f"renderers/visuals-v{version}.js"
    for version in ("1", "2")
}
_RENDER_ATTRIBUTES = {
    "data-managed-graph",
    "data-definition",
    "data-managed-slider",
    "data-slider-value",
    "data-plot",
    "data-values",
    "data-graph-targets",
    "data-render-error",
}
_SVG_NS = "http://www.w3.org/2000/svg"
_SVG_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "title",
    "desc",
}
_SVG_ATTRIBUTES = {
    "id",
    "viewBox",
    "preserveAspectRatio",
    "x",
    "y",
    "dx",
    "dy",
    "width",
    "height",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "x1",
    "x2",
    "y1",
    "y2",
    "d",
    "points",
    "transform",
    "fill",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-dasharray",
    "opacity",
    "fill-opacity",
    "stroke-opacity",
    "font-size",
    "font-family",
    "font-weight",
    "text-anchor",
    "dominant-baseline",
    "role",
    "aria-label",
}
_HTML_TAGS = set(
    [
        "html",
        "head",
        "title",
        "meta",
        "body",
        "main",
        "header",
        "footer",
        "nav",
        "article",
        "section",
        "aside",
        "div",
        "span",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "a",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "pre",
        "code",
        "blockquote",
        "strong",
        "em",
        "b",
        "i",
        "u",
        "s",
        "small",
        "sub",
        "sup",
        "abbr",
        "time",
        "mark",
        "kbd",
        "samp",
        "var",
        "details",
        "summary",
        "br",
        "hr",
        "img",
        "figure",
        "figcaption",
        "table",
        "caption",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "colgroup",
        "col",
        "style",
        "link",
        "input",
        "label",
        "output",
        "button",
    ]
)
_VOID = {"meta", "br", "hr", "img", "link", "input", "col"}
_MATH_NS = "http://www.w3.org/1998/Math/MathML"
_MATH_TAGS = set(
    [
        "math",
        "mrow",
        "mi",
        "mn",
        "mo",
        "mtext",
        "mspace",
        "ms",
        "mfrac",
        "msqrt",
        "mroot",
        "msub",
        "msup",
        "msubsup",
        "munder",
        "mover",
        "munderover",
        "mtable",
        "mtr",
        "mtd",
        "mstyle",
        "mpadded",
        "mphantom",
    ]
)
_MATH_ATTRIBUTES = set(
    [
        "id",
        "class",
        "style",
        "role",
        "aria-label",
        "aria-labelledby",
        "aria-describedby",
        "display",
        "mathvariant",
        "mathsize",
        "mathcolor",
        "mathbackground",
        "displaystyle",
        "scriptlevel",
        "stretchy",
        "symmetric",
        "largeop",
        "movablelimits",
        "accent",
        "accentunder",
        "fence",
        "separator",
        "lspace",
        "rspace",
        "minsize",
        "maxsize",
        "linethickness",
        "columnalign",
        "rowalign",
        "columnspacing",
        "rowspacing",
        "columnspan",
        "rowspan",
        "width",
        "height",
        "depth",
        "voffset",
    ]
)
# Compiler-owned containment, independent of global writer rules such as
# table{min-width:750px}. Wide source tables scroll inside their figure.
_FIGURE_CSS = (
    "figure[data-figure-id]{box-sizing:border-box!important;min-width:0!important;max-width:100%!important;overflow-x:auto!important;overflow-wrap:anywhere}"
    "figure[data-figure-id] [data-managed-graph]{min-width:0!important;max-width:100%!important;overflow-x:auto!important}"
    "figure[data-figure-id] svg,figure[data-figure-id] img{max-width:100%!important;height:auto!important}"
)


def _figure_style() -> str:
    return '<style data-visual-style="1">' + _FIGURE_CSS + "</style>"


def _policy_directives(policy: str) -> dict[str, frozenset[str]]:
    result = {}
    for directive in policy.split(";"):
        tokens = directive.split()
        if not tokens:
            continue
        name, *values = tokens
        name = name.lower()
        if name in result or len(values) != len(set(values)):
            raise ContractError("invalid_document")
        result[name] = frozenset(values)
    return result


_KINDS = {
    "measured": "Measured",
    "hypothetical": "Hypothetical example",
    "generated_concept": "Generated concept",
    "derived": "Derived",
    "reference": "Reference",
}


def validate_svg(content: bytes) -> None:
    """Reject rather than silently remove unsafe SVG elements/attributes."""
    try:
        if not 0 < len(content) <= MAX_ASSET_BYTES:
            raise ValueError("SVG size limit.")
        text = content.decode("utf-8")
        text = re.sub(r"^\s*<\?xml[^?]*\?>", "", text, count=1)
        if "<!" in text or "<?" in text:
            raise ValueError("SVG declarations are not allowed.")
        root = ET.fromstring(text)
        if root.tag != f"{{{_SVG_NS}}}svg":
            raise ValueError("SVG namespace required.")
        viewbox = [
            float(item)
            for item in root.attrib.get("viewBox", "").replace(",", " ").split()
        ]
        if (
            len(viewbox) != 4
            or not all(math.isfinite(x) for x in viewbox)
            or min(viewbox[2:]) <= 0
        ):
            raise ValueError("A finite positive SVG viewBox is required.")
        pending = [(root, 0)]
        count = 0
        ids: set[str] = set()
        while pending:
            node, depth = pending.pop()
            count += 1
            if (
                count > 10000
                or depth > 32
                or node.tag not in {f"{{{_SVG_NS}}}{tag}" for tag in _SVG_TAGS}
            ):
                raise ValueError("SVG element is not permitted.")
            for key, value in node.attrib.items():
                if key not in _SVG_ATTRIBUTES or re.search(
                    r"url\s*\(|javascript|@|\\|[\x00-\x1f]", value, re.I
                ):
                    raise ValueError("SVG attribute is not permitted.")
                if key in {"fill", "stroke"} and not re.fullmatch(
                    r"(?:#[a-fA-F0-9]{3,8}|[A-Za-z]+)", value
                ):
                    raise ValueError("Only literal SVG paint is allowed.")
                if key == "id":
                    if value in ids:
                        raise ValueError("Duplicate SVG ID.")
                    ids.add(value)
            pending.extend((child, depth + 1) for child in node)
    except (ValueError, UnicodeError, ET.ParseError) as exc:
        raise ContractError("invalid_document") from exc


def import_svg(
    content: bytes, *, asset_id: str, caption: str, alt: str, provenance: Provenance
) -> ImportedAsset:
    validate_svg(content)
    return ImportedAsset(
        asset=Asset(
            asset_id=asset_id,
            path=f"assets/{asset_id}.svg",
            media_type="image/svg+xml",
            sha256=hashlib.sha256(content).hexdigest(),
            kind="svg",
            caption=caption,
            alt=alt,
            provenance=provenance,
        ),
        content=content,
    )


def managed_renderer(version: str = RENDERER_VERSION) -> ImportedAsset:
    """Versions are immutable; v1 remains available for stored revisions."""
    if version not in _RENDERERS:
        raise ContractError("invalid_document")
    asset_id = f"visuals_v{version}"
    content = _RENDERERS[version].read_bytes()
    return ImportedAsset(
        asset=Asset(
            asset_id=asset_id,
            path=f"assets/{asset_id}.js",
            media_type="text/javascript",
            sha256=hashlib.sha256(content).hexdigest(),
            kind="renderer",
            caption=f"Managed observation-error and comparison renderer, version {version}.",
            alt="Interactive graphs with visible source tables.",
            provenance=Provenance(
                kind="reference",
                source=f"explain_html/renderers/visuals-v{version}.js, version {version}.",
            ),
        ),
        content=content,
    )


def _document_renderer(document: Document) -> ImportedAsset | None:
    graphs = [figure for figure in document.figures if figure.kind == "graph"]
    if not graphs:
        return None
    versions = {
        figure.renderer.version if figure.renderer else None for figure in graphs
    }
    if len(versions) != 1 or None in versions:
        raise ContractError("invalid_document")
    renderer = graphs[0].renderer
    assert renderer is not None
    return managed_renderer(renderer.version)


def _definition(graph: Graph) -> str:
    return json.dumps(
        graph.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
    )


def _renderer_script(document: Document) -> str:
    renderer = _document_renderer(document)
    if renderer is None:
        return ""
    targets = ""
    if renderer.asset.asset_id == "visuals_v2":
        manifest = [
            {
                "figure_id": figure.figure_id,
                "definition": _definition(parse_graph(figure.definition)),
            }
            for figure in document.figures
            if figure.kind == "graph"
        ]
        targets = (
            ' data-graph-targets="'
            + escape(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                quote=True,
            )
            + '"'
        )
    # file:// cannot perform SRI on an external script without CORS. Embed only
    # these exact managed bytes and authorize their digest in CSP. The same
    # bytes remain a declared revision asset for inspection and restoration.
    return f'<script data-asset-id="{renderer.asset.asset_id}"{targets}>{renderer.content.decode("utf-8")}</script>'


def content_security_policy(document: Document) -> str:
    script = "'none'"
    renderer = _document_renderer(document)
    if renderer is not None:
        digest = base64.b64encode(bytes.fromhex(renderer.asset.sha256)).decode("ascii")
        script = f"'sha256-{digest}'"
    return f"default-src 'none'; script-src {script}; img-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"


def _policy_meta(document: Document) -> str:
    return (
        '<meta data-visual-policy="1" http-equiv="Content-Security-Policy" content="'
        + escape(content_security_policy(document), quote=True)
        + '">'
    )


def _number(value: float) -> str:
    return format(value, ".9g")


def _axis(label: str, unit: str) -> str:
    return escape(f"{label} ({unit})")


def _plot(graph: Graph) -> str:
    x0, x1 = graph.x.domain
    y0, y1 = graph.y.domain
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 320" role="img" width="100%" aria-label="'
        + escape(
            f"{graph.y.label} ({graph.y.unit}) against {graph.x.label} ({graph.x.unit})",
            quote=True,
        )
        + '">',
        '<path d="M72 20V262H616" fill="none" stroke="currentColor"></path>',
    ]
    for i, series in enumerate(graph_series(graph)):
        color = ["#005a9c", "#a33500", "#367328", "#7a3589"][i % 4]
        points = " ".join(
            f"{_number(72 + (p.x - x0) / (x1 - x0) * 544)},{_number(262 - (p.y - y0) / (y1 - y0) * 242)}"
            for p in series.points
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"></polyline>'
        )
        for point in series.points:
            parts.append(
                f'<circle cx="{_number(72 + (point.x - x0) / (x1 - x0) * 544)}" cy="{_number(262 - (point.y - y0) / (y1 - y0) * 242)}" r="4" fill="{color}" data-x="{_number(point.x)}" data-y="{_number(point.y)}"><title>{escape(series.label)}: {_number(point.x)}, {_number(point.y)}</title></circle>'
            )
    parts.extend(
        [
            f'<text x="330" y="310" text-anchor="middle">{_axis(graph.x.label, graph.x.unit)}</text>',
            f'<text x="8" y="14">{_axis(graph.y.label, graph.y.unit)}</text>',
            f'<text x="72" y="282">{_number(x0)}</text>',
            f'<text x="586" y="282">{_number(x1)}</text>',
            f'<text x="28" y="262">{_number(y0)}</text>',
            f'<text x="28" y="30">{_number(y1)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _graph_html(graph: Graph, alt: str, version: str) -> str:
    definition = escape(_definition(graph), quote=True)
    result = f'<div data-managed-graph="{version}" role="group" aria-label="{escape(alt, quote=True)}" data-definition="{definition}">'
    if isinstance(graph, ObservationError):
        slider = graph.slider
        initial = graph_series(graph)[0]
        current = next(p.y for p in initial.points if p.x == slider.initial)
        result += f'<label>{escape(slider.label)} ({escape(slider.unit)}) <input data-managed-slider="1" type="range" min="{slider.min!r}" max="{slider.max!r}" step="{slider.step!r}" value="{slider.initial!r}"></label>'
        result += f'<output data-slider-value="1" aria-live="polite">{_number(slider.initial)} px; {escape(graph.y.label)} = {_number(current)} m</output>'
        result += f"<p>Rectified pinhole stereo: Z = f B / (d + error). f = {_number(graph.data.focal_px)} px; B = {_number(graph.data.baseline_m)} m; d = {_number(graph.data.disparity_px)} px. Error range: [{_number(slider.min)}, {_number(slider.max)}] px. Calibration, rectification and correct correspondence are assumed. This is a signed disparity-error example.</p>"
    result += '<div data-plot="1">' + _plot(graph) + "</div>"
    result += f'<table><caption>Source data and displayed values</caption><thead><tr><th scope="col">Series</th><th scope="col">{_axis(graph.x.label, graph.x.unit)}</th><th scope="col">{_axis(graph.y.label, graph.y.unit)}</th></tr></thead><tbody data-values="1">'
    for series in graph_series(graph):
        for point in series.points:
            result += f'<tr><th scope="row">{escape(series.label)}</th><td>{_number(point.x)}</td><td>{_number(point.y)}</td></tr>'
    return result + "</tbody></table></div>"


def _figure_html(figure: Figure, assets: dict[str, Asset]) -> str:
    if figure.kind == "graph":
        if figure.renderer is None:
            raise ContractError("invalid_document")
        expected = managed_renderer(figure.renderer.version).asset
        if (
            figure.renderer.asset_id != expected.asset_id
            or figure.asset_ids != [expected.asset_id]
            or assets.get(expected.asset_id) != expected
        ):
            raise ContractError("invalid_document")
        graph = parse_graph(figure.definition)
        if (
            isinstance(graph, ObservationError)
            and figure.provenance.kind != "hypothetical"
        ):
            raise ContractError("invalid_document")
        result = _graph_html(graph, figure.alt, figure.renderer.version)
    else:
        if (
            len(figure.asset_ids) != 1
            or figure.renderer is not None
            or figure.definition
        ):
            raise ContractError("invalid_document")
        asset = assets.get(figure.asset_ids[0])
        if (
            asset is None
            or asset.kind != figure.kind
            or asset.provenance != figure.provenance
        ):
            raise ContractError("broken_reference")
        result = f'<img data-asset-id="{asset.asset_id}" src="{escape(asset.path, quote=True)}" alt="{escape(figure.alt, quote=True)}" style="max-width:100%;height:auto">'
    return (
        result
        + f"<figcaption><p>{escape(figure.caption)}</p><p>{_KINDS[figure.provenance.kind]} — {escape(figure.provenance.source)}</p></figcaption>"
    )


def _css(text: str) -> None:
    reject_css_resources(text, code="invalid_document")
    if re.search(
        r"url\s*\(|expression\s*\(|@(?!media\b|supports\b|layer\b|container\b)|\\|/\*|(?<![\w-])(?:behavior|-moz-binding)\s*:|<",
        text,
        re.I,
    ):
        raise ContractError("invalid_document")


class _Markup(StrictHTMLParser):
    def __init__(self, document: Document):
        super().__init__(convert_charrefs=True)
        self.document = document
        self.stack: list[tuple[str, str | None, int, int]] = []
        self.slots: dict[str, tuple[int, int]] = {}
        self.scripts: list[tuple[int, int]] = []
        self.script_attributes: dict[int, dict[str, str | None]] = {}
        self.policies: list[tuple[int, int]] = []
        self.styles: list[tuple[int, int]] = []
        self.head_end: int | None = None
        self.has_body = False
        self.has_html = False
        self.lines = [0, *[m.end() for m in re.finditer("\n", document.html)]]
        self.feed(document.html)
        self.close()

    def source_offset(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        parent = self.stack[-1][0] if self.stack else None
        in_head = any(item[0] == "head" for item in self.stack)
        in_body = any(item[0] == "body" for item in self.stack)
        if tag == "html":
            if self.has_html or self.stack:
                raise ContractError("invalid_document")
            self.has_html = True
        elif tag == "head":
            if parent != "html" or self.head_end is not None or self.has_body:
                raise ContractError("invalid_document")
        elif tag == "body":
            if parent != "html" or self.head_end is None or self.has_body:
                raise ContractError("invalid_document")
            self.has_body = True
        elif not (in_head or in_body) or (
            in_head and tag not in {"meta", "title", "style", "link", "script"}
        ):
            raise ContractError("invalid_document")
        active = next((item[1] for item in reversed(self.stack) if item[1]), None)
        in_graph = any(
            f.figure_id == active and f.kind == "graph" for f in self.document.figures
        )
        in_math = any(item[0] == "math" for item in self.stack)
        if tag in _MATH_TAGS:
            if (tag != "math" and not in_math) or (tag == "math" and in_math):
                raise ContractError("invalid_document")
            if (
                set(attributes)
                - _MATH_ATTRIBUTES
                - ({"xmlns"} if tag == "math" else set())
            ):
                raise ContractError("invalid_document")
            if "xmlns" in attributes and attributes["xmlns"] != _MATH_NS:
                raise ContractError("invalid_document")
        elif (
            in_math
            or tag not in _HTML_TAGS | {"script"}
            and not (in_graph and tag in _SVG_TAGS)
        ):
            raise ContractError("invalid_document")
        if any(
            key.startswith("on")
            or key in {"srcdoc", "ping", "is", "formaction", "autofocus"}
            for key in attributes
        ):
            raise ContractError("invalid_document")
        reserved = {
            name
            for name in attributes
            if name in _RENDER_ATTRIBUTES
            or name.startswith(("data-managed-", "data-render-"))
        }
        if (
            reserved
            and not in_graph
            and not (tag == "script" and reserved == {"data-graph-targets"})
        ):
            raise ContractError("invalid_document")
        if "data-graph-targets" in attributes and tag != "script":
            raise ContractError("invalid_document")
        if "data-visual-style" in attributes and (
            tag != "style" or attributes != {"data-visual-style": "1"}
        ):
            raise ContractError("invalid_document")
        if "style" in attributes:
            _css(attributes["style"] or "")
        if tag == "input" and not in_graph:
            raise ContractError("invalid_document")
        if tag == "img" and (
            not active
            or not attributes.get("alt")
            or not attributes.get("data-asset-id")
        ):
            raise ContractError("invalid_document")
        if tag == "link" and attributes.get("rel") != "stylesheet":
            raise ContractError("invalid_document")
        start = self.source_offset()
        end = start + len(self.get_starttag_text() or "")
        if tag == "head":
            self.head_end = end
        if tag == "script":
            self.script_attributes[start] = attributes
        if tag == "meta" and "http-equiv" in attributes:
            if (
                not in_head
                or (attributes.get("http-equiv") or "").lower()
                != "content-security-policy"
                or set(attributes) - {"http-equiv", "content", "data-visual-policy"}
                or _policy_directives(attributes.get("content") or "")
                != _policy_directives(content_security_policy(self.document))
                or self.policies
            ):
                raise ContractError("invalid_document")
            self.policies.append((start, end))
        figure = attributes.get("data-figure-id")
        if figure and (tag != "figure" or active):
            raise ContractError("invalid_document")
        if tag not in _VOID:
            self.stack.append((tag, figure, start, end))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"html", "head", "body"}:
            raise ContractError("invalid_document")
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            raise ContractError("invalid_document")
        _, figure, start, inner_start = self.stack.pop()
        if figure:
            self.slots[figure] = (inner_start, self.source_offset())
        if (
            tag == "style"
            and "data-visual-style" in self.document.html[start:inner_start]
        ):
            end = self.document.html.find(">", self.source_offset()) + 1
            if self.document.html[start:end] != _figure_style() or self.styles:
                raise ContractError("invalid_document")
            self.styles.append((start, end))
        if tag == "script":
            end = self.document.html.find(">", self.source_offset()) + 1
            renderer = _document_renderer(self.document)
            attributes = self.script_attributes[start].copy()
            if renderer is None:
                raise ContractError("invalid_document")
            if (
                renderer.asset.asset_id == "visuals_v2"
                and attributes.pop("data-graph-targets", None) is None
            ):
                raise ContractError("invalid_document")
            # The compiler replaces the target manifest from Figure records.
            # Existing managed code may carry the previous definition manifest
            # while a legitimate edit is being compiled; validate_visuals still
            # requires the complete final markup to equal canonical output.
            if attributes != {
                "data-asset-id": renderer.asset.asset_id
            } or self.document.html[
                inner_start : self.source_offset()
            ] != renderer.content.decode("utf-8"):
                raise ContractError("invalid_document")
            self.scripts.append((start, end))

    def handle_data(self, data: str) -> None:
        if data.strip(" \t\r\n\f") and (
            not self.stack or self.stack[-1][0] in {"html", "head"}
        ):
            raise ContractError("invalid_document")
        if self.stack and self.stack[-1][0] == "style":
            _css(data)
        elif (
            self.stack
            and self.stack[-1][0] != "script"
            and re.search(r"\\[()[\]]|\$\$|\\(?:frac|sqrt|sum|begin)\b", data)
        ):
            raise ContractError("invalid_document")


def render_document(document: Document) -> Document:
    """Fill <figure data-figure-id> slots without prescribing page layout."""
    inspect_document(document)
    markup = _Markup(document)
    assets = {asset.asset_id: asset for asset in document.assets}
    # Managed components inside a figure disappear with its replacement;
    # do not apply overlapping source edits. Every runtime reference belongs to
    # the first graph's existing content block under the shared ownership policy.
    components_outside_figures = [
        (start, end)
        for start, end in markup.scripts + markup.policies + markup.styles
        if not any(
            inner_start <= start and end <= inner_end
            for inner_start, inner_end in markup.slots.values()
        )
    ]
    changes = [(start, end, "") for start, end in components_outside_figures]
    first_graph = next(
        (figure.figure_id for figure in document.figures if figure.kind == "graph"),
        None,
    )
    for figure in document.figures:
        if figure.figure_id not in markup.slots:
            raise ContractError("broken_reference")
        start, end = markup.slots[figure.figure_id]
        content = _figure_html(figure, assets)
        if figure.figure_id == first_graph:
            content += _renderer_script(document)
        changes.append((start, end, content))
    html = document.html
    for start, end, value in sorted(changes, reverse=True):
        html = html[:start] + value + html[end:]
    # A CSP meta precedes all resources. The hash-authorized classic script works
    # with file:// exports, without fetch(), ES module CORS or a CDN.
    if markup.head_end is None or not markup.has_body:
        raise ContractError("invalid_document")
    style = _figure_style() if document.figures else ""
    html = (
        html[: markup.head_end]
        + _policy_meta(document)
        + style
        + html[markup.head_end :]
    )
    result = document.model_copy(update={"html": html})
    inspect_document(result)
    return result


def validate_visuals(document: Document, payloads: dict[str, bytes]) -> None:
    """Validate the staged bundle; no side effects and no success verdict forged."""
    document = parse(Document, document.model_dump_json())
    inspect_document(document)
    assets = {asset.asset_id: asset for asset in document.assets}
    if set(payloads) != set(assets):
        raise ContractError("broken_reference")
    for asset in assets.values():
        content = payloads[asset.asset_id]
        if (
            not 0 < len(content) <= MAX_ASSET_BYTES
            or hashlib.sha256(content).hexdigest() != asset.sha256
        ):
            raise ContractError("invalid_document")
        if asset.kind == "image":
            extension, media_type = raster_type(content)
            if (
                asset.media_type != media_type
                or asset.path != f"assets/{asset.asset_id}.{extension}"
            ):
                raise ContractError("invalid_document")
        elif asset.kind == "svg":
            validate_svg(content)
            if asset.media_type != "image/svg+xml" or not asset.path.endswith(".svg"):
                raise ContractError("invalid_document")
        elif asset.kind == "renderer":
            matching = [
                managed_renderer(version)
                for version in _RENDERERS
                if asset.asset_id == f"visuals_v{version}"
            ]
            if (
                not matching
                or asset != matching[0].asset
                or content != matching[0].content
            ):
                raise ContractError("invalid_document")
        elif asset.kind == "stylesheet":
            try:
                _css(content.decode("utf-8"))
            except UnicodeError as exc:
                raise ContractError("invalid_document") from exc
            if asset.media_type != "text/css" or not asset.path.endswith(".css"):
                raise ContractError("invalid_document")
        else:
            # Graph data is in Figure.definition; no arbitrary executable data assets.
            if asset.media_type != "application/json" or not asset.path.endswith(
                ".json"
            ):
                raise ContractError("invalid_document")
            try:
                json.loads(content)
            except (ValueError, UnicodeError) as exc:
                raise ContractError("invalid_document") from exc
    if render_document(document) != document:
        raise ContractError("invalid_document")
