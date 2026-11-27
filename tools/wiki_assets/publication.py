"""Publication policy for generated text and static SVG/PNG pairs."""

from __future__ import annotations

import re
import struct
import zlib
from html import unescape
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree as ET  # noqa: N817

from PIL import Image

from .sources import require

MAX_ASSET_BYTES = 2 * 1024**2
MAX_DIMENSION = 4096
SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"
ET.register_namespace("", SVG[1:-1])
ET.register_namespace("xlink", XLINK[1:-1])

FORBIDDEN = {
    "local path": r"(?i)(?:/(?:home|mnt|Users|var|tmp|etc|opt|root|content|Volumes|workspace|private)/"
    r"|(?<![\w])[a-z]:[\\/]|~[\\/]|file:/|\.claude[\\/]worktrees|\.worktree-outputs|\.training_queue)",
    "email": r"(?i)(?<![\w.])[a-z0-9_.+-]+@[a-z0-9.-]+\.[a-z]{2,}",
    "credential": r"(?i)(?:\bgh[pousr]_[a-z\d_]{20,}|\bgithub_pat_[a-z\d_]{20,}|\bsk-[a-z\d_-]{20,}"
    r"|\bAKIA[A-Z0-9]{16}|\bAIza[a-z\d_-]{28,}|-----BEGIN [^-]*PRIVATE KEY-----"
    r"|\bBearer\s+[a-z\d_.-]{8,}|\b(?:token|api[_ -]?key|password)['\"]?\s*[=:]\s*['\"]?[a-z0-9_-]{8,})",
    "execution identifier": r"(?i)(?:\b\d+-\d{12,}_\d{4}-\d{2}-\d{2}T[\d.:-]+Z_\d+_[\w.-]+\.job\b"
    r"|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b"
    r"|\b(?:run|turn|decision|session|job)[_ -]?id['\"]?\s*[=:]\s*['\"]?[a-z0-9_.-]{8,})",
}


def scan_public_text(text: str, name: str) -> None:
    decoded = unquote(unescape(text))
    for category, pattern in FORBIDDEN.items():
        require(re.search(pattern, decoded) is None, f"{name}: found {category}")


def _asset_bytes(path: Path) -> bytes:
    require(
        path.is_file() and not path.is_symlink(), f"{path.name}: missing regular asset"
    )
    require(
        0 < path.stat().st_size <= MAX_ASSET_BYTES, f"{path.name}: invalid asset size"
    )
    return path.read_bytes()


SVG_ELEMENTS = {
    "svg",
    "g",
    "defs",
    "path",
    "rect",
    "use",
    "symbol",
    "clipPath",
    "pattern",
}
SVG_ATTRIBUTES = {
    "width",
    "height",
    "viewBox",
    "version",
    "id",
    "d",
    "x",
    "y",
    "fill",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-opacity",
    "fill-opacity",
    "opacity",
    "transform",
    "overflow",
    "clip-path",
    "style",
    "patternUnits",
    "patternTransform",
    "href",
}
CSS_ATTRIBUTES = {
    name for name in SVG_ATTRIBUTES if name.startswith(("fill", "stroke"))
} | {"opacity"}
FRAGMENT = r"[A-Za-z0-9_.:-]+"
XML_DECL = re.compile(r"^<\?xml [^<>]*\?>\s*")
MPL_DOCTYPE = (
    '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
    '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">'
)
MPL_STYLE = re.compile(
    r"\*\s*\{\s*stroke-linejoin:\s*round;\s*stroke-linecap:\s*butt;?\s*\}"
)


def _parse_svg(text: str, name: str) -> ET.Element:
    body = XML_DECL.sub("", text)
    require(
        "<!" not in body and "<?" not in body,
        f"{name}: unexpected SVG declaration or metadata",
    )
    root = ET.fromstring(body)
    require(root.tag == SVG + "svg", f"{name}: expected an SVG root")
    return root


def _safe_css_value(value: str, ids: set[str]) -> bool:
    url = re.fullmatch(rf"url\(#({FRAGMENT})\)", value)
    if url is not None:
        return url[1] in ids
    return re.fullmatch(r"[\w#.+%,\s-]+", value, re.ASCII) is not None


def _validate_style(value: str, ids: set[str]) -> bool:
    for declaration in value.rstrip("; ").split(";"):
        prop, separator, content = declaration.partition(":")
        if (
            not separator
            or prop.strip() not in CSS_ATTRIBUTES
            or not _safe_css_value(content.strip(), ids)
        ):
            return False
    return True


def _svg_size(root: ET.Element) -> tuple[float, float]:
    values = []
    for key in ("width", "height"):
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(pt|px)?", root.get(key, ""))
        if not match:
            raise ValueError(f"Invalid SVG {key}")
        length = float(match[1]) * (4 / 3 if match[2] == "pt" else 1)
        require(0 < length <= MAX_DIMENSION, f"Invalid SVG {key}")
        values.append(length)
    view_box = root.get("viewBox", "").split()
    require(
        len(view_box) == 4
        and all(re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", v) for v in view_box),
        "Invalid SVG viewBox",
    )
    require(
        float(view_box[2]) > 0 and float(view_box[3]) > 0,
        "Invalid SVG viewBox dimensions",
    )
    return values[0], values[1]


def _validate_svg(root: ET.Element, name: str) -> tuple[float, float]:
    dimensions = _svg_size(root)
    id_values = [node.attrib["id"] for node in root.iter() if "id" in node.attrib]
    ids = set(id_values)
    require(
        len(ids) == len(id_values)
        and all(re.fullmatch(FRAGMENT, value) for value in ids),
        f"{name}: invalid or duplicate SVG ids",
    )
    require(root.find(f".//{SVG}path") is not None, f"{name}: SVG must contain paths")
    for node in root.iter():
        kind = node.tag.removeprefix(SVG)
        require(
            node.tag == SVG + kind and kind in SVG_ELEMENTS,
            f"{name}: unexpected SVG element <{kind}>",
        )
        require(
            not (node.text or "").strip() and not (node.tail or "").strip(),
            f"{name}: text in SVG",
        )
        for key, value in node.attrib.items():
            name_without_ns = key.removeprefix(XLINK) if key == XLINK + "href" else key
            require(
                name_without_ns in SVG_ATTRIBUTES,
                f"{name}: unexpected SVG attribute {name_without_ns}",
            )
            if name_without_ns == "href":
                require(
                    kind == "use" and value.startswith("#") and value[1:] in ids,
                    f"{name}: external or dangling SVG reference",
                )
            elif name_without_ns == "style":
                require(_validate_style(value, ids), f"{name}: unsafe SVG style")
            elif re.search(r"url\s*\(", value, re.IGNORECASE):
                require(
                    name_without_ns in {"clip-path", "fill", "stroke"}
                    and _safe_css_value(value, ids),
                    f"{name}: external or dangling SVG resource",
                )
            else:
                require(
                    not re.search(
                        r"(?i)(?:https?://|data:|javascript:|@import)", value
                    ),
                    f"{name}: unexpected SVG resource",
                )
    return dimensions


def verify_svg(path: Path) -> tuple[float, float]:
    text = _asset_bytes(path).decode("utf-8")
    scan_public_text(text, path.name)
    return _validate_svg(_parse_svg(text, path.name), path.name)


def verify_png(path: Path) -> tuple[int, int]:
    raw = _asset_bytes(path)
    require(raw.startswith(b"\x89PNG\r\n\x1a\n"), f"{path.name}: invalid PNG signature")
    offset = 8
    types = []
    while offset < len(raw):
        require(len(raw) - offset >= 12, f"{path.name}: truncated PNG chunk")
        length, kind = struct.unpack_from(">I4s", raw, offset)
        end = offset + 12 + length
        require(
            end <= len(raw) and kind in {b"IHDR", b"IDAT", b"IEND"},
            f"{path.name}: unexpected PNG chunk",
        )
        require(
            zlib.crc32(raw[offset + 4 : end - 4])
            == struct.unpack_from(">I", raw, end - 4)[0],
            f"{path.name}: invalid PNG checksum",
        )
        types.append(kind)
        offset = end
    require(
        len(types) >= 3
        and types[0] == b"IHDR"
        and types[-1] == b"IEND"
        and all(t == b"IDAT" for t in types[1:-1]),
        f"{path.name}: invalid PNG chunk order",
    )
    require(
        struct.unpack_from(">I", raw, 8)[0] == 13 and raw[-12:-8] == b"\0" * 4,
        f"{path.name}: invalid IHDR or IEND length",
    )
    width, height, depth, color, compression, filtering, interlace = struct.unpack_from(
        ">IIBBBBB", raw, 16
    )
    require(
        (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0),
        f"{path.name}: expected 8-bit RGB PNG",
    )
    require(
        0 < max(width, height) <= MAX_DIMENSION and min(width, height) > 0,
        f"{path.name}: invalid PNG dimensions",
    )
    with Image.open(path) as image:
        image.load()
        require(
            image.format == "PNG" and image.mode == "RGB",
            f"{path.name}: expected RGB PNG",
        )
        corners = [(x, y) for x in (0, width - 1) for y in (0, height - 1)]
        require(
            all(image.getpixel(xy) == (255, 255, 255) for xy in corners),
            f"{path.name}: expected a white background",
        )
    return width, height


def verify_pair(svg: Path) -> None:
    width, height = verify_svg(svg)
    png_width, png_height = verify_png(svg.with_suffix(".png"))
    require(
        abs(png_height - height * png_width / width) <= 2,
        f"{svg.name}: PNG and SVG proportions differ",
    )


def prepare_pair(svg: Path) -> None:
    """Remove renderer metadata, then require the same format allowed on the Wiki."""
    text = svg.read_text(encoding="utf-8").replace(MPL_DOCTYPE, "", 1)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    root = _parse_svg(text, svg.name)
    for metadata in root.findall(SVG + "metadata"):
        root.remove(metadata)
    for parent in root.iter():
        for child in parent.findall(SVG + "style"):
            require(
                MPL_STYLE.fullmatch((child.text or "").strip()) is not None,
                f"{svg.name}: unexpected stylesheet",
            )
            parent.remove(child)
            for node in parent:
                node.attrib.setdefault("stroke-linejoin", "round")
                node.attrib.setdefault("stroke-linecap", "butt")
    _validate_svg(root, svg.name)
    svg.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")
    png = svg.with_suffix(".png")
    with Image.open(png) as image:
        image.load()
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image.convert("RGBA"))
        rgb = Image.frombytes("RGB", image.size, white.convert("RGB").tobytes())
    rgb.save(png, optimize=True)
    verify_pair(svg)
