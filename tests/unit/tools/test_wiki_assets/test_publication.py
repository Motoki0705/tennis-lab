from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tools.wiki_assets.markdown import verify_markdown
from tools.wiki_assets.publication import (
    MAX_ASSET_BYTES,
    scan_public_text,
    verify_pair,
    verify_png,
    verify_svg,
)

SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
SVG_PATH = '<path fill="#fff" d="M0 0h12v12H0z"/>'
ASSET = "assets/example/example.svg"
LINKS = f"![Comparison]({ASSET})\n\n[PNG]({ASSET.replace('.svg', '.png')})\n"


def test_svg_outline_and_local_glyph_reference_are_valid(tmp_path: Path) -> None:
    svg = tmp_path / "outline.svg"
    glyph = SVG_PATH.replace("path ", 'path id="glyph" ')
    svg.write_text(SVG_HEAD + f'<defs>{glyph}</defs><use href="#glyph"/></svg>')
    Image.new("RGB", (12, 12), "white").save(svg.with_suffix(".png"))
    verify_pair(svg)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("<text>人物</text>", "element <text>"),
        ("<metadata/>", "metadata"),
        ('<image href="https://example.org/a.png"/>', "element <image>"),
        ("<script/>", "element <script>"),
        ('<use href="https://example.org/icons.svg#label"/>', "external or dangling"),
        ('<use href="#missing"/>', "external or dangling"),
        ('<g onclick="example()"/>', "attribute onclick"),
        ('<g style="fill:url(https://example.org/paint)"/>', "unsafe SVG style"),
        ("<!-- author note -->", "declaration or metadata"),
    ],
)
def test_svg_rejects_text_active_elements_and_external_resources(
    tmp_path: Path, payload: str, message: str
) -> None:
    svg = tmp_path / "outline.svg"
    svg.write_text(f"{SVG_HEAD}{SVG_PATH}{payload}</svg>")
    with pytest.raises(ValueError, match=message):
        verify_svg(svg)


def test_svg_size_bound_precedes_xml_parse(tmp_path: Path) -> None:
    svg = tmp_path / "large.svg"
    svg.write_bytes(b" " * (MAX_ASSET_BYTES + 1))
    with pytest.raises(ValueError, match="asset size"):
        verify_svg(svg)


def test_png_rejects_alpha_metadata_and_corruption(tmp_path: Path) -> None:
    png = tmp_path / "table.png"
    Image.new("RGBA", (12, 12), "white").save(png)
    with pytest.raises(ValueError, match="8-bit RGB"):
        verify_png(png)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "Example metadata")
    Image.new("RGB", (12, 12), "white").save(png, pnginfo=metadata)
    with pytest.raises(ValueError, match="unexpected PNG chunk"):
        verify_png(png)
    Image.new("RGB", (12, 12), "white").save(png)
    raster = bytearray(png.read_bytes())
    raster[-1] ^= 1
    png.write_bytes(raster)
    with pytest.raises(ValueError, match="invalid PNG checksum"):
        verify_png(png)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("%2Fhome%2Fexample/file.txt", "local path"),
        (r"Example C:\Users\example\file.txt", "local path"),
        ("Someone <example@example.org>", "email"),
        ("github_pat_" + "x" * 25, "credential"),
        (
            "0-123456789012345_2026-02-01T00-00-00-000000Z_123_benchmark.job",
            "execution identifier",
        ),
    ],
)
def test_public_scan_rejects_private_identifiers(text: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        scan_public_text(text, "example")


def test_markdown_accepts_relative_assets_and_renders_nested_list(
    tmp_path: Path,
) -> None:
    page = tmp_path / "Comparison.md"
    svg = tmp_path / ASSET
    svg.parent.mkdir(parents=True)
    svg.touch()
    svg.with_suffix(".png").touch()
    page.write_text(
        "# Comparison\n\n![Comparison][my-image]\n\n[my-image]: ./"
        + ASSET
        + "\n\n[PNG](./"
        + ASSET.replace(".svg", ".png")
        + ")\n\n"
        "<details>\n<summary>Details</summary>\n\n- example\n  - nested\n\n</details>\n"
    )
    assert verify_markdown(page) == {svg}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "# Title\n\n| 項目 | 内容 |\n| --- | --- |\n| **想定読者** | all |\n",
            "opening metadata table",
        ),
        (
            "<details>\n<summary>Example</summary>\n- item\n\n</details>\n",
            "blank line after <summary>",
        ),
        (
            "<details><summary>Example</summary>\n\n- item\n</details>\n",
            "before and after </details>",
        ),
        ("![](" + ASSET + ")\n", "alt is empty"),
        (LINKS, "Missing relative Wiki asset"),
    ],
)
def test_markdown_checks_boilerplate_spacing_alt_and_links(
    tmp_path: Path, text: str, message: str
) -> None:
    page = tmp_path / "Comparison.md"
    page.write_text(text)
    with pytest.raises(ValueError, match=message):
        verify_markdown(page)


def test_markdown_refuses_displayed_svg_without_png_link(tmp_path: Path) -> None:
    page = tmp_path / "Comparison.md"
    svg = tmp_path / ASSET
    svg.parent.mkdir(parents=True)
    svg.touch()
    svg.with_suffix(".png").touch()
    page.write_text(LINKS.split("\n\n")[0])
    with pytest.raises(ValueError, match="PNG link missing"):
        verify_markdown(page)
