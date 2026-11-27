from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from tools.wiki_assets.__main__ import main
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
    assert verify_markdown(page) == {svg, svg.with_suffix(".png")}


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


@pytest.mark.parametrize(
    "pathname",
    [
        "/srv/tennis-lab/private-assets/report.csv",
        "/custom-root",
        "/日本語/表.csv",
        "[report](</data/report.csv>)",
        "[root](</data>)",
        r"\\server\share\report.csv",
        r"\\?\UNC\server\share\report.csv",
        "//server/share/report.csv",
        "&#47;&#47;server/share/report.csv",
    ],
)
def test_public_scan_rejects_absolute_paths(pathname: str) -> None:
    with pytest.raises(ValueError, match="local path"):
        scan_public_text(pathname, "example")


@pytest.mark.parametrize(
    "source",
    [
        "[Example](https://example.org/srv/project) http://example.org/a%20b/var/log",
        "https://example.org/a(b)/srv/project https://[2001:db8::1]/opt/page",
        "[asset](assets/example/v1/result.png) [doc](../docs/opt/README.md)",
        "`manifests/<page-slug>/manifest.json`",
        "ms/画像、人/秒、6,844/46,228、`SVG`/`PNG`",
        "<details>\n<summary>説明</summary>\n\n本文\n\n</details>",
        r'`re.compile(r"^/[a-z]+(?:/[^/]+)*$")`',
    ],
)
def test_public_scan_preserves_urls_and_non_path_slashes(source: str) -> None:
    scan_public_text(source, "example")


@pytest.mark.parametrize(
    "template",
    [
        "![Preview]({asset})",
        "[PNG file]({asset})",
        '<img src="{asset}" alt="Preview">',
        '<picture><source srcset="{asset} 2x"><img src="https://example.org/cover.png" alt="Preview"></picture>',
    ],
)
def test_direct_png_is_inspected_for_all_reference_styles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], template: str
) -> None:
    asset = ASSET.replace(".svg", ".png")
    png = tmp_path / asset
    png.parent.mkdir(parents=True)
    page = tmp_path / "PNG.md"
    page.write_text(template.format(asset=asset) + "\n")
    Image.new("RGB", (12, 12), "white").save(png)
    assert verify_markdown(page) == {png}
    assert main(["verify", "--wiki-page", str(page)]) == 0
    png.write_bytes(b"corrupt PNG")
    assert main(["verify", "--wiki-page", str(page)]) == 1
    assert "invalid PNG signature" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("RGBA", "expected 8-bit RGB PNG"),
        ("bytes", "invalid asset size"),
        ("dimensions", "invalid PNG dimensions"),
        ("text", "unexpected PNG chunk"),
    ],
)
def test_direct_png_rejects_unpublishable_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str, message: str
) -> None:
    asset = ASSET.replace(".svg", ".png")
    png = tmp_path / asset
    png.parent.mkdir(parents=True)
    page = tmp_path / "PNG.md"
    page.write_text(f"![Preview]({asset})\n")
    metadata = PngImagePlugin.PngInfo()
    if damage == "text":
        metadata.add_text("Comment", "Example metadata")
    size = (4097, 12) if damage == "dimensions" else (12, 12)
    Image.new("RGBA" if damage == "RGBA" else "RGB", size, "white").save(
        png, pnginfo=metadata
    )
    if damage == "bytes":
        with png.open("ab") as file:
            file.truncate(MAX_ASSET_BYTES + 1)
    assert main(["verify", "--wiki-page", str(page)]) == 1
    assert message in capsys.readouterr().err
