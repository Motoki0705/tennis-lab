import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from PIL import Image

from tools.wiki_assets.chart_renderer import render_chart
from tools.wiki_assets.chart_spec import Bars
from tools.wiki_assets.markdown import PARSER, References, snippet, verify_markdown
from tools.wiki_assets.publication import prepare_pair, verify_pair
from tools.wiki_assets.sources import ROOT, Sources, number, string
from tools.wiki_assets.table_spec import Column, Row, Table
from tools.wiki_assets.typst_runner import BINARY_ENV, resolve
from tools.wiki_assets.verify import (
    describe,
    file_digests,
    manifest_path,
    verify_bundle,
)

PAGE = "example"
REVISION = "c" * 40
RAW = b'{"batch":"B1","backend":"PT16","p50_ms":2.375}'
TABLE = "backend-speed"


@pytest.fixture
def projection() -> tuple[Sources, Bars]:
    sources = Sources(REVISION, lambda p: RAW)
    raw = sources.json("examples/cases.json")
    bars = Bars(
        Table(
            TABLE,
            "Latency comparison",
            "Fixed input population",
            (Column("Batch", 55), Column("Backend", 60), Column("P50", 65, 2, "ms")),
            (
                Row(
                    "case-a",
                    (
                        string(raw["batch"]),
                        string(raw["backend"]),
                        number(raw["p50_ms"]),
                    ),
                ),
            ),
        ),
        metrics=(2,),
        series=1,
    )
    return sources, bars


def test_chart_svg_and_rgb_png_are_deterministic(
    tmp_path: Path, projection: tuple[Sources, Bars]
) -> None:
    _, bars = projection
    for stem in ("first", "second"):
        render_chart(bars, tmp_path / stem, font="DejaVu Sans")
        prepare_pair(tmp_path / f"{stem}.svg")
    for ext in ("svg", "png"):
        assert (tmp_path / f"first.{ext}").read_bytes() == (
            tmp_path / f"second.{ext}"
        ).read_bytes()
    svg = tmp_path / "first.svg"
    assert "<path " in svg.read_text()
    assert "<text" not in svg.read_text() and "<metadata" not in svg.read_text()
    verify_pair(svg)
    with Image.open(svg.with_suffix(".png")) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert sum(image.getchannel("R").histogram()[:80]) > 100


def _write_bundle(output: Path, sources: Sources, bars: Bars) -> None:
    # Use small static files here. A separate test renders a real Matplotlib chart.
    svg = output / "assets" / PAGE / f"{TABLE}.svg"
    svg.parent.mkdir(parents=True)
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
        '<path fill="#fff" d="M0 0h12v12H0z"/></svg>'
    )
    Image.new("RGB", (12, 12), "white").save(svg.with_suffix(".png"))
    (output / f"{PAGE}.md").write_text(snippet(PAGE, sources, (bars,)))
    record = describe(PAGE, sources, (bars,))
    record["files"] = file_digests(output, PAGE, (bars,))
    record["tools"] = {"typst": "0.15.1", "matplotlib": "3.11.1", "pillow": "12.3.0"}
    target = manifest_path(output, PAGE)
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(record))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("number", "manifest numbers"),
        ("input digest", "manifest inputs"),
        ("text", "Markdown numbers"),
        ("PNG", "Missing relative Wiki asset"),
        ("image digest", "file digest differs"),
    ],
)
def test_verify_rejects_numeric_and_file_drift(
    tmp_path: Path, projection: tuple[Sources, Bars], damage: str, message: str
) -> None:
    sources, bars = projection
    _write_bundle(tmp_path, sources, bars)
    verify_bundle(tmp_path, PAGE, sources, (bars,))
    md = tmp_path / f"{PAGE}.md"
    svg = tmp_path / "assets" / PAGE / f"{TABLE}.svg"
    manifest = manifest_path(tmp_path, PAGE)
    if damage == "text":
        assert "2.38" in md.read_text()
        md.write_text(md.read_text().replace("2.38", "2.37"))
    elif damage in {"number", "input digest"}:
        changed = json.loads(manifest.read_text())
        if damage == "number":
            assert changed["assets"][TABLE]["rows"][0]["values"][2] == "2.375"
            changed["assets"][TABLE]["rows"][0]["values"][2] = "2.374"
        else:
            changed["inputs"]["knowledge/runs/examples/cases.json"] = "f" * 64
        manifest.write_text(json.dumps(changed))
    elif damage == "PNG":
        svg.with_suffix(".png").unlink()
    else:
        svg.write_text(svg.read_text().replace("#fff", "white"))
    with pytest.raises(ValueError, match=message):
        verify_bundle(tmp_path, PAGE, sources, (bars,))


def test_offline_typst_contract_no_ambient_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(BINARY_ENV, raising=False)
    with pytest.raises(ValueError, match="missing from the offline cache"):
        resolve(offline=True, cache=tmp_path)
    monkeypatch.setenv(BINARY_ENV, str(tmp_path / "not-installed"))
    with pytest.raises(ValueError, match="explicit Typst binary"):
        resolve(offline=True, cache=tmp_path)


def test_typst_rejects_changed_release_archive_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(BINARY_ENV, raising=False)

    def wrong_download(destination: Path) -> None:
        destination.write_bytes(b"public test data")

    monkeypatch.setattr("tools.wiki_assets.typst_runner._download", wrong_download)
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        resolve(cache=tmp_path)
    assert not list(tmp_path.rglob("typst"))


def test_real_typst_cli_generate_and_verify(tmp_path: Path) -> None:
    typst = os.environ.get(BINARY_ENV)
    if not typst:
        pytest.skip(f"set {BINARY_ENV} to run with Typst")
    from tools.wiki_assets.__main__ import main

    common = [
        "--page",
        "person-model-comparison-methods",
        "--offline",
        "--output",
        str(tmp_path),
    ]
    assert main(["generate", *common, "--typst-bin", typst]) == 0
    assert main(["verify", *common]) == 0
    output = tmp_path / "assets" / "person-model-comparison-methods"
    assert len(list(output.glob("*.svg"))) == len(list(output.glob("*.png"))) == 2
    synopsis = (tmp_path / "person-model-comparison-methods.md").read_text()
    assert "6,844 画像" in synopsis
    assert (
        "| Top-down pose | 384 | B1 / B4 / B8 | ms/人 | B1 は先頭 300 crop |"
        in synopsis
    )


def test_public_guides_have_resolvable_local_links_and_no_private_identifiers() -> None:
    paths = (
        ROOT / "docs/wiki/README.md",
        ROOT / "docs/wiki/rendering-comparison.md",
        ROOT / "tools/wiki_assets/README.md",
    )
    for path in paths:
        assert verify_markdown(path) == set()
        refs = References(path.name)
        refs.feed(PARSER.render(path.read_text(encoding="utf-8")))
        for href in refs.links:
            url = urlsplit(href)
            if not url.scheme and not url.netloc and url.path:
                assert (path.parent / unquote(url.path)).exists(), href
    inventory = paths[1].read_text()
    assert inventory.count("| [") == 38
