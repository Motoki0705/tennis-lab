"""Keep the public README languages, navigation and asset presentation aligned."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SECTION_IDS = ("overview", "model-downloads", "run-locally", "license", "citation")
# Contracts agreed for the independently maintained installation/license pages.
INSTALLATION_ANCHORS = {
    "requirements",
    "clone-and-sync",
    "dino-cuda-extension",
    "required-assets",
    "preflight",
    "run-the-sample",
    "visualize",
    "rehearse-clean-install",
}
LICENSE_ANCHORS = {
    "code-license-scope",
    "third-party-code",
    "third-party-weights",
    "smpl-auxiliary-assets",
    "third-party-data",
    "published-artifacts",
    "published-artifacts-hosting",
    "weight-license-basis",
    "sample-dataset-terms",
    "hf-license-metadata",
    "pipeline-outputs",
}


@pytest.fixture(params=("README.md", "README.ja.md"))
def readme(request: pytest.FixtureRequest) -> tuple[str, str]:
    filename = str(request.param)
    return filename, (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")


def test_readme_has_language_hub_and_license_badges_in_three_centered_rows(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    header = document.partition("\n## ")[0]
    assert "<hr>" in header, f"{filename}: missing rule above the badges"
    badges = re.search(
        r'<div align="center">(.*?)</div>', header.split("<hr>", 1)[1], re.DOTALL
    )
    assert badges is not None, f"{filename}: missing centered badges"
    suffix = ".ja.md" if filename.endswith(".ja.md") else ".md"
    expected_rows = (
        ["README.md", "README.ja.md"],
        [
            "https://huggingface.co/Motoki098/tennis-lab-models",
            "https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample",
        ],
        [
            "LICENSE",
            f"LICENSES{suffix}#published-artifacts",
            f"LICENSES{suffix}#sample-dataset-terms",
        ],
    )
    rows = re.split(r"<br\s*/?>", badges.group(1))
    assert len(rows) == len(expected_rows)
    for row, destinations in zip(rows, expected_rows, strict=True):
        assert re.findall(r'href="([^"]+)"', row) == destinations
        sources = re.findall(r'<img\b[^>]*src="([^"]+)"', row)
        assert len(sources) == len(destinations)
        assert all(source.startswith("https://img.shields.io/") for source in sources)
        assert len(re.findall(r'alt="[^"]+"', row)) == len(destinations)


def test_readme_opens_with_the_centered_logo_and_name_header(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    header = "assets/branding/tennis-lab-header.png"
    above_rule = document.partition("<hr>")[0]
    # The header image already carries the name, so no separate title follows it.
    block = re.fullmatch(r'<div align="center">\s*<img\b([^>]*)>\s*</div>\s*', above_rule)
    assert block is not None, f"{filename}: header image block must precede the badges"
    attributes = dict(re.findall(r'(\w+)="([^"]*)"', block.group(1)))
    assert attributes == {"src": header, "width": "420", "alt": "Tennis-Lab"}
    assert (REPOSITORY_ROOT / header).is_file()


def test_readme_has_five_numbered_sections_and_shared_explicit_anchors(
    readme: tuple[str, str],
) -> None:
    _, document = readme
    assert re.findall(r"^## (\d+)\. ", document, re.MULTILINE) == list("12345")
    assert re.findall(r'<a id="([^"]+)"></a>', document) == list(SECTION_IDS)
    contents = document.split('<a id="overview"></a>', maxsplit=1)[0]
    assert re.findall(r"\]\(#([^)]+)\)", contents) == list(SECTION_IDS)


def test_readme_has_both_asset_table_marker_pairs_and_no_mermaid_or_hash_values(
    readme: tuple[str, str],
) -> None:
    _, document = readme
    for marker in ("checkpoint-assets", "sample-assets"):
        start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
        assert document.count(start) == document.count(end) == 1
        assert document.index(start) < document.index(end)
    assert "mermaid" not in document.lower()
    assert re.search(r"[0-9a-f]{64}", document, re.IGNORECASE) is None
    assert re.search(r"^¹ ", document, re.MULTILINE) is not None
    assert "`src.utils.model_weights`" in document
    assert "`src.utils.sample_dataset`" in document


def test_readme_local_links_use_the_right_language_and_agreed_anchor_contracts(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    suffix = ".ja.md" if filename.endswith(".ja.md") else ".md"
    contracts = {
        f"docs/installation/README{suffix}": INSTALLATION_ANCHORS,
        f"LICENSES{suffix}": LICENSE_ANCHORS,
    }
    localized_paths = {
        "docs/installation/README.md",
        "docs/installation/README.ja.md",
        "LICENSES.md",
        "LICENSES.ja.md",
    }
    targets = re.findall(r"\]\(([^)\s]+)\)", document)
    targets += re.findall(r'href="([^"]+)"', document)
    for target in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        if url.path in localized_paths:
            assert url.path in contracts, f"{filename}: wrong language in {target}"
            if url.fragment:
                assert url.fragment in contracts[url.path], target
        elif not url.path:
            assert url.fragment in SECTION_IDS, target
        else:
            assert (REPOSITORY_ROOT / url.path).is_file(), target
            assert not url.fragment, f"No anchor contract for {target}"


def test_readme_attributes_the_gif_directly_below_the_image(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    image = "assets/tennis_scene/multiview_reconstruction.gif)"
    assert document.count(image) == 1
    attribution = document.split(image, maxsplit=1)[1].strip().splitlines()[0]
    assert attribution.startswith("© 2026 Motoki0705")
    assert "CC BY-NC-ND 4.0" in attribution
    suffix = ".ja.md" if filename.endswith(".ja.md") else ".md"
    assert f"(LICENSES{suffix}#sample-dataset-terms)" in attribution


def test_readme_preserves_the_repository_bibtex(readme: tuple[str, str]) -> None:
    _, document = readme
    assert document.count("```bibtex\n") == 1
    citation = document.split("```bibtex\n", maxsplit=1)[1].split("```", maxsplit=1)[0]
    assert citation == (
        "@software{tennis_lab,\n"
        "  author = {{Tennis-Lab contributors}},\n"
        "  title = {Tennis-Lab},\n"
        "  url = {https://github.com/Motoki0705/tennis-lab}\n"
        "}\n"
    )
