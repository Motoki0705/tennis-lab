"""Keep the public README languages, navigation and asset presentation aligned."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.tennis_scene.installation_assets import (
    PUBLIC_DOCUMENT_LANGUAGES,
    public_document_language_from_filename,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
README_DOCUMENTS = tuple(
    f"README{locale.suffix}" for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
)
LOCALIZED_PATHS = {
    f"{stem}{locale.suffix}": f"{stem}.md"
    for stem in ("docs/installation/README", "LICENSES")
    for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
}
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


def _read_readme_documents(root: Path) -> dict[str, str]:
    return {
        filename: (root / filename).read_text(encoding="utf-8")
        for filename in README_DOCUMENTS
    }


@pytest.fixture(params=README_DOCUMENTS)
def readme(request: pytest.FixtureRequest) -> tuple[str, str]:
    filename = str(request.param)
    return filename, _read_readme_documents(REPOSITORY_ROOT)[filename]


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
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    expected_rows = (
        list(README_DOCUMENTS),
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
    assert re.findall(r'alt="([^"]+)"', rows[0]) == [
        locale.label for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
    ]
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
    block = re.fullmatch(
        r'<div align="center">\s*<img\b([^>]*)>\s*</div>\s*', above_rule
    )
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


def _assert_readme_local_links(filename: str, document: str, root: Path) -> None:
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    contracts = {
        f"docs/installation/README{suffix}": INSTALLATION_ANCHORS,
        f"LICENSES{suffix}": LICENSE_ANCHORS,
    }
    targets = re.findall(r"\]\(([^)\s]+)\)", document)
    targets += re.findall(r'href="([^"]+)"', document)
    for target in targets:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        path, fragment = unquote(url.path), unquote(url.fragment)
        if path in LOCALIZED_PATHS:
            assert path in contracts, f"{filename}: wrong language in {target}"
            if fragment:
                assert fragment in contracts[path], target
        elif not path:
            assert fragment in SECTION_IDS, target
        else:
            assert not fragment, f"No anchor contract for {target}"
        destination = root / path if path else root / filename
        assert destination.is_file(), f"{filename}: missing file for {target}"
        if fragment:
            assert fragment in re.findall(
                r'<a id="([^"]+)"></a>', destination.read_text(encoding="utf-8")
            ), f"{filename}: missing anchor for {target}"


def test_readme_local_links_use_the_right_language_and_agreed_anchor_contracts(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    _assert_readme_local_links(filename, document, REPOSITORY_ROOT)


def test_readme_attributes_the_gif_directly_below_the_image(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    image = "assets/tennis_scene/multiview_reconstruction.gif)"
    assert document.count(image) == 1
    attribution = document.split(image, maxsplit=1)[1].strip().splitlines()[0]
    assert attribution.startswith("© 2026 Motoki0705")
    assert "CC BY-NC-ND 4.0" in attribution
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
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


def _normalized_references(document: str) -> Counter[str]:
    targets = []
    for target in re.findall(r"\]\(([^)\s]+)\)", document):
        url = urlsplit(target)
        if not url.scheme and not url.netloc:
            canonical = LOCALIZED_PATHS.get(unquote(url.path))
            if canonical is not None:
                target = url._replace(path=canonical).geturl()
        targets.append(target)
    return Counter(targets)


@pytest.mark.parametrize("filename", README_DOCUMENTS[1:])
def test_readme_translations_preserve_section_references_and_identifiers(
    filename: str,
) -> None:
    documents = _read_readme_documents(REPOSITORY_ROOT)
    english = documents[README_DOCUMENTS[0]]
    translated = documents[filename]
    anchor = re.compile(r'<a id="([^"]+)"></a>')
    assert anchor.findall(english) == anchor.findall(translated)
    for section_id, en, localized in zip(
        anchor.split(english)[1::2],
        anchor.split(english)[2::2],
        anchor.split(translated)[2::2],
        strict=True,
    ):
        context = f"{filename}: {section_id}"
        assert re.findall(r"^#{2,3} ", en, re.MULTILINE) == re.findall(
            r"^#{2,3} ", localized, re.MULTILINE
        ), f"{context}: heading levels"
        assert _normalized_references(en) == _normalized_references(localized), (
            f"{context}: references"
        )
        assert Counter(re.findall(r"`([^`\n]+)`", en)) == Counter(
            re.findall(r"`([^`\n]+)`", localized)
        ), f"{context}: inline code"
        if public_document_language_from_filename(filename) == "zh-CN":
            assert Counter(re.findall(r"[0-9]+(?:[.,][0-9]+)*", en)) == Counter(
                re.findall(r"[0-9]+(?:[.,][0-9]+)*", localized)
            ), f"{context}: numeric values"


@pytest.mark.parametrize("missing", README_DOCUMENTS)
def test_missing_readme_translation_is_detected(tmp_path: Path, missing: str) -> None:
    for filename, text in _read_readme_documents(REPOSITORY_ROOT).items():
        if filename != missing:
            (tmp_path / filename).write_text(text, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        _read_readme_documents(tmp_path)


@pytest.mark.parametrize("filename", README_DOCUMENTS)
@pytest.mark.parametrize(
    ("stem", "anchor"),
    (
        ("docs/installation/README", "required-assets"),
        ("LICENSES", "published-artifacts"),
    ),
)
def test_readme_links_reject_other_document_languages(
    filename: str, stem: str, anchor: str
) -> None:
    language = public_document_language_from_filename(filename)
    for target_language, locale in PUBLIC_DOCUMENT_LANGUAGES.items():
        text = f"[document]({stem}{locale.suffix}#{anchor})"
        if language == target_language:
            _assert_readme_local_links(filename, text, REPOSITORY_ROOT)
        else:
            with pytest.raises(AssertionError, match="wrong language"):
                _assert_readme_local_links(filename, text, REPOSITORY_ROOT)


@pytest.mark.parametrize("filename", README_DOCUMENTS)
@pytest.mark.parametrize(
    ("stem", "anchor"),
    (
        ("docs/installation/README", "required-assets"),
        ("LICENSES", "published-artifacts"),
    ),
)
@pytest.mark.parametrize("fault", ("missing file", "missing anchor"))
def test_readme_links_require_existing_translated_documents_and_anchors(
    tmp_path: Path, filename: str, stem: str, anchor: str, fault: str
) -> None:
    language = public_document_language_from_filename(filename)
    destination = f"{stem}{PUBLIC_DOCUMENT_LANGUAGES[language].suffix}"
    if fault == "missing anchor":
        path = tmp_path / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('<a id="another-section"></a>', encoding="utf-8")
    with pytest.raises(AssertionError, match=fault):
        _assert_readme_local_links(
            filename, f"[document]({destination}#{anchor})", tmp_path
        )
