"""Keep public README navigation, identifiers and required sections usable."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.tennis_scene.installation_assets import (
    PUBLIC_DOCUMENT_LANGUAGES,
    public_document_language_from_filename,
)
from tests.support.documents import assert_local_links, assert_sections, links

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
README_DOCUMENTS = tuple(
    f"README{locale.suffix}" for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
)
LOCALIZED_PATHS = {
    f"{stem}{locale.suffix}": f"{stem}.md"
    for stem in ("docs/installation/README", "LICENSES")
    for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
}
SECTION_IDS = {"overview", "model-downloads", "run-locally", "license", "citation"}
INSTALLATION_ANCHORS = {
    "requirements", "clone-and-sync", "dino-cuda-extension", "required-assets",
    "preflight", "run-the-sample", "visualize", "rehearse-clean-install",
}
LICENSE_ANCHORS = {
    "code-license-scope", "third-party-code", "third-party-weights",
    "smpl-auxiliary-assets", "third-party-data", "published-artifacts",
    "published-artifacts-hosting", "weight-license-basis", "sample-dataset-terms",
    "hf-license-metadata", "pipeline-outputs",
}


def _read_readme_documents(root: Path) -> dict[str, str]:
    return {
        filename: (root / filename).read_text(encoding="utf-8")
        for filename in README_DOCUMENTS
    }


@pytest.fixture(params=README_DOCUMENTS)
def readme(request: pytest.FixtureRequest) -> tuple[str, str]:
    filename = str(request.param)
    return filename, (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")


def test_readme_has_public_navigation_assets_sections_and_cli_names(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    assert set(README_DOCUMENTS) | {
        "https://huggingface.co/Motoki098/tennis-lab-models",
        "https://huggingface.co/datasets/Motoki098/tennis-lab-multiview-sample",
        "LICENSE",
        f"LICENSES{suffix}#published-artifacts",
        f"LICENSES{suffix}#sample-dataset-terms",
        f"docs/installation/README{suffix}",
        f"docs/installation/README{suffix}#required-assets",
        f"docs/installation/README{suffix}#preflight",
        "src/tennis_scene/README.md",
        "assets/branding/tennis-lab-header.png",
        "assets/tennis_scene/multiview_reconstruction.gif",
    } <= links(document), filename
    assert_sections(document, SECTION_IDS, context=filename)
    assert {f"#{section}" for section in SECTION_IDS} <= links(document), filename
    for public_name in ("src.utils.model_weights", "src.utils.sample_dataset"):
        assert f"`{public_name}`" in document, (filename, public_name)
    assert re.search(r"[0-9a-f]{64}", document, re.IGNORECASE) is None
    # Catalog-generated asset tables are checked in test_installation_assets.


def _assert_readme_local_links(filename: str, document: str, root: Path) -> None:
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    contracts = {
        f"docs/installation/README{suffix}": INSTALLATION_ANCHORS,
        f"LICENSES{suffix}": LICENSE_ANCHORS,
    }
    for target in links(document):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        path, fragment = unquote(parsed.path), unquote(parsed.fragment)
        if path in LOCALIZED_PATHS:
            assert path in contracts, f"{filename}: wrong language in {target}"
            if fragment:
                assert fragment in contracts[path], target
    assert_local_links(root / filename, document)


def test_readme_local_links_use_the_right_language_and_agreed_anchor_contracts(
    readme: tuple[str, str],
) -> None:
    filename, document = readme
    _assert_readme_local_links(filename, document, REPOSITORY_ROOT)


def test_readme_attributes_the_demonstration(readme: tuple[str, str]) -> None:
    filename, document = readme
    image = "assets/tennis_scene/multiview_reconstruction.gif)"
    assert image in document
    attribution = document.split(image, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert re.search(r"©\s+\d{4}\s+Motoki0705", attribution)
    assert "CC BY-NC-ND 4.0" in attribution
    language = public_document_language_from_filename(filename)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    assert f"LICENSES{suffix}#sample-dataset-terms" in links(attribution)


def test_readme_has_repository_citation_fields(readme: tuple[str, str]) -> None:
    _, document = readme
    match = re.search(r"```bibtex\s*\n(.*?)```", document, re.DOTALL)
    assert match is not None
    citation = match[1]
    assert re.search(r"@software\s*\{\s*tennis_lab\s*,", citation, re.IGNORECASE)
    fields = dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", citation, re.MULTILINE))
    assert {
        "author": "Tennis-Lab contributors",
        "title": "Tennis-Lab",
        "url": "https://github.com/Motoki0705/tennis-lab",
    }.items() <= {key: value.strip("{} ") for key, value in fields.items()}.items()


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
    (("docs/installation/README", "required-assets"), ("LICENSES", "published-artifacts")),
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
    (("docs/installation/README", "required-assets"), ("LICENSES", "published-artifacts")),
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
        _assert_readme_local_links(filename, f"[document]({destination}#{anchor})", tmp_path)
