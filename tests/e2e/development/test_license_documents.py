"""License translations preserve public inventories, machine blocks and links."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.tennis_scene.installation_assets import (
    PUBLIC_DOCUMENT_LANGUAGES,
    public_document_language_from_filename,
)
from src.utils.paths import PROJECT_ROOT
from tests.support.documents import assert_local_links, assert_sections, links

LICENSE_DOCUMENTS = tuple(
    f"LICENSES{locale.suffix}" for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
)
LOCALIZED_DOCUMENTS = {
    f"{stem}{locale.suffix}": f"{stem}.md"
    for stem in ("README", "docs/installation/README", "LICENSES")
    for locale in PUBLIC_DOCUMENT_LANGUAGES.values()
}
REQUIRED_ANCHORS = {
    "code-license-scope", "third-party-code", "third-party-weights",
    "smpl-auxiliary-assets", "third-party-data", "published-artifacts",
    "published-artifacts-hosting", "weight-license-basis", "sample-dataset-terms",
    "hf-license-metadata", "pipeline-outputs",
}


def _read_license_documents(root: Path) -> dict[str, str]:
    return {
        name: (root / name).read_text(encoding="utf-8") for name in LICENSE_DOCUMENTS
    }


def _normalized_links(text: str) -> set[str]:
    targets = set()
    for target in links(text):
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc:
            canonical = LOCALIZED_DOCUMENTS.get(unquote(parsed.path))
            if canonical is not None:
                target = parsed._replace(path=canonical).geturl()
        targets.add(target)
    return targets


def _table_references(text: str) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    references = set()
    for row in re.findall(r"^\|.*\|$", text, re.MULTILINE):
        identifiers = tuple(sorted(set(re.findall(r"`([^`\n]+)`", row))))
        targets = tuple(sorted(_normalized_links(row)))
        if identifiers or targets:
            references.add((identifiers, targets))
    return references


def _assert_license_translation(english: str, translated: str, name: str) -> None:
    for text in (english, translated):
        assert_sections(text, REQUIRED_ANCHORS, context=name)
    assert _normalized_links(english) == _normalized_links(translated), f"{name}: links"
    # Public source inventory and legal/machine-readable blocks have meaning;
    # prose, heading levels, table order and repeated references do not.
    inventory = r"`((?:src/|third_party/)[^`\n]+)`"
    assert set(re.findall(inventory, english)) == set(re.findall(inventory, translated)), (
        f"{name}: public source inventory"
    )
    assert _table_references(english) == _table_references(translated), f"{name}: table references"
    assert set(re.findall(r"```[\s\S]*?```", english)) == set(
        re.findall(r"```[\s\S]*?```", translated)
    ), f"{name}: fenced blocks"


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
def test_license_translations_preserve_structure_and_references(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    _assert_license_translation(documents[LICENSE_DOCUMENTS[0]], documents[name], name)


def _assert_local_link_languages(name: str, text: str) -> None:
    language = public_document_language_from_filename(name)
    suffix = PUBLIC_DOCUMENT_LANGUAGES[language].suffix
    for target in links(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        path = unquote(parsed.path)
        canonical = LOCALIZED_DOCUMENTS.get(path)
        if canonical is not None:
            expected = f"{canonical.removesuffix('.md')}{suffix}"
            assert path == expected, f"{name}: wrong document language: {target}"


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS)
def test_license_document_links_resolve_without_checksum_values(name: str) -> None:
    document = PROJECT_ROOT / name
    text = document.read_text(encoding="utf-8")
    introduction, _, body = text.partition("<a id=")
    assert set(LICENSE_DOCUMENTS) <= links(introduction), f"{name}: language links"
    _assert_local_link_languages(name, body)
    assert not re.search(r"\b[0-9a-fA-F]{64}\b", text)
    assert_local_links(document, text)


def test_repository_links_to_license_documents_use_existing_anchors() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=PROJECT_ROOT
    ).decode()
    checked = 0
    for relative in tracked.split("\0"):
        path = Path(relative)
        if (
            not relative
            or relative in LOCALIZED_DOCUMENTS
            or path.parts[0] == "knowledge"
            or path.parts[:2] == ("docs", "installation")
        ):
            continue
        document = PROJECT_ROOT / path
        for target in links(document.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme or Path(parsed.path).name not in LICENSE_DOCUMENTS:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            assert destination.parent == PROJECT_ROOT, (relative, target)
            assert_local_links(document, f"[license]({target})")
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("missing", LICENSE_DOCUMENTS)
def test_missing_license_translation_is_detected(tmp_path: Path, missing: str) -> None:
    for name, text in _read_license_documents(PROJECT_ROOT).items():
        if name != missing:
            (tmp_path / name).write_text(text, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        _read_license_documents(tmp_path)


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
def test_missing_license_section_is_detected(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    text = documents[name]
    start = text.index('<a id="sample-dataset-terms"></a>')
    end = text.index('<a id="hf-license-metadata"></a>')
    with pytest.raises(AssertionError, match="section"):
        _assert_license_translation(documents[LICENSE_DOCUMENTS[0]], text[:start] + text[end:], name)


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
@pytest.mark.parametrize(
    ("original", "replacement", "failure"),
    (
        ("](#pipeline-outputs)", "](#nonexistent-output)", "links"),
        ("All rights reserved.", "Some rights reserved.", "fenced blocks"),
        ("license: other", "license: apache-2.0", "fenced blocks"),
    ),
)
def test_license_translation_drift_is_detected(
    name: str, original: str, replacement: str, failure: str
) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    assert original in documents[name]
    changed = documents[name].replace(original, replacement, 1)
    with pytest.raises(AssertionError, match=failure):
        _assert_license_translation(documents[LICENSE_DOCUMENTS[0]], changed, name)


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
def test_license_translation_accepts_editorial_changes(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    rows = documents[name].replace("\n## ", "\n### ").splitlines(keepends=True)
    start = next(i for i, row in enumerate(rows) if row.startswith("| `src/"))
    rows[start], rows[start + 1] = rows[start + 1], rows[start]
    _assert_license_translation(
        documents[LICENSE_DOCUMENTS[0]], "".join(rows) + "\n追加说明 / 補足: 三维 (3D).\n", name
    )


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
def test_license_translation_rejects_reassigned_row_references(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    rows = documents[name].splitlines(keepends=True)
    source_rows = [i for i, row in enumerate(rows) if row.startswith("| `src/")]
    first, second = source_rows[:2]
    first_target, second_target = (sorted(links(rows[i]))[0] for i in (first, second))
    assert first_target != second_target
    rows[first] = rows[first].replace(first_target, second_target)
    rows[second] = rows[second].replace(second_target, first_target)
    changed = "".join(rows)
    assert _normalized_links(changed) == _normalized_links(documents[name])
    with pytest.raises(AssertionError, match="table references"):
        _assert_license_translation(documents[LICENSE_DOCUMENTS[0]], changed, name)


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS)
@pytest.mark.parametrize("stem", sorted(set(LOCALIZED_DOCUMENTS.values())))
def test_localized_license_links_require_the_document_language(name: str, stem: str) -> None:
    for locale in PUBLIC_DOCUMENT_LANGUAGES.values():
        target = f"{stem.removesuffix('.md')}{locale.suffix}#shared-anchor"
        text = f"[document]({target})"
        if name == f"LICENSES{locale.suffix}":
            _assert_local_link_languages(name, text)
        else:
            with pytest.raises(AssertionError, match="wrong document language"):
                _assert_local_link_languages(name, text)


def test_license_reference_comparison_preserves_anchors_and_external_urls() -> None:
    assert _normalized_links("[guide](docs/installation/README.zh-CN.md#required-assets)") == (
        _normalized_links("[guide](docs/installation/README.md#required-assets)")
    )
    assert _normalized_links("[guide](docs/installation/README.zh-CN.md#preflight)") != (
        _normalized_links("[guide](docs/installation/README.md#required-assets)")
    )
    assert _normalized_links("[upstream](https://example.org/README.zh-CN.md)") != (
        _normalized_links("[upstream](https://example.org/README.md)")
    )
