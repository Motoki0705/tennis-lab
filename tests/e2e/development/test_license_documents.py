"""License translations retain matching inventories and navigable references."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.utils.paths import PROJECT_ROOT

LANGUAGES = (("English", ".md"), ("日本語", ".ja.md"), ("简体中文", ".zh-CN.md"))
LICENSE_DOCUMENTS = tuple(f"LICENSES{suffix}" for _, suffix in LANGUAGES)
LOCALIZED_DOCUMENTS = {
    f"{stem}{suffix}": f"{stem}.md"
    for stem in ("README", "docs/installation/README", "LICENSES")
    for _, suffix in LANGUAGES
}
REQUIRED_ANCHORS = {
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
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _anchors(text: str) -> set[str]:
    headings = re.findall(r"^#+ (.+)$", text, re.MULTILINE)
    return set(ANCHOR.findall(text)) | {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in headings
    }


def _read_license_documents(root: Path) -> dict[str, str]:
    # Read every required language explicitly: missing translations must fail.
    return {
        name: (root / name).read_text(encoding="utf-8") for name in LICENSE_DOCUMENTS
    }


def _normalized_links(text: str) -> Counter[str]:
    targets = []
    for target in LINK.findall(text):
        parsed = urlsplit(target)
        if not parsed.scheme and not parsed.netloc:
            canonical = LOCALIZED_DOCUMENTS.get(unquote(parsed.path))
            if canonical is not None:
                target = parsed._replace(path=canonical).geturl()
        targets.append(target)
    return Counter(targets)


def _assert_license_translation(english: str, translated: str, name: str) -> None:
    english_ids = ANCHOR.findall(english)
    assert english_ids == ANCHOR.findall(translated), f"{name}: anchors"
    assert len(set(english_ids)) == len(english_ids)
    assert set(english_ids) >= REQUIRED_ANCHORS
    # Every section has a stable ID, including the additional self-link targets.
    for text in (english, translated):
        assert len(re.findall(r"^# ", text, re.MULTILINE)) == 1
        assert len(re.findall(r"^#{2,3} ", text, re.MULTILINE)) == len(english_ids), (
            f"{name}: heading count"
        )
        assert len(re.findall(r'<a id="[^"]+"></a>\n\n#{2,3} ', text)) == len(
            english_ids
        ), f"{name}: anchored headings"
    english_sections = ANCHOR.split(english)[2::2]
    translated_sections = ANCHOR.split(translated)[2::2]
    for anchor, en, localized in zip(
        english_ids, english_sections, translated_sections, strict=True
    ):
        context = f"{name}: {anchor}"
        assert re.findall(r"^#{2,3} ", en, re.MULTILINE) == re.findall(
            r"^#{2,3} ", localized, re.MULTILINE
        ), f"{context}: heading levels"
        assert _normalized_links(en) == _normalized_links(localized), (
            f"{context}: links"
        )
        assert Counter(re.findall(r"`([^`\n]+)`", en)) == Counter(
            re.findall(r"`([^`\n]+)`", localized)
        ), f"{context}: inline code"
        en_rows = re.findall(r"^\|.*\|$", en, re.MULTILINE)
        localized_rows = re.findall(r"^\|.*\|$", localized, re.MULTILINE)
        assert len(en_rows) == len(localized_rows), f"{context}: table rows"
        for en_row, localized_row in zip(en_rows, localized_rows, strict=True):
            assert en_row.count("|") == localized_row.count("|"), (
                f"{context}: table columns"
            )
            assert _normalized_links(en_row) == _normalized_links(localized_row), (
                f"{context}: table links"
            )
            assert Counter(re.findall(r"`([^`\n]+)`", en_row)) == Counter(
                re.findall(r"`([^`\n]+)`", localized_row)
            ), f"{context}: table inline code"
    # Keep full license text and machine-readable metadata identical.
    assert re.findall(r"```[\s\S]*?```", english) == re.findall(
        r"```[\s\S]*?```", translated
    ), f"{name}: fenced blocks"


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
def test_license_translations_preserve_structure_and_references(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    _assert_license_translation(documents[LICENSE_DOCUMENTS[0]], documents[name], name)


def _assert_local_link_languages(name: str, text: str) -> None:
    suffix = next(suffix for _, suffix in LANGUAGES if name == f"LICENSES{suffix}")
    for target in LINK.findall(text):
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
    hub = " | ".join(f"[{label}](LICENSES{suffix})" for label, suffix in LANGUAGES)
    assert hub in introduction.splitlines(), f"{name}: language links"
    _assert_local_link_languages(name, body)
    assert not re.search(r"\b[0-9a-fA-F]{64}\b", text)
    for target in LINK.findall(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        destination = (
            document.parent / unquote(parsed.path) if parsed.path else document
        ).resolve()
        assert destination.is_file(), (name, target)
        if parsed.fragment:
            assert unquote(parsed.fragment) in _anchors(
                destination.read_text(encoding="utf-8")
            ), (
                name,
                target,
            )


def test_repository_links_to_license_documents_use_existing_anchors() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=PROJECT_ROOT
    ).decode()
    checked = 0
    for relative in tracked.split("\0"):
        path = Path(relative)
        # These documents belong to the parallel README/installation tasks;
        # knowledge records intentionally retain their historical references.
        if (
            not relative
            or relative in LOCALIZED_DOCUMENTS
            or path.parts[0] == "knowledge"
            or path.parts[:2] == ("docs", "installation")
        ):
            continue
        document = PROJECT_ROOT / path
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            parsed = urlsplit(target)
            if parsed.scheme or Path(parsed.path).name not in LICENSE_DOCUMENTS:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            assert destination.parent == PROJECT_ROOT, (relative, target)
            assert destination.is_file(), (relative, target)
            assert unquote(parsed.fragment) in set(
                ANCHOR.findall(destination.read_text(encoding="utf-8"))
            ), (relative, target)
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
    with pytest.raises(AssertionError, match="anchors"):
        _assert_license_translation(
            documents[LICENSE_DOCUMENTS[0]], text[:start] + text[end:], name
        )


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS[1:])
@pytest.mark.parametrize(
    ("original", "replacement", "failure"),
    (
        ("\n## ", "\n### ", "heading levels"),
        ("](#pipeline-outputs)", "](#nonexistent-output)", "links"),
        ("`video_000/clip_000`", "`video_000/clip_001`", "inline code"),
        ("| --- | --- | --- |\n", "", "table rows"),
        ("| --- | --- |", "| --- |", "table columns"),
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
def test_reordered_license_table_rows_are_detected(name: str) -> None:
    documents = _read_license_documents(PROJECT_ROOT)
    rows = documents[name].splitlines(keepends=True)
    start = next(i for i, row in enumerate(rows) if row.startswith("| `src/"))
    rows[start], rows[start + 1] = rows[start + 1], rows[start]
    with pytest.raises(AssertionError, match="table links"):
        _assert_license_translation(
            documents[LICENSE_DOCUMENTS[0]], "".join(rows), name
        )


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS)
@pytest.mark.parametrize("stem", sorted(set(LOCALIZED_DOCUMENTS.values())))
def test_localized_license_links_require_the_document_language(
    name: str, stem: str
) -> None:
    for _, suffix in LANGUAGES:
        target = f"{stem.removesuffix('.md')}{suffix}#shared-anchor"
        text = f"[document]({target})"
        if name == f"LICENSES{suffix}":
            _assert_local_link_languages(name, text)
        else:
            with pytest.raises(AssertionError, match="wrong document language"):
                _assert_local_link_languages(name, text)


def test_license_reference_comparison_preserves_anchors_and_external_urls() -> None:
    assert _normalized_links(
        "[guide](docs/installation/README.zh-CN.md#required-assets)"
    ) == (_normalized_links("[guide](docs/installation/README.md#required-assets)"))
    assert _normalized_links(
        "[guide](docs/installation/README.zh-CN.md#preflight)"
    ) != (_normalized_links("[guide](docs/installation/README.md#required-assets)"))
    assert _normalized_links("[upstream](https://example.org/README.zh-CN.md)") != (
        _normalized_links("[upstream](https://example.org/README.md)")
    )
