"""License translations retain matching inventories and navigable references."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from src.utils.paths import PROJECT_ROOT

LICENSE_DOCUMENTS = ("LICENSES.md", "LICENSES.ja.md")
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


def test_license_translations_preserve_structure_and_references() -> None:
    english, japanese = (
        (PROJECT_ROOT / name).read_text() for name in LICENSE_DOCUMENTS
    )
    english_ids, japanese_ids = ANCHOR.findall(english), ANCHOR.findall(japanese)
    assert english_ids == japanese_ids
    assert len(set(english_ids)) == len(english_ids)
    assert set(english_ids) >= REQUIRED_ANCHORS
    # Every section has a stable ID, including the additional self-link targets.
    for text in (english, japanese):
        assert len(re.findall(r"^#{2,3} ", text, re.MULTILINE)) == len(english_ids)
        assert len(re.findall(r'<a id="[^"]+"></a>\n\n#{2,3} ', text)) == len(
            english_ids
        )
    english_sections = ANCHOR.split(english)[2::2]
    japanese_sections = ANCHOR.split(japanese)[2::2]
    for anchor, en, ja in zip(
        english_ids, english_sections, japanese_sections, strict=True
    ):
        assert re.findall(r"^#{2,3} ", en, re.MULTILINE) == re.findall(
            r"^#{2,3} ", ja, re.MULTILINE
        ), anchor
        assert Counter(LINK.findall(en)) == Counter(LINK.findall(ja)), anchor
        assert Counter(re.findall(r"`([^`\n]+)`", en)) == Counter(
            re.findall(r"`([^`\n]+)`", ja)
        ), anchor
        en_rows = re.findall(r"^\|.*\|$", en, re.MULTILINE)
        ja_rows = re.findall(r"^\|.*\|$", ja, re.MULTILINE)
        assert len(en_rows) == len(ja_rows), anchor
        for en_row, ja_row in zip(en_rows, ja_rows, strict=True):
            assert en_row.count("|") == ja_row.count("|"), anchor
            assert Counter(LINK.findall(en_row)) == Counter(LINK.findall(ja_row)), (
                anchor,
                en_row,
                ja_row,
            )
    # Keep full license text and machine-readable metadata identical.
    assert re.findall(r"```[\s\S]*?```", english) == re.findall(
        r"```[\s\S]*?```", japanese
    )


@pytest.mark.parametrize("name", LICENSE_DOCUMENTS)
def test_license_document_links_resolve_without_checksum_values(name: str) -> None:
    document = PROJECT_ROOT / name
    text = document.read_text()
    introduction = ANCHOR.split(text)[0]
    for language in LICENSE_DOCUMENTS:
        assert language in LINK.findall(introduction)
    assert not re.search(r"\b[0-9a-fA-F]{64}\b", text)
    for target in LINK.findall(text):
        parsed = urlsplit(target)
        if parsed.scheme:
            continue
        destination = (
            document.parent / unquote(parsed.path) if parsed.path else document
        ).resolve()
        assert destination.exists(), (name, target)
        if parsed.fragment:
            assert unquote(parsed.fragment) in _anchors(destination.read_text()), (
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
            or relative in (*LICENSE_DOCUMENTS, "README.md", "README.ja.md")
            or path.parts[0] == "knowledge"
            or path.parts[:2] == ("docs", "installation")
        ):
            continue
        document = PROJECT_ROOT / path
        for target in LINK.findall(document.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or Path(parsed.path).name not in LICENSE_DOCUMENTS:
                continue
            destination = (document.parent / unquote(parsed.path)).resolve()
            assert destination.parent == PROJECT_ROOT, (relative, target)
            assert destination.is_file(), (relative, target)
            assert unquote(parsed.fragment) in set(
                ANCHOR.findall(destination.read_text())
            ), (relative, target)
            checked += 1
    assert checked > 0
