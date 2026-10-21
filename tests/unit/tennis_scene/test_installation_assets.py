"""Keep localized asset tables in every tracked Markdown consumer authoritative."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from src.tennis_scene import installation_assets
from src.tennis_scene.installation_assets import (
    INSTALLATION_ASSETS,
    AssetTableLanguage,
    asset_table,
    sample_asset_table,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.sample_dataset import DATASET_REPO, PUBLISHED_FILES, PublishedSampleFile

# Other tracked Markdown with either marker is also checked.
REQUIRED_ASSET_TABLE_DOCUMENTS = (
    # Add one consumer per line; each required document must contain both tables.
    "docs/installation/README.md",
)
TABLE_RENDERERS = {
    "checkpoint-assets": asset_table,
    "sample-assets": sample_asset_table,
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _table_block(marker: str, table: str) -> str:
    return f"<!-- {marker}:start -->\n{table}\n<!-- {marker}:end -->"


def _assert_markdown_asset_tables(
    root: Path, required_documents: tuple[str, ...]
) -> None:
    tracked = _git(root, "ls-files", "-z", "--", "*.md").split("\0")
    tracked = [filename for filename in tracked if filename]
    problems = [
        f"{filename}: required asset-table document is not tracked Markdown"
        for filename in sorted(set(required_documents) - set(tracked))
    ]
    for filename in tracked:
        path = root / filename
        if not path.is_file():
            problems.append(f"{filename}: tracked Markdown file is missing")
            continue
        document = path.read_text(encoding="utf-8")
        language: AssetTableLanguage = "ja" if filename.endswith(".ja.md") else "en"
        for marker, render in TABLE_RENDERERS.items():
            start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
            if (
                filename not in required_documents
                and start not in document
                and end not in document
            ):
                continue
            expected = _table_block(marker, render(language))
            if document.count(start) != 1 or document.count(end) != 1:
                problems.append(
                    f"{filename}: expected exactly one start/end pair for {marker}.\n"
                    f"Paste this expected block:\n{expected}"
                )
                continue
            actual = document[document.index(start) : document.index(end) + len(end)]
            if actual != expected:
                problems.append(
                    f"{filename}: {marker} differs from the {language} catalog.\n"
                    f"Paste this expected block:\n{expected}"
                )
    if problems:
        # Preserve the generated Markdown verbatim for copying into each file.
        raise AssertionError("\n\n".join(problems))


@pytest.mark.parametrize(
    ("language", "header", "source_label"),
    [
        (
            "en",
            "| Asset | Destination (relative to `ckpt/`) | Source | License / terms |",
            "Source",
        ),
        (
            "ja",
            "| 資産 | 配置先（`ckpt/`相対） | 入手先 | ライセンス・条件 |",
            "配布元",
        ),
    ],
)
def test_checkpoint_table_has_all_assets_and_localized_terms_without_hashes(
    language: AssetTableLanguage,
    header: str,
    source_label: str,
) -> None:
    table = asset_table(language)
    lines = table.splitlines()
    assert lines[0] == header
    assert len(lines) == len(INSTALLATION_ASSETS) + 2
    assert all(len(line.split("|")) == 6 for line in lines)
    for asset, line in zip(INSTALLATION_ASSETS, lines[2:], strict=True):
        terms = asset.license_en if language == "en" else asset.license_ja
        assert f"| {asset.name} | `{asset.filename}` |" in line
        assert f"[{source_label}]({asset.source})" in line
        assert line.endswith(f"| {terms} |")
    assert "SHA" not in table
    assert re.search(r"[0-9a-f]{64}", table) is None
    if language == "en":
        assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", table) is None


def test_license_footnotes_identify_the_same_assets_in_both_languages() -> None:
    marked_assets = []
    for asset in INSTALLATION_ASSETS:
        assert ("¹" in asset.license_en) == ("¹" in asset.license_ja)
        if "¹" in asset.license_en:
            marked_assets.append(asset.name)
    assert marked_assets == ["detector", "vitpose", "hmr2", "tracking_encoder"]


@pytest.mark.parametrize(
    ("language", "header", "source_label"),
    [
        ("en", "| Destination | Source / terms |", "Published dataset"),
        ("ja", "| 配置先 | 入手先・条件 |", "公開Dataset"),
    ],
)
def test_sample_table_uses_the_acquisition_manifest_without_hashes(
    language: AssetTableLanguage,
    header: str,
    source_label: str,
) -> None:
    table = sample_asset_table(language)
    lines = table.splitlines()
    assert lines[0] == header
    assert len(lines) == len(PUBLISHED_FILES) + 2
    assert all(len(line.split("|")) == 4 for line in lines)
    for sample, line in zip(PUBLISHED_FILES, lines[2:], strict=True):
        assert f"`data/samples/{sample.filename}`" in line
        assert (
            f"[{source_label}](https://huggingface.co/datasets/{DATASET_REPO})" in line
        )
        assert line.endswith("/ CC BY-NC-ND 4.0 |")
    assert "SHA" not in table
    assert re.search(r"[0-9a-f]{64}", table) is None


def test_sample_table_reads_the_manifest_instead_of_a_separate_file_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        installation_assets,
        "PUBLISHED_FILES",
        (PublishedSampleFile("added.json", "0" * 64),),
    )
    table = sample_asset_table("en")
    assert "`data/samples/added.json`" in table
    assert "cam0.mp4" not in table
    assert len(table.splitlines()) == 3


@pytest.mark.parametrize("render", TABLE_RENDERERS.values())
def test_table_rejects_unsupported_language(
    render: Callable[[AssetTableLanguage], str],
) -> None:
    with pytest.raises(ValueError, match="Unsupported asset table language"):
        render(cast(AssetTableLanguage, "fr"))


def test_tracked_markdown_asset_tables_match_authoritative_catalogs() -> None:
    _assert_markdown_asset_tables(PROJECT_ROOT, REQUIRED_ASSET_TABLE_DOCUMENTS)


@pytest.fixture
def markdown_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    return tmp_path


def _write_tracked_document(root: Path, filename: str, content: str) -> None:
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(root, "add", "--", filename)


@pytest.mark.parametrize(
    ("filename", "language"),
    [
        ("README.md", "en"),
        ("nested/README.ja.md", "ja"),
        ("nested/notes.md", "en"),
    ],
)
def test_markdown_check_discovers_consumers_and_uses_filename_language(
    markdown_repo: Path,
    filename: str,
    language: AssetTableLanguage,
) -> None:
    blocks = "\n\n".join(
        _table_block(marker, render(language))
        for marker, render in TABLE_RENDERERS.items()
    )
    _write_tracked_document(markdown_repo, filename, blocks)
    _assert_markdown_asset_tables(markdown_repo, (filename,))


@pytest.mark.parametrize("marker", TABLE_RENDERERS)
def test_markdown_check_also_checks_optional_consumers_with_one_table(
    markdown_repo: Path, marker: str
) -> None:
    _write_tracked_document(
        markdown_repo,
        "nested/notes.md",
        _table_block(marker, TABLE_RENDERERS[marker]("en")),
    )
    _assert_markdown_asset_tables(markdown_repo, ())


@pytest.mark.parametrize(
    ("filename", "language"), [("extra.md", "en"), ("extra.ja.md", "ja")]
)
def test_markdown_check_reports_modified_rows_and_pasteable_expected_blocks(
    markdown_repo: Path,
    filename: str,
    language: AssetTableLanguage,
) -> None:
    expected = _table_block("sample-assets", sample_asset_table(language))
    _write_tracked_document(
        markdown_repo, filename, expected.replace("CC BY-NC-ND 4.0", "wrong terms", 1)
    )
    with pytest.raises(AssertionError) as error:
        _assert_markdown_asset_tables(markdown_repo, ())
    assert filename in str(error.value)
    assert expected in str(error.value)


def test_markdown_check_reports_all_stale_tracked_files_and_ignores_untracked_files(
    markdown_repo: Path,
) -> None:
    for filename in ("first.md", "nested/second.md"):
        _write_tracked_document(
            markdown_repo,
            filename,
            "<!-- checkpoint-assets:start -->\nstale\n<!-- checkpoint-assets:end -->",
        )
    (markdown_repo / "untracked.md").write_text(
        "<!-- sample-assets:start -->", encoding="utf-8"
    )
    with pytest.raises(AssertionError) as error:
        _assert_markdown_asset_tables(markdown_repo, ())
    message = str(error.value)
    assert "first.md" in message and "nested/second.md" in message
    assert "untracked.md" not in message


@pytest.mark.parametrize("marker", TABLE_RENDERERS)
@pytest.mark.parametrize(
    "fault", ["missing-start", "missing-end", "reversed", "duplicate"]
)
def test_markdown_check_rejects_incomplete_or_ambiguous_markers(
    markdown_repo: Path, marker: str, fault: str
) -> None:
    expected = _table_block(marker, TABLE_RENDERERS[marker]("en"))
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    broken = {
        "missing-start": expected.replace(start, ""),
        "missing-end": expected.replace(end, ""),
        "reversed": f"{end}\n{start}",
        "duplicate": f"{expected}\n{expected}",
    }[fault]
    _write_tracked_document(markdown_repo, "broken.md", broken)
    with pytest.raises(AssertionError) as error:
        _assert_markdown_asset_tables(markdown_repo, ())
    assert expected in str(error.value)


@pytest.mark.parametrize("fault", ["absent", "untracked", "no-markers", "one-table"])
def test_markdown_check_requires_tracked_consumers_with_both_tables(
    markdown_repo: Path, fault: str
) -> None:
    if fault == "untracked":
        (markdown_repo / "required.md").write_text("# Guide", encoding="utf-8")
    elif fault == "no-markers":
        _write_tracked_document(markdown_repo, "required.md", "# Guide")
    elif fault == "one-table":
        _write_tracked_document(
            markdown_repo,
            "required.md",
            _table_block("checkpoint-assets", asset_table("en")),
        )
    with pytest.raises(AssertionError, match="required.md"):
        _assert_markdown_asset_tables(markdown_repo, ("required.md",))
