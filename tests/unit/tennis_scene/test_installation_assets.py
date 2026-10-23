"""Keep localized asset tables in every tracked Markdown consumer authoritative."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast, get_args
from urllib.parse import unquote, urlsplit

import pytest

from src.tennis_scene import installation_assets
from src.tennis_scene.installation_assets import (
    ASSET_TABLE_LANGUAGE_SUFFIXES,
    INSTALLATION_ASSETS,
    AssetTableLanguage,
    asset_table,
    asset_table_language_from_filename,
    sample_asset_table,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.sample_dataset import DATASET_REPO, PUBLISHED_FILES, PublishedSampleFile

# Other tracked Markdown with either marker is also checked.
REQUIRED_ASSET_TABLE_DOCUMENTS = (
    # Add one consumer per line; each required document must contain both tables.
    "README.md",
    "README.ja.md",
    "docs/installation/README.md",
    "docs/installation/README.ja.md",
    "docs/installation/README.zh-CN.md",
)
INSTALLATION_GUIDE_DOCUMENTS = tuple(
    f"docs/installation/README{suffix}"
    for suffix in ASSET_TABLE_LANGUAGE_SUFFIXES.values()
)
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
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
        consumed_tables = {
            marker: render
            for marker, render in TABLE_RENDERERS.items()
            if filename in required_documents
            or f"<!-- {marker}:start -->" in document
            or f"<!-- {marker}:end -->" in document
        }
        if not consumed_tables:
            continue
        try:
            language = asset_table_language_from_filename(filename)
        except ValueError as error:
            problems.append(f"{filename}: {error}")
            continue
        for marker, render in consumed_tables.items():
            start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
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
        (
            "zh-CN",
            "| 资产 | 保存位置（相对于 `ckpt/`） | 下载来源 | 许可证 / 条款 |",
            "下载来源",
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
        terms = {
            "en": asset.license_en,
            "ja": asset.license_ja,
            "zh-CN": asset.license_zh_cn,
        }[language]
        assert f"| {asset.name} | `{asset.filename}` |" in line
        assert f"[{source_label}]({asset.source})" in line
        assert line.endswith(f"| {terms} |")
    assert "SHA" not in table
    assert re.search(r"[0-9a-f]{64}", table) is None
    if language == "en":
        assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", table) is None


def test_license_footnotes_identify_the_same_assets_in_all_languages() -> None:
    marked_assets = []
    for asset in INSTALLATION_ASSETS:
        assert ("¹" in asset.license_en) == ("¹" in asset.license_ja)
        assert ("¹" in asset.license_en) == ("¹" in asset.license_zh_cn)
        if "¹" in asset.license_en:
            marked_assets.append(asset.name)
    assert marked_assets == ["detector", "vitpose", "hmr2", "tracking_encoder"]


@pytest.mark.parametrize(
    ("language", "header", "source_label"),
    [
        ("en", "| Destination | Source / terms |", "Published dataset"),
        ("ja", "| 配置先 | 入手先・条件 |", "公開Dataset"),
        ("zh-CN", "| 保存位置 | 下载来源 / 条款 |", "公开数据集"),
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


@pytest.mark.parametrize("language", ASSET_TABLE_LANGUAGE_SUFFIXES)
def test_sample_table_reads_the_manifest_instead_of_a_separate_file_list(
    monkeypatch: pytest.MonkeyPatch,
    language: AssetTableLanguage,
) -> None:
    monkeypatch.setattr(
        installation_assets,
        "PUBLISHED_FILES",
        (PublishedSampleFile("added.json", "0" * 64),),
    )
    table = sample_asset_table(language)
    assert "`data/samples/added.json`" in table
    assert "cam0.mp4" not in table
    assert len(table.splitlines()) == 3


@pytest.mark.parametrize("render", TABLE_RENDERERS.values())
@pytest.mark.parametrize("language", ("fr", "zh", "zh-TW", "zh-cn", ""))
def test_table_rejects_unsupported_language(
    render: Callable[[AssetTableLanguage], str],
    language: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported asset table language"):
        render(cast(AssetTableLanguage, language))


def test_public_filename_suffixes_cover_every_supported_language() -> None:
    assert set(ASSET_TABLE_LANGUAGE_SUFFIXES) == set(get_args(AssetTableLanguage))
    assert len(set(ASSET_TABLE_LANGUAGE_SUFFIXES.values())) == len(
        ASSET_TABLE_LANGUAGE_SUFFIXES
    )


@pytest.mark.parametrize(
    ("filename", "language"),
    [
        ("README.md", "en"),
        ("nested/notes.md", "en"),
        ("nested/README.ja.md", "ja"),
        ("nested/README.zh-CN.md", "zh-CN"),
        ("LICENSES.zh-CN.md", "zh-CN"),
        ("directory.ja/notes.md", "en"),
        ("directory.zh-TW/README.zh-CN.md", "zh-CN"),
    ],
)
@pytest.mark.parametrize("as_path", (False, True))
def test_public_filename_language_uses_only_the_document_suffix(
    filename: str, language: AssetTableLanguage, as_path: bool
) -> None:
    value = Path(filename) if as_path else filename
    assert asset_table_language_from_filename(value) == language


@pytest.mark.parametrize(
    "filename",
    (
        "README.zh.md",
        "README.zh-TW.md",
        "README.zh-cn.md",
        "README.fr.md",
        "README.en.md",
        "README.ja.zh-CN.md",
        "README.md.bak",
        "README",
    ),
)
def test_public_filename_language_rejects_unknown_suffixes(filename: str) -> None:
    with pytest.raises(ValueError, match="Unsupported asset table document suffix"):
        asset_table_language_from_filename(filename)


def test_tracked_markdown_asset_tables_match_authoritative_catalogs() -> None:
    _assert_markdown_asset_tables(PROJECT_ROOT, REQUIRED_ASSET_TABLE_DOCUMENTS)


@pytest.mark.parametrize("filename", INSTALLATION_GUIDE_DOCUMENTS[1:])
def test_installation_translations_preserve_sections_commands_and_references(
    filename: str,
) -> None:
    english = (PROJECT_ROOT / INSTALLATION_GUIDE_DOCUMENTS[0]).read_text(
        encoding="utf-8"
    )
    translated = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
    language = asset_table_language_from_filename(filename)
    assert re.findall(r"^(#{1,3}) ", english, re.MULTILINE) == re.findall(
        r"^(#{1,3}) ", translated, re.MULTILINE
    )
    assert ANCHOR.findall(english) == ANCHOR.findall(translated)
    assert len(set(ANCHOR.findall(translated))) == len(ANCHOR.findall(translated))
    assert re.findall(r"^```[^\n]*\n.*?^```", english, re.MULTILINE | re.DOTALL) == (
        re.findall(r"^```[^\n]*\n.*?^```", translated, re.MULTILINE | re.DOTALL)
    )
    english_sections = ANCHOR.split(english)
    translated_sections = ANCHOR.split(translated)
    for anchor, en, localized in zip(
        english_sections[1::2],
        english_sections[2::2],
        translated_sections[2::2],
        strict=True,
    ):
        # Japanese already presents this variable/value as one inline assignment.
        # Executable fenced commands above must still match byte for byte.
        comparable_inline = localized.replace(
            "`CUDA_HOME=/usr/local/cuda`", "`CUDA_HOME` `/usr/local/cuda`"
        )
        assert Counter(re.findall(r"`([^`\n]+)`", en)) == Counter(
            re.findall(r"`([^`\n]+)`", comparable_inline)
        ), anchor
        normalized_links = [
            target.replace(
                f"LICENSES{ASSET_TABLE_LANGUAGE_SUFFIXES[language]}", "LICENSES.md"
            )
            for target in LINK.findall(localized)
        ]
        assert Counter(LINK.findall(en)) == Counter(normalized_links), anchor


def test_chinese_installation_preserves_numbers_in_each_english_section() -> None:
    english, chinese = (
        ANCHOR.split((PROJECT_ROOT / filename).read_text(encoding="utf-8"))
        for filename in (
            "docs/installation/README.md",
            "docs/installation/README.zh-CN.md",
        )
    )
    for anchor, en, zh in zip(english[1::2], english[2::2], chinese[2::2], strict=True):
        # Compare literal numeric values, including versions, units and timings;
        # word boundaries differ between English and Chinese prose.
        assert Counter(re.findall(r"[0-9]+(?:[.,][0-9]+)*", en)) == Counter(
            re.findall(r"[0-9]+(?:[.,][0-9]+)*", zh)
        ), anchor


@pytest.mark.parametrize("filename", INSTALLATION_GUIDE_DOCUMENTS)
def test_installation_guides_have_language_navigation_and_valid_local_links(
    filename: str,
) -> None:
    document = PROJECT_ROOT / filename
    text = document.read_text(encoding="utf-8")
    introduction = ANCHOR.split(text)[0]
    navigation = " / ".join(
        f"[{label}](README{suffix})"
        for label, suffix in zip(
            ("English", "日本語", "简体中文"),
            ASSET_TABLE_LANGUAGE_SUFFIXES.values(),
            strict=True,
        )
    )
    assert navigation in introduction
    assert re.search(r"[0-9a-fA-F]{64}", text) is None
    language = asset_table_language_from_filename(filename)
    license_links = []
    for target in LINK.findall(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        destination = (
            document.parent / unquote(parsed.path) if parsed.path else document
        ).resolve()
        if destination.name.startswith("LICENSES"):
            assert destination.name == (
                f"LICENSES{ASSET_TABLE_LANGUAGE_SUFFIXES[language]}"
            ), target
            license_links.append(parsed.fragment)
            # The Chinese LICENSES document is supplied by a parallel child.
            # Its existence/content is covered by that child's license tests;
            # here we enforce the agreed path and common-anchor contract.
            if language == "zh-CN":
                continue
        assert destination.is_file(), target
        if parsed.fragment:
            content = destination.read_text(encoding="utf-8")
            headings = re.findall(r"^#+ (.+)$", content, re.MULTILINE)
            anchors = set(ANCHOR.findall(content)) | {
                re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                for heading in headings
            }
            assert unquote(parsed.fragment) in anchors, target
    assert license_links == [
        "code-license-scope",
        "third-party-weights",
        "published-artifacts",
    ]


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
        ("nested/README.zh-CN.md", "zh-CN"),
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


@pytest.mark.parametrize("language", ASSET_TABLE_LANGUAGE_SUFFIXES)
@pytest.mark.parametrize("marker", TABLE_RENDERERS)
def test_markdown_check_also_checks_optional_consumers_with_one_table(
    markdown_repo: Path, marker: str, language: AssetTableLanguage
) -> None:
    _write_tracked_document(
        markdown_repo,
        f"nested/notes{ASSET_TABLE_LANGUAGE_SUFFIXES[language]}",
        _table_block(marker, TABLE_RENDERERS[marker](language)),
    )
    _assert_markdown_asset_tables(markdown_repo, ())


def test_markdown_check_ignores_unrelated_markdown_suffixes(
    markdown_repo: Path,
) -> None:
    _write_tracked_document(
        markdown_repo, ".github/agents/example.agent.md", "# No asset tables"
    )
    _assert_markdown_asset_tables(markdown_repo, ())


@pytest.mark.parametrize(
    ("filename", "language"),
    [("extra.md", "en"), ("extra.ja.md", "ja"), ("extra.zh-CN.md", "zh-CN")],
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


@pytest.mark.parametrize("filename", ("README.zh.md", "README.zh-TW.md"))
def test_markdown_check_reports_unknown_language_suffixes(
    markdown_repo: Path, filename: str
) -> None:
    _write_tracked_document(
        markdown_repo, filename, _table_block("checkpoint-assets", asset_table("en"))
    )
    with pytest.raises(
        AssertionError, match="Unsupported asset table document suffix"
    ) as error:
        _assert_markdown_asset_tables(markdown_repo, ())
    assert filename in str(error.value)


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
@pytest.mark.parametrize("language", ASSET_TABLE_LANGUAGE_SUFFIXES)
def test_markdown_check_requires_tracked_consumers_with_both_tables(
    markdown_repo: Path, fault: str, language: AssetTableLanguage
) -> None:
    filename = f"required{ASSET_TABLE_LANGUAGE_SUFFIXES[language]}"
    if fault == "untracked":
        (markdown_repo / filename).write_text("# Guide", encoding="utf-8")
    elif fault == "no-markers":
        _write_tracked_document(markdown_repo, filename, "# Guide")
    elif fault == "one-table":
        _write_tracked_document(
            markdown_repo,
            filename,
            _table_block("checkpoint-assets", asset_table(language)),
        )
    with pytest.raises(AssertionError, match=re.escape(filename)):
        _assert_markdown_asset_tables(markdown_repo, (filename,))
