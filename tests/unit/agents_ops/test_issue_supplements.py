"""Insertion-only content contract; no agent, GitHub, subprocess or real clock."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops"))
api: Any = importlib.import_module("issue_rewrite.invariants")
formatting: Any = importlib.import_module("issues.format")


@pytest.fixture(scope="module")
def original() -> str:
    return str(
        formatting.render_parent_body(
            {
                "目的・背景": (
                    "目的の原文。  \n続きは #17 と 2.5 秒、@reviewer、`code`。\n\n"
                    "> 引用の原文\nlazy continuation\n\n"
                    "<!-- metadata: a  b -->\n\n"
                    "~~~python\nvalue = 17\n~~~\n\n"
                    "<details>\n<summary>既存の詳細</summary>\n\n既存の内容。\n\n</details>\n\n"
                    "最後の段落。"
                ),
                "全体のスコープ": "1. 推論\n2. 評価",
                "全体方針・合意事項": "まだ合意なし。",
                "全体の完了条件": "- [ ] 確認する。",
                "進捗概要": "未着手。",
                "依存関係・進め方": "依存なし。",
            }
        )
    )


def details(content: str) -> str:
    # Expected markup is specified independently of the production renderer.
    return (
        "\n\n<details>\n<summary>Geminiによる補足・背景情報</summary>\n\n"
        + content
        + "\n\n</details>\n"
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("ending", ["", "\n", "  \n"])
def test_multiple_insertions_preserve_every_original_character(
    original: str, newline: str, ending: str
) -> None:
    source = original.rstrip("\n") + ending
    source = source.replace("\n", newline)
    supplements = [
        {
            "after": "最後の段落。",
            "content": "背景は #99 と 8.7 秒。https://example.org/new",
        },
        {
            "after": "## 目的・背景",
            "content": "既存部品は [README](https://example.org/readme) にある。",
        },
    ]
    expected = source.replace(
        "最後の段落。", "最後の段落。" + details(supplements[0]["content"])
    ).replace("## 目的・背景", "## 目的・背景" + details(supplements[1]["content"]))
    assert api.render_supplements(source, supplements) == expected
    assert api.check_invariants(source, expected, supplements=supplements) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("目的の原文。", "分かりやすい目的。"),
        ("。  \n", "。\n"),
        ("#17", "#18"),
        ("2.5", "2.6"),
        ("@reviewer", "@other"),
        ("`code`", "`other`"),
        ("value = 17", "value=17"),
        ("a  b", "a b"),
        ("lazy continuation", "changed continuation"),
        ("1. 推論", "9. 推論"),
        ("[ ]", "[x]"),
        ("未着手。", "着手済み。"),
        ("既存の内容。", ""),
        ("依存なし。", "依存なし。追加要件。"),
    ],
)
def test_any_source_edit_fails_even_when_a_supplement_is_valid(
    original: str, old: str, new: str
) -> None:
    supplements = [{"after": "## 目的・背景", "content": "根拠のある補足。"}]
    body = original.replace(
        "## 目的・背景", "## 目的・背景" + details("根拠のある補足。")
    )
    errors = api.check_invariants(
        original, body.replace(old, new, 1), supplements=supplements
    )
    assert any(error["rule"] == "original_preservation" for error in errors)


def test_empty_supplements_keep_existing_details_and_eof_exactly(original: str) -> None:
    assert api.render_supplements(original, []) == original
    assert api.check_invariants(original, original, supplements=[]) == []
    assert api.check_invariants(original, original + "\n", supplements=[])
    hidden = details(original)
    assert api.check_invariants(original, hidden, supplements=[])


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        ("不存在", "unique"),
        ("原文", "unique"),
        ("目的の", "complete_markdown_block"),
        ("引用の原文", "complete_markdown_block"),
        ("value = 17", "complete_markdown_block"),
        ("<!-- metadata: a  b -->", "complete_markdown_block"),
        ("既存の内容。", "complete_markdown_block"),
        ("1. 推論", "complete_markdown_block"),
        ("未着手。", "machine_managed"),
        ("## 子タスクのチェックリスト", "machine_managed"),
    ],
)
def test_ambiguous_or_structurally_unsafe_positions_are_rejected(
    original: str, after: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        api.render_supplements(original, [{"after": after, "content": "背景。"}])


def test_duplicate_positions_and_overlapping_anchor_matches_are_rejected(
    original: str,
) -> None:
    with pytest.raises(ValueError, match="duplicate_supplement_position"):
        api.render_supplements(
            original,
            [
                {"after": "## 目的・背景", "content": "背景1。"},
                {"after": "目的・背景", "content": "背景2。"},
            ],
        )
    with pytest.raises(ValueError, match="unique"):
        api.render_supplements(
            original + "\naaaa", [{"after": "aaa", "content": "背景。"}]
        )


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (" \n", "empty"),
        ("背景。</details>本文外", "details_delimiter"),
        ("<DETAILS>入れ子</DETAILS>", "details_delimiter"),
        ("`</summary>`", "details_delimiter"),
        ("<!-- 未閉鎖", "raw_html"),
        ("<div>背景</div>", "raw_html"),
        ("## 追加条件", "issue_sections"),
        ("見出し\n=====", "issue_sections"),
        ("- [ ] タスク追加", "task_checkboxes"),
        ("[元リンク]: https://other.invalid", "reference_definition"),
        ("~~~python\nopen = True", "unclosed_code_fence"),
        ("```", "unclosed_code_fence"),
        ("- item\n\n    ~~~\n    open = True", "unclosed_code_fence"),
    ],
)
def test_supplements_cannot_escape_or_change_the_original_rendering(
    original: str, content: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        api.render_supplements(
            original, [{"after": "## 目的・背景", "content": content}]
        )


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085", "\v", "\f", "\0"])
def test_commonmark_line_maps_do_not_split_unicode_separators(
    original: str, separator: str
) -> None:
    paragraph = f"a{separator}b{separator}c"
    source = original.replace("最後の段落。", paragraph)
    assert api.render_supplements(
        source, [{"after": paragraph, "content": "背景。"}]
    ) == source.replace(paragraph, paragraph + details("背景。"))


def test_artifact_metadata_must_explain_every_added_character(original: str) -> None:
    supplements = [{"after": "最後の段落。", "content": "背景。"}]
    expected = original.replace("最後の段落。", "最後の段落。" + details("背景。"))
    assert api.check_invariants(original, expected, supplements=[])
    assert api.check_invariants(
        original, expected + "任意の追加", supplements=supplements
    )
    assert api.check_invariants(
        original,
        expected.replace("<details>", "<details open>", 1),
        supplements=supplements,
    )
