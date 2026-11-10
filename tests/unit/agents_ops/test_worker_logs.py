"""Public record, privacy and merge-policy equivalence classes; no Git/HTTP."""

from __future__ import annotations

import copy
import importlib
import json
from typing import Any
from urllib.parse import quote

import pytest

from tests.support.agents_ops.worker_logs import journal, record_source, schema, sources

safety: Any = importlib.import_module("worker_logs.safety")
board_safety: Any = importlib.import_module("board.safety")


@pytest.mark.parametrize(
    "value",
    [
        "観測/補間/遮蔽推定など",
        "「遮蔽（座標あり/なし）」",
        "カメラ/ボール",
        "あり/なし",
        "観測/補間",
        "カタカナ/テスト",
        "観測/補間をv3に保存",
        "𠮷/漢字",
        "Ａ/Ｂ",
        "A/B tests/example.py:12 https://example.com/report",
        "https://example.com/?mode=あり/なし#観測/補間",
        "https://example.com/?mode=%E3%81%82%E3%82%8A%2F%E3%81%AA%E3%81%97",
    ],
)
def test_japanese_word_separators_preserve_public_text(value: str) -> None:
    board_safety.public(value, {}, {})
    result = safety.Guard().sanitize(value)
    assert result.text == value
    assert result.replacements == 0 and result.local_paths == {}


@pytest.mark.parametrize(
    ("value", "expected", "kind"),
    [
        ("保存先 /mnt/unknown/report.txt。", "保存先 [local path]。", "posix"),
        ("パスは/home/user/x", "パスは[local path]", "posix"),
        ("/home/利用者/非公開.txt", "[local path]", "posix"),
        ("/資料/非公開.txt", "[local path]", "posix"),
        ("（/.ssh/config）", "（[local path]）", "posix"),
        ("（/./private/file）", "（[local path]）", "posix"),
        ("（/+private/file）", "（[local path]）", "posix"),
        ("設定は~/.ssh/config", "設定は[local path]", "home"),
        ("~/.config/app/settings", "[local path]", "home"),
        (r"参照 C:\Users\unknown\report.txt", "参照 [local path]", "windows"),
        ("C:/Users/unknown/report.txt", "[local path]", "windows"),
        (r"\\server\private\report.txt", "[local path]", "unc"),
        ("git -C/unknown/repo status", "git -C[local path] status", "posix"),
        ("compiler -I/unknown/header", "compiler -I[local path]", "posix"),
        (r"compiler -IC:\unknown\header", "compiler -I[local path]", "windows"),
        (r"compiler -I\unknown/header", "compiler -I[local path]", "windows"),
        ("/", "[local path]", "posix"),
        ("保存先 %2Fhome%2Fuser%2Fx", "保存先 [local path]", "posix"),
        ("%252Fhome%252Fuser%252Fx", "[local path]", "posix"),
        ("保存先 ／home/user/x", "保存先 [local path]", "posix"),
        (
            "%E4%BF%9D%E5%AD%98%20%2Fhome%2Fuser%2Fx%20%E7%B5%82%E4%BA%86",
            "%E4%BF%9D%E5%AD%98%20[local path]%20%E7%B5%82%E4%BA%86",
            "posix",
        ),
        ('参照 "/home/user/a b.txt"。', '参照 "[local path]"。', "posix"),
        ("[資料](/unknown/report.txt)", "[資料]([local path])", "posix"),
        ("[資料](/unknown/(data)/file.txt)", "[資料]([local path])", "posix"),
        ("/home/[unknown]/report.txt", "[local path]", "posix"),
        (
            "保存 `/unknown/run\n次の指示 `",
            "保存 `[local path]\n次の指示 `",
            "posix",
        ),
    ],
)
def test_local_paths_are_detected_then_masked(
    value: str, expected: str, kind: str
) -> None:
    guard = safety.Guard()
    with pytest.raises(journal.PublicationError, match="absolute_path"):
        guard.check(value)
    result = guard.sanitize(value)
    assert result.text == expected
    assert result.replacements == 1 and result.local_paths == {kind: 1}
    guard.check(result.text)


@pytest.mark.parametrize(
    "value",
    ["パスは/home/user/x", "参照先：/private/state", "設定は~/.ssh", "設定は~/.config"],
)
def test_board_keeps_rejecting_real_paths_next_to_japanese(value: str) -> None:
    with pytest.raises(board_safety.BoardError, match="private path"):
        board_safety.public(value, {}, {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[資料](/資料/非公開.txt)", "[資料]([local path])"),
        ("`/資料/非公開.txt`", "`[local path]`"),
        ("保存先:/資料/非公開.txt", "保存先:[local path]"),
        ("保存先：/資料/非公開.txt", "保存先：[local path]"),
        ("（/資料/x）", "（[local path]）"),
        ("(/資料/x)", "([local path])"),
        ("[/資料/x]", "[[local path]]"),
        ("資料=/資料/x", "資料=[local path]"),
        (" /資料/x", " [local path]"),
        ('"/資料/x"', '"[local path]"'),
        ("'/資料/x'", "'[local path]'"),
        ("保存先:/ひみつ/ファイル.txt", "保存先:[local path]"),
        ("保存先：／資料/x", "保存先：[local path]"),
        ("[資料](%2F資料%2F非公開.txt)", "[資料]([local path])"),
        ("%60%2F資料%2F非公開.txt%60", "%60[local path]%60"),
        ("保存先:%252F資料%252F非公開.txt", "保存先:[local path]"),
        ("保存先%EF%BC%9A%2F資料%2F非公開.txt", "保存先%EF%BC%9A[local path]"),
        ("＋/－", "＋[local path]"),
        ("（/）", "（[local path]）"),
        ("○/×", "○[local path]"),
    ],
)
def test_japanese_named_paths_at_boundaries_are_rejected_or_masked(
    value: str, expected: str
) -> None:
    with pytest.raises(board_safety.BoardError, match="private path"):
        board_safety.public(value, {}, {})
    guard = safety.Guard()
    with pytest.raises(journal.PublicationError, match="absolute_path"):
        guard.check(value)
    result = guard.sanitize(value)
    assert result.text == expected
    assert result.replacements == 1 and result.local_paths == {"posix": 1}
    guard.check(result.text)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/?file=/資料/非公開.txt",
        "https://example.com/#保存先:/資料/非公開.txt",
        "https://example.com/?file=（/資料/x）",
        "https://example.com/?file=%2F資料%2F非公開.txt",
        "https://example.com/#保存先%EF%BC%9A%252F資料%252F非公開.txt",
        "https://example.com/?file=%60%2F資料%2F非公開.txt%60",
        "https://example.com/#%5B%2F資料%2F非公開.txt%5D",
        "%68%74%74%70%73%3A%2F%2Fexample.com%2F%3Ffile%3D%2F資料%2F非公開.txt",
    ],
)
def test_japanese_named_paths_in_url_queries_and_fragments_are_rejected(
    value: str,
) -> None:
    with pytest.raises(board_safety.BoardError, match="private path"):
        board_safety.public(value, {}, {})
    guard = safety.Guard()
    with pytest.raises(journal.PublicationError, match="unsafe_url"):
        guard.check(value)
    with pytest.raises(journal.PublicationError, match="unsafe_url"):
        guard.sanitize(value)


@pytest.mark.parametrize("encoding_layers", [4, 5])
def test_path_decoding_limit_is_shared(encoding_layers: int) -> None:
    value = "/資料/x"
    for _ in range(encoding_layers):
        value = quote(value, safe="")
    with pytest.raises(board_safety.BoardError):
        board_safety.public(value, {}, {})
    guard = safety.Guard()
    if encoding_layers == 5:
        with pytest.raises(journal.PublicationError, match="ambiguous_encoding"):
            guard.sanitize(value)
    else:
        result = guard.sanitize(value)
        assert result.text == "[local path]"
        assert result.replacements == 1 and result.local_paths == {"posix": 1}


@pytest.mark.parametrize(
    "value",
    [
        "保存 /unknown/report token=undisclosed",
        "/unknown/ghp_" + "x" * 30,
        "/unknown/%67%68%70%5F" + "x" * 30,
        "/unknown/report\x1b[0m",
        "/unknown/01234567-89ab-cdef-0123-456789abcdef",
        "https://example.com/?file=%2Fhome%2Fuser%2Fx",
        "https://example.com/#%252Fhome%252Fuser%252Fx",
        "https://example.com/?file=パスは%2Fhome%2Fuser%2Fx",
    ],
)
def test_path_masking_keeps_rejecting_other_secrets_and_unsafe_urls(value: str) -> None:
    with pytest.raises(journal.PublicationError):
        safety.Guard().sanitize(value)


def test_masked_output_is_rechecked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(safety, "mask_local_paths", lambda text: (text, {"posix": 1}))
    with pytest.raises(journal.PublicationError, match="absolute_path"):
        safety.Guard().sanitize("保存 /home/unknown/report.txt")


def test_known_identifier_inside_unknown_path_is_fully_masked() -> None:
    result = safety.Guard(["private-session-value"]).sanitize(
        "保存 /unknown/private-session-value/report.txt。"
    )
    assert result.text == "保存 [local path]。"
    assert result.replacements == 2 and result.local_paths == {"posix": 1}


@pytest.mark.parametrize(
    "case", ["retry", "autonomous_continuation", "new_lineage", "legacy_lineage"]
)
def test_public_keys_follow_logical_lineage(case: str) -> None:
    phase = {"lineage_id": "private-lineage", "delegation_id": "private-delegation"}
    first = sources.public_key(phase)
    if case == "autonomous_continuation":
        phase["delegation_id"] = "different-private-delegation"
    if case == "new_lineage":
        other = {"lineage_id": "other-private-lineage"}
        assert sources.public_key(other) != first
    elif case == "legacy_lineage":
        assert phase["public_key"] == first
        assert first not in {phase["lineage_id"], phase["delegation_id"]}
    assert sources.public_key(phase) == first
    assert len(first) == 32


@pytest.mark.parametrize(
    "case",
    ["valid", "version", "unknown", "duplicate", "digest"],
)
def test_schema_and_receipts_fail_closed(case: str) -> None:
    source, receipt = record_source()
    files = sources.render(source, receipt)
    if case == "valid":
        meta = schema.validate(files)
        assert meta["sources"]["issue"]["url"] == "https://github.com/o/r/issues/10"
        assert files["instruction.md"].decode() == source["instruction"]
        assert meta["artifacts"]["prs"] == [42]
        assert meta["worker"]["selected"] == {"model": "gpt-6.1-sol", "effort": "max"}
        assert meta["sanitization"] == {
            "version": 1,
            "instruction": 0,
            "final": 0,
            "additional": 0,
        }
        return
    meta = json.loads(files["meta.json"])
    if case == "version":
        meta["schema_version"] = 2
    elif case == "unknown":
        meta["private_session"] = "must not be accepted"
    elif case == "duplicate":
        files["meta.json"] = files["meta.json"].replace(
            b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1'
        )
    else:
        for name in ("instruction.md", "final.md"):
            modified = {**files, name: files[name] + b"changed"}
            with pytest.raises(ValueError):
                schema.validate(modified)
        return
    if case != "duplicate":
        files["meta.json"] = schema.json_bytes(meta)
    with pytest.raises(ValueError):
        schema.validate(files)


def test_mask_counts_cover_instruction_final_and_additional_text() -> None:
    source, receipt = record_source()
    source["instruction"] = "観測/補間。パスは/home/unknown/report.txt。"
    source["additional"] = ["設定は~/.config/app", "参照は~/.ssh/config"]
    receipt["final"] = (
        r"確認は C:\Users\unknown\report.txt と \\server\private\report.txt"
    )
    receipt["final_sha256"] = schema.sha(receipt["final"].encode())
    files = sources.render(source, receipt)
    meta = schema.validate(files)
    assert files["instruction.md"].decode() == "観測/補間。パスは[local path]。"
    assert files["final.md"].decode() == "確認は [local path] と [local path]"
    assert meta["instruction"]["additional_instructions"] == [
        "設定は[local path]",
        "参照は[local path]",
    ]
    assert meta["sanitization"] == {
        "version": 2,
        "instruction": 1,
        "final": 2,
        "additional": 2,
        "local_paths": {"posix": 1, "home": 2, "windows": 1, "unc": 1},
    }


@pytest.mark.parametrize(
    "counts",
    [
        {"posix": 0},
        {"home": -1},
        {"windows": True},
        {"unc": 2},
        {"private-path": 1},
        {},
    ],
)
def test_invalid_mask_counts_cannot_cross_publication_boundary(
    counts: dict[str, int],
) -> None:
    source, receipt = record_source()
    source["instruction"] = "保存 /unknown/report.txt"
    files = sources.render(source, receipt)
    meta = json.loads(files["meta.json"])
    meta["sanitization"]["local_paths"] = counts
    files["meta.json"] = schema.json_bytes(meta)
    with pytest.raises(journal.PublicationError, match="record_sanitization"):
        schema.validate(files)
    with pytest.raises(journal.PublicationError, match="record_sanitization"):
        journal.view(
            {
                "source": {"public_key": "a" * 32, "sequence": 1},
                "state": "ready",
                "files": {name: value.decode() for name, value in files.items()},
            }
        )


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "checkout_path",
        "known_identity",
        "external_path",
        "credential",
        "encoded_link",
        "private_link",
        "url_credentials",
        "controls",
        "metadata",
    ],
)
def test_publication_safety_preserves_or_blocks(case: str) -> None:
    guard = safety.Guard(["private-value"], ["/private/checkout"])
    value: Any = "日本語の根拠。tests/example.py:12 https://github.com/o/r/pull/42\n"
    if case == "checkout_path":
        result = guard.sanitize("確認: /private/checkout/tests/example.py:12\n")
        assert (
            result.text == "確認: ./tests/example.py:12\n" and result.replacements == 1
        )
        references = safety.Guard(
            [], ["/private/checkout"], repo="o/r", commit="a" * 40
        )
        linked = references.sanitize("[根拠](/private/checkout/tests/example.py:12)")
        assert (
            linked.text
            == "[根拠](https://github.com/o/r/blob/"
            + "a" * 40
            + "/tests/example.py#L12)"
        )
        return
    if case == "known_identity":
        result = guard.sanitize("実行は private-value で確認。")
        assert (
            result.text == "実行は [private identifier] で確認。"
            and result.replacements == 1
        )
        return
    if case == "ordinary":
        assert guard.sanitize(value) == safety.Sanitized(value, 0, {})
        for url in (
            "HTTPS://EXAMPLE.COM/path",
            "Http://example.com/path",
            "%48%54%54%50%53%3A%2F%2Fexample.com/path",
        ):
            assert guard.sanitize(url) == safety.Sanitized(url, 0, {})
        return
    if case == "url_credentials":
        for url in (
            "https://user:pass@example.com/path",
            "HTTPS://user:pass@example.com/path",
            "Http://user:pass@example.com/path",
            "%48%54%54%50%53%3A%2F%2Fuser%3Apass%40example.com/path",
        ):
            with pytest.raises(journal.PublicationError, match="unsafe_url"):
                guard.check(url)
        # Non-HTTP schemes must remain subject to the local-path check.
        for url in (
            "file:///home/unknown/report.md",
            "FiLe://user:pass@example.com/path",
            "%46%49%4C%45%3A%2F%2F%2Fhome/unknown/report.md",
            "FTP://user:pass@example.com/path",
        ):
            with pytest.raises(journal.PublicationError, match="absolute_path"):
                guard.check(url)
        return
    value = {
        "external_path": "保存先 /home/unknown/run",
        "credential": "ghp_" + "x" * 30,
        "encoded_link": "https://example.com/?x=private%2dvalue",
        "private_link": "https://example.com/?file=/home/unknown/file",
        "controls": "visible\x1b[0m",
        "metadata": {"https://example.com/private-value": ["otherwise ordinary"]},
    }[case]
    if case == "external_path":
        for command in (
            "git -C/home/unknown/repository status",
            "compiler -I/private/unknown/header",
        ):
            with pytest.raises(journal.PublicationError):
                guard.check(command)
    with pytest.raises(journal.PublicationError):
        guard.check(value)
    assert "private-value" not in str(journal.PublicationError("private_value"))
    if case == "metadata":
        assert (
            str(journal.PublicationError("privatevaluewithoutpunctuation"))
            == "publication_failed"
        )
        bad = {
            "source": {"public_key": "a" * 32, "sequence": 1},
            "state": "blocked",
            "error": "privatevaluewithoutpunctuation",
        }
        with pytest.raises(journal.PublicationError, match="invalid_error_code"):
            journal.view(bad)


@pytest.mark.parametrize(
    "case",
    [
        "outside",
        "modify",
        "delete",
        "rename",
        "symlink",
        "executable",
        "gitlink",
        "repository",
        "base",
        "head",
        "head.ref",
        "body",
    ],
)
def test_merge_policy_rejects_unsafe_diff_or_identity(case: str) -> None:
    expected = {"logs/instruction.md", "logs/final.md", "logs/meta.json"}
    raw = b"".join(
        f":000000 100644 {'0' * 40} {'a' * 40} A\0{p}\0".encode()
        for p in sorted(expected)
    )
    schema.additions(raw, expected)
    if case in {"repository", "base", "head", "head.ref", "body"}:
        if case == "repository":
            normalize = importlib.import_module(
                "worker_logs.publication"
            ).canonical_remote
            assert normalize("https://github.com/o/r") == normalize(
                "https://github.com/O/R.git"
            )
            assert normalize("git@github.com:o/r.git") == normalize(
                "git@github.com:O/R"
            )
            assert normalize("https://github.com/other/r.git") != normalize(
                "https://github.com/o/r"
            )
            assert normalize("https://user:pass@github.com/o/r.git") != normalize(
                "https://github.com/o/r"
            )
        row = {
            "source": {"repo": "o/r"},
            "branch": "worker-log/d-abc/0001",
            "head": "a" * 40,
            "pr_body": "marker",
        }
        pr: dict[str, Any] = {
            "number": 7,
            "html_url": "https://github.com/o/r/pull/7",
            "draft": False,
            "body": "marker",
            "head": {
                "ref": row["branch"],
                "sha": row["head"],
                "repo": {"full_name": "o/r"},
            },
            "base": {"ref": "main", "repo": {"full_name": "o/r"}},
        }
        schema.identity(pr, row, "o/r")
        bad = copy.deepcopy(pr)
        if case == "repository":
            bad["head"]["repo"]["full_name"] = "other/r"
        elif case == "base":
            bad["base"]["ref"] = "release"
        elif case == "head":
            bad["head"]["sha"] = "b" * 40
        elif case == "head.ref":
            bad["head"]["ref"] = "other-branch"
        else:
            bad["body"] = "other marker"
        with pytest.raises(
            journal.PublicationError, match="pr_identity_changed"
        ) as exc:
            schema.identity(bad, row, "o/r")
        field = {"repository": "head.repo", "base": "base.ref", "head": "head.sha"}.get(
            case, case
        )
        assert field in exc.value.detail
        return
    changes = {
        "outside": (b"logs/final.md", b"code.py"),
        "modify": (b":000000", b":100644"),
        "delete": (b" A\0", b" D\0"),
        "rename": (b" A\0", b" R100\0"),
        "symlink": (b" 100644 ", b" 120000 "),
        "executable": (b" 100644 ", b" 100755 "),
        "gitlink": (b" 100644 ", b" 160000 "),
    }
    before, after = changes[case]
    with pytest.raises(journal.PublicationError, match="unsafe_diff"):
        schema.additions(raw.replace(before, after, 1), expected)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "outside",
        "modified",
        "removed",
        "renamed",
        "extra",
        "missing",
        "duplicate",
    ],
)
def test_github_changed_files_must_be_exactly_the_three_additions(change: str) -> None:
    expected = {"logs/instruction.md", "logs/final.md", "logs/meta.json"}
    changes = [{"filename": name, "status": "added"} for name in sorted(expected)]
    if change == "none":
        schema.pr_additions(changes, expected)
        return
    if change == "outside":
        changes[0]["filename"] = "outside.txt"
    elif change in {"modified", "removed", "renamed"}:
        changes[0]["status"] = change
        if change == "renamed":
            changes[0]["previous_filename"] = "existing.txt"
    elif change == "extra":
        changes.append({"filename": "extra.txt", "status": "added"})
    elif change == "missing":
        changes.pop()
    else:
        changes[0] = changes[1]
    with pytest.raises(journal.PublicationError, match="unsafe_diff"):
        schema.pr_additions(changes, expected)


@pytest.mark.parametrize("response", ["three_files", "full_page", "object"])
def test_github_file_endpoint_cannot_hide_a_larger_diff(
    monkeypatch: pytest.MonkeyPatch, response: str
) -> None:
    github = importlib.import_module("worker_logs.publication").GitHub("o/r")
    expected = {"logs/instruction.md", "logs/final.md", "logs/meta.json"}
    changes: Any = [{"filename": name, "status": "added"} for name in sorted(expected)]
    if response == "full_page":
        changes += [
            {"filename": f"extra-{i}.txt", "status": "added"} for i in range(97)
        ]
    elif response == "object":
        changes = {"message": "invalid response"}
    requests = []

    def request(endpoint: str) -> Any:
        requests.append(endpoint)
        return changes

    monkeypatch.setattr(github, "request", request)
    if response == "three_files":
        schema.pr_additions(github.files(7), expected)
    else:
        with pytest.raises(
            journal.PublicationError,
            match="github_invalid_response" if response == "object" else "unsafe_diff",
        ):
            schema.pr_additions(github.files(7), expected)
    assert requests == ["repos/o/r/pulls/7/files?per_page=100"]
