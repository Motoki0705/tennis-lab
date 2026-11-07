"""Public record, privacy and merge-policy equivalence classes; no Git/HTTP."""

from __future__ import annotations

import copy
import importlib
import json
from typing import Any

import pytest

from tests.support.agents_ops.worker_logs import journal, record_source, schema, sources

safety: Any = importlib.import_module("worker_logs.safety")


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
        text, count = guard.sanitize("確認: /private/checkout/tests/example.py:12\n")
        assert text == "確認: ./tests/example.py:12\n" and count == 1
        references = safety.Guard(
            [], ["/private/checkout"], repo="o/r", commit="a" * 40
        )
        linked, _ = references.sanitize("[根拠](/private/checkout/tests/example.py:12)")
        assert (
            linked
            == "[根拠](https://github.com/o/r/blob/"
            + "a" * 40
            + "/tests/example.py#L12)"
        )
        return
    if case == "known_identity":
        text, count = guard.sanitize("実行は private-value で確認。")
        assert text == "実行は [private identifier] で確認。" and count == 1
        return
    if case == "ordinary":
        assert guard.sanitize(value) == (value, 0)
        return
    value = {
        "external_path": "保存先 /home/unknown/run",
        "credential": "ghp_" + "x" * 30,
        "encoded_link": "https://example.com/?x=private%2dvalue",
        "private_link": "https://example.com/?file=/home/unknown/file",
        "url_credentials": "https://user:pass@example.com/path",
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
    ],
)
def test_merge_policy_rejects_unsafe_diff_or_identity(case: str) -> None:
    expected = {"logs/instruction.md", "logs/final.md", "logs/meta.json"}
    raw = b"".join(
        f":000000 100644 {'0' * 40} {'a' * 40} A\0{p}\0".encode()
        for p in sorted(expected)
    )
    schema.additions(raw, expected)
    if case in {"repository", "base", "head"}:
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
        else:
            bad["head"]["sha"] = "b" * 40
        with pytest.raises(journal.PublicationError, match="pr_identity_changed"):
            schema.identity(bad, row, "o/r")
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
