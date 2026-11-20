"""Contracts for the shared-memory source schema, tree checks and CLI."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS))
schema = importlib.import_module("shared_memory.schema")
render = importlib.import_module("shared_memory.render")
checks = importlib.import_module("shared_memory.validate")
cli = importlib.import_module("shared_memory.cli_impl")

TODAY = date(2026, 11, 20)
MEMORY = ".agents/memory"
ENTRY = f"{MEMORY}/workflow/example.md"
SUMMARY = f"{MEMORY}/SUMMARY.md"
INDEX = f"{MEMORY}/workflow/index.md"


def _entry(**fields: object) -> dict[str, object]:
    return {
        "title": "作業の確認",
        "summary": "作業結果を確認する。",
        "applies_to": "all",
        "source": "#2324",
        "created": date(2026, 8, 1),
        "last_verified": TODAY,
        "evidence": "コードを読む",
        "citations": [],
        **fields,
    }


def _document(metadata: dict[str, object], body: str = "観察の本文。\n") -> str:
    header = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    return f"---\n{header}---\n\n{body}"


def _tree(entries: dict[str, str]) -> dict[str, str | bytes]:
    parsed = [
        schema.parse_entry(
            Path(path).parent.name, Path(path).name, document, today=TODAY
        )
        for path, document in entries.items()
    ]
    return {
        ".agents/skills/memory/SKILL.md": "# Memory\n\n## 形式\n",
        # The dedicated observations have their own schema and content checks.
        f"{MEMORY}/ops-report/record.md": "Dedicated entry without shared metadata.\n",
        "code.py": "def verify_contract(): ...\n",
        "guide.md": (
            "# 手順の **確認**!\n\n# 手順の **確認**!\n\n"
            "<a name='old-path'></a>\n\n```md\n# Hidden heading\n```\n"
        ),
        **entries,
        **render.render_documents(parsed),
    }


def _write_tree(root: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            content.encode("utf-8") if isinstance(content, str) else content
        )


def test_schema_accepts_gemini_iso_strings_and_full_summary_length() -> None:
    meta = _entry(
        applies_to="gemini",
        last_verified="2026-11-20",
        summary="確" * 160,
        citations=[{"path": "code.py", "symbol": "verify_contract", "note": "確認先"}],
    )
    entry = schema.parse_entry("workflow", "example.md", _document(meta), today=TODAY)
    assert entry.last_verified == TODAY
    assert entry.citations == (schema.Citation("code.py", "verify_contract", "確認先"),)
    assert entry.summary == "確" * 160 and entry.applies_to == "gemini"


@pytest.mark.parametrize("missing", ["summary", "citations"])
def test_schema_requires_shared_fields(missing: str) -> None:
    meta = _entry()
    del meta[missing]
    with pytest.raises(ValueError, match=missing):
        schema.parse_entry("decision", "example.md", _document(meta), today=TODAY)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("title", "one\ntwo", "title must be on one line"),
        ("title", "title\n", "title must be on one line"),
        ("summary", "  ", "summary must be a non-empty string"),
        ("summary", 12, "summary must be a non-empty string"),
        ("summary", "one\ntwo", "summary must be on one line"),
        ("summary", "界" * 161, "summary exceeds 160 characters"),
        ("applies_to", "other", "applies_to must be one of"),
        ("applies_to", ["gemini"], "applies_to must be a non-empty string"),
        ("source", None, "source must be a non-empty string"),
        ("evidence", {}, "evidence must be a non-empty string"),
        ("created", "20260801", "created must be an ISO date"),
        ("last_verified", "2026-11-20T00:00:00", "last_verified must be an ISO date"),
        ("last_verified", "2026-02-30", "last_verified must be an ISO date"),
        ("last_verified", "2026-11-21", "created <= last_verified <= today"),
        ("created", "2026-11-21", "created <= last_verified <= today"),
        ("type", "decision", "type is supplied by the layer directory"),
    ],
)
def test_schema_rejects_invalid_metadata(
    field: str, value: object, message: str
) -> None:
    doc = _document(_entry(**{field: value}))
    with pytest.raises(ValueError, match=message):
        schema.parse_entry("decision", "example.md", doc, today=TODAY)


@pytest.mark.parametrize(
    ("filename", "text", "message"),
    [
        ("bad_name.md", _document(_entry()), "kebab-case"),
        ("example.md", "# Missing header\n", "missing YAML frontmatter"),
        ("example.md", "---\n- one\n---\n", "frontmatter must be a mapping"),
        ("example.md", "---\nsummary: [\n---\n", "invalid YAML frontmatter"),
        (
            "example.md",
            _document(_entry()).replace("summary:", "summary: Duplicate\nsummary:", 1),
            "YAML mapping keys must be unique strings",
        ),
    ],
)
def test_schema_rejects_malformed_entry_files(
    filename: str, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        schema.parse_entry("gotcha", filename, text, today=TODAY)


@pytest.mark.parametrize(
    ("citations", "message"),
    [
        (None, "citations must be a list"),
        ({"path": "guide.md"}, "citations must be a list"),
        (["guide.md"], "requires path and accepts only"),
        ([{"symbol": "check"}], "requires path and accepts only"),
        ([{"path": "guide.md", "range": "1-3"}], "requires path and accepts only"),
        ([{"path": []}], "path must be a non-empty string"),
        ([{"path": "/outside.py"}], "repository-relative path"),
        ([{"path": "docs/../guide.md"}], "repository-relative path"),
        ([{"path": "docs\\guide.md"}], "repository-relative path"),
        ([{"path": "C:/outside.py"}], "repository-relative path"),
        ([{"path": "guide.md", "symbol": None}], "symbol must be a non-empty string"),
        ([{"path": "guide.md", "note": " "}], "note must be a non-empty string"),
    ],
)
def test_schema_rejects_malformed_citations(citations: object, message: str) -> None:
    with pytest.raises(schema.CitationError, match=message):
        schema.parse_entry(
            "workflow",
            "example.md",
            _document(_entry(citations=citations)),
            today=TODAY,
        )


def test_file_map_resolves_code_symbols_and_rendered_markdown_anchors() -> None:
    citations = [
        {"path": "code.py", "symbol": "verify_contract"},
        {"path": "guide.md", "symbol": "#手順の-確認-1"},
        {"path": "guide.md", "symbol": "#old-path"},
    ]
    body = (
        "[guide][g]\n\n[g]: ../../../guide.md#手順の-確認\n\n"
        "[index](index.md)\n\n"
        "```md\n[example for prose](missing.md)\n```\n"
    )
    files = _tree({ENTRY: _document(_entry(citations=citations), body)})
    assert checks.validate(files, today=TODAY) == []


def test_broken_citations_and_relative_links_report_the_owning_entry() -> None:
    citations = [
        {"path": "gone.py"},
        {"path": "code.py", "symbol": "verify_other"},
        {"path": "guide.md", "symbol": "#hidden-heading"},
        {"path": ".agents/memory"},
    ]
    body = (
        "[anchor](../../../guide.md#missing)\n\n"
        "[file](missing.md)\n\n"
        "[outside](../../../../outside.md)\n"
    )
    files = _tree({ENTRY: _document(_entry(citations=citations), body)})
    report = checks.audit(files, today=TODAY)
    assert [(issue.path, issue.message) for issue in report.broken_citations] == [
        (ENTRY, "citations[0]: target file does not exist"),
        (ENTRY, "citations[1]: symbol was not found"),
        (ENTRY, "citations[2]: Markdown anchor was not found"),
        (ENTRY, "citations[3]: target file does not exist"),
    ]
    assert [issue.message for issue in report.problems if issue.code == "link"] == [
        "../../../guide.md#missing: Markdown anchor was not found",
        "missing.md: target does not exist",
        "../../../../outside.md: path resolves outside the repository",
    ]


@pytest.mark.parametrize(
    ("target", "text", "symbol", "message"),
    [
        pytest.param(
            "guide.md",
            "[Old heading](#removed-heading)\n\n```md\n#removed-heading\n```\n",
            "#removed-heading",
            "Markdown anchor was not found",
            id="removed-heading-remains-in-links-and-code",
        ),
        pytest.param(
            "code.py",
            "#removed-heading\n",
            "#removed-heading",
            "anchor symbols require a Markdown file",
            id="anchor-on-code-file",
        ),
        pytest.param(
            "guide.md",
            "## Real Heading\n",
            "real-heading",
            "symbol was not found",
            id="plain-symbol-is-a-text-search",
        ),
    ],
)
def test_citations_distinguish_text_from_anchor_selectors(
    target: str, text: str, symbol: str, message: str
) -> None:
    files = _tree(
        {ENTRY: _document(_entry(citations=[{"path": target, "symbol": symbol}]))}
    )
    files[target] = text
    for problems in (
        checks.validate(files, today=TODAY),
        checks.plan_sync(files, today=TODAY)[1],
    ):
        assert [(issue.path, issue.code, issue.message) for issue in problems] == [
            (ENTRY, "citation", f"citations[0]: {message}")
        ]


def test_checkout_checks_symlinks_for_citations_and_relative_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    outside = tmp_path / "other"
    (outside / "nested").mkdir(parents=True)
    (outside / "nested/file.py").write_text("outside contents\n")
    (outside / "return").symlink_to(root, target_is_directory=True)
    citations = [
        {"path": "internal.py", "symbol": "verify_contract"},
        {"path": "escape/nested/file.py"},
        {"path": "escape/return/code.py"},
    ]
    body = "[outside](../../../escape/nested/file.py)\n"
    _write_tree(root, _tree({ENTRY: _document(_entry(citations=citations), body)}))
    (root / "internal.py").symlink_to("code.py")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    report = checks.audit(root, today=TODAY)
    assert [issue.message for issue in report.broken_citations] == [
        "citations[1]: path resolves outside the repository",
        "citations[2]: path resolves outside the repository",
    ]
    assert any(
        issue.code == "link" and issue.path == ENTRY for issue in report.problems
    )


def test_checkout_checks_relative_link_components_before_parent_segments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    outside = tmp_path / "other"
    outside.mkdir()
    link = "../../../.venv/bin/python/../../../AGENTS.md"
    files = _tree({ENTRY: _document(_entry(), f"[Guide]({link})\n")})
    files["AGENTS.md"] = "# Agent guidelines\n"
    _write_tree(root, files)
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(outside, target_is_directory=True)
    for problems in (
        checks.validate(root, today=TODAY),
        checks.plan_sync(root, today=TODAY)[1],
    ):
        assert [(issue.path, issue.code, issue.message) for issue in problems] == [
            (ENTRY, "link", f"{link}: path resolves outside the repository")
        ]


def test_checkout_validates_the_anchor_on_the_resolved_internal_path(
    tmp_path: Path,
) -> None:
    body = "[Refer to the guide](../../../alias/../guide.md#actual-destination)\n"
    files = _tree({ENTRY: _document(_entry(), body)})
    files["docs/guide.md"] = "# Actual Destination\n"
    _write_tree(tmp_path, files)
    (tmp_path / "docs/nested").mkdir()
    (tmp_path / "alias").symlink_to("docs/nested", target_is_directory=True)
    assert checks.validate(tmp_path, today=TODAY) == []


def test_shared_entries_and_summary_reject_dedicated_memory_references() -> None:
    dedicated = f"{MEMORY}/ops-report/index.md"
    files = _tree(
        {
            ENTRY: _document(
                _entry(
                    summary="[Digest](ops-report/index.md#digest)",
                    citations=[{"path": dedicated, "symbol": "#digest"}],
                ),
                "[Digest](../ops-report/index.md#digest)\n",
            )
        }
    )
    files[dedicated] = "# Digest\n"
    for problems in (
        checks.validate(files, today=TODAY),
        checks.plan_sync(files, today=TODAY)[1],
    ):
        assert {(issue.path, issue.code) for issue in problems} == {
            (ENTRY, "dedicated_reference"),
            (ENTRY, "citation"),
            (ENTRY, "link"),
            (SUMMARY, "dedicated_reference"),
            (SUMMARY, "link"),
        }


def test_dedicated_memory_references_through_a_checkout_alias_are_rejected(
    tmp_path: Path,
) -> None:
    files = _tree(
        {
            ENTRY: _document(
                _entry(citations=[{"path": "digest-alias/record.md"}]),
                "[Digest](../../../digest-alias/record.md)\n",
            )
        }
    )
    _write_tree(tmp_path, files)
    (tmp_path / "digest-alias").symlink_to(
        f"{MEMORY}/ops-report", target_is_directory=True
    )
    for problems in (
        checks.validate(tmp_path, today=TODAY),
        checks.plan_sync(tmp_path, today=TODAY)[1],
    ):
        assert {(issue.path, issue.code) for issue in problems} == {
            (ENTRY, "link"),
            (ENTRY, "citation"),
        }


def test_validate_reports_dedicated_memory_mentioned_in_a_stored_summary() -> None:
    files = _tree({ENTRY: _document(_entry())})
    files[SUMMARY] = "# Shared memory\n\nSee also ops-report.\n"
    assert (SUMMARY, "dedicated_reference") in {
        (issue.path, issue.code) for issue in checks.validate(files, today=TODAY)
    }


def test_layout_rejects_legacy_stray_and_nested_paths() -> None:
    files = _tree({ENTRY: _document(_entry())})
    paths = [
        f"{MEMORY}/review/old.md",
        f"{MEMORY}/loose.md",
        f"{MEMORY}/workflow/data.txt",
        f"{MEMORY}/workflow/nested.md/entry.md",
    ]
    for path in paths:
        files[path] = "stray"
    assert {issue.path for issue in checks.validate(files, today=TODAY)} == {
        f"{MEMORY}/review",
        f"{MEMORY}/loose.md",
        f"{MEMORY}/workflow/data.txt",
        f"{MEMORY}/workflow/nested.md",
    }


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("fixture.user@example.invalid", "email address"),
        ("ghp_" + "f" * 36, "token or API key"),
        ("github_pat_" + "f" * 36, "token or API key"),
        ("sk-" + "f" * 36, "token or API key"),
        ("/home/fixture/settings.txt", "machine-specific path"),
    ],
)
def test_private_data_in_body_is_detected_without_echoing_it(
    value: str, kind: str
) -> None:
    files = _tree({ENTRY: _document(_entry(), f"```\n{value}\n```\n")})
    problems = checks.validate(files, today=TODAY)
    assert [(issue.code, issue.message) for issue in problems] == [
        ("private_data", f"contains {kind}")
    ]
    assert value not in repr(problems)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        pytest.param("[Local](http://localhost:9000/ui)", "local URL", id="localhost"),
        pytest.param("[Local](http://127.0.0.1/ui)", "local URL", id="ipv4-loopback"),
        pytest.param("[Local](http://[::1]:9000/ui)", "local URL", id="ipv6-loopback"),
        pytest.param("[Local](//service.internal/ui)", "local URL", id="private-host"),
        pytest.param("[Local](http://%6cocalhost/ui)", "local URL", id="encoded-host"),
        pytest.param(
            "[Local](file:///home/fixture/settings.txt)",
            "machine-specific path",
            id="file-url",
        ),
        pytest.param("token = fixture-value", "credential assignment", id="token"),
        pytest.param("api_key: fixture-value", "credential assignment", id="api-key"),
        pytest.param("secret=fixture-value", "credential assignment", id="secret"),
        pytest.param(
            '"password": "fixture-value"', "credential assignment", id="password"
        ),
    ],
)
def test_shared_memory_rejects_local_urls_and_credential_assignments(
    text: str, kind: str
) -> None:
    files = _tree({ENTRY: _document(_entry(), f"{text}\n")})
    for problems in (
        checks.validate(files, today=TODAY),
        checks.plan_sync(files, today=TODAY)[1],
    ):
        assert [(issue.code, issue.message) for issue in problems] == [
            ("private_data", f"contains {kind}")
        ]
        assert text not in repr(problems)


def test_renderer_is_order_independent_uses_filenames_and_includes_each_layer() -> None:
    entries = [
        schema.parse_entry(
            layer,
            name,
            _document(_entry(title=title, summary=summary, applies_to=scope)),
            today=TODAY,
        )
        for layer, name, title, summary, scope in [
            ("gotcha", "first.md", "D", "gotcha one", "all"),
            ("decision", "first.md", "C", "decision one", "all"),
            ("workflow", "zulu.md", "A", "workflow two", "all"),
            ("environment", "first.md", "B", "environment one", "claude"),
            ("workflow", "alpha.md", "Z", "workflow one", "gemini"),
        ]
    ]
    rendered = render.render_documents(entries)
    assert rendered == render.render_documents(reversed(entries))
    assert rendered[INDEX].splitlines()[-2:] == [
        "- [Z](alpha.md) (gemini)",
        "- [A](zulu.md)",
    ]
    assert [
        line
        for line in rendered[SUMMARY].splitlines()
        if line.startswith(("## ", "- "))
    ] == [
        "## [workflow](workflow/index.md) — 作業のコツ",
        "- workflow one",
        "- workflow two",
        "## [decision](decision/index.md) — ユーザーの判断",
        "- decision one",
        "## [environment](environment/index.md) — 環境の癖",
        "- environment one",
        "## [gotcha](gotcha/index.md) — 落とし穴",
        "- gotcha one",
    ]


def test_limits_distinguish_utf8_bytes_from_lines_and_report_excess() -> None:
    files = _tree(
        {
            f"{MEMORY}/workflow/entry-{index}.md": _document(_entry(summary="界" * 120))
            for index in range(23)
        }
    )
    report = checks.audit(files, today=TODAY)
    assert [(usage.path, usage.unit) for usage in report.limits if usage.exceeded] == [
        (SUMMARY, "bytes")
    ]
    assert [(issue.path, issue.code) for issue in report.problems] == [
        (SUMMARY, "limit")
    ]
    usage = next(
        row for row in report.limits if row.path == SUMMARY and row.unit == "bytes"
    )
    assert usage.maximum == 8192 and usage.used > 8192 and usage.ratio > 1.0
    assert any(
        row.path == SUMMARY and row.unit == "lines" and row.used < 80
        for row in report.limits
    )


def test_sync_checks_generated_entry_limits_before_writing() -> None:
    files = _tree(
        {
            f"{MEMORY}/workflow/entry-{index}.md": _document(_entry(summary=str(index)))
            for index in range(31)
        }
    )
    problems = checks.plan_sync(files, today=TODAY)[1]
    assert [(issue.path, issue.message) for issue in problems] == [
        (INDEX, "31 entries exceeds the limit of 30")
    ]


def test_summary_and_entry_line_limits_are_independent() -> None:
    files = _tree({ENTRY: _document(_entry(), "body\n" * 40)})
    files[SUMMARY] = "line\n" * 81
    report = checks.audit(files, today=TODAY)
    assert {
        (usage.path, usage.unit, usage.maximum)
        for usage in report.limits
        if usage.exceeded
    } == {(SUMMARY, "lines", 80), (ENTRY, "lines", 40)}


def test_audit_signals_only_ages_older_than_ninety_days() -> None:
    expired_path = f"{MEMORY}/decision/expired.md"
    files = _tree(
        {
            ENTRY: _document(_entry(last_verified=TODAY - timedelta(days=90))),
            expired_path: _document(_entry(last_verified=TODAY - timedelta(days=91))),
        }
    )
    report = checks.audit(files, today=TODAY)
    assert not report.problems
    assert [(row.path, row.last_verified, row.age_days) for row in report.stale] == [
        (expired_path, "2026-08-21", 91)
    ]


def test_cli_sync_check_compares_bytes_and_preserves_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = _tree({ENTRY: _document(_entry())})
    _write_tree(tmp_path, files)
    summary = tmp_path / SUMMARY
    old = summary.read_bytes().replace(b"\n", b"\r\n")
    summary.write_bytes(old)
    index = tmp_path / INDEX
    index.unlink()

    argv = ["sync-index", "--repo", str(tmp_path)]
    assert cli.main([*argv, "--check"]) == 1
    assert summary.read_bytes() == old and not index.exists()
    assert SUMMARY in capsys.readouterr().err
    assert cli.main(argv) == 0
    assert summary.read_bytes() != old and index.is_file()
    before = summary.stat().st_ino
    assert cli.main([*argv, "--check"]) == 0
    assert cli.main(argv) == 0 and summary.stat().st_ino == before
    assert cli.main(["validate", "--repo", str(tmp_path)]) == 0


def test_sync_refuses_to_write_through_a_generated_symlink(tmp_path: Path) -> None:
    files = _tree({ENTRY: _document(_entry())})
    _write_tree(tmp_path, files)
    index = tmp_path / INDEX
    summary = tmp_path / SUMMARY
    summary.write_text("out of sync\n")
    target = tmp_path / "code.py"
    before = target.read_bytes()
    index.unlink()
    index.symlink_to(target)
    assert cli.main(["sync-index", "--repo", str(tmp_path)]) == 1
    assert summary.read_text() == "out of sync\n"
    assert index.is_symlink() and target.read_bytes() == before


def test_module_entrypoint_validates_and_emits_audit_json(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        _tree({ENTRY: _document(_entry(citations=[{"path": "code.py"}]))}),
    )
    (tmp_path / "code.py").unlink()
    validate_result = subprocess.run(
        [sys.executable, "-m", "shared_memory", "validate"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(OPS)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate_result.returncode == 1
    assert f"{ENTRY}: [citation]" in validate_result.stderr
    audit_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shared_memory",
            "audit",
            "--repo",
            str(tmp_path),
            "--today",
            "2027-02-19",
        ],
        env={**os.environ, "PYTHONPATH": str(OPS)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit_result.returncode == 1
    report = json.loads(audit_result.stdout)
    assert report["today"] == "2027-02-19"
    assert report["stale"] == [
        {"path": ENTRY, "last_verified": "2026-11-20", "age_days": 91}
    ]
    assert report["broken_citations"][0]["path"] == ENTRY
    assert report["limits"]
