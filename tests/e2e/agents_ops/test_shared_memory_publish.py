"""Exercise the dedicated publisher with real Git objects and a fake GitHub API."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
checks = importlib.import_module("shared_memory.validate")
operations = importlib.import_module("shared_memory.operations")
publish = importlib.import_module("shared_memory.publish")
github = importlib.import_module("shared_memory.github")

MEMORY = ".agents/memory"
SUMMARY = f"{MEMORY}/SUMMARY.md"
CURRENT = f"{MEMORY}/workflow/current.md"
OLD = f"{MEMORY}/decision/old.md"
LAYERS = ("workflow", "decision", "environment", "gotcha")
PR_ENDPOINT = "repos/test-owner/tennis-memory/pulls/17"
MAIN_ENDPOINT = "repos/test-owner/tennis-memory/git/ref/heads/main"


def add(
    layer: str = "workflow", slug: str = "new", **fields: object
) -> dict[str, object]:
    return {
        "op": "add",
        "layer": layer,
        "slug": slug,
        "title": "New observation",
        "summary": "New summary.",
        "applies_to": "codex",
        "source": "Test observation",
        "evidence": "Check the documented behavior again.",
        "citations": [{"path": "reference.md", "symbol": "#current-heading"}],
        "body": "See [the reference](../../../reference.md#current-heading).",
        **fields,
    }


def update(slug: str = "current", **fields: object) -> dict[str, object]:
    return {
        "op": "update",
        "layer": "workflow",
        "slug": slug,
        "reason": "根拠を再確認",
        **fields,
    }


def delete(
    layer: str = "decision", slug: str = "old", **fields: object
) -> dict[str, object]:
    return {
        "op": "delete",
        "layer": layer,
        "slug": slug,
        "reason": "観察の役目を終えた",
        **fields,
    }


def _write_entry(repo: Path, path: str, **fields: object) -> None:
    metadata = {**add(), **fields}
    body = metadata.pop("body")
    for key in ("op", "layer", "slug"):
        metadata.pop(key)
    metadata.setdefault("created", "2026-01-01")
    metadata.setdefault("last_verified", "2026-01-01")
    header = yaml.safe_dump(metadata, sort_keys=False)
    (repo / path).write_text(f"---\n{header}---\n\n{body}\n", encoding="utf-8")


def _sync(repo: Path) -> None:
    edits, problems = checks.plan_sync(repo)
    assert not problems, problems
    for path, data in edits.items():
        (repo / path).write_bytes(data)


@dataclass(frozen=True)
class Repository:
    tmp: Path
    source: Path
    origin: Path
    real_git: str

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.check_output(
            [
                self.real_git,
                "-c",
                "user.name=Tester",
                "-c",
                "user.email=tests@example.test",
                *args,
            ],
            cwd=cwd or self.source,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    def bare(self, *args: str) -> str:
        return self.git(f"--git-dir={self.origin}", *args)

    def cli(
        self, *args: str, input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "shared_memory", *args, "--repo", str(self.source)],
            cwd=self.tmp,
            input=input,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )

    def batch(
        self, specs: list[dict[str, object]], *args: str
    ) -> subprocess.CompletedProcess[str]:
        return self.cli("batch", "--spec", "-", *args, input=json.dumps(specs))

    def api_calls(self) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in (self.tmp / "api-log").read_text().splitlines()
        ]

    def git_calls(self, name: str) -> list[str]:
        return [
            line
            for line in (self.tmp / "git-log").read_text().splitlines()
            if f" {name} " in line
        ]

    def assert_no_publication(self) -> None:
        assert self.api_calls() == []
        assert self.git_calls("push") == []
        assert self.bare("for-each-ref", "refs/heads/shared-memory") == ""

    def snapshot(self) -> tuple[str, str, str, bytes, bytes]:
        return (
            self.git("status", "--porcelain", "--branch"),
            self.git("rev-parse", "HEAD", "refs/remotes/origin/main"),
            self.git("worktree", "list", "--porcelain"),
            (self.source / ".git/index").read_bytes(),
            (self.source / ".git/FETCH_HEAD").read_bytes(),
        )

    def parallel_commit(self, kind: str, *, ref: str = "parallel") -> str:
        other = self.tmp / "other"
        self.git("clone", str(self.origin), str(other))
        if kind == "new-entry":
            _write_entry(
                other, f"{MEMORY}/workflow/parallel.md", summary="Parallel observation."
            )
            _sync(other)
        elif kind == "updated-summary":
            _write_entry(
                other, CURRENT, title="Current observation", summary="New summary."
            )
            _sync(other)
        else:
            assert kind == "removed-anchor"
            (other / "reference.md").write_text("# Changed heading\n")
        self.git("add", "-A", cwd=other)
        self.git("commit", "-qm", "Parallel change", cwd=other)
        self.git("push", "origin", f"HEAD:refs/heads/{ref}", cwd=other)
        return self.git("rev-parse", "HEAD", cwd=other)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repository:
    real_git = shutil.which("git")
    assert real_git
    for key in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        monkeypatch.setenv(key, os.devnull)
    source, origin = tmp_path / "source", tmp_path / "origin.git"
    source.mkdir()
    fixture = Repository(tmp_path, source, origin, real_git)
    fixture.git("init", "-q", "-b", "main")
    fixture.git("init", "-q", "--bare", "-b", "main", str(origin))
    fixture.git("config", "user.name", "Tester")
    fixture.git("config", "user.email", "tests@example.test")
    fixture.git("remote", "add", "origin", str(origin))
    for layer in (*LAYERS, "ops-report"):
        (source / MEMORY / layer).mkdir(parents=True)
    (source / MEMORY / "ops-report/SUMMARY.md").write_text("Independent contents.\n")
    skill = source / ".agents/skills/memory/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# 共有memory\n\n## 形式\n")
    (source / "reference.md").write_text("# Current heading\n\nVerifiable detail.\n")
    (source / "escaped.md").symlink_to("../outside.md")
    _write_entry(
        source,
        CURRENT,
        title="Current observation",
        summary="Current summary.",
        citations=[],
    )
    _write_entry(
        source,
        OLD,
        title="Old observation",
        summary="Old summary.",
        applies_to="all",
        source="Earlier trial of annotation review",
        evidence="Compare the archived settings in the decision record.",
        citations=[],
        body="A former workaround for reviewer handoff required a second pass.\n\nThe current workflow keeps the pending decisions in the same queue.",
    )
    _sync(source)
    fixture.git("add", "-A")
    fixture.git("commit", "-qm", "Seed shared memory")
    fixture.git("push", "origin", "main")
    fixture.git("switch", "-qc", "callers-branch")
    (source / ".git/FETCH_HEAD").write_text("caller fetch marker\n")

    tools = tmp_path / "bin"
    tools.mkdir()
    git_wrapper = tools / "git"
    git_wrapper.write_text(
        "#!/bin/bash\n"
        'case " $* " in\n'
        '  *" remote get-url --push "*) printf "%s\\n" "${MEMORY_TEST_PUSH_URL:-git@github.com:test-owner/tennis-memory.git}"; exit 0;;\n'
        '  *" remote get-url "*) printf "%s\\n" "git@github.com:test-owner/tennis-memory.git"; exit 0;;\n'
        '  *" fetch "*|*" push "*) printf "%s\\n" "$*" >> "$MEMORY_TEST_GIT_LOG";;\n'
        "esac\n"
        'exec "$MEMORY_TEST_GIT" "$@"\n'
    )
    git_wrapper.chmod(0o755)
    gh = tools / "gh"
    helper = Path(__file__).with_name("fake_shared_memory_gh.py")
    gh.write_text(f"#!{sys.executable}\n" + helper.read_text(encoding="utf-8"))
    gh.chmod(0o755)
    (tmp_path / "api-log").touch()
    (tmp_path / "git-log").touch()
    for key, value in {
        "MEMORY_TEST_GIT": real_git,
        "MEMORY_TEST_ORIGIN": str(origin),
        "MEMORY_TEST_PR": str(tmp_path / "pr.json"),
        "MEMORY_TEST_API_LOG": str(tmp_path / "api-log"),
        "MEMORY_TEST_GIT_LOG": str(tmp_path / "git-log"),
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(OPS),
    }.items():
        monkeypatch.setenv(key, value)
    return fixture


def _report(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report: dict[str, Any] = json.loads(result.stdout)
    assert report["validation"] == "passed"
    return report


@pytest.mark.parametrize(
    "command", ["add", "update", "delete-flags", "delete-spec", "batch"]
)
def test_cli_publishes_one_verified_commit(repo: Repository, command: str) -> None:
    (repo.source / "reference.md").write_text(
        "Uncommitted change removes the anchor.\n"
    )
    (repo.source / "staged.txt").write_text("Owned by caller.\n")
    repo.git("add", "staged.txt")
    local = repo.snapshot()
    base = repo.bare("rev-parse", "main")
    addition = add("environment", reason="環境の挙動を実測")
    revision = update(title="Rechecked observation", summary="Rechecked summary.")
    if command == "batch":
        result = repo.batch(
            [
                addition,
                add("gotcha", "separate", summary="Separate note."),
                revision,
                delete(),
            ]
        )
    elif command == "delete-flags":
        result = repo.cli(
            "delete", "--layer", "decision", "--slug", "old", "--reason", "廃止を確認"
        )
    else:
        spec = {"add": addition, "update": revision, "delete-spec": delete()}[
            command
        ].copy()
        spec.pop("op")
        spec_path = repo.tmp / "spec.json"
        spec_path.write_text(json.dumps(spec))
        result = repo.cli(command.removesuffix("-spec"), "--spec", str(spec_path))
    report = _report(result)
    head, merged = report["head"], report["merge_sha"]
    assert (
        report["base"] == base and report["dry_run"] is False and report["retries"] == 0
    )
    assert report["pr"] == 17 and report["url"].endswith("/pull/17")
    assert report["branch_cleanup_error"] is None
    assert repo.bare("rev-list", "--parents", "-n", "1", "main").split() == [
        merged,
        base,
        head,
    ]
    assert repo.bare("rev-list", "--parents", "-n", "1", head).split() == [head, base]
    assert repo.bare("for-each-ref", "refs/heads/shared-memory") == ""
    assert (
        repo.bare("show", f"main:{MEMORY}/ops-report/SUMMARY.md")
        == "Independent contents."
    )
    assert repo.snapshot() == local

    expected = {SUMMARY: "modified"}
    if command in {"add", "batch"}:
        path = f"{MEMORY}/environment/new.md"
        index = f"{MEMORY}/environment/index.md"
        expected.update({path: "added", index: "modified"})
        assert "- New summary." in repo.bare("show", f"main:{SUMMARY}")
        assert "[New observation](new.md) (codex)" in repo.bare("show", f"main:{index}")
        assert "#current-heading" in repo.bare("show", f"main:{path}")
    if command in {"update", "batch"}:
        expected.update(
            {CURRENT: "modified", f"{MEMORY}/workflow/index.md": "modified"}
        )
        assert "- Rechecked summary." in repo.bare("show", f"main:{SUMMARY}")
        assert "source: Test observation" in repo.bare("show", f"main:{CURRENT}")
    if command.startswith("delete") or command == "batch":
        expected.update({OLD: "removed", f"{MEMORY}/decision/index.md": "modified"})
        assert "Old summary." not in repo.bare("show", f"main:{SUMMARY}")
        assert repo.bare("ls-tree", "main", "--", OLD) == ""
    if command == "batch":
        expected.update(
            {
                f"{MEMORY}/gotcha/separate.md": "added",
                f"{MEMORY}/gotcha/index.md": "modified",
            }
        )
        assert len(report["operations"]) == 4
    assert report["files"] == expected
    calls = repo.api_calls()
    assert [(row["method"], row["path"]) for row in calls] == [
        ("POST", PR_ENDPOINT.rsplit("/", 1)[0]),
        ("GET", PR_ENDPOINT),
        ("GET", MAIN_ENDPOINT),
        ("GET", f"{PR_ENDPOINT}/files?per_page=100"),
        ("GET", PR_ENDPOINT),
        ("GET", MAIN_ENDPOINT),
        ("PUT", f"{PR_ENDPOINT}/merge"),
    ]
    assert calls[0]["data"]["draft"] is False
    assert calls[0]["data"]["head"].startswith("shared-memory/")
    pr_body = calls[0]["data"]["body"]
    assert "機械検査" in pr_body
    assert all(path in pr_body for path in expected)
    assert all(op["reason"] in pr_body for op in report["operations"])
    assert calls[-1]["data"] == {"merge_method": "merge", "sha": head}


def test_dry_run_uses_latest_main_and_leaves_the_caller_index_and_fetch_head(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    newest = repo.parallel_commit("new-entry", ref="main")
    snapshot = repo.snapshot()
    with monkeypatch.context() as context:
        context.setenv("GIT_INDEX_FILE", str(repo.source / ".git/index"))
        context.setenv("GIT_WORK_TREE", str(repo.tmp))
        report = _report(repo.batch([add()], "--dry-run"))
    assert report["base"] == newest and report["dry_run"] is True
    assert (
        report["pr"] is None and report["url"] is None and report["merge_sha"] is None
    )
    assert report["files"][f"{MEMORY}/workflow/new.md"] == "added"
    generated = repo.git("show", f"{report['head']}:{SUMMARY}")
    assert "- Parallel observation." in generated and "- New summary." in generated
    assert repo.bare("rev-parse", "main") == newest
    assert repo.snapshot() == snapshot
    repo.assert_no_publication()


def test_python_api_returns_the_same_validated_edits_without_writing(
    repo: Repository,
) -> None:
    spec = [
        add("environment", body="## Evidence\n\nObserved behavior."),
        add(
            "gotcha",
            "related",
            citations=[{"path": f"{MEMORY}/environment/new.md", "symbol": "#evidence"}],
        ),
        delete(),
    ]
    snapshot = repo.snapshot()
    edits = operations.apply_operations(repo.source, spec)
    result = publish.publish_operations(repo.source, spec, dry_run=True)
    assert edits[OLD] is None and set(result.files) == edits.keys()
    for path, content in edits.items():
        if content is not None:
            assert repo.git("show", f"{result.head}:{path}") == content.decode().strip()
    assert repo.snapshot() == snapshot
    repo.assert_no_publication()


@pytest.mark.parametrize("status", [405, 409])
def test_merge_refusal_reapplies_to_the_advanced_main_with_exact_lease(
    repo: Repository, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    latest = repo.parallel_commit("new-entry")
    monkeypatch.setenv(
        "MEMORY_TEST_CASE", json.dumps({"conflicts": [status], "advance_to": latest})
    )
    report = _report(repo.batch([add()]))
    assert report["retries"] == 1 and report["base"] == latest
    creates = [row for row in repo.api_calls() if row["method"] == "POST"]
    merges = [row["data"]["sha"] for row in repo.api_calls() if row["method"] == "PUT"]
    assert (
        len(creates) == 1
        and len(merges) == 2
        and merges[0] != merges[1] == report["head"]
    )
    branch = creates[0]["data"]["head"]
    initial, retry, cleanup = repo.git_calls("push")
    assert f"{merges[0]}:refs/heads/{branch}" in initial
    assert f"--force-with-lease=refs/heads/{branch}:{merges[0]}" in retry
    assert f"{merges[1]}:refs/heads/{branch}" in retry
    assert f"--force-with-lease=refs/heads/{branch}:{merges[1]}" in cleanup
    assert len(repo.git_calls("fetch")) == 2
    assert "- Parallel observation." in repo.bare("show", f"main:{SUMMARY}")
    assert "- New summary." in repo.bare("show", f"main:{SUMMARY}")


def test_main_advance_detected_by_get_refreshes_the_pr_body_and_diff(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest = repo.parallel_commit("updated-summary")
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps({"advance_on_read": latest}))
    report = _report(
        repo.batch([update(summary="New summary.", body="Rechecked the evidence.")])
    )
    assert report["retries"] == 1 and report["files"] == {CURRENT: "modified"}
    calls = repo.api_calls()
    assert [row["method"] for row in calls].count("PUT") == 1
    assert SUMMARY in calls[0]["data"]["body"]
    patch = next(row for row in calls if row["method"] == "PATCH")
    assert f"`{SUMMARY}`" not in patch["data"]["body"]
    assert f"`{CURRENT}`" in patch["data"]["body"]


@pytest.mark.parametrize(
    ("case", "error", "attempts", "fetches"),
    [
        ({"conflicts": [405]}, "remote main has not advanced", 1, 2),
        (
            {"conflicts": [409] * 4, "advance_empty": True},
            "retry limit reached (3)",
            4,
            4,
        ),
        ({"response_lost_after_merge": True}, "HTTP 503", 1, 1),
        (
            {"conflicts": [409], "advance_empty": True, "broken_conflict_body": True},
            "ambiguous HTTP 409 response",
            1,
            1,
        ),
    ],
)
def test_merge_failure_stops_without_cleanup_or_extra_mutations(
    repo: Repository,
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, object],
    error: str,
    attempts: int,
    fetches: int,
) -> None:
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps(case))
    result = repo.batch([add()])
    assert result.returncode == 1 and not result.stdout
    assert "PR #17" in result.stderr and error in result.stderr
    assert sum(row["method"] == "PUT" for row in repo.api_calls()) == attempts
    assert sum(row["method"] == "POST" for row in repo.api_calls()) == 1
    assert len(repo.git_calls("fetch")) == fetches
    assert len(repo.git_calls("push")) == attempts
    assert repo.bare("for-each-ref", "refs/heads/shared-memory")


def test_retry_rejects_citation_broken_on_new_main_without_force_push(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    latest = repo.parallel_commit("removed-anchor")
    monkeypatch.setenv(
        "MEMORY_TEST_CASE", json.dumps({"conflicts": [405], "advance_to": latest})
    )
    result = repo.batch([add()])
    assert (
        result.returncode == 1
        and "[citation]" in result.stderr
        and "PR #17" in result.stderr
    )
    assert len(repo.git_calls("fetch")) == 2 and len(repo.git_calls("push")) == 1
    assert sum(row["method"] == "PUT" for row in repo.api_calls()) == 1
    assert repo.bare("rev-parse", "main") == latest


def test_cleanup_uses_lease_and_keeps_confirmed_merge_on_failure(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "MEMORY_TEST_CASE", json.dumps({"change_branch_after_merge": True})
    )
    report = _report(repo.batch([add()]))
    assert report["branch_cleanup_error"] and report["merge_sha"] == repo.bare(
        "rev-parse", "main"
    )
    assert (
        repo.bare("for-each-ref", "--format=%(objectname)", "refs/heads/shared-memory")
        == report["merge_sha"]
    )


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ({"pr_fields": {"head.sha": "a" * 40}}, "identity or state"),
        ({"pr_fields": {"base.ref": "other"}}, "identity or state"),
        ({"pr_fields": {"draft": True}}, "identity or state"),
        (
            {"pr_fields": {"head.repo": {"full_name": "fork/memory"}}},
            "unexpected repository",
        ),
        ({"pr_fields": {"changed_files": 99}}, "file count"),
        ({"head_after_files": True}, "identity or state"),
        (
            {"file_fields": {"filename": f"{MEMORY}/ops-report/SUMMARY.md"}},
            "files differ",
        ),
        ({"file_fields": {"status": "removed"}}, "files differ"),
        ({"file_fields": {"previous_filename": CURRENT}}, "files differ"),
        ({"duplicate_file": True}, "files differ"),
    ],
)
def test_github_mismatch_stops_before_merge(
    repo: Repository,
    monkeypatch: pytest.MonkeyPatch,
    case: dict[str, object],
    error: str,
) -> None:
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps(case))
    result = repo.batch([add()])
    assert result.returncode == 1 and not result.stdout
    assert error in result.stderr and "PR #17" in result.stderr
    assert len(repo.git_calls("push")) == 1
    assert all(row["method"] != "PUT" for row in repo.api_calls())


@pytest.mark.parametrize(
    ("specs", "error"),
    [
        ([add("review")], "layer must be"),
        ([add("ops-report")], "layer must be"),
        ([add(slug="../../file")], "slug must be kebab-case"),
        ([add(slug="index")], "cannot be index"),
        ([add(path="code.py")], "unknown add fields"),
        ([add(title=42)], "[entry_schema] title"),
        ([add(summary="a" * 161)], "summary exceeds 160"),
        (
            [add(citations=[{"path": "reference.md", "symbol": "#removed-heading"}])],
            "Markdown anchor was not found",
        ),
        ([add(citations=[{"path": ".git"}])], "target file does not exist"),
        (
            [add(citations=[{"path": f"{MEMORY}/ops-report/SUMMARY.md"}])],
            "dedicated ops-report memory",
        ),
        (
            [add(body="[digest](../ops-report/SUMMARY.md)")],
            "dedicated ops-report memory",
        ),
        (
            [add(body="[escape](../../../escaped.md)")],
            "path resolves outside the repository",
        ),
        ([add(body="ghp_" + "x" * 25)], "token or API key"),
        ([add(body="[local](http://localhost:8000/)")], "local URL"),
        ([add(body="[private](file:///home/example/doc.md)")], "machine-specific path"),
        ([add(body="password=fixture-only")], "credential assignment"),
        ([add(reason="api_key=fixture-only")], "reason contains credential assignment"),
        ([add(body="Long entry.\n" * 41)], "lines exceeds the limit of 40"),
        ([add(slug="current")], "entry already exists"),
        ([update("absent", title="Other")], "entry does not exist"),
        ([delete(slug="absent")], "entry does not exist"),
        ([delete(reason="  ")], "reason must be a non-empty string"),
        ([add(), delete("workflow", "new")], "duplicate operation"),
        ([add(), update(summary="Current summary.")], "update has no changes"),
        ([add(citations=[{"path": OLD}]), delete()], "target file does not exist"),
        ([], "non-empty sequence"),
    ],
)
def test_bad_batch_is_rejected_before_any_remote_write(
    repo: Repository, specs: list[dict[str, object]], error: str
) -> None:
    snapshot = repo.snapshot()
    result = repo.batch(specs)
    assert result.returncode == 1 and not result.stdout and error in result.stderr
    assert repo.snapshot() == snapshot
    repo.assert_no_publication()


@pytest.mark.parametrize(
    ("count", "summary", "budget"), [(66, "Tiny note.", 80), (24, "界" * 120, 8192)]
)
def test_summary_size_failure_is_caught_after_generation(
    repo: Repository, count: int, summary: str, budget: int
) -> None:
    result = repo.batch(
        [add(LAYERS[i % 4], f"long-{i}", summary=summary) for i in range(count)]
    )
    assert result.returncode == 1 and f"exceeds the limit of {budget}" in result.stderr
    assert f"{SUMMARY}: [limit]" in result.stderr
    repo.assert_no_publication()


@pytest.mark.parametrize(
    "path", [".github/workflows/ci.yml", f"{MEMORY}/workflow/extra.md"]
)
def test_actual_commit_diff_must_match_the_validated_files(
    repo: Repository, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    original = publish.Git.build_commit

    def polluted(git: Any, base: str, edits: dict[str, bytes | None]) -> str:
        head: str = original(git, base, {**edits, path: b"unrequested change\n"})
        return head

    monkeypatch.setattr(publish.Git, "build_commit", polluted)
    with pytest.raises(ValueError, match="committed diff"):
        publish.publish_operations(repo.source, [add()])
    repo.assert_no_publication()


def test_git_detected_rename_is_rejected_before_push(repo: Repository) -> None:
    result = repo.batch(
        [
            add(
                slug="relocated",
                title="Current observation",
                summary="Current summary.",
                citations=[],
                created="2026-01-01",
                last_verified="2026-01-01",
            ),
            delete("workflow", "current"),
        ]
    )
    assert result.returncode == 1 and "rename detected" in result.stderr
    repo.assert_no_publication()


def test_changed_executable_file_is_rejected_using_raw_git_modes(
    repo: Repository,
) -> None:
    (repo.source / CURRENT).chmod(0o755)
    repo.git("update-index", "--chmod=+x", CURRENT)
    repo.git("commit", "-qm", "Change file mode")
    repo.git("push", "origin", "HEAD:main")
    result = repo.batch([update(title="New title")])
    assert result.returncode == 1 and "unsafe file mode" in result.stderr
    repo.assert_no_publication()


def test_invalid_frontmatter_on_untouched_main_entry_prevents_publication(
    repo: Repository,
) -> None:
    (repo.source / OLD).write_text("---\ntitle: [invalid\n---\n")
    repo.git("add", "--", OLD)
    repo.git("commit", "-qm", "Break a shared memory entry")
    repo.git("push", "origin", "HEAD:main")
    result = repo.batch([add()])
    assert (
        result.returncode == 1
        and f"{OLD}: [entry_schema] invalid YAML" in result.stderr
    )
    repo.assert_no_publication()


def test_cli_rejects_duplicate_spec_keys(repo: Repository) -> None:
    result = repo.cli(
        "add", "--spec", "-", input='{"layer":"workflow","layer":"decision"}'
    )
    assert result.returncode == 1 and "JSON keys must be unique" in result.stderr
    assert repo.git_calls("fetch") == []
    repo.assert_no_publication()


def test_remote_identity_mismatch_is_rejected_before_fetch(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEMORY_TEST_PUSH_URL", "https://github.com/another/repo.git")
    result = repo.batch([add()])
    assert result.returncode == 1 and "different repositories" in result.stderr
    assert repo.git_calls("fetch") == []
    repo.assert_no_publication()


def test_github_files_pagination_requires_the_complete_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [f"{MEMORY}/{LAYERS[i % 4]}/note-{i}.md" for i in range(101)]
    pages = [
        [{"filename": path, "status": "added"} for path in paths[:100]],
        [{"filename": paths[-1], "status": "added"}],
    ]
    endpoints = []

    def page(method: str, endpoint: str) -> list[dict[str, str]]:
        assert method == "GET"
        endpoints.append(endpoint)
        return pages[int(endpoint.endswith("&page=2"))]

    client = github.GitHub("test-owner/tennis-memory")
    monkeypatch.setattr(client, "request", page)
    expected = dict.fromkeys(paths, "added")
    client.verify_files(17, expected)
    assert endpoints == [
        f"{PR_ENDPOINT}/files?per_page=100",
        f"{PR_ENDPOINT}/files?per_page=100&page=2",
    ]
    pages[1].clear()
    with pytest.raises(ValueError, match="file count"):
        client.verify_files(17, expected)


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_ci_ignores_only_shared_memory_markdown(event: str) -> None:
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    ignored = workflow["on"][event]["paths-ignore"]
    assert {pattern for pattern in ignored if pattern.startswith(MEMORY)} == {
        SUMMARY,
        *(f"{MEMORY}/{layer}/*.md" for layer in LAYERS),
    }
    for included in (
        f"{MEMORY}/ops-report/SUMMARY.md",
        ".agents/ops/shared_memory/publish.py",
        ".agents/skills/memory/SKILL.md",
        "tests/e2e/agents_ops/test_agents_docs.py",
        ".github/workflows/ci.yml",
    ):
        assert not any(fnmatchcase(included, pattern) for pattern in ignored)
