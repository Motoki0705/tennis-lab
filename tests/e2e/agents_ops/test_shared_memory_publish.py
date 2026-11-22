"""Exercise the dedicated publisher with real Git objects and a fake GitHub API."""

from __future__ import annotations

import importlib
import json
import os
import select
import signal
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.shared_memory import (
    CURRENT,
    LAYERS,
    MEMORY,
    OLD,
    SUMMARY,
    Repository,
    add,
    delete,
    memory_repo,
    update,
)

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
operations = importlib.import_module("shared_memory.operations")
publish = importlib.import_module("shared_memory.publish")
github = importlib.import_module("shared_memory.github")

PR_ENDPOINT = "repos/test-owner/tennis-memory/pulls/17"
MAIN_ENDPOINT = "repos/test-owner/tennis-memory/git/ref/heads/main"
repo = memory_repo


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
            "delete",
            "--layer",
            "decision",
            "--slug",
            "old",
            "--reason",
            "廃止を確認",
            "--expected-sha256",
            sha256((repo.source / OLD).read_bytes()).hexdigest(),
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
    assert repo.bare("rev-parse", "main") == merged == head
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
        ("POST", "graphql"),
        ("GET", PR_ENDPOINT),
    ]
    assert calls[0]["data"]["draft"] is False
    assert calls[0]["data"]["head"].startswith("shared-memory/")
    pr_body = calls[0]["data"]["body"]
    assert "機械検査" in pr_body
    assert all(path in pr_body for path in expected)
    assert all(op["reason"] in pr_body for op in report["operations"])
    assert repo.transactions() == [
        [
            {
                "name": "refs/heads/main",
                "beforeOid": base,
                "afterOid": head,
                "force": False,
            },
            {
                "name": f"refs/heads/{calls[0]['data']['head']}",
                "beforeOid": head,
                "afterOid": head,
                "force": False,
            },
        ]
    ]


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


@pytest.mark.parametrize(
    "status", [405, 409, None], ids=["405", "409", "ref-transaction"]
)
def test_merge_refusal_reapplies_to_the_advanced_main_with_exact_lease(
    repo: Repository, monkeypatch: pytest.MonkeyPatch, status: int | None
) -> None:
    latest = repo.parallel_commit("new-entry")
    monkeypatch.setenv(
        "MEMORY_TEST_CASE",
        json.dumps({"conflicts": [status] if status else [], "advance_to": latest}),
    )
    report = _report(repo.batch([add()]))
    assert report["retries"] == 1 and report["base"] == latest
    creates = [
        row for row in repo.api_calls() if row["path"] == PR_ENDPOINT.rsplit("/", 1)[0]
    ]
    merges = [transaction[0]["afterOid"] for transaction in repo.transactions()]
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
    assert len(repo.transactions()) == 1
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
            {"unconfirmed_merge": True},
            "main update acknowledged but merge confirmation failed",
            1,
            1,
        ),
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
    assert len(repo.transactions()) == attempts
    assert (
        sum(row["path"] == PR_ENDPOINT.rsplit("/", 1)[0] for row in repo.api_calls())
        == 1
    )
    assert len(repo.git_calls("fetch")) == fetches
    assert len(repo.git_calls("push")) == attempts
    assert repo.bare("for-each-ref", "refs/heads/shared-memory")


def test_atomic_transaction_rejects_last_moment_main_change_and_rechecks_citations(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = repo.bare("rev-parse", "main")
    latest = repo.parallel_commit("removed-anchor")
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps({"advance_to": latest}))
    result = repo.batch([add()])
    assert (
        result.returncode == 1
        and "[citation]" in result.stderr
        and "PR #17" in result.stderr
    )
    assert len(repo.git_calls("fetch")) == 2 and len(repo.git_calls("push")) == 1
    (transaction,) = repo.transactions()
    assert transaction[0]["beforeOid"] == base
    assert repo.bare("rev-parse", "main") == latest
    assert repo.bare("ls-tree", "main", "--", f"{MEMORY}/workflow/new.md") == ""


def test_atomic_transaction_rejects_head_change_after_last_pr_read(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = repo.bare("rev-parse", "main")
    switched = repo.parallel_commit("new-entry")
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps({"change_head_to": switched}))
    result = repo.batch([add()])
    assert result.returncode == 1 and "remote main has not advanced" in result.stderr
    (transaction,) = repo.transactions()
    assert transaction[1]["beforeOid"] == transaction[0]["afterOid"] != switched
    assert repo.bare("rev-parse", "main") == base
    assert repo.bare("rev-parse", transaction[1]["name"]) == switched
    assert len(repo.git_calls("push")) == 1


@pytest.mark.parametrize("op", ["update", "delete"])
@pytest.mark.parametrize("when", ["initial", "retry"])
def test_entry_version_guard_refuses_a_plan_for_an_entry_reverified_on_main(
    repo: Repository, monkeypatch: pytest.MonkeyPatch, op: str, when: str
) -> None:
    path = CURRENT if op == "update" else OLD
    expected = sha256((repo.source / path).read_bytes()).hexdigest()
    spec = (
        update(
            expected_sha256=expected,
            body="Original plan would replace the observation.",
        )
        if op == "update"
        else delete(expected_sha256=expected)
    )
    if when == "initial":
        latest = repo.parallel_commit("refreshed-entry", ref="main", target=path)
    else:
        latest = repo.parallel_commit("refreshed-entry", target=path)
        monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps({"advance_to": latest}))

    result = repo.batch([spec])
    assert result.returncode == 1 and not result.stdout
    assert f"{path}: entry changed since it was inspected; reassess" in result.stderr
    assert repo.bare("rev-parse", "main") == latest
    assert "Reverified against the latest behavior." in repo.bare(
        "show", f"main:{path}"
    )
    if when == "initial":
        repo.assert_no_publication()
    else:
        assert len(repo.git_calls("fetch")) == 2
        assert len(repo.git_calls("push")) == len(repo.transactions()) == 1


def test_index_repair_publishes_only_drifted_generated_documents(
    repo: Repository,
) -> None:
    index = f"{MEMORY}/workflow/index.md"
    expected = {path: repo.bare("show", f"main:{path}") for path in (SUMMARY, index)}
    for path in expected:
        (repo.source / path).write_text("Outdated generated content.\n")
    repo.git("add", "--", *expected)
    repo.git("commit", "-qm", "Drift in generated files")
    repo.git("push", "origin", "HEAD:main")
    snapshot = repo.snapshot()

    plan = publish.publish_index_repair(
        repo.source, reason="索引のずれを修復", dry_run=True
    )
    assert plan.files == {SUMMARY: "modified", index: "modified"}
    repo.assert_no_publication()
    report = _report(repo.cli("repair-index", "--reason", "索引のずれを修復"))
    assert report["files"] == plan.files and report["operations"] == []
    assert report["index_repair_reason"] == "索引のずれを修復"
    assert "索引のずれを修復" in repo.api_calls()[0]["data"]["body"]
    assert repo.snapshot() == snapshot
    for path, contents in expected.items():
        assert repo.bare("show", f"main:{path}") == contents


def test_index_repair_rejects_noop_before_publication(repo: Repository) -> None:
    result = repo.cli("repair-index", "--reason", "索引のずれを修復")
    assert result.returncode == 1 and not result.stdout
    assert "generated documents are already up to date (no-op)" in result.stderr
    repo.assert_no_publication()


def test_sigterm_during_git_preparation_leaves_no_temporary_paths_or_registration(
    repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = repo.tmp / "temp"
    temporary.mkdir()
    ready, block = repo.tmp / "ready", repo.tmp / "block"
    os.mkfifo(ready)
    os.mkfifo(block)
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setenv("MEMORY_TEST_SIGNAL_READY", str(ready))
    monkeypatch.setenv("MEMORY_TEST_SIGNAL_BLOCK", str(block))
    snapshot = repo.snapshot()
    refs = repo.git("for-each-ref")
    fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    try:
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shared_memory",
                "batch",
                "--spec",
                "-",
                "--repo",
                str(repo.source),
            ],
            cwd=repo.tmp,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as cli:
            assert cli.stdin is not None
            try:
                cli.stdin.write(json.dumps([add()]).encode())
                cli.stdin.close()
                assert select.select([fd], [], [], 5)[0], (
                    "git mktree did not reach the signal checkpoint"
                )
                git_pid = int(os.read(fd, 32).strip())
                cli.terminate()
                assert cli.wait(timeout=5) == 128 + signal.SIGTERM
                with pytest.raises(ProcessLookupError):
                    os.kill(git_pid, 0)
            finally:
                if cli.poll() is None:
                    cli.kill()
    finally:
        os.close(fd)
    assert repo.snapshot() == snapshot and repo.git("for-each-ref") == refs
    assert list(temporary.iterdir()) == []
    repo.assert_no_publication()


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
        == report["base"]
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
    assert repo.transactions() == []


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
        ([delete(expected_sha256="abc")], "expected_sha256 must be"),
        (
            [add(expected_sha256="a" * 64)],
            "expected_sha256 is available for updates and deletions",
        ),
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


@pytest.mark.parametrize("path", ["reference.md", f"{MEMORY}/workflow/extra.md"])
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
