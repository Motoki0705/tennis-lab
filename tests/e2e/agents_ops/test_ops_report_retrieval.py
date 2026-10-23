"""Pinned audit retrieval, page boundaries, GET-only authority and durable evidence."""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / ".agents/ops/ops-report"
sys.path.insert(0, str(PACKAGE))
retrieval: Any = importlib.import_module("opsreport.retrieval")
server: Any = importlib.import_module("opsreport.mcp_server")

FAKE_GH = r"""
import json, os, sys, time
from urllib.parse import urlsplit, parse_qs
from pathlib import Path
args = sys.argv[1:]
calls = Path(os.environ["RETRIEVAL_GH_CALLS"])
with calls.open("a") as f: f.write(json.dumps(args)+"\n")
assert args[:2] == ["api", "--hostname"]
assert args[args.index("--method")+1] == "GET"
assert args[args.index("--hostname")+1] == "github.com"
assert "--paginate" not in args and "--input" not in args
state = json.loads(Path(os.environ["RETRIEVAL_GH_STATE"]).read_text())
failure = state.get("failure")
if failure == "timeout": time.sleep(2)
if failure in ("403", "500", "404"):
    print("HTTP/2.0 "+failure+" Error\n\n{}")
    print("gh: failed (HTTP "+failure+")", file=sys.stderr)
    sys.exit(1)
if failure == "invalid":
    print("HTTP/2.0 200 OK\n\ninvalid")
    sys.exit(0)
if failure == "duplicate":
    print('HTTP/2.0 200 OK\n\n{"total_count":0,"total_count":1}')
    sys.exit(0)
if failure == "huge":
    print("HTTP/2.0 200 OK\n\n" + "x"*20000)
    sys.exit(0)
parsed = urlsplit(args[-1])
path, query = parsed.path, parse_qs(parsed.query)
page = int(query.get("page", [1])[0]); size = int(query.get("per_page", [50])[0])
head = "a"*40
def emit(value, total=None, text=False):
    print("HTTP/2.0 200 OK")
    if total is not None and page*size < total:
        print('Link: <https://api.github.com/next?page='+str(page+1)+'>; rel="next"')
    print()
    print(value if text else json.dumps(value, ensure_ascii=False), end="" if text else "\n")
def page_rows(total, make):
    return [make(i) for i in range((page-1)*size+1, min(page*size, total)+1)]
def issue(i):
    return {"id":1000+i,"number":i,"title":"issue "+str(i),"body":"UNREQUESTED_BODY"*1000,
            "state":"open","user":{"login":"researcher"},"labels":[{"name":"audit"}],
            "html_url":"https://github.com/o/r/issues/"+str(i),"updated_at":"2026-10-24T00:00:00Z"}
def job(i):
    return {"id":i,"run_id":70,"head_sha":head,"name":"unit "+str(i),"status":"completed","conclusion":"failure"}
if path == "search/issues":
    total = state.get("total", 421)
    rows = page_rows(total, issue)
    if failure == "empty_page": rows = []
    if failure == "huge_titles":
        for row in rows: row["title"] = "t"*30000
    emit({"total_count": total,"incomplete_results":failure == "incomplete", "items":rows}, total)
elif path == "repos/o/r/issues/17":
    row=issue(17);row["body"]="first line\n"+"長"*20000+"\nlast line\n"
    if failure == "missing_body": del row["body"]
    if failure == "wrong_body": row["body"] = 9
    if failure == "edited_body": row["body"] = "edited since first range\n"
    if failure == "wrong_kind": row["pull_request"]={}
    emit(row)
elif path == "repos/o/r/pulls/17":
    if "Accept: application/vnd.github.diff" in args:
        count = 1999 if failure == "partial_diff" else 2000
        emit("diff --git a/model.py b/model.py\n--- a/model.py\n+++ b/model.py\n@@ -0,0 +1,2000 @@\n"+"+change\n"*count, text=True)
    else:
        prior = [json.loads(l) for l in calls.read_text().splitlines()]
        meta = [c for c in prior if c[-1] == path and "Accept: application/vnd.github+json" in c]
        actual = "b"*40 if failure == "head_moves" and len(meta)>1 else head
        emit({**issue(17),"head":{"sha":actual},"base":{"sha":"c"*40},"changed_files":1})
elif path in ("repos/o/r/issues/17/comments","repos/o/r/pulls/17/comments","repos/o/r/pulls/17/reviews"):
    emit(page_rows(113, lambda i: {"id":i,"body":"COMMENT_BODY"*10000,"user":{"login":"reviewer"}}),113)
elif path in ("repos/o/r/issues/comments/9","repos/o/r/pulls/comments/9","repos/o/r/pulls/17/reviews/9"):
    number = 18 if failure == "association" else 17
    emit({"id":9,"body":"comment line 1\ncomment line 2\n","issue_url":"https://api.github.com/repos/o/r/issues/"+str(number), "pull_request_url":"https://api.github.com/repos/o/r/pulls/"+str(number)})
elif path == "repos/o/r/actions/runs":
    total=state.get("total",211)
    emit({"total_count":total,"workflow_runs":page_rows(total,lambda i:{"id":i,"head_sha":head,"run_attempt":1,"status":"completed","conclusion":"failure"})},total)
elif path == "repos/o/r/actions/runs/70":
    emit({"id":70,"head_sha":head,"run_attempt":2})
elif path == "repos/o/r/actions/runs/70/attempts/2/jobs":
    emit({"total_count":211,"jobs":page_rows(211,job)},211)
elif path == "repos/o/r/actions/jobs/9":
    emit(job(9))
elif path == "repos/o/r/actions/jobs/9/logs":
    if failure == "empty_log": emit("",text=True)
    else: emit("".join("line "+str(i)+" テスト\n" for i in range(1,50001)),text=True)
else:
    print("unexpected endpoint "+path,file=sys.stderr);sys.exit(91)
"""


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@dataclass
class Scenario:
    repo: Path
    base: str
    later: str
    state: Path
    calls_path: Path
    journal: Path
    client: Any

    def fail(self, failure: str, **extra: Any) -> None:
        self.state.write_text(json.dumps({"failure": failure, **extra}))

    def calls(self) -> list[list[str]]:
        return (
            [json.loads(line) for line in self.calls_path.read_text().splitlines()]
            if self.calls_path.exists()
            else []
        )

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (self.journal / "retrieval.jsonl").read_text().splitlines()
        ]


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scenario:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    for path, content in {
        "model.py": "needle = 1\nneedle = 2\nlast = 3\n",
        "odd: name.py": "needle = 4\n",
        "knowledge/finding.md": "pinned knowledge\n",
        ".agents/memory/SUMMARY.md": "pinned shared memory\n",
        ".agents/memory/decision/index.md": "decision index\n",
        ".agents/memory/ops-report/SUMMARY.md": "old dedicated memory\n",
        "binary.bin": "binary\0source",
        "long.txt": "長" * 25000 + "\n",
    }.items():
        file = repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    (repo / "escape").symlink_to(tmp_path / "unreadable-secret")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "model.py").write_text("newer code\n")
    (repo / ".agents/memory/ops-report/SUMMARY.md").write_text(
        "new pinned dedicated memory\n"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "later")
    later = git(repo, "rev-parse", "HEAD")
    (repo / "model.py").write_text("dirty worktree\n")
    (repo / "knowledge/finding.md").write_text("dirty knowledge\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text("#!" + sys.executable + "\n" + FAKE_GH)
    gh.chmod(0o700)
    state = tmp_path / "state.json"
    state.write_text("{}")
    calls = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("RETRIEVAL_GH_STATE", str(state))
    monkeypatch.setenv("RETRIEVAL_GH_CALLS", str(calls))
    journal = tmp_path / "round" / "receipts"
    client = retrieval.Retrieval(
        repo, base, "o/r", "round-1", journal, ops_memory_sha=later
    )
    return Scenario(repo, base, later, state, calls, journal, client)


def ok(scenario: Scenario, name: str, **arguments: Any) -> dict[str, Any]:
    response: dict[str, Any] = scenario.client.call(name, arguments)
    assert response["ok"], response
    assert response["base_sha"] == scenario.base
    assert response["sources"] and response["retrieved_at"]
    assert "untrusted research" in response["notice"]
    assert response["pagination"]["omitted"] is False
    return response


def error(scenario: Scenario, name: str, code: str, **arguments: Any) -> dict[str, Any]:
    response: dict[str, Any] = scenario.client.call(name, arguments)
    assert response["ok"] is False
    assert "data" not in response
    assert response["error"]["code"] == code, response
    assert response["sources"] and response["retrieved_at"] and response["notice"]
    return response


def test_code_reads_pinned_objects_and_pages_search(scenario: Scenario) -> None:
    response = ok(scenario, "code_file", path="model.py", start_line=2, line_count=1)
    assert response["data"]["text"] == "needle = 2\n"
    assert response["sources"][0]["sha"] == scenario.base
    first = ok(scenario, "code_search", text="needle", per_page=2)
    last = ok(scenario, "code_search", **first["pagination"]["next"])
    assert [r["line"] for r in first["data"]] == [1, 2]
    assert last["data"][0]["path"] == "odd: name.py"
    assert last["pagination"]["has_more"] is False
    assert ok(scenario, "code_search", text="dirty worktree")["data"] == []


def test_paths_history_and_diff_stay_at_base(scenario: Scenario) -> None:
    response = ok(scenario, "code_paths", prefix="knowledge/", per_page=1)
    assert [row["path"] for row in response["data"]] == ["knowledge/finding.md"]
    history = ok(scenario, "git_history", path="model.py", per_page=1)
    assert [row["sha"] for row in history["data"]] == [scenario.base]
    assert history["pagination"]["has_more"] is False
    diff = ok(scenario, "commit_diff", commit_sha=scenario.base, path="model.py")
    assert "+needle = 1" in diff["data"]["text"]
    error(scenario, "commit_diff", "UPSTREAM_ERROR", commit_sha=scenario.later)


def test_history_pagination_on_multiple_ancestors(scenario: Scenario) -> None:
    client = retrieval.Retrieval(
        scenario.repo,
        scenario.later,
        "o/r",
        "history",
        scenario.journal.parent / "history",
    )
    first = client.call("git_history", {"per_page": 1})
    assert first["ok"] and first["data"][0]["sha"] == scenario.later
    second = client.call("git_history", first["pagination"]["next"])
    assert second["ok"] and second["data"][0]["sha"] == scenario.base
    assert not second["pagination"]["has_more"]


def test_git_replace_and_external_diff_do_not_change_source(scenario: Scenario) -> None:
    git(scenario.repo, "replace", scenario.base, scenario.later)
    git(scenario.repo, "config", "diff.external", "does-not-exist")
    assert "needle" in ok(scenario, "code_file", path="model.py")["data"]["text"]
    assert (
        "+needle"
        in ok(scenario, "commit_diff", commit_sha=scenario.base, path="model.py")[
            "data"
        ]["text"]
    )


def test_records_use_scope_and_separate_memory_sha(scenario: Scenario) -> None:
    assert (
        ok(scenario, "record_file", scope="knowledge", path="finding.md")["data"][
            "text"
        ]
        == "pinned knowledge\n"
    )
    assert (
        ok(scenario, "record_file", scope="memory", path="SUMMARY.md")["data"]["text"]
        == "pinned shared memory\n"
    )
    dedicated = ok(scenario, "record_file", scope="ops_memory", path="SUMMARY.md")
    assert dedicated["sources"][0]["sha"] == scenario.later
    assert dedicated["data"]["text"] == "new pinned dedicated memory\n"
    shared = ok(scenario, "record_paths", scope="memory")
    assert all("ops-report/" not in row["path"] for row in shared["data"])
    assert ok(scenario, "record_paths", scope="ops_memory")["data"][0]["path"].endswith(
        "ops-report/SUMMARY.md"
    )
    error(
        scenario,
        "record_file",
        "INVALID_ARGUMENT",
        scope="memory",
        path="ops-report/SUMMARY.md",
    )


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "/etc/passwd",
        "model.py/../secret",
        "./model.py",
        "a//b",
        ".git/config",
        "a\\b",
        "a\nfile",
    ],
)
def test_path_escape_is_rejected(scenario: Scenario, path: str) -> None:
    error(scenario, "code_file", "INVALID_ARGUMENT", path=path)
    assert not scenario.calls()


def test_symlink_binary_missing_and_ranges_are_explicit_errors(
    scenario: Scenario,
) -> None:
    error(scenario, "code_file", "UNSUPPORTED_OBJECT", path="escape")
    error(scenario, "code_file", "BINARY_SOURCE", path="binary.bin")
    error(scenario, "code_file", "NOT_FOUND", path="absent")
    error(scenario, "code_file", "RANGE_ERROR", path="model.py", start_line=100)
    error(scenario, "code_file", "RANGE_ERROR", path="model.py", offset=10000)


def test_long_unicode_line_can_be_acquired_without_silent_truncation(
    scenario: Scenario,
) -> None:
    first = ok(scenario, "code_file", path="long.txt", line_count=1, max_chars=10000)
    second = ok(scenario, "code_file", **first["pagination"]["next"])
    third = ok(scenario, "code_file", **second["pagination"]["next"])
    assert (
        "".join(r["data"]["text"] for r in (first, second, third))
        == "長" * 25000 + "\n"
    )
    assert not third["pagination"]["has_more"]


def test_large_issue_list_pages_and_literal_filters(scenario: Scenario) -> None:
    first = ok(
        scenario,
        "github_items",
        kind="issue",
        text="repo:other/repo OR ignored",
        label="audit",
        author="tester",
        updated_after="2026-10-01",
        per_page=100,
    )
    assert len(first["data"]) == 100 and first["data"][-1]["number"] == 100
    second = ok(scenario, "github_items", **first["pagination"]["next"])
    assert second["data"][0]["number"] == 101
    last = ok(scenario, "github_items", kind="pr", page=5, per_page=100)
    assert len(last["data"]) == 21 and last["pagination"]["has_more"] is False
    assert "UNREQUESTED_BODY" not in json.dumps(first)
    query = parse_qs(urlsplit(scenario.calls()[0][-1]).query)["q"][0]
    assert (
        query.startswith("repo:o/r is:issue")
        and '"repo:other/repo OR ignored"' in query
    )
    assert 'label:"audit"' in query and "updated:>=2026-10-01" in query
    assert parse_qs(urlsplit(scenario.calls()[1][-1]).query)["q"][0] == query


def test_issue_and_pr_body_ranges_include_head_and_long_line_continuation(
    scenario: Scenario,
) -> None:
    first = ok(
        scenario,
        "github_item",
        kind="issue",
        number=17,
        start_line=2,
        line_count=1,
        max_chars=12000,
    )
    second = ok(scenario, "github_item", **first["pagination"]["next"])
    assert first["data"]["text"] + second["data"]["text"] == "長" * 20000 + "\n"
    response = ok(scenario, "github_item", kind="pr", number=17, max_chars=10)
    assert response["sources"][0]["pr_head_sha"] == "a" * 40
    assert response["sources"][0]["pr_base_sha"] == "c" * 40


@pytest.mark.parametrize("kind", ["discussion", "review_comment", "review"])
def test_comment_lists_and_selected_bodies(scenario: Scenario, kind: str) -> None:
    first = ok(scenario, "github_comments", number=17, kind=kind, per_page=100)
    last = ok(scenario, "github_comments", **first["pagination"]["next"])
    assert len(last["data"]) == 13 and not last["pagination"]["has_more"]
    assert "COMMENT_BODY" not in json.dumps(first)
    body = ok(
        scenario,
        "github_comment",
        number=17,
        comment_id=9,
        kind=kind,
        start_line=2,
        line_count=1,
    )
    assert body["data"]["text"] == "comment line 2\n"


def test_comment_wrong_issue_is_rejected(scenario: Scenario) -> None:
    scenario.fail("association")
    error(scenario, "github_comment", "SOURCE_MISMATCH", number=17, comment_id=9)


def test_pr_diff_range_is_versioned_and_every_request_is_get(
    scenario: Scenario,
) -> None:
    response = ok(
        scenario, "pr_diff", number=17, head_sha="a" * 40, start_line=1000, line_count=2
    )
    assert response["data"]["text"] == "+change\n+change\n"
    assert all(source["pr_head_sha"] == "a" * 40 for source in response["sources"])
    assert len(scenario.calls()) == 3
    assert all(call[call.index("--method") + 1] == "GET" for call in scenario.calls())


def test_pr_head_mismatch_and_mid_acquisition_update_are_errors(
    scenario: Scenario,
) -> None:
    error(scenario, "pr_diff", "HEAD_CHANGED", number=17, head_sha="b" * 40)
    assert len(scenario.calls()) == 1
    scenario.calls_path.unlink()
    scenario.fail("head_moves")
    error(scenario, "pr_diff", "HEAD_CHANGED", number=17, head_sha="a" * 40)


def test_partial_diff_is_not_returned_as_complete(scenario: Scenario) -> None:
    scenario.fail("partial_diff")
    error(scenario, "pr_diff", "INCOMPLETE_DIFF", number=17, head_sha="a" * 40)


def test_body_edit_between_ranges_requires_fresh_acquisition(
    scenario: Scenario,
) -> None:
    first = ok(scenario, "github_item", kind="issue", number=17, max_chars=4)
    scenario.fail("edited_body")
    response = error(
        scenario, "github_item", "SOURCE_CHANGED", **first["pagination"]["next"]
    )
    assert (
        response["error"]["details"]["expected_sha256"]
        == first["pagination"]["content_sha256"]
    )
    assert response["sources"][0]["url"].endswith("issues/17")


def test_empty_log_and_wrong_body_type_are_errors(scenario: Scenario) -> None:
    scenario.fail("empty_log")
    error(scenario, "ci_log", "EMPTY_LOG", job_id=9)
    scenario.fail("wrong_body")
    error(scenario, "github_item", "INVALID_SOURCE", kind="issue", number=17)


def test_shallow_history_does_not_claim_complete_ancestry(scenario: Scenario) -> None:
    (scenario.repo / ".git/shallow").write_text(scenario.base + "\n")
    error(scenario, "git_history", "INCOMPLETE_HISTORY")


def test_ci_runs_jobs_and_large_log_ranges_keep_code_identity(
    scenario: Scenario,
) -> None:
    runs = ok(
        scenario,
        "ci_runs",
        branch="main",
        head_sha="a" * 40,
        status="failure",
        event="push",
        created_after="2026-10-01",
        page=3,
        per_page=100,
    )
    assert len(runs["data"]) == 11 and not runs["pagination"]["has_more"]
    jobs = ok(scenario, "ci_jobs", run_id=70, page=2, per_page=100)
    assert jobs["data"][0]["id"] == 101
    assert jobs["sources"][1]["run_attempt"] == 2
    log = ok(scenario, "ci_log", job_id=9, start_line=49999, line_count=2)
    assert log["data"]["text"] == "line 49999 テスト\nline 50000 テスト\n"
    assert log["pagination"]["total_lines"] == 50000
    assert not log["pagination"]["has_more"]
    assert all(
        source["head_sha"] == "a" * 40 and source["run_id"] == 70
        for source in log["sources"]
    )
    assert len(retrieval.encode(log)) < retrieval.MAX_RESPONSE_BYTES


@pytest.mark.parametrize("status", ["403", "500", "404"])
def test_github_failures_are_not_empty_successes(
    scenario: Scenario, status: str
) -> None:
    scenario.fail(status)
    response = error(scenario, "github_items", "UPSTREAM_ERROR", kind="issue")
    assert response["error"]["details"]["http_status"] == int(status)
    assert response["sources"][0]["method"] == "GET"
    assert scenario.records()[-1]["response"] == response


@pytest.mark.parametrize(
    "failure,code",
    [
        ("invalid", "INVALID_SOURCE"),
        ("duplicate", "INVALID_SOURCE"),
        ("incomplete", "INCOMPLETE_SEARCH"),
        ("empty_page", "INCOMPLETE_PAGE"),
        ("missing_body", "INVALID_SOURCE"),
        ("wrong_kind", "SOURCE_MISMATCH"),
    ],
)
def test_malformed_and_incomplete_sources_fail_closed(
    scenario: Scenario, failure: str, code: str
) -> None:
    scenario.fail(failure)
    if failure in ("missing_body", "wrong_kind"):
        error(scenario, "github_item", code, kind="issue", number=17)
    else:
        error(scenario, "github_items", code, kind="issue")


def test_search_and_run_ceiling_is_not_silent_omission(scenario: Scenario) -> None:
    scenario.fail("none", total=1201)
    for name, args in [("github_items", {"kind": "issue"}), ("ci_runs", {})]:
        response = error(scenario, name, "UPSTREAM_LIMIT", **args)
        assert response["pagination"]["omitted"] is True
        assert response["error"]["details"]["total"] == 1201


def test_timeout_and_source_limit_are_recorded(scenario: Scenario) -> None:
    scenario.client.command.timeout = 0.1
    scenario.fail("timeout")
    error(scenario, "github_items", "TIMEOUT", kind="issue")
    scenario.client.command.timeout = 5
    scenario.client.command.max_bytes = 4096
    scenario.fail("huge")
    response = error(scenario, "github_items", "SOURCE_LIMIT", kind="issue")
    assert response["pagination"]["omitted"] is True
    assert [
        r["response"]["error"]["code"]
        for r in scenario.records()
        if r["event"] == "received"
    ] == ["TIMEOUT", "SOURCE_LIMIT"]


def test_response_limit_is_not_truncated_success(scenario: Scenario) -> None:
    scenario.fail("huge_titles")
    response = error(
        scenario, "github_items", "RESPONSE_LIMIT", kind="issue", per_page=3
    )
    assert len(retrieval.encode(response)) < retrieval.MAX_RESPONSE_BYTES
    assert "data" not in response


@pytest.mark.parametrize(
    "name,args",
    [
        ("execute", {"command": "touch forbidden"}),
        ("github_items", {"kind": "issue", "method": "POST"}),
        ("github_items", {"kind": "issue", "endpoint": "repos/other/repo/issues"}),
        ("github_items", {"kind": "issue", "text": '" repo:other/repo'}),
        ("code_file", {"path": "model.py", "command": "cat"}),
        ("code_paths", {"page": 0}),
        ("ci_log", {"job_id": 9, "line_count": 1001}),
        ("ci_runs", {"head_sha": "main"}),
    ],
)
def test_unapproved_operations_and_arguments_do_not_run_commands(
    scenario: Scenario, name: str, args: dict[str, Any]
) -> None:
    error(
        scenario,
        name,
        "UNKNOWN_TOOL" if name == "execute" else "INVALID_ARGUMENT",
        **args,
    )
    assert scenario.calls() == []
    assert scenario.records()[-1]["event"] == "received"


def test_journal_saves_exact_response_hash_and_private_mode(scenario: Scenario) -> None:
    response = ok(scenario, "github_item", kind="issue", number=17, max_chars=10)
    records = scenario.records()
    assert [r["event"] for r in records] == ["context", "requested", "received"]
    assert records[-1]["response"] == response
    assert records[1]["receipt"] == records[-1]["receipt"] == response["receipt"]
    assert (
        records[-1]["response_sha256"]
        == retrieval.hashlib.sha256(retrieval.encode(response)).hexdigest()
    )
    assert (scenario.journal / "retrieval.jsonl").stat().st_mode & 0o777 == 0o600


def test_recording_failure_prevents_fetch_and_does_not_publish_unrecorded_data(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(value: dict[str, Any], **kwargs: Any) -> None:
        raise OSError("read-only journal")

    monkeypatch.setattr(scenario.client.journal, "append", broken)
    error(scenario, "github_items", "RECORDING_FAILED", kind="issue")
    assert not scenario.calls()


def test_completion_record_failure_is_not_success(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = scenario.client.journal.append

    def broken(value: dict[str, Any], **kwargs: Any) -> None:
        if value["event"] == "received":
            raise OSError("disk full")
        original(value, **kwargs)

    monkeypatch.setattr(scenario.client.journal, "append", broken)
    error(scenario, "github_items", "RECORDING_FAILED", kind="issue")
    assert len(scenario.calls()) == 1
    assert scenario.records()[-1]["event"] == "requested"


def test_journal_rejects_mixed_context_and_partial_append(scenario: Scenario) -> None:
    with pytest.raises(ValueError, match="context mismatch"):
        retrieval.Retrieval(
            scenario.repo, scenario.base, "o/r", "other-round", scenario.journal
        )
    with (scenario.journal / "retrieval.jsonl").open("ab") as file:
        file.write(b'{"partial":')
    error(scenario, "code_paths", "RECORDING_FAILED")


def test_journal_must_be_external_and_cannot_be_symlink(
    scenario: Scenario, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="outside"):
        retrieval.Retrieval(
            scenario.repo, scenario.base, "o/r", "bad", scenario.repo / "records"
        )
    directory = tmp_path / "linked"
    directory.mkdir(mode=0o700)
    (directory / "retrieval.jsonl").symlink_to(scenario.journal / "retrieval.jsonl")
    with pytest.raises(OSError):
        retrieval.Retrieval(scenario.repo, scenario.base, "o/r", "round-1", directory)


def request(
    method: str, params: dict[str, Any] | None = None, identifier: int = 1
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": identifier,
        "method": method,
        "params": params or {},
    }


def test_mcp_lists_only_read_operations_and_rejects_command_routes(
    scenario: Scenario,
) -> None:
    response = server.dispatch(scenario.client, request("tools/list"))
    tools = response["result"]["tools"]
    metadata = server.dispatch(
        scenario.client,
        request("tools/list", {"cursor": None, "_meta": {"progressToken": "probe"}}),
    )
    assert metadata["result"]["tools"] == tools
    assert {t["name"] for t in tools} == {
        "code_paths",
        "code_search",
        "code_file",
        "git_history",
        "commit_diff",
        "record_paths",
        "record_file",
        "github_items",
        "github_item",
        "github_comments",
        "github_comment",
        "pr_diff",
        "ci_runs",
        "ci_jobs",
        "ci_log",
    }
    assert all(
        t["annotations"]["readOnlyHint"] is True
        and t["inputSchema"]["additionalProperties"] is False
        for t in tools
    )
    assert (
        server.dispatch(scenario.client, request("execute", {"command": "id"}))[
            "error"
        ]["code"]
        == -32601
    )
    assert (
        server.dispatch(scenario.client, request("tools/list", {"cursor": "ignored"}))[
            "error"
        ]["code"]
        == -32602
    )
    denied = server.dispatch(
        scenario.client,
        request("tools/call", {"name": "shell", "arguments": {"command": "id"}}),
    )
    assert denied["result"]["isError"] is True
    assert not scenario.calls()


def test_stdio_recovers_after_duplicate_json_and_oversize_request(
    scenario: Scenario,
) -> None:
    raw = (
        b'{"jsonrpc":"2.0","id":1,"id":2,"method":"ping"}\n'
        + b"x" * (server.MAX_REQUEST_BYTES + 1)
        + b"\n"
    )
    raw += (
        retrieval.encode(
            request(
                "tools/call",
                {
                    "name": "code_file",
                    "arguments": {"path": "model.py", "line_count": 1},
                },
                3,
            )
        )
        + b"\n"
    )
    target = io.BytesIO()
    server.serve(scenario.client, io.BytesIO(raw), target)
    rows = [json.loads(line) for line in target.getvalue().splitlines()]
    assert [r.get("error", {}).get("code") for r in rows[:2]] == [-32700, -32600]
    assert (
        json.loads(rows[2]["result"]["content"][0]["text"])["data"]["text"]
        == "needle = 1\n"
    )


def test_deleted_journal_does_not_silently_restart_evidence(scenario: Scenario) -> None:
    (scenario.journal / "retrieval.jsonl").unlink()
    error(scenario, "github_items", "RECORDING_FAILED", kind="issue")
    assert not scenario.calls()


def test_real_stdio_server_processes_share_append_only_journal(
    scenario: Scenario,
) -> None:
    env = {**os.environ, "PYTHONPATH": str(PACKAGE)}
    argv = [
        sys.executable,
        "-m",
        "opsreport.mcp_server",
        "--repo",
        str(scenario.repo),
        "--base-sha",
        scenario.base,
        "--repository",
        "o/r",
        "--round-id",
        "round-1",
        "--ops-memory-sha",
        scenario.later,
        "--journal-dir",
        str(scenario.journal),
    ]
    payload = (
        b"\n".join(
            retrieval.encode(
                request(
                    "tools/call",
                    {"name": "code_paths", "arguments": {"per_page": 2}},
                    i,
                )
            )
            for i in range(1, 4)
        )
        + b"\n"
    )

    def run(_: int) -> list[dict[str, Any]]:
        result = subprocess.run(
            argv, input=payload, capture_output=True, env=env, check=True
        )
        return [json.loads(row) for row in result.stdout.splitlines()]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, range(3)))
    assert all(
        len(rows) == 3 and all(not row["result"]["isError"] for row in rows)
        for rows in results
    )
    records = scenario.records()
    requested = {row["receipt"] for row in records if row["event"] == "requested"}
    received = {row["receipt"] for row in records if row["event"] == "received"}
    assert len(requested) == 9 and requested == received
    assert len(records) == 19
