"""Explicit migration against frozen v1 fixtures, fake GitHub and real git/files."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))
audit: Any = importlib.import_module("opsreport.audit")
legacy: Any = importlib.import_module("opsreport.legacy_checkpoint")
migration: Any = importlib.import_module("opsreport.migration")
proposals: Any = importlib.import_module("opsreport.proposals")
ledger: Any = importlib.import_module("opsreport.ledger")
cli: Any = importlib.import_module("opsreport.cli")
SLOT = "20261012"
NUMBER = 80
ARGS = ("migrate-checkpoint", "--slot", SLOT, "--report-issue", str(NUMBER))


def _path(ops: Any) -> Path:
    return (
        Path(ops.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/ops-report/audit"
        / f"{SLOT}.json"
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _issue(ops: Any) -> dict[str, Any]:
    return next(r for r in ops.state["issues"] if r["number"] == NUMBER)


def _write(ops: Any, data: dict[str, Any], *, rehash: bool = True) -> None:
    if rehash:
        data["collected"]["audit_input_sha256"] = audit.input_digest(
            {k: v for k, v in data["collected"].items() if k != "audit_runtime"}
        )
    _path(ops).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


def _seed(
    ops: Any, *, completed: bool = False, findings: bool = False
) -> dict[str, Any]:
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ops.repo, text=True
    ).strip()
    text = (Path(__file__).parent / "fixtures/legacy_checkpoint.json").read_text()
    data: dict[str, Any] = json.loads(
        text.replace("REPO_BASE_SHORT", base[:10]).replace("REPO_BASE", base)
    )
    data["collected"]["audit_plan"] = legacy.plan(ops.repo, base)
    if not completed:
        data["results"] = {}
    elif not findings:
        data["results"]["code"]["findings"] = []
    _path(ops).parent.mkdir(parents=True)
    _write(ops, data)
    meta = data["collected"]["meta"]
    proposal, _ = proposals.render_proposals(
        [
            proposals.Proposal(
                section="その他の助言", title="旧提案", evidence="固定版の根拠"
            )
        ],
        SLOT,
    )
    marker = proposals.report_marker(
        slot=SLOT,
        since=datetime.fromisoformat(meta["period_start"]),
        until=datetime.fromisoformat(meta["period_end"]),
        proposals=1,
        status="incomplete",
        base=base,
        audit=data["collected"]["audit_input_sha256"],
        agent="codex",
    )
    state = ops.state
    state["issues"].append(
        {
            "number": NUMBER,
            "title": "運用レポート 2026-10-12",
            "state": "OPEN",
            "author": {"login": "bot"},
            "labels": ["ops-report"],
            "url": f"https://github.com/o/r/issues/{NUMBER}",
            "createdAt": "2026-10-12T00:00:00Z",
            "body": marker
            + "\n旧本文は維持\n"
            + proposals.PROPOSALS_BEGIN
            + "\n"
            + proposal
            + "\n"
            + proposals.PROPOSALS_END
            + "\n",
        }
    )
    ops.write_state(state)
    return data


def _book(ops: Any) -> Any:
    value = audit.load_checkpoint(_path(ops), ops.repo)
    return ledger.Ledger.load(
        Path(value["ledger"]),
        slot=SLOT,
        base_sha=value["collected"]["meta"]["base_sha"],
    )


def _edits(ops: Any) -> list[list[str]]:
    return [c for c in ops.calls() if c[:2] == ["issue", "edit"]]


def test_empty_checkpoint_migrates_same_round_with_digest_mapping_and_original_bytes(
    ops_env: Any,
) -> None:
    old = _seed(ops_env)
    path = _path(ops_env)
    original = path.read_bytes()
    old_issue = _issue(ops_env)
    schedule = path.parent.parent / "schedule.json"
    before = schedule.read_bytes()
    result = ops_env.run(*ARGS)
    assert result.returncode == 0, result.stderr + result.stdout
    output = json.loads(result.stdout)
    assert output["status"] == "migrated"
    saved = audit.load_checkpoint(path, ops_env.repo)
    assert saved["version"] == 2 and saved["attempts"] == []
    new = saved["collected"]
    assert new["meta"] == {**old["collected"]["meta"], "repository": "o/r"}
    assert new["ops_memory"] == old["collected"]["ops_memory"]
    assert new["migration"]["input_sha256"] == old["collected"]["audit_input_sha256"]
    assert new["audit_input_sha256"] == audit.input_digest(new)
    assert output["legacy_input_sha256"] == new["migration"]["input_sha256"]
    archive = migration.archive_path(path)
    assert (archive / "source.json").read_bytes() == original
    assert _json(archive / "issue-before.json")["body"] == old_issue["body"]
    book = _book(ops_env)
    assert book.snapshot["run"]["report_issue"] == NUMBER
    assert book.snapshot["run"]["status"] == "incomplete"
    assert book.snapshot["tasks"] == {}
    assert len(book.snapshot["evidence"]) == 1
    new_issue = _issue(ops_env)
    assert new_issue["body"].partition("\n")[2] == old_issue["body"].partition("\n")[2]
    marker = proposals.parse_report_meta(new_issue["body"])
    assert marker.audit == new["audit_input_sha256"] and marker.status == "incomplete"
    assert f"legacy_audit={output['legacy_input_sha256']}" in new_issue["body"]
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()
    assert not (path.parent.parent / "memory.json").exists()
    assert schedule.read_bytes() == before
    assert not any(c[:2] == ["issue", "create"] for c in ops_env.calls())
    same = path.read_bytes(), Path(saved["ledger"]).read_bytes(), new_issue
    again = ops_env.run(*ARGS)
    assert again.returncode == 0, again.stderr + again.stdout
    assert json.loads(again.stdout)["status"] == "already_migrated"
    assert same == (
        path.read_bytes(),
        Path(saved["ledger"]).read_bytes(),
        _issue(ops_env),
    )
    assert len(_edits(ops_env)) == 1


def test_completed_results_are_imported_with_original_origin_and_unverified_findings(
    ops_env: Any,
) -> None:
    old = _seed(ops_env, completed=True, findings=True)
    result = ops_env.run(*ARGS)
    assert result.returncode == 0, result.stderr + result.stdout
    book = _book(ops_env)
    task = book.snapshot["tasks"]["legacy_code"]
    source = old["results"]["code"]
    assert set(book.snapshot["tasks"]) == {"legacy_code"}
    for key in ("status", "started_at", "finished_at", "agent_id", "conclusion"):
        assert task[key] == source[key]
    assert task["origin"] == {"kind": "legacy", "id": "legacy_code_origin"}
    assert migration.verify_import(book)[0] == {"legacy_code"}
    text = json.dumps(book.snapshot["evidence"], ensure_ascii=False)
    for value in [*source["evidence"], *source["counterevidence"]]:
        assert value in text
    assert book.snapshot["findings"]["legacy_code_code-path"]["status"] == "unverified"
    assert not book.snapshot["handoffs"]
    assert migration.source_original(book) == old
    # The unfinished result is preserved, but never represented as completed.
    assert (
        migration.source_original(book)["results"]["platform"]["status"]
        == "interrupted"
    )


@pytest.mark.parametrize("completed", [False, True])
def test_migrated_run_resumes_same_issue_without_reexecuting_legacy_task(
    ops_env: Any, completed: bool
) -> None:
    old = _seed(ops_env, completed=completed)
    result = ops_env.run(*ARGS)
    assert result.returncode == 0, result.stderr + result.stdout
    pinned = _json(_path(ops_env))["collected"]
    before = _book(ops_env).snapshot["tasks"]
    resumed = ops_env.run("report", "--if-due", "--now", "2026-10-21T08:00:00+09:00")
    assert resumed.returncode == 0, resumed.stderr
    book = _book(ops_env)
    assert _json(_path(ops_env))["collected"] == pinned
    assert book.snapshot["base_sha"] == old["collected"]["meta"]["base_sha"]
    assert book.snapshot["run"]["status"] == "completed"
    assert _issue(ops_env)["number"] == NUMBER
    for key, value in before.items():
        assert book.snapshot["tasks"][key] == value
    assert not any(c[:2] == ["issue", "create"] for c in ops_env.calls())


def test_preview_and_report_dry_run_leave_source_state_and_issue_unchanged(
    ops_env: Any,
) -> None:
    _seed(ops_env, completed=True)
    path = _path(ops_env)
    before = path.read_bytes(), ops_env.state
    result = ops_env.run(*ARGS, "--dry-run")
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)["status"] == "preview"
    assert not migration.archive_path(path).exists()
    assert not path.with_suffix(".ledger.json").exists()
    assert before == (path.read_bytes(), ops_env.state)
    assert ops_env.run(*ARGS).returncode == 0
    book = _book(ops_env)
    state_before_report = path.read_bytes(), book.path.read_bytes(), ops_env.state
    report = ops_env.run("report", "--dry-run")
    assert report.returncode == 0, report.stderr
    assert state_before_report == (
        path.read_bytes(),
        book.path.read_bytes(),
        ops_env.state,
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "corrupt",
        "duplicate-json",
        "hash",
        "inventory",
        "memory",
        "base",
        "boundary",
        "digest",
        "closed",
        "complete",
        "selected",
        "option",
        "child",
        "body-v1",
        "wrong-issue",
        "duplicate-slot",
        "role",
        "duplicate-agent",
        "result-base",
        "gap",
        "result-time",
        "version",
    ],
)
def test_invalid_migration_stops_with_facts_and_resume_without_replacing_input(
    ops_env: Any, problem: str
) -> None:
    source = _seed(ops_env, completed=True)
    path = _path(ops_env)
    state = ops_env.state
    issue = next(r for r in state["issues"] if r["number"] == NUMBER)
    if problem == "missing":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b'{"version":1,')
    elif problem == "duplicate-json":
        path.write_bytes(
            path.read_bytes().replace(b'"version": 1', b'"version": 1, "version": 1')
        )
    elif problem == "hash":
        source["collected"]["meta"]["period_start"] = "2026-10-08T08:00:00+09:00"
        _write(ops_env, source, rehash=False)
    elif problem in {"inventory", "memory", "base"}:
        if problem == "inventory":
            source["collected"]["audit_plan"]["assignments"][0]["targets"].append(
                "not-in-git.py"
            )
        elif problem == "memory":
            del source["collected"]["ops_memory"]["decisions"]
        else:
            source["collected"]["meta"]["base_sha"] = "0" * 40
        _write(ops_env, source)
    elif problem == "boundary":
        issue["body"] = issue["body"].replace("since=2026-10-09", "since=2026-10-08")
    elif problem == "digest":
        issue["body"] = issue["body"].replace(
            source["collected"]["audit_input_sha256"], "0" * 64
        )
    elif problem == "closed":
        issue["state"] = "CLOSED"
    elif problem == "complete":
        issue["body"] = issue["body"].replace("status=incomplete", "status=complete")
    elif problem == "selected":
        issue["body"] = issue["body"].replace("- [ ] **R-", "- [x] **R-")
    elif problem == "option":
        issue["body"] = issue["body"].replace(
            proposals.PROPOSALS_END, "  - [x] 採用\n" + proposals.PROPOSALS_END
        )
    elif problem == "child":
        issue["body"] = issue["body"].replace("** 旧提案", "** 旧提案 → #90")
    elif problem == "body-v1":
        issue["body"] = issue["body"].replace(
            "ops-report:v2 slot=", "ops-report:v1 week="
        )
    elif problem == "wrong-issue":
        issue["number"] = NUMBER + 1
    elif problem == "duplicate-slot":
        state["issues"].append({**issue, "number": NUMBER + 1})
    elif problem == "version":
        source["version"] = True
        _write(ops_env, source)
    else:
        row = source["results"]["code"]
        if problem == "role":
            row["agent_type"] = "parent"
        elif problem == "duplicate-agent":
            row["agent_id"] = source["results"]["platform"]["agent_id"]
        elif problem == "result-base":
            row["base_sha"] = "0" * 40
        elif problem == "gap":
            row["unconfirmed"] = source["results"]["platform"]["unconfirmed"]
        else:
            row["finished_at"] = row["started_at"]
        _write(ops_env, source)
    ops_env.write_state(state)
    before = path.read_bytes() if path.exists() else None
    result = ops_env.run(*ARGS)
    assert result.returncode == 1, result.stdout
    failure = json.loads(result.stdout)
    assert failure["status"] == "stopped"
    assert failure["reason"] and failure["facts"] and failure["resume"]
    assert (path.read_bytes() if path.exists() else None) == before
    assert not _edits(ops_env)
    assert ops_env.state == state
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()


@pytest.mark.parametrize(
    "env", [{"FAKE_GH_FAIL": "issue edit"}, {"FAKE_GH_EDIT_FAIL_AFTER_SAVE": "1"}]
)
def test_issue_failure_and_lost_response_resume_from_same_prepared_conversion(
    ops_env: Any, env: dict[str, str]
) -> None:
    _seed(ops_env)
    original = _path(ops_env).read_bytes()
    result = ops_env.run(*ARGS, **env)
    assert result.returncode == 1
    assert _path(ops_env).read_bytes() == original
    archive = migration.archive_path(_path(ops_env))
    prepared = {p.name: p.read_bytes() for p in archive.iterdir()}
    stopped = ops_env.run("report", "--if-due")
    assert stopped.returncode == 1 and "migrate-checkpoint" in stopped.stderr
    result = ops_env.run(*ARGS)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.name: p.read_bytes() for p in archive.iterdir()} == prepared
    assert len(_edits(ops_env)) == (1 if "FAKE_GH_EDIT_FAIL_AFTER_SAVE" in env else 2)
    assert len([r for r in ops_env.state["issues"] if "ops-report" in r["labels"]]) == 1


@pytest.mark.parametrize(
    "failure_at", ["source", "ledger", "checkpoint", "after-checkpoint"]
)
def test_atomic_local_failures_preserve_original_and_retry(
    ops_env: Any, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    _seed(ops_env, completed=True)
    path = _path(ops_env)
    original = path.read_bytes()
    for key, value in ops_env.env.items():
        monkeypatch.setenv(key, value)
    atomic = migration.atomic_bytes

    def failing(target: Path, data: bytes) -> None:
        fail = (
            (failure_at == "source" and target.name == "source.json")
            or (failure_at == "ledger" and target == path.with_suffix(".ledger.json"))
            or (failure_at in {"checkpoint", "after-checkpoint"} and target == path)
        )
        if fail:
            if failure_at == "after-checkpoint":
                atomic(target, data)
            raise OSError("injected migration disk failure")
        atomic(target, data)

    monkeypatch.setattr(migration, "atomic_bytes", failing)
    result = cli.main(
        [
            *ARGS,
            "--repo-root",
            str(ops_env.repo),
            "--log-dir",
            str(ops_env.tmp / "in-process"),
        ]
    )
    assert result == 1
    if failure_at != "after-checkpoint":
        assert path.read_bytes() == original
    else:
        assert _json(path)["version"] == 2
    monkeypatch.setattr(migration, "atomic_bytes", atomic)
    retry = ops_env.run(*ARGS)
    assert retry.returncode == 0, retry.stderr + retry.stdout
    assert (migration.archive_path(path) / "source.json").read_bytes() == original
    assert _book(ops_env).snapshot["tasks"]["legacy_code"]["status"] == "completed"


def test_concurrent_migrations_publish_once(ops_env: Any) -> None:
    _seed(ops_env)
    one = ops_env.start(*ARGS)
    two = ops_env.start(*ARGS)
    first, first_err = one.communicate(timeout=40)
    second, second_err = two.communicate(timeout=40)
    assert one.returncode == two.returncode == 0, first_err + second_err
    assert {json.loads(first)["status"], json.loads(second)["status"]} == {
        "migrated",
        "already_migrated",
    }
    assert len(_edits(ops_env)) == 1


@pytest.mark.parametrize(
    "change", ["archive", "legacy-task", "legacy-evidence", "input"]
)
def test_changed_migration_provenance_stops_normal_run(
    ops_env: Any, change: str
) -> None:
    _seed(ops_env, completed=True)
    assert ops_env.run(*ARGS).returncode == 0
    path = _path(ops_env)
    if change == "archive":
        (migration.archive_path(path) / "source.json").write_text("{}")
    elif change == "input":
        data = _json(path)
        data["collected"]["migration"]["input_sha256"] = "0" * 64
        _write(ops_env, data)
    else:
        ledger_path = path.with_suffix(".ledger.json")
        data = _json(ledger_path)
        for event in data["events"]:
            row = event["data"]
            if (
                change == "legacy-task"
                and row.get("id") == "legacy_code"
                and row["status"] == "completed"
            ):
                row["conclusion"] = "tampered conclusion"
            if (
                change == "legacy-evidence"
                and row.get("id") == "legacy_code_evidence_0"
            ):
                row["reference"] = "tampered reference"
        ledger_path.write_text(json.dumps(data))
    before = ops_env.state
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1, result.stderr
    assert not Path(ops_env.env["FAKE_AGENT_USED"]).exists()
    assert ops_env.state == before


def test_body_change_after_preparation_is_never_overwritten_on_retry(
    ops_env: Any,
) -> None:
    _seed(ops_env)
    assert ops_env.run(*ARGS, FAKE_GH_FAIL="issue edit").returncode == 1
    state = ops_env.state
    next(r for r in state["issues"] if r["number"] == NUMBER)["body"] += (
        "人間が追記した説明\n"
    )
    ops_env.write_state(state)
    original = _path(ops_env).read_bytes()
    retry = ops_env.run(*ARGS)
    assert retry.returncode == 1 and "issue changed" in retry.stdout
    assert ops_env.state == state
    assert _path(ops_env).read_bytes() == original


def test_previous_migrated_round_preserves_checkpoint_original_and_source_slot(
    ops_env: Any,
) -> None:
    old = _seed(ops_env, completed=True)
    migrated = ops_env.run(*ARGS)
    assert migrated.returncode == 0, migrated.stderr + migrated.stdout
    previous = _book(ops_env)
    previous.resume("Finish the migrated round", at=audit.timestamp())
    previous.end(
        {
            "status": "completed",
            "reason": "Imported results verified",
            "checked_scope": ["big.py"],
            "excluded_scope": ["external submodule internals"],
            "unverified_risks": [],
            "further_investigation": [],
        },
        at=audit.timestamp(),
    )
    current_slot = "20261015"
    current = ledger.Ledger.create(
        previous.path.with_name(f"{current_slot}.ledger.json"),
        slot=current_slot,
        base_sha=previous.snapshot["base_sha"],
        goal="Reobserve the completed migrated round",
        at=audit.timestamp(),
    )
    pinned = ops_env.tmp / "next-input.json"
    pinned.write_text(
        json.dumps(
            {
                "previous_reports": [
                    {
                        "slot": SLOT,
                        "number": NUMBER,
                        "base_sha": previous.snapshot["base_sha"],
                        "status": "complete",
                    }
                ]
            }
        )
    )
    tools = importlib.import_module("opsreport.audit_tools")
    retrieval = importlib.import_module("opsreport.retrieval")
    server = tools.AuditTools(
        retrieval.Retrieval(
            ops_env.repo,
            current.snapshot["base_sha"],
            "o/r",
            current_slot,
            ops_env.tmp / "next-sources",
        ),
        ledger_path=current.path,
        runtime_dir=ops_env.tmp / "next-runtime",
        input_path=pinned,
        ending_path=ops_env.tmp / "next-ending.json",
        max_threads=3,
    )
    records = server.call("audit_records", {"entity": "evidence", "slot": SLOT})
    assert records["ok"], records
    ref = next(
        row["ref"] for row in records["data"] if row["id"] == migration.SOURCE_ID
    )
    original = server.call("audit_original", {"ref": ref, "max_chars": 12000})
    assert original["ok"], original
    assert not original["pagination"]["has_more"]
    assert json.loads(original["data"]["text"])["original_checkpoint"] == old
    assert {"kind": "audit-ledger", "slot": SLOT} in original["sources"]


def test_imported_originals_are_bounded_and_available_through_mcp(ops_env: Any) -> None:
    _seed(ops_env, completed=True, findings=True)
    assert ops_env.run(*ARGS).returncode == 0
    book = _book(ops_env)
    tools = importlib.import_module("opsreport.audit_tools")
    retrieval = importlib.import_module("opsreport.retrieval")
    server = tools.AuditTools(
        retrieval.Retrieval(
            ops_env.repo,
            book.snapshot["base_sha"],
            "o/r",
            SLOT,
            ops_env.tmp / "sources",
        ),
        ledger_path=book.path,
        runtime_dir=ops_env.tmp / "runtime",
        input_path=migration.archive_path(_path(ops_env)) / "source.json",
        ending_path=ops_env.tmp / "ending.json",
        max_threads=3,
    )
    status = server.call("audit_status", {"page": 1, "per_page": 1})
    assert status["ok"], status
    imported = status["data"]["legacy_import"]
    assert imported["total"] == 1
    assert imported["completed_tasks"][0]["id"] == "legacy_code"
    original = server.call(
        "audit_original",
        {"ref": imported["source"], "offset": 0, "max_chars": 300},
    )
    assert original["ok"], original
    assert len(original["data"]["text"]) == 300
    assert original["pagination"]["has_more"]
    rejected = server.call(
        "audit_plan_task",
        {
            "id": "forged_import",
            "question": "new",
            "scope": ["big.py"],
            "origin": {"kind": "legacy", "id": "legacy_code_origin"},
        },
    )
    assert not rejected["ok"]
    assert "reserved" in rejected["error"]["message"]
    report = importlib.import_module("opsreport.report")
    c = _json(_path(ops_env))["collected"]
    _, body, ids = report.render_ledger_report(
        ledger=book,
        agent_name="codex",
        since=datetime.fromisoformat(c["meta"]["period_start"]),
        until=datetime.fromisoformat(c["meta"]["period_end"]),
        input_sha256=c["audit_input_sha256"],
        ledger_reference="保存台帳",
    )
    assert "legacy:legacy_code_origin" in body and "unverified（1件）" in body
    assert ids == []


@pytest.mark.parametrize(
    "change",
    [
        "closed",
        "selected",
        "child",
        "replaced-checkpoint",
        "archive-corrupt",
        "destination-conflict",
    ],
)
def test_retry_refuses_changed_inputs_and_never_discards_prepared_state(
    ops_env: Any, change: str
) -> None:
    _seed(ops_env, completed=True)
    assert ops_env.run(*ARGS, FAKE_GH_FAIL="issue edit").returncode == 1
    path = _path(ops_env)
    state = ops_env.state
    issue = next(r for r in state["issues"] if r["number"] == NUMBER)
    if change == "closed":
        issue["state"] = "CLOSED"
    elif change == "selected":
        issue["body"] = issue["body"].replace("- [ ] **R-", "- [X] **R-")
    elif change == "child":
        issue["body"] = issue["body"].replace("** 旧提案", "** 旧提案 → #90")
    elif change == "replaced-checkpoint":
        value = _json(path)
        value["results"] = {}
        _write(ops_env, value)
    elif change == "archive-corrupt":
        (migration.archive_path(path) / "checkpoint.json").write_text("{")
    else:
        path.with_suffix(".ledger.json").write_text("foreign ledger")
    ops_env.write_state(state)
    original = path.read_bytes()
    archive = {p.name: p.read_bytes() for p in migration.archive_path(path).iterdir()}
    retry = ops_env.run(*ARGS)
    assert retry.returncode == 1, retry.stdout
    assert len(_edits(ops_env)) == 1
    assert path.read_bytes() == original
    assert {
        p.name: p.read_bytes() for p in migration.archive_path(path).iterdir()
    } == archive
    assert ops_env.state == state


def test_runtime_metadata_is_archived_without_entering_legacy_digest(
    ops_env: Any,
) -> None:
    value = _seed(ops_env)
    old_digest = value["collected"]["audit_input_sha256"]
    value["collected"]["audit_runtime"] = {
        "checkout": "/old/checkout",
        "requested": ["code"],
        "previous_results": {},
    }
    _write(ops_env, value)
    assert value["collected"]["audit_input_sha256"] == old_digest
    result = ops_env.run(*ARGS)
    assert result.returncode == 0, result.stdout
    book = _book(ops_env)
    assert (
        migration.source_original(book)["collected"]["audit_runtime"]
        == value["collected"]["audit_runtime"]
    )
    assert "audit_runtime" not in _json(_path(ops_env))["collected"]


def test_issue_change_just_before_edit_is_detected(
    ops_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(ops_env)
    for key, value in ops_env.env.items():
        monkeypatch.setenv(key, value)
    real_issue = migration._issue
    reads = 0

    def concurrent_edit(repo: Path, slot: str, number: int) -> dict[str, Any]:
        nonlocal reads
        reads += 1
        if reads == 2:
            state = ops_env.state
            next(r for r in state["issues"] if r["number"] == NUMBER)["body"] += (
                "concurrent human edit\n"
            )
            ops_env.write_state(state)
        result: dict[str, Any] = real_issue(repo, slot, number)
        return result

    monkeypatch.setattr(migration, "_issue", concurrent_edit)
    original = _path(ops_env).read_bytes()
    assert (
        cli.main(
            [
                *ARGS,
                "--repo-root",
                str(ops_env.repo),
                "--log-dir",
                str(ops_env.tmp / "in-process"),
            ]
        )
        == 1
    )
    assert not _edits(ops_env)
    assert _path(ops_env).read_bytes() == original
    assert _issue(ops_env)["body"].endswith("concurrent human edit\n")
