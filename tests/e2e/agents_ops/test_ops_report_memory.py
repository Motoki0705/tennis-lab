"""Memory handoff across publications, with fake APIs/agents and real git."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/ops-report"))
memory: Any = importlib.import_module("opsreport.memory")
memory_pr: Any = importlib.import_module("opsreport.memory_pr")
agent: Any = importlib.import_module("opsreport.agent")
DIRECTORY = ".agents/memory/ops-report"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
    ).strip()


def _update(version: int = 1) -> dict[str, Any]:
    return {
        "summary": f"CIの継続観察 {version}",
        "upsert": [
            {
                "slug": "ci-wait",
                "title": "CI待ち時間を再検証する",
                "focus": "CI/依存",
                "observation": f"検証した観察 {version}",
                "evidence": f"small.py:1 / #7 / 計測{version}",
                "hypothesis": "実行順に起因する可能性（未確認）",
                "deferred_reason": "#50 の人間の判断で、CI構成変更まで対策を見送る",
                "recheck_when": "CI構成が変わったら再計測する",
                "references": ["#50", "R-20261012-02"],
                "last_verified": "2026-10-12" if version == 1 else "2026-10-15",
            }
        ],
        "delete": [],
    }


@pytest.fixture
def memory_env(ops_env: Any, agent_output: dict[str, Any]) -> Any:
    shutil.copytree(ROOT / ".agents/memory", ops_env.repo / ".agents/memory")
    _git(ops_env.repo, "config", "user.name", "test")
    _git(ops_env.repo, "config", "user.email", "test@example.com")
    _git(ops_env.repo, "add", ".agents/memory")
    _git(ops_env.repo, "commit", "-qm", "seed shared and dedicated memory")
    upstream = ops_env.tmp / "upstream.git"
    _git(ops_env.tmp, "clone", "--bare", str(ops_env.repo), str(upstream))
    _git(ops_env.repo, "remote", "add", "origin", str(upstream))
    _git(ops_env.repo, "fetch", "origin")
    agent_output["memory_update"] = _update()
    ops_env.set_agent_output(agent_output)
    return ops_env


def _journal(ops: Any) -> dict[str, Any]:
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _reports(ops: Any) -> list[dict[str, Any]]:
    return [r for r in ops.state["issues"] if "ops-report" in r["labels"]]


def _main_files(ops: Any) -> dict[str, str]:
    return memory.read_tree(ops.repo, "origin/main")  # type: ignore[no-any-return]


def _shared_hashes(ops: Any, ref: str = "origin/main") -> dict[str, str]:
    paths = _git(
        ops.repo, "ls-tree", "-r", "--name-only", ref, "--", ".agents/memory"
    ).splitlines()
    return {
        p: hashlib.sha256(
            subprocess.check_output(["git", "show", f"{ref}:{p}"], cwd=ops.repo)
        ).hexdigest()
        for p in paths
        if not p.startswith(DIRECTORY + "/")
    }


def _payload(ops: Any, version: int = 1) -> None:
    payload = json.loads(Path(ops.env["FAKE_AGENT_OUTPUT"]).read_text())
    payload["memory_update"] = _update(version)
    ops.set_agent_output(payload)


def _collected(ops: Any, slot: str) -> dict[str, Any]:
    for run in ops.run_dirs():
        path = run / "collected.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["meta"]["period_end"].startswith(slot):
                return data  # type: ignore[no-any-return]
    raise AssertionError(f"no collected data for {slot}")


def _external_merge(ops: Any, slot: str = "20261012") -> None:
    """Remote fixture only: the exporter must observe, never perform this write.

    Formal gate acceptance/refusal is covered separately by test_memory_gate.
    """
    row = _journal(ops)["records"][slot]
    _git(
        ops.tmp,
        "--git-dir",
        str(ops.tmp / "upstream.git"),
        "update-ref",
        "refs/heads/main",
        row["head"],
    )
    state = ops.state
    pr = next(p for p in state["memory_prs"] if p["url"] == row["pr_url"])
    pr["state"] = "MERGED"
    work = next(i for i in state["issues"] if i["url"] == row["work_issue"])
    work["state"] = "CLOSED"
    ops.write_state(state)


def test_two_reports_use_main_observations_and_human_decisions(memory_env: Any) -> None:
    ops = memory_env
    shared = _shared_hashes(ops)
    original = _main_files(ops)
    checkout = _git(ops.repo, "rev-parse", "HEAD")
    first = ops.run("report")
    assert first.returncode == 0, first.stderr
    assert _main_files(ops) == original  # PR creation alone is not memory success
    assert len(ops.state["memory_prs"]) == 1
    assert _journal(ops)["records"]["20261012"]["state"] == "pending"
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    # Both selected proposals and explicit deferrals enter the second prompt.
    state = ops.state
    report = next(r for r in state["issues"] if r["number"] == 100)
    report["body"] = report["body"].replace(
        "- [ ] **R-20261012-02**", "- [x] **R-20261012-02**"
    )
    # A local snapshot cannot invent a human deferral absent from GitHub.
    head, tail = report["body"].split("**R-20261012-03**", 1)
    report["body"] = (
        head
        + "**R-20261012-03**"
        + tail.replace("見送り理由: 未記入", "見送り理由: 構成変更まで保留", 1)
    )
    ops.write_state(state)
    decisions = ops.tmp / "decisions.json"
    decisions.write_text(
        json.dumps(
            [
                {
                    "proposal_id": "R-20261012-03",
                    "decision": "deferred",
                    "reason": "構成変更まで保留",
                    "source": "https://github.com/o/r/issues/100 / R-20261012-03",
                }
            ]
        )
    )
    _external_merge(ops)
    _payload(ops, 2)
    second = ops.run(
        "report",
        "--now",
        "2026-10-15T08:00:00+09:00",
        "--memory-decisions",
        str(decisions),
    )
    assert second.returncode == 0, second.stderr
    data = _collected(ops, "2026-10-15")
    assert "検証した観察 1" in data["ops_memory"]["files"]["ci-wait.md"]
    assert "対策を見送る" in data["ops_memory"]["files"]["ci-wait.md"]
    assert data["ops_memory"]["base_sha"] != checkout
    assert data["ops_memory"]["persistence"][0]["state"] == "merged"
    assert {
        d["proposal_id"]: d["decision"] for d in data["ops_memory"]["decisions"]
    } == {
        "R-20261012-02": "selected",
        "R-20261012-03": "deferred",
    }
    assert (
        data["previous_reports"]["reports"][-1]["proposals"][1]["state"] == "selected"
    )
    assert all(
        not entry["path"].startswith(DIRECTORY) for entry in data["memory"]["entries"]
    )
    assert data["memory"]["entries_total"] == len(
        [p for p in shared if not p.endswith(("index.md", "SUMMARY.md"))]
    )
    assert _shared_hashes(ops) == shared
    assert len(ops.state["memory_prs"]) == 2

    # Daily triage observes the ordinary gate's completed merge, without an agent.
    _external_merge(ops, "20261015")
    Path(ops.env["FAKE_AGENT_USED"]).unlink()
    # Avoid filing a child in this test; adoption/requirements are #1223's job.
    state = ops.state
    for r in state["issues"]:
        if "ops-report" in r["labels"]:
            r["state"] = "CLOSED"
    ops.write_state(state)
    synced = ops.run("triage")
    assert synced.returncode == 0, synced.stderr
    assert not Path(ops.env["FAKE_AGENT_USED"]).exists()
    entry = _main_files(ops)["ci-wait.md"]
    assert "検証した観察 2" in entry and "検証した観察 1" not in entry
    meta = yaml.safe_load(entry.split("---\n")[1])
    assert meta["created"] == date(2026, 10, 12)
    assert meta["last_verified"] == date(2026, 10, 15)
    assert meta["source"] == _reports(ops)[-1]["url"]
    assert _shared_hashes(ops) == shared
    # The caller's checkout is never advanced or edited by memory persistence.
    assert _git(ops.repo, "rev-parse", "HEAD") == checkout
    assert _git(ops.repo, "status", "--porcelain") == ""
    assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")
    assert "ops-report-memory-20261015" not in _git(ops.repo, "worktree", "list")


def test_dry_run_never_prepares_memory_or_writes_git(memory_env: Any) -> None:
    ops = memory_env
    before = _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)")
    result = ops.run("report", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert _reports(ops) == [] and not ops.state.get("memory_prs")
    assert not (
        Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    ).exists()
    assert _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)") == before


@pytest.mark.parametrize("failure", ["agent", "publication", "push", "pr"])
def test_partial_failures_preserve_last_main_and_recover(
    memory_env: Any, failure: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    env = {
        "agent": {"FAKE_AGENT_FAIL": "1"},
        "publication": {"FAKE_GH_FAIL": "issue create"},
        "pr": {"FAKE_GH_FAIL": "pr create"},
        "push": {},
    }[failure]
    hook = ops.tmp / "upstream.git/hooks/pre-receive"
    if failure == "push":
        hook.write_text("#!/bin/sh\necho 'injected push failure' >&2\nexit 1\n")
        hook.chmod(0o755)
    failed = ops.run("report", **env)
    assert failed.returncode == 1, failed.stderr
    assert _main_files(ops) == original
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    if failure in {"agent", "publication"}:
        assert len(_reports(ops)) == (1 if failure == "agent" else 0)
        assert ops.state.get("memory_prs", []) == []
        assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")
    else:
        row = _journal(ops)["records"]["20261012"]
        assert row["state"] == "pending" and row["error"]
        assert row["pr_url"] is None
        assert len(_reports(ops)) == 1
        schedule_path = (
            Path(ops.env["XDG_STATE_HOME"])
            / "tennis-lab-agents/ops-report/schedule.json"
        )
        assert (
            json.loads(schedule_path.read_text())["last_success"]["url"]
            == _reports(ops)[0]["url"]
        )
    if failure == "push":
        hook.unlink()
    retry = ops.run("report", "--if-due")
    assert retry.returncode == 0, retry.stderr
    assert len(_reports(ops)) == 1
    assert len(ops.state["memory_prs"]) == 1
    assert _main_files(ops) == original
    _external_merge(ops)
    complete = ops.run("report", "--if-due")
    assert complete.returncode == 0, complete.stderr
    assert "検証した観察 1" in _main_files(ops)["ci-wait.md"]


@pytest.mark.parametrize("ci", ["PENDING", "MISSING", "FAILURE"])
def test_pending_or_failed_memory_is_disclosed_and_main_is_used(
    memory_env: Any, ci: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    assert ops.run("report").returncode == 0
    _payload(ops, 2)
    second = ops.run("report", "--now", "2026-10-15T08:00:00+09:00", FAKE_MEMORY_CI=ci)
    assert second.returncode == 0, second.stderr
    assert _main_files(ops) == original
    data = _collected(ops, "2026-10-15")
    assert data["ops_memory"]["files"] == original
    assert "**専用メモリ未反映**" in _reports(ops)[-1]["body"]
    assert "最後の有効版" in _reports(ops)[-1]["body"]
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    assert _journal(ops)["records"]["20261012"]["state"] == "pending"


def test_changed_head_after_handoff_fails_closed(memory_env: Any) -> None:
    ops = memory_env
    original = _main_files(ops)
    assert ops.run("report").returncode == 0
    result = ops.run("triage", FAKE_MEMORY_CHANGE_HEAD="1")
    assert result.returncode == 1
    assert "HEAD/path contract changed" in result.stderr
    assert _main_files(ops) == original
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())


def test_concurrent_reports_and_syncs_only_create_one_handoff(
    memory_env: Any,
) -> None:
    ops = memory_env
    runs = [ops.start("report", "--if-due", FAKE_GH_CREATE_DELAY="1") for _ in range(2)]
    for proc in runs:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, (out, err)
    assert len(_reports(ops)) == 1
    assert len(ops.state["memory_prs"]) == 1
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    row = _journal(ops)["records"]["20261012"]
    assert row["state"] == "pending" and row["stage"] == "review_handoff"
    assert (
        len(
            [i for i in ops.state["issues"] if "ops-report-memory-task:v1" in i["body"]]
        )
        == 1
    )


def test_interrupted_save_leaves_main_valid_and_requires_explicit_dirty_repair(
    memory_env: Any,
) -> None:
    ops = memory_env
    original = _main_files(ops)
    marker = ops.tmp / "commit-started"
    hook = ops.repo / ".git/hooks/pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nsleep 300\n")
    hook.chmod(0o755)
    proc = ops.start("report")
    try:
        deadline = time.monotonic() + 30
        while (
            not marker.exists() and proc.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists(), proc.communicate(timeout=10)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        hook.unlink()
    assert proc.returncode != 0
    assert _main_files(ops) == original
    assert len(_reports(ops)) == 1 and not ops.state.get("memory_prs")
    row = _journal(ops)["records"]["20261012"]
    assert row["state"] == "pending" and row["stage"] == "commit"
    retry = ops.run("report", "--if-due")
    assert retry.returncode == 1
    assert "interrupted/dirty memory worktree" in retry.stderr
    assert _main_files(ops) == original
    assert _journal(ops)["records"]["20261012"]["error"]


def test_crash_after_publication_is_recovered_from_verified_history(
    memory_env: Any,
) -> None:
    ops = memory_env
    assert ops.run("report", FAKE_GH_FAIL="pr create").returncode == 1
    # Reproduce the durable state immediately before gh issue create returned.
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    journal = _journal(ops)
    row = journal["records"]["20261012"]
    row.update(
        state="prepared",
        stage="publication_unconfirmed",
        report_url=None,
        head=None,
        error=None,
    )
    path.write_text(json.dumps(journal))
    Path(ops.env["FAKE_AGENT_USED"]).unlink()
    result = ops.run("report", "--if-due")
    assert result.returncode == 0, result.stderr
    assert len(_reports(ops)) == 1 and len(ops.state["memory_prs"]) == 1
    assert not Path(ops.env["FAKE_AGENT_USED"]).exists()


@pytest.mark.parametrize("interruption", ["none", "before_edit", "after_edit"])
def test_partial_report_completion_binds_verified_publication(
    memory_env: Any, interruption: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    assert ops.run("report", FAKE_AUDIT_FAIL="platform").returncode == 1
    partial = _reports(ops)[0]
    journal_path = (
        Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    )
    assert not journal_path.exists() and not ops.state.get("memory_prs")
    # Completion must publish the prepared title, even if the partial was renamed.
    state = ops.state
    next(r for r in state["issues"] if r["number"] == partial["number"])["title"] = (
        "調査再開待ち"
    )
    ops.write_state(state)
    env = {
        "none": {},
        "before_edit": {"FAKE_GH_FAIL": "issue edit"},
        "after_edit": {"FAKE_GH_EDIT_FAIL_AFTER_SAVE": "1"},
    }[interruption]

    result = ops.run("report", "--if-due", **env)

    assert result.returncode == (0 if interruption == "none" else 1), result.stderr
    assert _main_files(ops) == original
    row = _journal(ops)["records"]["20261012"]
    assert row["report_url"] == partial["url"]
    if interruption != "none":
        assert row["state"] == "prepared" and row["head"] is None
        assert not ops.state.get("memory_prs")
        if interruption == "before_edit":
            assert "status=incomplete" in _reports(ops)[0]["body"].splitlines()[0]
            # Triage leaves a failed edit pending without persisting observations.
            triaged = ops.run("triage")
            assert triaged.returncode == 0, triaged.stderr
            assert _journal(ops)["records"]["20261012"] == row
            assert not ops.state.get("memory_prs")
        Path(ops.env["FAKE_AGENT_USED"]).unlink()
        recovered = ops.run("report", "--if-due")
        assert recovered.returncode == 0, recovered.stderr
        if interruption == "after_edit":
            assert not Path(ops.env["FAKE_AGENT_USED"]).exists()

    complete = _reports(ops)[0]
    row = _journal(ops)["records"]["20261012"]
    assert len(_reports(ops)) == 1 and complete["number"] == partial["number"]
    assert "status=complete" in complete["body"].splitlines()[0]
    assert row["publication"] == {
        "title": complete["title"],
        "body_sha256": hashlib.sha256(complete["body"].encode("utf-8")).hexdigest(),
        "author": complete["author"]["login"],
    }
    assert complete["title"] == "運用レポート 2026-10-12"
    assert row["state"] == "pending" and row["report_url"] == partial["url"]
    assert len(ops.state["memory_prs"]) == 1
    assert sum(c[:2] == ["issue", "create"] for c in ops.calls()) == 2
    assert sum(c[:2] == ["issue", "edit"] for c in ops.calls()) == (
        2 if interruption == "before_edit" else 1
    )
    _external_merge(ops)
    merged = ops.run("triage")
    assert merged.returncode == 0, merged.stderr
    assert _journal(ops)["records"]["20261012"]["state"] == "merged"
    entry = _main_files(ops)["ci-wait.md"]
    assert "検証した観察 1" in entry and partial["url"] in entry


@pytest.mark.parametrize("command", ["report", "triage"])
@pytest.mark.parametrize("mismatch", ["author", "body", "title", "url"])
def test_failed_completion_rejects_forged_publication(
    memory_env: Any, command: str, mismatch: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    refs = _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)")
    assert ops.run("report", FAKE_AUDIT_FAIL="platform").returncode == 1
    failed = ops.run("report", "--if-due", FAKE_GH_FAIL="issue edit")
    assert failed.returncode == 1, failed.stderr
    row = _journal(ops)["records"]["20261012"]
    assert row["state"] == "prepared" and row["report_url"] == _reports(ops)[0]["url"]
    report_file = next(
        p
        for run in ops.run_dirs()
        if (p := run / "report.md").exists()
        and "status=complete" in p.read_text().splitlines()[0]
    )
    state = ops.state
    forged = next(r for r in state["issues"] if "ops-report" in r["labels"])
    forged.update(title=row["publication"]["title"], body=report_file.read_text())
    if mismatch == "author":
        forged["author"] = {"login": "another-user"}
    elif mismatch == "body":
        forged["body"] += "\n公開予定とは異なる調査結果。\n"
    elif mismatch == "title":
        forged["title"] = "同slotの別レポート"
    else:
        # All content and the author match, but this is a different issue.
        forged.update(number=101, url="https://github.com/o/r/issues/101")
        state["next_number"] = 102
    ops.write_state(state)
    Path(ops.env["FAKE_AGENT_USED"]).unlink()

    args = ("report", "--if-due") if command == "report" else ("triage",)
    result = ops.run(*args)

    assert result.returncode == 1, result.stderr
    assert "publication" in result.stderr and "mismatch" in result.stderr
    assert _journal(ops)["records"]["20261012"] == row
    assert _main_files(ops) == original
    assert _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert "ops-report-memory/" not in _git(ops.repo, "ls-remote", "--heads", "origin")
    assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")
    assert not ops.state.get("memory_prs")
    assert not Path(ops.env["FAKE_AGENT_USED"]).exists()


@pytest.mark.parametrize("command", ["report", "triage"])
@pytest.mark.parametrize("mismatch", ["author", "body", "title"])
def test_failed_publication_rejects_same_slot_forged_issue(
    memory_env: Any, command: str, mismatch: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    refs = _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)")
    assert ops.run("report", FAKE_GH_FAIL="issue create").returncode == 1
    body = next(
        p for run in ops.run_dirs() if (p := run / "report.md").exists()
    ).read_text()
    assert _journal(ops)["records"]["20261012"]["publication"] == {
        "title": "運用レポート 2026-10-12",
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "author": "bot",
    }
    forged: dict[str, Any] = {
        "number": 100,
        "title": "運用レポート 2026-10-12",
        "body": body,
        "author": {"login": "bot"},
        "labels": ["ops-report"],
        "state": "OPEN",
        "createdAt": "2026-10-12T00:00:00Z",
        "updatedAt": "2026-10-12T00:00:00Z",
        "url": "https://github.com/o/r/issues/100",
    }
    if mismatch == "author":
        forged["author"] = {"login": "another-user"}
    elif mismatch == "body":
        forged["body"] = body + "\n公開予定とは異なる調査結果。\n"
    else:
        forged["title"] = "同slotの別レポート"
    state = ops.state
    state["issues"].append(forged)
    state["next_number"] = 101
    ops.write_state(state)
    Path(ops.env["FAKE_AGENT_USED"]).unlink()

    args = ("report", "--if-due") if command == "report" else ("triage",)
    result = ops.run(*args)

    assert result.returncode == 1, result.stderr
    assert "publication" in result.stderr and "mismatch" in result.stderr
    row = _journal(ops)["records"]["20261012"]
    assert row["state"] == "prepared" and row["report_url"] is None
    assert row["head"] is None and row["pr_url"] is None
    assert _main_files(ops) == original
    assert _git(ops.repo, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert "ops-report-memory/" not in _git(ops.repo, "ls-remote", "--heads", "origin")
    assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")
    assert not ops.state.get("memory_prs")
    assert not Path(ops.env["FAKE_AGENT_USED"]).exists()


@pytest.mark.parametrize("mismatch", ["url", "body", "author"])
def test_publication_response_must_match_prepared_report(
    memory_env: Any, mismatch: str
) -> None:
    ops = memory_env
    original = _main_files(ops)
    env = {
        "url": {"FAKE_GH_CREATE_URL": "https://github.com/o/r/issues/999"},
        "body": {
            "FAKE_GH_CREATE_EDIT": json.dumps([["今回は", "書き換えられた結果は"]])
        },
        "author": {"FAKE_GH_CREATE_AUTHOR": "another-user"},
    }[mismatch]

    result = ops.run("report", **env)

    assert result.returncode == 1, result.stderr
    assert "publication" in result.stderr and "mismatch" in result.stderr
    row = _journal(ops)["records"]["20261012"]
    assert row["state"] == "prepared" and row["head"] is None
    assert _main_files(ops) == original
    assert "ops-report-memory/" not in _git(ops.repo, "ls-remote", "--heads", "origin")
    assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")
    assert not ops.state.get("memory_prs")
    # A retry cannot disregard the successful create response or accept the issue.
    retry = ops.run("report", "--if-due")
    assert retry.returncode == 1, retry.stderr
    assert _journal(ops)["records"]["20261012"]["state"] == "prepared"
    assert not ops.state.get("memory_prs")


@pytest.mark.parametrize("change", ["old_version", "missing_identity"])
def test_unverifiable_publication_journal_fails_closed(
    memory_env: Any, change: str
) -> None:
    ops = memory_env
    assert ops.run("report", FAKE_GH_FAIL="issue create").returncode == 1
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    journal = _journal(ops)
    if change == "old_version":
        journal["version"] = 1
    else:
        del journal["records"]["20261012"]["publication"]
    original = json.dumps(journal)
    path.write_text(original)

    result = ops.run("triage")

    assert result.returncode == 1 and "corrupt memory journal" in result.stderr
    assert path.read_text() == original
    assert not _reports(ops) and not ops.state.get("memory_prs")
    assert "ops-report-memory-20261012" not in _git(ops.repo, "worktree", "list")


def test_pr_creation_journal_gap_reuses_existing_pr(memory_env: Any) -> None:
    ops = memory_env
    assert ops.run("report").returncode == 0
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    journal = _journal(ops)
    journal["records"]["20261012"].update(pr_url=None, stage="pr_create")
    path.write_text(json.dumps(journal))
    assert ops.run("triage").returncode == 0
    assert len(ops.state["memory_prs"]) == 1
    assert sum(c[:2] == ["pr", "create"] for c in ops.calls()) == 1


def test_memory_change_on_main_during_pending_pr_is_not_overwritten(
    memory_env: Any,
) -> None:
    ops = memory_env
    assert ops.run("report").returncode == 0
    other = ops.tmp / "other"
    _git(ops.tmp, "clone", str(ops.tmp / "upstream.git"), str(other))
    (other / DIRECTORY / "SUMMARY.md").write_text("# human observation\n")
    _git(other, "add", DIRECTORY)
    _git(
        other,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "human observation",
    )
    _git(other, "push", "origin", "main")
    result = ops.run("triage")
    assert result.returncode == 1 and "main memory advanced" in result.stderr
    assert _main_files(ops)["SUMMARY.md"] == "# human observation\n"
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    assert _journal(ops)["records"]["20261012"]["state"] == "failed"


@pytest.mark.parametrize("change", ["CLOSED", "CONFLICTING", "fork", "draft"])
def test_unsafe_or_closed_pr_preserves_main(memory_env: Any, change: str) -> None:
    ops = memory_env
    original = _main_files(ops)
    assert ops.run("report").returncode == 0
    state = ops.state
    pr = state["memory_prs"][0]
    if change == "CLOSED":
        pr["state"] = change
    elif change == "CONFLICTING":
        pr["mergeable"] = change
    elif change == "fork":
        pr["isCrossRepository"] = True
    else:
        pr["isDraft"] = True
    ops.write_state(state)
    result = ops.run("triage")
    assert result.returncode == 1
    assert _main_files(ops) == original
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    if change == "CLOSED":
        assert _journal(ops)["records"]["20261012"]["state"] == "failed"
        assert (
            ops.run("triage").returncode == 0
        )  # retain failure, don't retry an obsolete PR


def test_path_scope_checks_deleted_and_renamed_sources(memory_env: Any) -> None:
    ops = memory_env
    original = _git(ops.repo, "rev-parse", "HEAD")
    _git(ops.repo, "mv", "small.py", DIRECTORY + "/moved.md")
    _git(ops.repo, "commit", "-qm", "unexpected rename")
    head = _git(ops.repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="confined"):
        memory_pr.Store(ops.repo, ops.tmp / "path-scope-memory.json")._verify_commit(
            {"head": head, "base_sha": original}, {}
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda u: u["upsert"][0].update(slug="../../workflow/oops"),
        lambda u: u["upsert"][0].update(slug="summary"),
        lambda u: u["upsert"][0].update(last_verified="2999-01-01"),
        lambda u: u["upsert"][0].pop("evidence"),
        lambda u: u["delete"].append({"slug": "ci-wait", "reason": "duplicate"}),
        lambda u: u.update(files={"../SUMMARY.md": "oops"}),
    ],
)
def test_invalid_patch_is_incomplete_and_never_persisted(
    memory_env: Any, mutate: Any
) -> None:
    ops = memory_env
    payload = json.loads(Path(ops.env["FAKE_AGENT_OUTPUT"]).read_text())
    mutate(payload["memory_update"])
    ops.set_agent_output(payload)
    result = ops.run("report")
    assert result.returncode == 1
    assert len(_reports(ops)) == 1 and not ops.state.get("memory_prs")
    assert "status=incomplete" in _reports(ops)[0]["body"].splitlines()[0]


def test_patch_requires_explicit_no_change_and_deletion_reason(
    agent_output: dict[str, Any],
) -> None:
    agent_output.pop("memory_update")
    with pytest.raises(agent.AgentOutputError, match="memory_update is required"):
        agent.parse_agent_output(json.dumps(agent_output), {7})
    update = _update()
    update["delete"] = [{"slug": "old", "reason": ""}]
    with pytest.raises(ValueError, match="deletion reason"):
        memory.parse_update(update)


def test_delete_compress_and_reverify_keep_index_consistent() -> None:
    files = {"SUMMARY.md": "initial", "index.md": "initial"}
    first = memory.apply_update(
        files, _update(), source="#100", today=date(2026, 10, 12)
    )
    update = _update(2)
    update["upsert"][0]["slug"] = "new-focus"
    update["delete"] = [{"slug": "ci-wait", "reason": "CI構成変更で反証"}]
    second = memory.apply_update(first, update, source="#102", today=date(2026, 10, 15))
    assert "ci-wait.md" not in second and "ci-wait" not in second["index.md"]
    assert "(new-focus.md)" in second["index.md"]
    assert "CIの継続観察 2" in second["SUMMARY.md"]
    update["upsert"][0]["last_verified"] = "2026-10-11"
    update["delete"] = []
    with pytest.raises(ValueError, match="monotonically"):
        memory.apply_update(second, update, source="#103", today=date(2026, 10, 15))


def test_corrupt_journal_fails_closed_without_reset(memory_env: Any) -> None:
    ops = memory_env
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    path.write_text("{")
    result = ops.run("report")
    assert result.returncode == 1 and "corrupt memory journal" in result.stderr
    assert path.read_text() == "{" and not _reports(ops)


def test_atomic_journal_replace_failure_keeps_previous_version(
    tmp_path: Path, monkeypatch: Any
) -> None:
    path = tmp_path / "memory.json"
    original = '{"last_valid": true}\n'
    path.write_text(original)

    def interrupted_replace(source: Path, target: Path) -> None:
        assert target == path
        raise OSError("injected interruption before replace")

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="interruption"):
        memory_pr._atomic_json(path, {"next": "unconfirmed"})
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "row",
    [
        {
            "proposal_id": "R-20261012-02",
            "decision": "unchecked",
            "reason": "no check",
            "source": "#100",
        },
        {
            "proposal_id": "R-20261012-02",
            "decision": "deferred",
            "reason": "",
            "source": "#100",
        },
    ],
)
def test_invalid_decision_inlet_fails_before_agent(
    memory_env: Any, row: dict[str, Any]
) -> None:
    ops = memory_env
    path = ops.tmp / "decisions.json"
    path.write_text(json.dumps([row]))
    result = ops.run("report", "--memory-decisions", str(path))
    assert result.returncode == 1
    assert not Path(ops.env["FAKE_AGENT_USED"]).exists()
    assert not _reports(ops)


def test_incomplete_scout_defers_all_memory_until_same_report_completes(
    memory_env: Any,
) -> None:
    ops = memory_env
    original = _main_files(ops)
    partial = ops.run("report", FAKE_AUDIT_FAIL="platform")
    assert partial.returncode == 1, partial.stderr
    assert _main_files(ops) == original
    assert not ops.state.get("memory_prs")
    assert not (
        Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    ).exists()
    number = _reports(ops)[0]["number"]
    retry = ops.run("report", "--if-due")
    assert retry.returncode == 0, retry.stderr
    assert len(_reports(ops)) == 1 and _reports(ops)[0]["number"] == number
    assert len(ops.state["memory_prs"]) == 1
    assert _journal(ops)["records"]["20261012"]["state"] == "pending"
    assert _main_files(ops) == original
    _external_merge(ops)
    merged = ops.run("report", "--if-due")
    assert merged.returncode == 0, merged.stderr
    assert _journal(ops)["records"]["20261012"]["state"] == "merged"


@pytest.mark.parametrize("merged", [False, True])
def test_v2_journal_reconciles_before_migrating_to_review_handoff(
    memory_env: Any, merged: bool
) -> None:
    ops = memory_env
    assert ops.run("report").returncode == 0
    journal = _journal(ops)
    before = journal["records"]["20261012"].copy()
    if merged:
        _external_merge(ops)
    state = ops.state
    state["issues"] = [i for i in state["issues"] if i["url"] != before["work_issue"]]
    state["memory_prs"][0]["body"] = (
        state["memory_prs"][0]["body"].split("\n", 1)[0] + "\n\n旧CI待ちの本文\n"
    )
    ops.write_state(state)
    journal["version"] = 2
    row = journal["records"]["20261012"]
    for key in ("work_issue", "validation", "work_issue_create_attempted"):
        row.pop(key)
    row["stage"] = "ci_wait"
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    path.write_text(json.dumps(journal))
    response = ops.run("triage")
    assert response.returncode == 0, response.stderr
    after = _journal(ops)
    row = after["records"]["20261012"]
    assert after["version"] == 3
    assert all(
        row[k] == before[k]
        for k in ("head", "base_sha", "files", "update", "publication", "pr_url")
    )
    assert row["stage"] == ("merged" if merged else "review_handoff")
    assert (row["work_issue"] is None) == merged
    assert not any(c[:2] == ["pr", "merge"] for c in ops.calls())
    if not merged:
        assert (
            f"Closes #{row['work_issue'].rsplit('/', 1)[1]}"
            in ops.state["memory_prs"][0]["body"]
        )


def test_work_issue_response_loss_reuses_marker_and_never_resends_unknown(
    memory_env: Any,
) -> None:
    ops = memory_env
    assert ops.run("report").returncode == 0
    journal = _journal(ops)
    issue = journal["records"]["20261012"]["work_issue"]
    journal["records"]["20261012"]["work_issue"] = None
    path = Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    path.write_text(json.dumps(journal))
    before = sum(c[:2] == ["issue", "create"] for c in ops.calls())
    assert ops.run("triage").returncode == 0
    assert _journal(ops)["records"]["20261012"]["work_issue"] == issue
    path.write_text(json.dumps(journal))
    state = ops.state
    state["issues"] = [i for i in state["issues"] if i["url"] != issue]
    ops.write_state(state)
    assert ops.run("triage").returncode == 1
    assert sum(c[:2] == ["issue", "create"] for c in ops.calls()) == before
