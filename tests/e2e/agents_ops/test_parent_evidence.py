"""Evidence issues through the normal CLI, fake GitHub and throwaway git."""

from __future__ import annotations

import copy
import importlib
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_comments,
    evidence_number,
    untracked_legacy_fixture,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.github_http import install


def evidence_issue(env: ParentEnv, n: int = 9, *, state: str = "open") -> dict:
    return {
        **copy.deepcopy(env.state()["issues"]["1"]),
        "id": n + 10000,
        "number": n,
        "title": "Parent evidence #1",
        "body": '<!-- parent-evidence:v1 {"parent": 1, "repo": "o/r"} -->\n\nMachine evidence.',
        "labels": [{"name": "parent-evidence"}],
        "user": {"login": "operator"},
        "state": state,
    }


@pytest.mark.parametrize("lost_response", [False, True])
def test_creation_is_idempotent_after_response_loss_and_parallel_retries(
    parent_env: ParentEnv,
    lost_response: bool,
) -> None:
    env = parent_env
    if lost_response:
        result = env.cli(
            "evidence", "--create", extra={"PARENT_FAIL_AFTER_CREATE": "1"}, ok=False
        )
        assert "uncertain" in result.stderr
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: env.cli("evidence", "--create"), range(2)))
    created = [
        r
        for r in env.state()["issues"].values()
        if r["labels"] == [{"name": "parent-evidence"}]
    ]
    assert len(created) == 1
    assert {json.loads(r.stdout)["number"] for r in results} == {created[0]["number"]}
    posts = [c for c in env.calls() if "POST" in c and "repos/o/r/issues" in c]
    assert len(posts) == 1
    assert not any(env.state()["comments"].values())


def test_uncertain_creation_without_remote_match_never_posts_again(
    parent_env: ParentEnv,
) -> None:
    env = parent_env
    result = env.cli(
        "evidence", "--create", extra={"PARENT_FAIL_PATH": "repos/o/r/issues"}, ok=False
    )
    assert "uncertain" in result.stderr
    again = env.cli("evidence", "--create", ok=False)
    assert "uncertain" in again.stderr
    assert len([c for c in env.calls() if "POST" in c and "repos/o/r/issues" in c]) == 1


def test_register_recovers_after_an_explicit_evidence_creation_rejection(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    failed = p.cli("register", extra={"PARENT_REJECT_EVIDENCE_CREATE": "1"}, ok=False)
    receipt = json.loads(failed.stdout)
    assert receipt["parent_action_applied"] is True
    assert "register" in receipt["assignment_evidence"]["retry"]
    assert "HTTP 403" in receipt["assignment_evidence"]["error"]
    assert len(p.state()["issues"]) == 2
    assert not p.comments()

    p.cli("register")
    p.cli("register")
    assert len(p.state()["issues"]) == 3
    [assigned] = evidence_comments(p.state())
    assert '"event": "assigned", "generation": 1' in assigned["body"].splitlines()[0]
    assert len([c for c in p.calls() if "POST" in c and "repos/o/r/issues" in c]) == 2


def test_register_verifies_evidence_while_labelled_listing_lags(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["unlisted_evidence"] = [9000]
    p.save(state)

    first = json.loads(p.cli("register").stdout)
    assert first["assignment"] == "active"
    assert first["parent_action_applied"] is True
    journal = p.root / "private-state/tennis-lab-agents/parent-sessions/evidence-issues"
    [binding] = journal.glob("*.json")
    assert json.loads(binding.read_text())["state"] == "bound"
    assert json.loads(p.cli("evidence").stdout)["number"] == 9000
    p.cli("register")
    [assigned] = evidence_comments(p.state())
    assert '"event": "assigned"' in assigned["body"].splitlines()[0]

    calls = p.calls()
    creation = [
        i for i, c in enumerate(calls) if "POST" in c and "repos/o/r/issues" in c
    ]
    assert len(creation) == 1
    assert any(
        "GET" in c and "repos/o/r/issues/9000" in c for c in calls[creation[0] + 1 :]
    )

    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    p.save(state)
    assert "evidence issue binding changed" in p.cli("evidence", ok=False).stderr


def test_evidence_creation_requires_a_verified_direct_issue_read(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["unlisted_evidence"] = [9000]
    p.save(state)
    failed = p.cli(
        "register",
        extra={"PARENT_FAIL_GET_PATH": "repos/o/r/issues/9000"},
        ok=None,
    )
    assert failed.returncode == 1, failed.stderr
    receipt = json.loads(failed.stdout)
    assert receipt["parent_action_applied"] is True
    assert "HTTP 503" in receipt["assignment_evidence"]["error"]
    assert receipt["board_sync"]["status"] == "assigned"
    assert receipt["display_sync"]["status"] == "not_configured"
    assert not any(p.state()["comments"].values())

    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    p.save(state)
    conflict = json.loads(p.cli("register", ok=False).stdout)
    assert "evidence issue binding changed" in conflict["assignment_evidence"]["error"]
    assert not any(p.state()["comments"].values())

    del state["issues"]["9"]
    p.save(state)
    p.cli("register")
    assert json.loads(p.cli("evidence").stdout)["number"] == 9000
    assert len(evidence_comments(p.state())) == 1
    assert len([c for c in p.calls() if "POST" in c and "repos/o/r/issues" in c]) == 1


def _ensure_in_process(
    p: ParentEnv, directory: Path, ready: Any, contender: Any, results: Any, role: str
) -> None:
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        os.environ[key] = p.env[key]
    # A fork has its own fake-gh environment; use a server in that process
    # instead of inheriting the parent pytest server and its environment.
    with closing(install(importlib.import_module("issues.transport"))):
        store = importlib.import_module("parent.state").Store(directory)
        gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
        original_get = gh.get
        synchronized = False

        def get(endpoint: str, *, missing_parent: bool = False) -> Any:
            nonlocal synchronized
            if (
                role == "first"
                and endpoint == "repos/o/r/issues/1"
                and not synchronized
            ):
                synchronized = True
                with store.unlocked_read():
                    ready.set()
                    assert contender.wait(10), (
                        "second process never entered its registry"
                    )
            return original_get(endpoint, missing_parent=missing_parent)

        gh.get = get

        if role == "second":
            assert ready.wait(10), "first process never reached GitHub I/O"
        with store.locked():
            store.focus("o/r#1")
            if role == "second":
                contender.set()
            results.put(gh.evidence_issues().ensure(1, "operator").number)


def test_creation_releases_registry_before_waiting_for_another_creator(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    ctx = multiprocessing.get_context("fork")
    ready, contender, results = ctx.Event(), ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(
            target=_ensure_in_process,
            args=(p, p.root / "shared-creation", ready, contender, results, role),
        )
        for role in ("first", "second")
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(15)
        assert all(not process.is_alive() for process in processes), (
            "creation deadlocked while reacquiring the registry"
        )
        assert all(process.exitcode == 0 for process in processes)
        assert len({results.get(timeout=2) for _ in processes}) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        results.close()
    assert (
        len(
            [
                call
                for call in p.calls()
                if "POST" in call and "repos/o/r/issues" in call
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "tamper",
    ["duplicate", "label", "extra-label", "publisher", "parent", "repo", "number"],
)
def test_binding_is_revalidated_instead_of_trusting_the_local_number(
    parent_env: ParentEnv, tamper: str
) -> None:
    env = parent_env
    state = env.state()
    state["issues"]["9"] = evidence_issue(env)
    env.save(state)
    assert json.loads(env.cli("evidence", "--create").stdout)["number"] == 9
    state = env.state()
    if tamper == "duplicate":
        state["issues"]["10"] = evidence_issue(env, 10)
    elif tamper == "label":
        state["issues"]["9"]["labels"] = []
    elif tamper == "extra-label":
        state["issues"]["9"]["labels"].append({"name": "issue:parent"})
    elif tamper == "publisher":
        state["issues"]["9"]["user"]["login"] = "other"
    elif tamper == "number":
        state["issues"]["9"]["number"] = 10
    else:
        state["issues"]["9"]["body"] = (
            state["issues"]["9"]["body"].replace('"parent": 1', '"parent": 2')
            if tamper == "parent"
            else state["issues"]["9"]["body"].replace('"o/r"', '"other/repo"')
        )
    env.save(state)
    result = env.cli("evidence", ok=False)
    assert "evidence" in result.stderr
    assert not [c for c in env.calls() if "POST" in c]


@pytest.mark.parametrize("state", ["open", "closed"])
def test_evidence_cannot_be_assigned_or_read_as_a_work_family(
    parent_env: ParentEnv, state: str
) -> None:
    env = parent_env
    snapshot = env.state()
    snapshot["issues"]["9"] = evidence_issue(env, state=state)
    env.save(snapshot)
    result = env.cli("register", parent="9", ok=False)
    assert "evidence" in result.stderr
    assert not [c for c in env.calls() if "POST" in c or "PATCH" in c]


def move_machine_records(env: ParentEnv) -> dict[int, int]:
    """Build mixed legacy/evidence history without creating a second container."""
    state = env.state()
    dest = str(evidence_number(state))
    originals = state["comments"][dest]
    machines = [
        c
        for c in originals
        if c["body"].startswith(("<!-- parent-session:", "<!-- parent-review:"))
    ]
    first = 1 + max(c["id"] for rows in state["comments"].values() for c in rows)
    legacy = [
        c
        for c in machines
        if any(
            f'"event": "{event}"' in c["body"].splitlines()[0]
            for event in ("assigned", "checking")
        )
    ]
    moved = {
        c["id"]: c["id"] if c in legacy else first + i for i, c in enumerate(machines)
    }
    state["comments"][dest] = [
        {**c, "id": moved[c["id"]]} for c in machines if c not in legacy
    ]
    state["comments"]["1"].extend(legacy)
    env.save(state)
    untracked_legacy_fixture(env)
    return moved


def test_readonly_client_uses_the_selected_state_cache_without_session_identity(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    custom = p.root / "selected-state"
    value = json.loads(
        p.cli(
            "--state-dir",
            str(custom),
            "evidence",
            "--create",
            extra={"CLAUDE_CODE_SESSION_ID": "", "CODEX_SESSION_ID": ""},
        ).stdout
    )
    state = p.state()
    row = state["issues"][str(value["number"])]
    row["labels"], row["body"] = [], "binding removed"
    p.save(state)
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        monkeypatch.setenv(key, p.env[key])
    gh = importlib.import_module("parent.github").GitHub("o/r")
    store = importlib.import_module("parent.state").Store(custom)
    session = importlib.import_module("parent.session").ParentSession(
        gh, store, 1, None
    )
    before = {f.name: f.read_bytes() for f in (custom / "evidence-issues").iterdir()}
    with pytest.raises(ValueError, match="evidence"):
        session.history()
    assert {
        f.name: f.read_bytes() for f in (custom / "evidence-issues").iterdir()
    } == before


def test_evidence_agreement_keeps_state_and_same_login_human_input(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    before = json.loads(p.cli("status").stdout)
    inputs = json.loads(p.cli("basis-input").stdout)
    move_machine_records(p)
    assert json.loads(p.cli("status").stdout) == before
    assert json.loads(p.cli("basis-input").stdout) == inputs
    state = p.state()
    n = 1 + max(c["id"] for rows in state["comments"].values() for c in rows)
    state["comments"]["1"] = [
        {"id": n, "user": {"login": "operator"}, "body": "次の方針を確認したい。"}
    ]
    p.save(state)
    after = json.loads(p.cli("status").stdout)
    assert after["input_changes"]["comments"][0]["id"] == n
    assert p.state() == state


@pytest.mark.parametrize("state", ["open", "closed"])
def test_listings_and_explicit_work_consumers_exclude_evidence(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    p = parent_env
    raw = p.state()
    raw["issues"]["9"] = evidence_issue(p, state=state)
    for issue in raw["issues"].values():
        issue["assignees"] = []
    p.save(raw)
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        monkeypatch.setenv(key, p.env[key])
    gh = importlib.import_module("parent.github").GitHub("o/r")
    assert {i["number"] for i in gh.issue_rows()} == {1, 2}
    with pytest.raises(ValueError, match="evidence"):
        gh.family(9)
    store = importlib.import_module("parent.state").Store(p.root / "collection")
    collector = importlib.import_module("dispatcher.collect")
    transaction_gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    before = len(p.calls())
    with store.locked():
        snapshot = collector.collect(transaction_gh, store, p.repo)
    assert (
        sum(
            "repos/o/r/issues?state=all&labels=parent-evidence&per_page=100&page=1"
            in call
            for call in p.calls()[before:]
        )
        == 1
    )
    assert snapshot["subjects"] == [1, 2] and set(snapshot["issues"]) == {"1", "2"}
    snapshot["issues"]["9"] = {
        **raw["issues"]["9"],
        "parent": None,
        "children": [],
        "comments": [],
    }
    assert (
        importlib.import_module("dispatcher.schema").guard(snapshot, 9)["disposition"]
        == "out_of_scope"
    )
    api = SimpleNamespace(issue=lambda *_: raw["issues"]["9"])
    with pytest.raises(ValueError, match="evidence"):
        importlib.import_module("project_progress.progress").read_tree(api, "o/r", 9)
    assert not [c for c in p.calls() if "POST" in c or "PATCH" in c]


def test_issue_workflows_exclude_evidence_before_side_effects() -> None:
    root = Path(__file__).resolve().parents[3]
    for workflow, job in (
        ("issue-supplement.yml", "prepare"),
        ("issue-labeler.yml", "label"),
    ):
        config: dict[str, Any] = yaml.safe_load(
            (root / ".github/workflows" / workflow).read_text()
        )
        condition = config["jobs"][job]["if"]
        assert (
            "!contains(github.event.issue.labels.*.name, 'parent-evidence')"
            in condition
        )
        assert (
            "!startsWith(github.event.issue.body, '<!-- parent-evidence:')" in condition
        )
    steps = config["jobs"]["label"]["steps"]
    label_step = next(
        s for s in steps if str(s.get("uses", "")).startswith("github/issue-labeler@")
    )
    assert "steps.target.outputs.is_work_issue" in label_step["if"]
