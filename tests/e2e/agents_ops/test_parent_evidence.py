"""Evidence issues through the normal CLI, fake GitHub and throwaway git."""

from __future__ import annotations

import copy
import importlib
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_number,
    untracked_legacy_fixture,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


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


def _ensure_in_process(
    p: ParentEnv, directory: Path, ready: Any, contender: Any, results: Any, role: str
) -> None:
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        os.environ[key] = p.env[key]
    store = importlib.import_module("parent.state").Store(directory)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    original_get = gh.get
    synchronized = False

    def get(endpoint: str, *, missing_parent: bool = False) -> Any:
        nonlocal synchronized
        if role == "first" and endpoint == "repos/o/r/issues/1" and not synchronized:
            synchronized = True
            with store.unlocked_read():
                ready.set()
                assert contender.wait(10), "second process never entered its registry"
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


def test_review_gate_and_inspection_read_evidence_permalink_without_new_agent(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    before = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    moved = move_machine_records(p)
    execution_path = g.review.run_dir / "execution.json"
    execution = json.loads(execution_path.read_text())
    old_url = execution["parent_run_url"]
    cid = moved[int(old_url.rsplit("-", 1)[1])]
    new_url = (
        f"https://github.com/o/r/issues/{evidence_number(p.state())}#issuecomment-{cid}"
    )
    execution["parent_run_url"] = new_url
    execution["stages"]["publication"].update(
        parent_comment_id=cid, parent_run_url=new_url
    )
    execution_path.write_text(json.dumps(execution))
    state = p.state()
    for rows in state["comments"].values():
        for c in rows:
            records = g.policy.records([c], "operator")
            if records and records[0]["kind"] == "pr":
                c["body"] = g.policy.marker({**records[0], "parent_run_url": new_url})
    p.save(state)
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0")) == before
    view = json.loads(p.cli("review-inspect", "--prs", "100").stdout)
    run = view["prs"][0]["history"][0]
    assert run["parent_run_url"] == new_url and run["publication_errors"] == []
    assert len(g.review.calls()) == 1 and not g.merge_calls()
    state = p.state()
    result = next(
        c
        for c in state["comments"][str(evidence_number(state))]
        if g.policy.records([c], "operator")
        and g.policy.records([c], "operator")[0]["kind"] == "result"
    )
    state["comments"]["1"].append(
        {
            **result,
            "id": max(c["id"] for rows in state["comments"].values() for c in rows) + 1,
        }
    )
    p.save(state)
    blocked = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert (
        blocked.returncode == 2 and "duplicate parent evidence record" in blocked.stderr
    )
    assert len(g.review.calls()) == 1 and not g.merge_calls()


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
    before = len(p.calls())
    with store.locked():
        snapshot = collector.collect(gh, store, p.repo)
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
