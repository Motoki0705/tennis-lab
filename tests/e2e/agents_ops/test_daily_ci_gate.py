"""Daily freeze enforcement through the unchanged formal review requirements."""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from tests.e2e.agents_ops.test_daily_ci import (
    Clock,
    FakeActions,
    FakeHooks,
    git,
)
from tests.e2e.agents_ops.test_merge_gate import (
    GateEnv,
)
from tests.e2e.agents_ops.test_merge_gate import (
    gate_env as gate_env,
)
from tests.e2e.agents_ops.test_merge_gate import (
    parent_env as parent_env,
)


def connect_gate(g: GateEnv, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Clock]:
    p = g.review.p
    for name in (
        "CODEX_THREAD_ID",
        "CODEX_SESSION_ID",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CODE_PARENT_SESSION_ID",
        "CLAUDE_CODE_AGENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in p.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AGENT_AUTO_MERGE", "1")
    monkeypatch.setenv("PARENT_MERGE_ADVANCE_BASE", "1")
    monkeypatch.chdir(p.repo)
    module = importlib.import_module("merge_gate")
    package = importlib.import_module("daily_ci")
    controller = importlib.import_module("daily_ci.controller").Controller
    clock = Clock()
    api = package.CycleAPI(package.CycleStore("o/r"), clock=clock)
    api.initialize(git(p.repo, "rev-parse", "HEAD"))
    runner = controller(api, FakeActions(p.repo), hooks=FakeHooks())
    runner.enable()
    runner.tick()
    monkeypatch.setattr(
        module, "CycleAPI", lambda store: package.CycleAPI(store, clock=clock)
    )
    fake = p.root / "bin/gh"
    shebang, body = fake.read_text().split("\n", 1)
    # Real gh subprocess checks the same repo mutex is held during the remote write.
    intercept = f"""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, {str(module.HERE.parent)!r})
from daily_ci import CycleStore
if "repos/o/r/git/ref/heads/main" in sys.argv:
    if "--include" in sys.argv:
        print("HTTP/1.1 200 OK\\n")
    print(__import__("json").dumps({{"object": {{"sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}}}}))
    raise SystemExit(0)
if sys.argv[1:3] == ["pr", "merge"]:
    try:
        with CycleStore("o/r").repository_lock():
            raise RuntimeError("formal gate did not hold repository merge mutex")
    except ValueError as exc:
        assert "busy" in str(exc)
"""
    fake.write_text(shebang + "\n" + intercept + body)
    return module, api, clock


def call_gate(
    module: Any, capsys: pytest.CaptureFixture[str], cycle_id: str | None = None
) -> tuple[int, dict[str, Any], str]:
    args = ["--repo", "o/r", "--parent", "1"]
    if cycle_id:
        args.extend(["--daily-ci-cycle", cycle_id])
    code = module.main([*args, "merge"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out else {}, captured.err


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_formal_gate_freezes_normal_merges_then_reopens_at_deadline(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, clock = connect_gate(g, monkeypatch)
    code, value, error = call_gate(module, capsys)
    assert code == 1, error
    assert all(
        "daily_ci_normal_merge_frozen" in row["reasons"] for row in value["results"]
    )
    assert not g.merge_calls()
    # Ordinary merges still require no CI, including after an unresolved deadline.
    state = g.review.p.state()
    for pr in state["prs"]:
        pr["statusCheckRollup"] = []
    g.review.p.save(state)
    clock.set("2026-10-25T09:00:00+09:00")
    api.expire()
    code, value, error = call_gate(module, capsys)
    assert code == 0, error
    assert len(g.merge_calls()) == 3
    assert all(row["action"] == "merged" for row in value["results"])


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_registered_repair_passes_formal_gate_and_records_new_main(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, _ = connect_gate(g, monkeypatch)
    cycle_id = api.snapshot()["current_cycle_id"]
    head = g.review.p.state()["prs"][0]["head"]["sha"]
    api.register_repair(cycle_id, 100, head, ["failing-test"])
    old_target = api.snapshot()["cycles"][cycle_id]["target_sha"]
    code, value, error = call_gate(module, capsys, cycle_id)
    assert code == 1, error  # Two unregistered unrelated PRs remain blocked.
    rows = {r["pr"]: r for r in value["results"]}
    assert rows[100]["action"] == "merged"
    assert all(
        "daily_ci_repair_not_registered" in rows[n]["reasons"] for n in (101, 102)
    )
    assert len(g.merge_calls()) == 1
    cycle = api.snapshot()["cycles"][cycle_id]
    assert cycle["target_sha"] != old_target
    assert cycle["target_sha"] == git(g.review.p.repo, "rev-parse", "HEAD")
    assert cycle["generations"][-1]["kind"] == "repair"
    assert cycle["repair_prs"]["100"]["merge_state"] == "merged"


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_registered_repair_keeps_formal_draft_check(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, _ = connect_gate(g, monkeypatch)
    cycle_id = api.snapshot()["current_cycle_id"]
    state = g.review.p.state()
    state["prs"][0]["draft"] = True
    g.review.p.save(state)
    api.register_repair(cycle_id, 100, state["prs"][0]["head"]["sha"], ["failing-test"])
    code, value, error = call_gate(module, capsys, cycle_id)
    assert code == 1, error
    assert not g.merge_calls()
    row = next(r for r in value["results"] if r["pr"] == 100)
    assert row["decision"] == "blocked" and "draft" in row["reasons"]


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_delayed_repair_worker_cannot_merge_as_daytime_work(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, clock = connect_gate(g, monkeypatch)
    cycle_id = api.snapshot()["current_cycle_id"]
    head = g.review.p.state()["prs"][0]["head"]["sha"]
    api.register_repair(cycle_id, 100, head, ["failing-test"])
    clock.set("2026-10-25T09:00:00+09:00")
    # No watchdog/controller call is needed for this deadline check.
    monkeypatch.setenv("DAILY_CI_CYCLE_ID", cycle_id)
    code, value, error = call_gate(module, capsys)
    assert code == 1, error
    assert all(
        "daily_ci_cycle_expired_or_inactive" in row["reasons"]
        for row in value["results"]
    )
    assert not g.merge_calls()


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_window_is_rechecked_after_fresh_formal_collection(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, clock = connect_gate(g, monkeypatch)
    clock.set("2026-10-25T05:59:59+09:00")
    original = module.collect
    collections = 0

    def collect(*args: Any, **kwargs: Any) -> Any:
        nonlocal collections
        collections += 1
        result = original(*args, **kwargs)
        if collections == 2:
            clock.set("2026-10-25T06:00:00+09:00")
        return result

    monkeypatch.setattr(module, "collect", collect)
    code, value, error = call_gate(module, capsys)
    assert code == 1, error
    assert not g.merge_calls()
    assert all(
        "daily_ci_normal_merge_frozen" in row["reasons"] for row in value["results"]
    )


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
@pytest.mark.parametrize("repair", [False, True])
def test_window_closure_after_journaling_does_not_send_merge(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    repair: bool,
) -> None:
    g = gate_env
    module, api, clock = connect_gate(g, monkeypatch)
    cycle_id = api.snapshot()["current_cycle_id"]
    if repair:
        head = g.review.p.state()["prs"][0]["head"]["sha"]
        api.register_repair(cycle_id, 100, head, ["failing-test"])
    clock.set("2026-10-25T08:59:59+09:00" if repair else "2026-10-25T05:59:59+09:00")
    original = module.Store.write_json

    def write(path: Any, value: Any) -> None:
        original(path, value)
        if "merges" in path.parts and value.get("status") == "attempting":
            clock.set(
                "2026-10-25T09:00:00+09:00" if repair else "2026-10-25T06:00:00+09:00"
            )

    monkeypatch.setattr(module.Store, "write_json", staticmethod(write))
    code, _, error = call_gate(module, capsys, cycle_id if repair else None)
    assert code == 1, error
    assert not g.merge_calls()
    if not repair:
        # A provably unsent request must not poison this head after 09:00.
        monkeypatch.setattr(module.Store, "write_json", staticmethod(original))
        clock.set("2026-10-25T09:00:00+09:00")
        code, value, error = call_gate(module, capsys)
        assert code == 0, error
        assert all(row["action"] == "merged" for row in value["results"])


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_repair_gate_detects_external_main_update(
    gate_env: GateEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g = gate_env
    module, api, _ = connect_gate(g, monkeypatch)
    cycle_id = api.snapshot()["current_cycle_id"]
    head = g.review.p.state()["prs"][0]["head"]["sha"]
    api.register_repair(cycle_id, 100, head, ["failing-test"])
    git(g.review.p.repo, "commit", "--allow-empty", "-m", "unexpected main push")
    code, value, error = call_gate(module, capsys, cycle_id)
    assert code == 1, error
    assert not g.merge_calls()
    assert (
        "daily_ci_main_changed"
        in next(r for r in value["results"] if r["pr"] == 100)["reasons"]
    )
