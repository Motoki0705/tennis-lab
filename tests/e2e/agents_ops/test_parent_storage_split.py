"""Ownership-preserving migration and independent parent transactions."""

from __future__ import annotations

import importlib
import json

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def record(session: str) -> dict:
    return {
        "session": session,
        "status": "active",
        "generation": 1,
        "tasks": {"10": {"evidence": "kept"}},
        "reviews": [{"request": {"schema_version": 1}}],
    }


def test_migration_preserves_records_and_only_changes_selected_file(
    parent_env: ParentEnv,
) -> None:
    module = importlib.import_module("parent.state")
    directory = parent_env.root / "split"
    original = {
        "version": 1,
        "parents": {"o/r#1": record("claude:one"), "o/r#2": record("claude:two")},
    }
    module.Store.write_json(directory / "registry.json", original)
    with module.Store(directory).locked() as store:
        store.migrate()
        assert store.parents == original["parents"]
    assert json.loads((directory / "registry-v1.backup.json").read_text()) == original
    index = (directory / "registry.json").read_bytes()
    peer = directory / "parents" / (module.digest("o/r#2") + ".json")
    peer_stamp = peer.stat().st_mtime_ns
    with module.Store(directory).locked() as store:
        store.parents["o/r#1"]["tasks"]["10"]["evidence"] = "updated"
        store.save()
    assert (directory / "registry.json").read_bytes() == index
    assert peer.stat().st_mtime_ns == peer_stamp
    with module.Store(directory).locked() as store:
        assert store.parents["o/r#1"]["tasks"]["10"]["evidence"] == "updated"
        store.migrate()  # Idempotent; no loss of newer evidence.


@pytest.mark.parametrize("migrate", [False, True])
def test_unrelated_save_does_not_abort_or_overwrite_peer(
    parent_env: ParentEnv, migrate: bool
) -> None:
    module = importlib.import_module("parent.state")
    directory = parent_env.root / "independent"
    module.Store.write_json(
        directory / "registry.json",
        {
            "version": 1,
            "parents": {"o/r#1": record("claude:one"), "o/r#2": record("claude:two")},
        },
    )
    with module.Store(directory).locked() as first:
        if migrate:
            first.migrate()
        first.focus("o/r#1")
        live = first.parents["o/r#1"]
        live["tasks"]["10"]["evidence"] = "unsaved local"
        with first.unlocked_read(), module.Store(directory).locked() as second:
            second.parents["o/r#2"]["tasks"]["10"]["evidence"] = "peer write"
            second.save()
        assert first.parents["o/r#1"] is live
        first.save()
    with module.Store(directory).locked() as result:
        assert result.parents["o/r#2"]["tasks"]["10"]["evidence"] == "peer write"
        assert result.parents["o/r#1"]["tasks"]["10"]["evidence"] == "unsaved local"


def test_same_parent_change_still_rejects_stale_data(parent_env: ParentEnv) -> None:
    module = importlib.import_module("parent.state")
    directory = parent_env.root / "conflict"
    module.Store.write_json(
        directory / "registry.json",
        {"version": 1, "parents": {"o/r#1": record("claude:one")}},
    )
    with module.Store(directory).locked() as first:
        first.migrate()
        first.focus("o/r#1")
        with (
            pytest.raises(ValueError, match="registry changed"),
            first.unlocked_read(),
            module.Store(directory).locked() as second,
        ):
            second.parents["o/r#1"]["generation"] += 1
            second.save()


def test_malformed_legacy_state_is_not_migrated(parent_env: ParentEnv) -> None:
    module = importlib.import_module("parent.state")
    path = parent_env.root / "bad" / "registry.json"
    original = {
        "version": 1,
        "parents": {"o/r#1": record("same"), "o/r#2": record("same")},
    }
    module.Store.write_json(path, original)
    with (
        pytest.raises(ValueError, match="multiple active"),
        module.Store(path.parent).locked() as store,
    ):
        store.migrate()
    assert json.loads(path.read_text()) == original
    assert not (path.parent / "registry-v1.backup.json").exists()


def test_metrics_are_private_and_requests_are_reused(parent_env: ParentEnv) -> None:
    p = parent_env
    p.cli("register")
    log = p.root / "metrics.jsonl"
    result = p.cli(
        "status", extra={"PARENT_METRICS_FILE": str(log), "PARENT_METRICS_ATTEMPT": "2"}
    )
    metrics = json.loads(log.read_text())
    assert metrics["cache_hits"] > 0
    assert metrics["github_requests"] > 0
    assert metrics["attempt"] == 2 and metrics["retry_count"] == 1
    assert metrics["exit_code"] == 0
    assert metrics["elapsed_seconds"] >= metrics["github_seconds"]
    assert p.env["CLAUDE_CODE_SESSION_ID"] not in log.read_text()
    assert str(p.root) not in log.read_text()
    assert result.returncode == 0


def test_parent_does_not_enumerate_repository_pulls(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    p.child("bounded-fetch")
    state = p.state()
    state["prs"] = [{"number": n, "body": "Closes #99999"} for n in range(1000, 3000)]
    p.save(state)
    p.calls_path.write_text("")
    p.cli("status")
    calls = p.calls_path.read_text()
    assert "pulls?state=all" not in calls
    assert "issues/10/timeline" in calls
    assert len(calls.splitlines()) < 15


def test_unstable_pagination_restarts_and_preserves_all_rows(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("parent.github")
    gh = module.GitHub("o/r")
    calls: list[str] = []

    def page(endpoint: str, **kwargs: object) -> object:
        calls.append(endpoint)
        if len(calls) == 2:
            return [{"id": 99}, {"id": 100}]  # New insertion shifted a page boundary.
        if endpoint.endswith("&page=1"):
            return [{"id": n} for n in range(100)]
        return [{"id": 100}, {"id": 101}]

    monkeypatch.setattr(gh, "get", page)
    assert [row["id"] for row in gh.rows("repos/o/r/issues/1/comments")] == list(
        range(102)
    )
    assert len(calls) == 4


def test_persistent_duplicate_pages_never_become_success(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("parent.github")
    gh = module.GitHub("o/r")
    monkeypatch.setattr(gh, "get", lambda *a, **kw: [{"id": 1}, {"id": 1}])
    with pytest.raises(ValueError, match="three bounded attempts"):
        gh.rows("repos/o/r/issues/1/comments")


def test_new_transaction_never_reuses_old_github_facts(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("parent.github")
    state = importlib.import_module("parent.state")
    store = state.Store(parent_env.root / "cache-transactions")
    gh = module.GitHub("o/r", store=store)
    gh.parent = 1
    calls: list[str] = []

    def fetch(self: object, endpoint: str, **kwargs: object) -> object:
        calls.append(endpoint)
        return {"revision": len(calls)}

    monkeypatch.setattr(module.IssueGitHub, "get", fetch)
    with store.locked():
        assert gh.get("user") == {"revision": 1}
        assert gh.get("user") == {"revision": 1}
    with store.locked():
        assert gh.get("user") == {"revision": 2}
