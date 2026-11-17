"""Evidence discovery cost and freshness through fake GitHub's real pagination."""

from __future__ import annotations

import importlib
import json

import pytest

from tests.e2e.agents_ops.test_parent_evidence import evidence_issue
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)


def listing_calls(calls: list[list[str]]) -> list[str]:
    return [
        arg for call in calls for arg in call if arg.startswith("repos/o/r/issues?")
    ]


@pytest.mark.parametrize("duplicate", [False, True])
def test_labelled_discovery_reads_later_pages_and_rejects_duplicate_bindings(
    parent_env: ParentEnv, duplicate: bool
) -> None:
    p = parent_env
    state = p.state()
    for n in range(100, 200):
        row = evidence_issue(p, n, state="closed")
        row["body"] = row["body"].replace('"parent": 1', f'"parent": {n + 1000}')
        state["issues"][str(n)] = row
    state["issues"]["9"] = evidence_issue(p, state="closed")
    if duplicate:
        state["issues"]["10"] = evidence_issue(p, 10)
    p.save(state)
    result = p.cli("evidence", ok=not duplicate)
    if duplicate:
        assert "duplicate evidence issues" in result.stderr
    else:
        assert json.loads(result.stdout)["number"] == 9
        assert json.loads(result.stdout)["state"] == "closed"
    assert len(listing_calls(p.calls())) == 2


def test_unlabelled_marker_duplicate_is_outside_discovery(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    state["issues"]["10"] = {**evidence_issue(p, 10), "labels": []}
    p.save(state)
    assert json.loads(p.cli("evidence", "--create").stdout)["number"] == 9
    assert json.loads(p.cli("evidence").stdout)["number"] == 9
    assert not any("POST" in call for call in p.calls())


@pytest.mark.parametrize("boundary", ["scope", "fresh", "write"])
def test_discovery_scope_shares_reads_and_refreshes_at_boundaries(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["9"] = evidence_issue(p)
    p.save(state)
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS", "XDG_STATE_HOME"):
        monkeypatch.setenv(key, p.env[key])
    store = importlib.import_module("parent.state").Store(p.root / "discovery-state")
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    gh.parent = 1
    manager = gh.evidence_issues()
    with store.locked():
        store.focus("o/r#1")
        with gh.history_read_scope():
            assert manager.find(1, "operator").number == 9
            assert manager.find(1, "operator").number == 9
            assert len(listing_calls(p.calls())) == 1
            state["issues"]["10"] = evidence_issue(p, 10)
            p.save(state)
            if boundary == "write":
                # An unrelated comment write must also refresh the evidence index.
                # Its endpoint does not evict the listing from the GET cache.
                gh.write(
                    "POST",
                    "repos/o/r/issues/1/comments",
                    {"body": "Parent observation recorded"},
                )
            if boundary == "fresh":
                with (
                    gh.fresh_reads(),
                    pytest.raises(ValueError, match="duplicate evidence issues"),
                ):
                    manager.find(1, "operator")
            if boundary != "scope":
                with pytest.raises(ValueError, match="duplicate evidence issues"):
                    manager.find(1, "operator")
        if boundary == "scope":
            with (
                gh.history_read_scope(),
                pytest.raises(ValueError, match="duplicate evidence issues"),
            ):
                manager.find(1, "operator")
    assert len(listing_calls(p.calls())) == 2
