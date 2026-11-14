"""One family observation per launch plan, including its fresh admission check."""

from __future__ import annotations

import importlib

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def test_fresh_launch_plan_observes_each_family_relation_once(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    p.agreed()
    p.child("implementation")
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    directory = importlib.import_module("parent.state").state_directory()
    store = importlib.import_module("parent.state").Store(directory)
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    try:
        with store.locked():
            session = importlib.import_module("parent.session").ParentSession(
                gh, store, 1, "claude:private-session-a-unique"
            )
            for _ in range(2):
                offset = len(p.calls())
                with gh.fresh_reads():
                    assert session.launch_plan()["tasks"][0]["child"] == 10
                calls = p.calls()[offset:]
                endpoints = [a for c in calls for a in c if a.startswith("repos/")]
                for n in (1, 10):
                    assert (
                        endpoints.count(
                            f"repos/o/r/issues/{n}/sub_issues?per_page=100&page=1"
                        )
                        == 1
                    )
                    assert endpoints.count(f"repos/o/r/issues/{n}/parent") == 1
                    assert (
                        endpoints.count(
                            f"repos/o/r/issues/{n}/timeline?per_page=100&page=1"
                        )
                        == 1
                    )
    finally:
        gh.close()
