"""Current-head Python CI and human control, without Git/agent setup."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

HEAD = "a" * 40
CI_ERRORS = {
    "success": (),
    "failure": ("ci_python_failure",),
    "pending": ("ci_python_pending",),
    "cancelled": ("ci_python_cancelled",),
    "missing": ("ci_python_missing",),
    "head_mismatch": ("ci_python_head_mismatch",),
}


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    ops = Path(__file__).resolve().parents[3] / ".agents/ops"
    monkeypatch.syspath_prepend(str(ops))
    monkeypatch.syspath_prepend(str(ops / "review"))
    return importlib.import_module("merge_gate_policy")


def ci_runs(state: str, head: str = HEAD) -> list[dict[str, Any]]:
    if state == "missing":
        return []
    status, conclusion = {
        "success": ("completed", "success"),
        "failure": ("completed", "failure"),
        "pending": ("queued", None),
        "cancelled": ("completed", "cancelled"),
        "head_mismatch": ("completed", "success"),
    }[state]
    return [
        {
            "name": "Python",
            "head_sha": "0" * 40 if state == "head_mismatch" else head,
            "status": status,
            "conclusion": conclusion,
            "started_at": "2026-10-30T01:00:00Z",
        }
    ]


@pytest.mark.parametrize("state,errors", list(CI_ERRORS.items()))
def test_python_ci_errors(policy: Any, state: str, errors: tuple[str, ...]) -> None:
    assert policy.python_ci_errors(ci_runs(state), HEAD) == errors


@pytest.mark.parametrize("value", ["", "true", "2", " 1"])
def test_auto_merge_policy_rejects_invalid_values(policy: Any, value: str) -> None:
    with pytest.raises(ValueError, match="AGENT_AUTO_MERGE"):
        policy.auto_merge(value)


@pytest.mark.parametrize(
    "old,new_status,new_conclusion,new_started,expected",
    [
        ("failure", "completed", "success", "2026-10-30T02:00:00Z", ()),
        (
            "success",
            "completed",
            "failure",
            "2026-10-30T02:00:00Z",
            ("ci_python_failure",),
        ),
        (
            "success",
            "completed",
            "cancelled",
            "2026-10-30T02:00:00Z",
            ("ci_python_cancelled",),
        ),
        (
            "failure",
            "completed",
            "success",
            "2026-10-30T01:00:00Z",
            ("ci_python_failure",),
        ),
        ("failure", "completed", "success", None, ("ci_python_failure",)),
        (
            "failure",
            "completed",
            "success",
            "0001-01-01T00:00:00Z",
            ("ci_python_failure",),
        ),
        ("success", "queued", None, None, ("ci_python_pending",)),
    ],
)
def test_python_reruns_use_latest_time_and_pessimism_when_unordered(
    policy: Any,
    old: str,
    new_status: str,
    new_conclusion: str | None,
    new_started: str | None,
    expected: tuple[str, ...],
) -> None:
    earlier = {**ci_runs("success")[0], "conclusion": old}
    latest = {
        **earlier,
        "status": new_status,
        "conclusion": new_conclusion,
        "started_at": new_started,
    }
    assert policy.python_ci_errors([earlier, latest], HEAD) == expected


@pytest.mark.parametrize("field", ["head_sha", "status", "conclusion"])
def test_unknown_python_check_fields_are_not_success(policy: Any, field: str) -> None:
    run = {**ci_runs("success")[0], field: "unknown"}
    with pytest.raises(ValueError):
        policy.python_ci_errors([run], HEAD)


@pytest.mark.parametrize("tail", ["success", "failure", "missing_page"])
def test_required_check_run_pages_are_all_checked(
    policy: Any,
    tmp_path: Path,
    tail: str,
) -> None:
    Store = importlib.import_module("parent.state").Store
    LiveBackend = importlib.import_module("review.progression_backend").LiveBackend
    backend = LiveBackend(
        {"request": {"repo": "o/r", "parent_issue": 1}}, Store(tmp_path)
    )
    calls: list[str] = []

    class Fake:
        def get(self, endpoint: str) -> dict:
            calls.append(endpoint)
            if endpoint.endswith("page=1"):
                return {
                    "total_count": 101,
                    "check_runs": [ci_runs("success")[0] for _ in range(100)],
                }
            assert endpoint.endswith("page=2")
            return {
                "total_count": 101,
                "check_runs": [] if tail == "missing_page" else ci_runs(tail),
            }

    backend.gh = Fake()
    if tail == "missing_page":
        with pytest.raises(ValueError, match="incomplete"):
            backend.ci(11, HEAD)
    else:
        assert backend.ci(11, HEAD)["status"] == tail
    assert len(calls) == 2


@pytest.mark.parametrize(
    "case,expected",
    [
        ("research", True),
        ("merged", True),
        ("extra_closed_unmerged", True),
        ("extra_open", False),
        ("closed_unmerged", False),
        ("nondefault", False),
        ("issue_open", False),
        ("cancelled", False),
        ("duplicate", False),
        ("body_stopped", False),
    ],
)
def test_dependency_keeps_completion_and_default_branch_contract(
    policy: Any,
    tmp_path: Path,
    case: str,
    expected: bool,
) -> None:
    Store = importlib.import_module("parent.state").Store
    Backend = importlib.import_module("review.progression_backend").LiveBackend
    backend = Backend({"request": {"repo": "o/r", "parent_issue": 1}}, Store(tmp_path))
    item: dict[str, Any] = {"state": "closed", "state_reason": "completed", "body": ""}
    pr: dict[str, Any] = {
        "number": 30,
        "state": "closed",
        "body": "Closes #3",
        "merged": True,
        "base": {"ref": "main", "repo": {"full_name": "o/r", "default_branch": "main"}},
    }
    prs = [pr]
    if case == "research":
        prs = []
    elif case == "extra_closed_unmerged":
        prs.append({**pr, "number": 31, "merged": False})
    elif case == "extra_open":
        prs.append({**pr, "number": 31, "merged": False, "state": "open"})
    elif case == "closed_unmerged":
        pr["merged"] = False
    elif case == "nondefault":
        pr["base"]["ref"] = "codex/dependency"
    elif case == "issue_open":
        item.update(state="open", state_reason=None)
    elif case in {"cancelled", "duplicate"}:
        item["state_reason"] = "not_planned" if case == "cancelled" else "duplicate"
    elif case == "body_stopped":
        item["body"] = importlib.import_module("parent.policy").replace_section(
            "",
            "作業状況・停止事項",
            "<!-- parent-task-status:v1 state=stopped detail=" + "a" * 64 + " -->",
        )

    class Fake:
        def issue(self, n: int) -> dict:
            assert n == 3
            return item

        def pulls(self, numbers: set[int]) -> list[dict]:
            assert numbers == {3}
            return prs

        def get(self, endpoint: str) -> dict:
            return next(p for p in prs if endpoint.endswith("/" + str(p["number"])))

    backend.gh = Fake()
    assert (
        backend.dependency(
            {"dependencies": [{"issue": 3, "prs": [p["number"] for p in prs]}]}
        )
        is expected
    )
