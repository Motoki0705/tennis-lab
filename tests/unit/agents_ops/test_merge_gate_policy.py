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
    return importlib.import_module("lib.ops_review.merge_gate_policy")


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
    LiveBackend = importlib.import_module(
        "lib.ops_review.progression_backend"
    ).LiveBackend
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


@pytest.fixture
def main_ci_backend(policy: Any, tmp_path: Path) -> Any:
    Store = importlib.import_module("parent.state").Store
    LiveBackend = importlib.import_module(
        "lib.ops_review.progression_backend"
    ).LiveBackend
    state = {
        "request": {"repo": "o/r", "parent_issue": 1},
        "targets": {"11": {"base_ref": "trunk"}},
    }
    return LiveBackend(state, Store(tmp_path))


@pytest.mark.parametrize(
    ("scenario", "status", "reason"),
    [
        ("no_jobs", "cancelled_before_start", "ci_python_cancelled_before_start"),
        ("no_python", "failure", "ci_python_cancelled"),
        ("python_cancelled", "failure", "ci_python_cancelled"),
        ("python_passed", "success", None),
        ("stale_python", "failure", "ci_python_cancelled"),
        ("rerun_queued", "pending", "ci_python_missing"),
        ("other_workflow", "pending", "ci_python_missing"),
    ],
)
def test_main_ci_checks_the_latest_push_attempt_even_without_checks(
    main_ci_backend: Any, scenario: str, status: str, reason: str | None
) -> None:
    latest = {
        "id": 17,
        "name": "CI",
        "path": ".github/workflows/ci.yml@refs/heads/trunk",
        "head_sha": HEAD,
        "head_branch": "trunk",
        "event": "push",
        "check_suite_id": 12,
        "run_number": 30,
        "run_attempt": 2,
        "status": "completed",
        "conclusion": "cancelled",
    }
    unrelated = {
        **latest,
        "path": ".github/workflows/other.yml",
        "id": 18,
        "run_number": 31,
    }
    runs = [unrelated]
    if scenario != "other_workflow":
        runs.extend(
            [
                {**latest, "run_attempt": 1, "conclusion": "success"},
                {**latest, "id": 15, "run_number": 29, "run_attempt": 8},
                {**latest, "id": 16, "run_number": 32, "event": "pull_request"},
                latest,
            ]
        )
    checks = [
        {**check, "id": 40, "check_suite": {"id": 12}}
        for check in ci_runs(
            {
                "python_cancelled": "cancelled",
                "python_passed": "success",
                "stale_python": "success",
                "rerun_queued": "success",
                "other_workflow": "success",
            }.get(scenario, "missing")
        )
    ]
    if scenario == "rerun_queued":
        latest.update(status="queued", conclusion=None)
    jobs = (
        []
        if scenario in {"no_jobs", "rerun_queued"}
        else [{"id": 40 if scenario in {"python_passed", "python_cancelled"} else 42}]
    )
    observations: list[str] = []

    class Fake:
        def get(self, endpoint: str) -> dict[str, Any]:
            observations.append(endpoint)
            if "/check-runs?" in endpoint:
                return {"total_count": len(checks), "check_runs": checks}
            if (
                endpoint
                == f"repos/o/r/actions/runs?head_sha={HEAD}&event=push&branch=trunk&per_page=100&page=1"
            ):
                return {"total_count": len(runs), "workflow_runs": runs}
            assert (
                endpoint
                == "repos/o/r/actions/runs/17/attempts/2/jobs?per_page=100&page=1"
            )
            return {"total_count": len(jobs), "jobs": jobs}

    main_ci_backend.gh = Fake()
    result = main_ci_backend.ci(11, HEAD, main=True)
    assert result["status"] == status
    assert result["reasons"] == ([reason] if reason else [])
    assert len(observations) == (2 if scenario == "other_workflow" else 3)
    if scenario == "no_jobs":
        assert (result["run"]["id"], result["run"]["run_attempt"]) == (17, 2)


@pytest.mark.parametrize("status", [201, 503])
def test_rerun_workflow_accepts_empty_201_and_submits_once(
    main_ci_backend: Any, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    HTTPResponse = importlib.import_module("issues.transport").HTTPResponse
    GitHubWriteError = importlib.import_module("parent.github").GitHubWriteError
    requests = []

    def response(method: str, endpoint: str, payload: dict[str, Any]) -> Any:
        requests.append((method, endpoint, payload))
        return HTTPResponse(status, {}, "")

    monkeypatch.setattr(main_ci_backend.gh._http, "request_response", response)
    if status == 201:
        main_ci_backend.rerun_workflow(17)
    else:
        with pytest.raises(GitHubWriteError, match="HTTP 503"):
            main_ci_backend.rerun_workflow(17)
    assert requests == [("POST", "repos/o/r/actions/runs/17/rerun", {})]


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
    Backend = importlib.import_module("lib.ops_review.progression_backend").LiveBackend
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
