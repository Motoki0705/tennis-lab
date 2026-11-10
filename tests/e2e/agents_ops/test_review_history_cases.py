"""Five public historical pairs, contrasted with synthetic current-contract runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_merge_gate import ci_rollup
from tests.e2e.agents_ops.test_parent_session import (
    ReviewEnv,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_inspection import inspect
from tests.e2e.agents_ops.test_review_recovery import gate_for

CASES = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "support/agents_ops/review_history_cases.json"
    ).read_text()
)["cases"]


def legacy_result(case: dict[str, Any], policy: Any) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "review_run_id": f"parent-{case['parent']}-" + "a" * 32,
        "repo": "Motoki0705/tennis-lab",
        "parent_issue": case["parent"],
        "reason": "Public historical fixture",
        "prs": [case["source_snapshot"]],
    }
    return {
        "schema_version": 1,
        "kind": "result",
        "review_run_id": request["review_run_id"],
        "request": request,
        "request_sha256": policy.digest(request),
        "reviewer": "codex",
        "profile": "review",
        "started_at": "2026-10-31T00:00:00+00:00",
        "finished_at": "2026-10-31T00:01:00+00:00",
        "status": case["source_status"],
        "verdict": "incomplete",
        "summary": "Public evidence: " + case["source_url"],
        "error": case["cause"],
        "exception": None,
        "cross_findings": [],
        "prs": [
            {
                "number": case["pr"],
                "child_issues": [],
                "coverage": case["source_coverage"],
                "summary": case["source_url"],
                "findings": [],
            }
        ],
    }


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_public_pair_is_not_promoted_but_current_evidence_path_adds_no_agent(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, case: dict[str, Any]
) -> None:
    # Only public machine-record facts are copied. This does not claim to recover
    # a historical runtime or turn a past not_reviewed into reviewed.
    assert case["source_snapshot"] == case["next_snapshot"]
    assert case["source_findings"] == case["next_findings"] == 0
    policy = review_policy_module(monkeypatch)
    old = legacy_result(case, policy)
    policy.validate_result(old)
    coverage = policy.evaluate_coverage(
        old, old["request"]["prs"], [], {}, fix_diff_digests={}
    )
    assert not coverage.covered and f"pr_{case['pr']}_not_reviewed" in coverage.reasons
    forged = copy.deepcopy(old)
    forged["prs"][0]["evidence_waits"] = [
        {"question": case["cause"], "required_evidence": "CI evidence"}
    ]
    with pytest.raises(ValueError, match="completed code review"):
        policy.validate_result(forged)

    r = single_review_env
    g = gate_for(r, monkeypatch)
    if case["cause"] == "ci_pending":
        state = r.p.state()
        state["prs"][0]["statusCheckRollup"] = ci_rollup("pending")
        r.p.save(state)
    output = json.loads(r.output.read_text())
    output["prs"][0]["evidence_waits"] = [
        {
            "question": case["cause"],
            "required_evidence": "Synthetic fixture: same-HEAD CI/test evidence",
        }
    ]
    r.output.write_text(json.dumps(output))
    r.run()
    g.confirm()
    _, before = inspect(r, 100)
    assert before["prs"][0]["history"][0]["coverage_check"]["reasons"] == [
        "pr_100_evidence_1_pending"
    ]
    assert (
        "pr_100_evidence_1_pending"
        in g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["reasons"]
    )
    calls = len(r.calls())
    state = r.p.state()
    state["prs"][0]["statusCheckRollup"] = ci_rollup("success")
    r.p.save(state)
    snapshot = r.request["prs"][0]
    evidence = r.p.root / "completion.json"
    evidence.write_text(
        json.dumps(
            {
                "ci": "Synthetic fixture: same HEAD Python success",
                "regression_tests": "Synthetic fixed-HEAD evidence confirmed",
                "resolution": "Parent checked the outstanding evidence only",
                "reviewed_paths": run_git(
                    r.p.repo,
                    "diff",
                    "--name-only",
                    snapshot["merge_base_sha"],
                    snapshot["head_sha"],
                ).splitlines(),
            }
        )
    )
    r.cli(
        "complete-evidence",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--wait",
        "1",
        "--evidence",
        str(evidence),
    )
    _, after = inspect(r, 100)
    assert after["prs"][0]["history"][0]["coverage_check"]["covered"] is True
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    assert len(r.calls()) == calls == 1  # no second fake agent, no real model
