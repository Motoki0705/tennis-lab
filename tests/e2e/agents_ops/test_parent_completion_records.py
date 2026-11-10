"""Manual completion records do not require a fictitious child or new execution."""

from __future__ import annotations

import json
from typing import Any

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.improvements import candidate_spec


def agree(p: ParentEnv) -> None:
    state = p.state()
    policy = __import__("parent.policy", fromlist=["replace_section"])
    state["issues"]["1"]["body"] = policy.replace_section(
        state["issues"]["1"]["body"],
        "全体の完了条件",
        "- [ ] 結果を保存できる。\n- [ ] 失敗を表示できる。",
    )
    p.save(state)
    p.agreed()


def finish_spec(p: ParentEnv, candidates: list[str] | None = None) -> dict[str, Any]:
    required = json.loads(p.cli("status").stdout)["completion"]["requirements"]
    assert required["available"]
    assert [r["id"] for r in required["criteria"]] == ["agreement", "ac-1", "ac-2"]
    return {
        "schema_version": 1,
        "state": "completed",
        "basis_ref": required["basis_ref"],
        "acceptance": [
            {
                "id": "agreement",
                "state": "met",
                "evidence": "合意したfakeでの検証を完了。",
            },
            {
                "id": "ac-1",
                "state": "met",
                "evidence": "保存された結果を再読込して照合。",
            },
            {
                "id": "ac-2",
                "state": "met",
                "evidence": "意図した失敗を注入し表示を確認。",
            },
        ],
        "remaining_required": [],
        "improvements": {"reviewed": True, "candidates": candidates or []},
        "summary": "親が全条件の結果と根拠を確認した。",
    }


def test_unsplit_manual_parent_finishes_with_no_candidate_and_retries_without_reactivation(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    agree(p)
    before = p.state()["issues"]["1"]["body"]
    spec = finish_spec(p)
    response = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert (
        response["assignment"] == "ended"
        and response["completion"]["acceptance"] == "completed"
    )
    count = len(p.state()["comments"]["1"])
    response = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert (
        response["assignment"] == "ended" and len(p.state()["comments"]["1"]) == count
    )
    assert p.state()["issues"]["1"]["body"] == before and response["tasks"] == {}
    assert "次回改善: なし。" in "\n".join(
        c["body"] for c in p.state()["comments"]["1"]
    )


def test_unmet_acceptance_exported_to_candidate_cannot_finish(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    agree(p)
    candidate = candidate_spec()
    candidate["unmet_acceptance"] = ["ac-2"]
    p.cli("improvement", "--key", "failure-display", "--spec", p.spec(candidate))
    spec = finish_spec(p, ["I-1-failure-display"])
    denied = p.cli("release", "--ended", "--spec", p.spec(spec), ok=False)
    assert "unfinished acceptance" in denied.stderr
    status = json.loads(p.cli("status").stdout)
    assert status["assignment"] == "active" and status["completion"]["record"] is None
    p.cli("improvement-publish", "--candidate", "I-1-failure-display", ok=False)


def test_recorded_completion_and_failed_proposal_publication_are_independent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    agree(p)
    p.cli("improvement", "--key", "next", "--spec", p.spec(candidate_spec()))
    p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    p.cli("improvement-publish", "--candidate", "I-1-next", ok=False)
    del p.env["PARENT_FAIL_AFTER_CREATE"]
    spec = finish_spec(p, ["I-1-next"])
    status = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert (
        status["completion"]["acceptance"] == "completed"
        and status["improvements"][0]["publication"] == "outcome_unknown"
    )
    p.cli("improvement-publish", "--candidate", "I-1-next")
    status = json.loads(p.cli("status").stdout)
    assert (
        status["assignment"] == "ended"
        and status["completion"]["acceptance"] == "completed"
    )


def test_note_only_end_remains_unverified_and_unknown_legacy_criteria_are_not_migrated(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    status = json.loads(p.cli("status").stdout)
    assert not status["completion"]["requirements"]["available"]
    status = json.loads(p.cli("release", "--ended", "--note-file", p.note()).stdout)
    assert (
        status["assignment"] == "ended"
        and status["completion"]["acceptance"] == "unverified"
    )


def test_handwritten_completed_workflow_json_cannot_replace_a_finish(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    agree(p)
    response = p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "自己申告",
                "resume_condition": "なし",
                "evidence": "すべて完了した",
            }
        ),
        ok=False,
    )
    assert "structured finish" in response.stderr
    assert json.loads(p.cli("status").stdout)["assignment"] == "active"
