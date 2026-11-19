"""Manual completion records do not require a fictitious child or new execution."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv, make_gate_env
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_improvements import decision_spec
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.decision_logs import navigation_snapshot
from tests.support.agents_ops.improvements import candidate_spec, finish_for


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


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("gate_env", ["parent_self"], indirect=True)
def test_unsplit_manual_parent_finishes_with_no_candidate_and_retries_without_reactivation(
    gate_env: GateEnv,
    combined: bool,
) -> None:
    g, p = gate_env, gate_env.review.p
    requirement_body = importlib.import_module("parent.execution").requirement_body
    before = requirement_body(p.state()["issues"]["1"]["body"])
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    spec = finish_for(p)
    if combined:
        evidence = json.loads(
            p.cli("note", "--body-file", "-", stdin="実mergeを確認した。").stdout
        )
        spec.update(schema_version=2, evidence=[evidence["url"]])
    else:
        p.announce_completion()
    response = json.loads(
        p.cli("release", "--ended", "--spec", "-", stdin=json.dumps(spec)).stdout
    )
    assert (
        response["assignment"] == "ended"
        and response["completion"]["acceptance"] == "completed"
    )
    count = len(p.state()["comments"]["1"])
    response = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert (
        response["assignment"] == "ended" and len(p.state()["comments"]["1"]) == count
    )
    assert requirement_body(p.state()["issues"]["1"]["body"]) == before
    assert set(response["tasks"]) == {"1"} and p.state()["parents"] == {}
    assert len(g.merge_calls()) == len(g.review.calls()) == 1
    assert "次回改善: なし。" in "\n".join(c["body"] for c in p.comments())
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    public = navigation_snapshot(directory)
    assert "最後の親の受入（明示記録時点）: completed" in public
    assert "担当終了" in public and "#issuecomment-" in public


def test_extracted_candidate_groups_complete_through_structured_finish(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    extracted_body = []

    def agree_extracted() -> None:
        p.cli("register", parent="2")
        p.cli(
            "agree",
            "--spec",
            p.spec(
                {
                    "purpose": "元の成果を実装する。",
                    "scope": "元の親の範囲。",
                    "behavior": "合意後に実装する。",
                    "acceptance": "fakeで検証する。",
                    "human_confirmation": "この範囲で進めてください。",
                }
            ),
            parent="2",
        )
        p.env["PARENT_CREATED_AT"] = "2099-01-01T00:00:00Z"
        validate_spec = importlib.import_module("issues.improvement").validate_spec
        for key in ("first", "second"):
            candidate = candidate_spec()
            candidate["acceptance"][0] = "保存できる。\n  再読込して内容が一致する。"
            validate_spec(candidate)
            p.cli(
                "improvement",
                "--key",
                key,
                "--spec",
                p.spec(candidate),
                parent="2",
            )
        p.cli("improvement-propose", parent="2")
        decision = decision_spec()
        del decision["revision"], decision["previous_decision"]
        decision["human_confirmation"]["source"] = (
            "人間との会話で元親 #2 の2候補を確認。"
        )
        decision["candidates"] = {
            key: {"revision": 1, "previous_decision": None}
            for key in ("I-2-first", "I-2-second")
        }
        approved = json.loads(
            p.cli(
                "improvement-decide",
                "--candidate",
                "I-2-first",
                "--candidate",
                "I-2-second",
                "--action",
                "approve",
                "--spec",
                p.spec(decision),
                parent="2",
            ).stdout
        )["decisions"][0]["decision_id"]
        # The shared merge fixture operates on #1. Let the real extraction CLI
        # create that issue, then restore the fake server's normal allocation.
        state = p.state()
        next_number = state["next"]
        del state["issues"]["1"]
        state["next"] = 1
        p.save(state)
        extracted = json.loads(
            p.cli(
                "improvement-extract",
                "--candidates",
                "I-2-first",
                "I-2-second",
                parent="2",
            ).stdout
        )
        assert extracted["issue"] == 1
        state = p.state()
        extracted_body.append(state["issues"]["1"]["body"])
        assert "- [ ] 保存できる。\n  再読込して内容が一致する。" in extracted_body[0]
        state["next"] = next_number
        p.save(state)
        p.cli("release", "--ended", "--note-file", p.note(), parent="2")
        p.cli("register")
        p.cli("agree", "--improvement-approval", approved)
        required = json.loads(p.cli("status").stdout)["completion"]["requirements"]
        assert required["available"], required
        assert [row["id"] for row in required["criteria"]] == [
            "agreement",
            "ac-I-2-first-1",
            "ac-I-2-first-2",
            "ac-I-2-second-1",
            "ac-I-2-second-2",
        ]
        assert [row["criterion"] for row in required["criteria"][1:]] == [
            "[I-2-first] 保存できる。 再読込して内容が一致する。",
            "[I-2-first] 取得失敗を表示し、保存済み結果を保持する。",
            "[I-2-second] 保存できる。 再読込して内容が一致する。",
            "[I-2-second] 取得失敗を表示し、保存済み結果を保持する。",
        ]

    monkeypatch.setattr(p, "agreed", agree_extracted)
    g = make_gate_env(p, monkeypatch, "parent_self")
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    p.announce_completion()
    spec = finish_for(p)
    p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 2,
                "state": "completed",
                "finish": spec,
                "reason": "取り出した全候補の受入を確認した。",
                "resume_condition": "完了。",
                "evidence": "PR #100 の実merge・closeと各候補の検証を確認した。",
            }
        ),
    )
    result = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert result["completion"]["acceptance"] == "completed"
    assert result["completion"]["record"]["state"] == "completed"
    assert result["assignment"] == "ended"
    snapshot = importlib.import_module("parent.completion").snapshot
    assert snapshot(p.state()["issues"]["1"]["body"]) == snapshot(extracted_body[0])


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
    p.cli("improvement-propose", ok=False)


@pytest.mark.parametrize("gate_env", ["parent_self"], indirect=True)
def test_recorded_completion_and_failed_proposal_publication_are_independent(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    p.cli("improvement", "--key", "next", "--spec", p.spec(candidate_spec()))
    p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    p.cli("improvement-propose", ok=False)
    del p.env["PARENT_FAIL_AFTER_CREATE"]
    p.announce_completion()
    spec = finish_for(p, candidates=["I-1-next"])
    status = json.loads(p.cli("release", "--ended", "--spec", p.spec(spec)).stdout)
    assert (
        status["completion"]["acceptance"] == "completed"
        and status["improvements"][0]["publication"] == "outcome_unknown"
    )
    assert status["unpublished_improvements"] == ["I-1-next"]
    p.cli("improvement-propose")
    status = json.loads(p.cli("status").stdout)
    assert (
        status["assignment"] == "ended"
        and status["completion"]["acceptance"] == "completed"
    )
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


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
