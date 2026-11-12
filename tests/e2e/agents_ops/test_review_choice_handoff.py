"""Parent CLI choices retain provenance through publication, recovery and gate."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ReviewEnv
from tests.e2e.agents_ops.test_parent_session import pair_review_env as pair_review_env
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_inspection import inspect
from tests.e2e.agents_ops.test_review_recovery import (
    gate_for,
    recover,
    recovery_request,
)

CONFIG = Path(__file__).resolve().parents[3] / ".agents/ops/lib/agents.conf"


def select_config(r: ReviewEnv) -> Path:
    config = Path(r.p.root) / "select.conf"
    config.write_text(
        CONFIG.read_text().replace(
            'REVIEW_SELECTION_MODE="fixed"', 'REVIEW_SELECTION_MODE="select"'
        )
    )
    return config


def request(r: ReviewEnv, *flags: str) -> ReviewEnv:
    manifest = json.loads(
        r.p.cli(
            "review-prepare",
            "--prs",
            "100",
            "--reason-file",
            r.p.note("親が必要な追加確認を判断した。"),
            *flags,
        ).stdout
    )
    instructed = json.loads(
        r.p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            r.p.note(
                "固定した受入条件とPR単位のcoverageを確認する。部分確認はnot_reviewedとする。"
            ),
        ).stdout
    )
    path = r.p.root / (manifest["review_run_id"] + ".json")
    path.write_text(json.dumps(instructed))
    return replace(r, request=instructed, request_path=path)


def run(r: ReviewEnv, config: Path, *, ok: bool = True, **extra: str) -> Any:
    return r.cli(
        "run",
        "--request",
        str(r.request_path),
        "--agent-exec",
        str(r.fake),
        "--review-config-file",
        str(config),
        ok=ok,
        extra=extra,
    )


@pytest.mark.parametrize("human", [False, True])
def test_parent_choice_is_frozen_published_inspected_and_accepted_by_gate(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, human: bool
) -> None:
    r = single_review_env
    config = select_config(r)
    model = "gpt-6-astra" if human else "gpt-6.1-sol"
    flags = [
        "--review-config-file",
        str(config),
        "--review-model",
        model,
        "--model-reason-file",
        r.p.note("人間の指定を優先" if human else "定型的な確認"),
    ]
    if human:
        flags += [
            "--review-model-source",
            "human_instruction",
            "--human-model-reference",
            "親が人間の元指示を確認済み",
        ]
    chosen = request(r, *flags)
    result = json.loads(run(chosen, config).stdout)["result"]
    choice = result["request"]["review_selection"]
    argv = chosen.calls()[-1]["args"]
    assert (
        json.loads(Path(argv[argv.index("--review-selection-file") + 1]).read_text())
        == choice
    )
    assert Path(argv[argv.index("--config-file") + 1]) == config.resolve()
    assert result["request"]["schema_version"] == 3
    assert choice["specified_by"] == ("human_instruction" if human else "parent_choice")
    assert result["runtime_settings"]["effective"]["model"]["value"] == model
    execution = json.loads((chosen.run_dir / "execution.json").read_text())
    assert execution["review_selection"] == choice
    state_dir = chosen.run_dir.parents[1]
    before = {
        str(p.relative_to(state_dir)): p.read_bytes()
        for p in state_dir.rglob("*")
        if p.is_file()
    }
    response, view = inspect(chosen, 100)
    assert response.returncode == 0, response.stdout
    latest = view["prs"][0]["history"][-1]
    assert latest["review_selection"] == choice
    assert latest["runtime_settings"]["effective"]["model"]["value"] == model
    assert before == {
        str(p.relative_to(state_dir)): p.read_bytes()
        for p in state_dir.rglob("*")
        if p.is_file()
    }
    g = gate_for(chosen, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0
    capture = importlib.import_module("decision_logs.review_capture")
    operation = capture.previous_run(
        state_dir, "o/r", 1, chosen.request["review_run_id"]
    )
    assert (
        operation is not None
        and operation.row["model_specified_by"] == choice["specified_by"]
    )
    observations = importlib.import_module("decision_logs.observations")
    runtime = json.loads((chosen.run_dir / "agent-runtime/result.json").read_text())
    model_observation = observations.runtime_auxiliary(
        runtime, operation.row["counter_key"], operation.row["model_specified_by"]
    )["selected_model"]
    assert model_observation["value"]["model"] == model
    assert model_observation["value"]["specified_by"] == choice["specified_by"]


def test_fixed_and_post_prepare_changes_are_rejected_without_an_agent(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    denied = r.p.cli(
        "review-prepare",
        "--prs",
        "100",
        "--review-model",
        "gpt-6.1-sol",
        "--model-reason-file",
        r.p.note("通常の確認"),
        ok=False,
    )
    assert "fixed" in denied.stderr
    denied = r.p.cli(
        "review-request",
        "--run-id",
        r.request["review_run_id"],
        "--instruction-file",
        r.p.note("再送"),
        "--review-config-file",
        str(select_config(r)),
        "--review-model",
        "gpt-6.1-sol",
        "--model-reason-file",
        r.p.note("新しい選択"),
        ok=False,
    )
    assert "frozen" in denied.stderr
    assert r.calls() == []


def test_sol_recovery_records_source_model_under_current_fixed_policy(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    config = select_config(r)
    chosen = request(
        r,
        "--review-config-file",
        str(config),
        "--review-model",
        "gpt-6.1-sol",
        "--model-reason-file",
        r.p.note("定型確認"),
    )
    run(chosen, config, ok=False, REVIEW_AGENT_FINAL_EXIT="3")
    count = len(chosen.calls())
    adopted = recovery_request(chosen)
    assert adopted.request["review_selection"]["kind"] == "recover"
    result = json.loads(recover(chosen, adopted).stdout)["result"]
    assert result["runtime_settings"]["selected"]["model"] == "gpt-6.1-sol"
    assert result["runtime_settings"]["effective"]["model"]["value"] == "gpt-6.1-sol"
    assert len(chosen.calls()) == count
    assert "cannot launch" in adopted.run(ok=False).stderr
    g = gate_for(adopted, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0


def test_handoff_records_source_but_partial_review_stays_blocked(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    config = select_config(r)
    sol = request(
        r,
        "--review-config-file",
        str(config),
        "--review-model",
        "gpt-6.1-sol",
        "--model-reason-file",
        r.p.note("定型確認"),
    )
    run(sol, config)
    astra = request(
        sol,
        "--review-config-file",
        str(config),
        "--review-model",
        "gpt-6-astra",
        "--model-reason-file",
        r.p.note("未解決の統合境界"),
        "--handoff-from-run",
        sol.request["review_run_id"],
        "--handoff-scope-file",
        r.p.note("残る並行処理の疑問を確認"),
    )
    output = json.loads(astra.output.read_text())
    output["verdict"] = "incomplete"
    output["prs"][0]["coverage"] = "not_reviewed"
    astra.output.write_text(json.dumps(output))
    result = json.loads(run(astra, config, ok=False).stdout)["result"]
    assert result["exception"]["reason"] == astra.request["reason"]
    assert (
        result["request"]["review_selection"]["handoff"]["source_run"]
        == sol.request["review_run_id"]
    )
    assert (
        astra.calls()[-1]["context"]["review_handoff"]["source_result"][
            "runtime_settings"
        ]["selected"]["model"]
        == "gpt-6.1-sol"
    )
    assert len(astra.calls()) == 2
    g = gate_for(astra, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode != 0


def test_all_consumers_reject_selected_model_changed_in_saved_runtime(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    r.run()
    g = gate_for(r, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0
    path = r.run_dir / "agent-runtime/result.json"
    runtime = json.loads(path.read_text())
    runtime["contract"]["settings"]["selected"]["model"] = "gpt-6.1-sol"
    runtime["runtime"]["effective"]["model"]["value"] = "gpt-6.1-sol"
    path.write_text(json.dumps(runtime))
    calls = len(r.calls())
    rejected = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert rejected.returncode == 2 and "selection" in rejected.stderr
    response, view = inspect(r, 100)
    assert response.returncode == 2
    assert any(
        error["scope"] == "effective_settings"
        for error in view["prs"][0]["history"][-1]["errors"]
    )
    assert (
        r.cli(
            "resume-publication", "--run", r.request["review_run_id"], ok=False
        ).returncode
        != 0
    )
    adopted = recovery_request(r)
    assert recover(r, adopted, ok=False).returncode != 0
    assert len(r.calls()) == calls


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_request_is_not_rewritten_or_given_an_invented_choice(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    r = single_review_env
    policy = importlib_policy(monkeypatch)
    original = {k: v for k, v in r.request.items() if k != "review_state"}
    manifest = copy.deepcopy(original)
    del manifest["review_selection"]
    manifest["schema_version"] = version
    if version == 1:
        del manifest["authorization"], manifest["requirements"]
    old_digest, new_digest = policy.digest(original), policy.digest(manifest)
    registry_path = r.run_dir.parents[1] / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry["parents"]["o/r#1"]["reviews"][0]["request"] = manifest
    registry_path.write_text(json.dumps(registry))
    state = r.p.state()
    for rows in state["comments"].values():
        for comment in rows:
            comment["body"] = comment["body"].replace(old_digest, new_digest)
    r.p.save(state)
    r.request.clear()
    r.request.update(manifest, review_state="instructed")
    r.request_path.write_text(json.dumps(r.request))
    path = r.run_dir / "request.json"
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    result = json.loads(r.run().stdout)["result"]
    assert result["request_sha256"] == new_digest
    assert (
        "runtime_settings" not in result and "review_selection" not in result["request"]
    )
    response, view = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    assert view["prs"][0]["history"][0]["review_selection"] == {
        "status": "legacy_unrecorded"
    }
    assert path.read_bytes() == before
    adopted = recovery_request(r)
    recovered = json.loads(recover(r, adopted).stdout)["result"]
    assert recovered["runtime_settings"]["selected"]["model"] == "review-fixture"
    assert recovered["runtime_settings"]["selected"]["effort"] == "max"
    assert path.read_bytes() == before and len(r.calls()) == 1
    g = gate_for(adopted, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0


def importlib_policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests.e2e.agents_ops.test_parent_session import review_policy_module

    return review_policy_module(monkeypatch)


@pytest.mark.parametrize("changed", ["model", "effort", "digest"])
def test_partial_review_requires_runtime_proof_at_gate_and_publication(
    pair_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    r = pair_review_env
    output = json.loads(r.output.read_text())
    output["verdict"] = "incomplete"
    output["prs"][1]["coverage"] = "not_reviewed"
    output["prs"][0]["evidence_waits"] = [
        {"question": "独立したPRの確認", "required_evidence": "同HEADのテスト結果"}
    ]
    r.output.write_text(json.dumps(output))
    result = json.loads(r.run(ok=False).stdout)["result"]
    assert result["status"] == "failed"
    g = gate_for(r, monkeypatch)
    g.confirm()
    wait_proof = r.p.root / "partial-evidence.json"
    wait_proof.write_text(
        json.dumps(
            {
                "ci": "同HEADのPython成功",
                "regression_tests": "対象テスト成功",
                "resolution": "親が要求された証拠を確認",
                "reviewed_paths": ["change.txt"],
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
        str(wait_proof),
    )
    assert g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"] == "merge"
    proof = r.p.root / "partial-review-confirmation.json"
    proof.write_text(
        json.dumps(
            {
                "ci": "同HEADのPython成功",
                "regression_tests": "reviewed側PRの対象テスト成功",
                "resolution": "親が独立したreviewed側PRの範囲を確認",
                "reviewed_paths": [],
            }
        )
    )
    r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "changes",
        "--fixed-head",
        r.request["prs"][0]["head_sha"],
        "--evidence",
        str(proof),
    )
    path = r.run_dir / "agent-runtime/result.json"
    runtime = json.loads(path.read_text())
    if changed == "model":
        runtime["runtime"]["effective"]["model"]["value"] = "gpt-6.1-sol"
    elif changed == "effort":
        runtime["runtime"]["effective"]["effort"]["value"] = "max"
    else:
        runtime["finished_at"] += 1
    path.write_text(json.dumps(runtime))
    count = len(r.calls())
    github = copy.deepcopy(r.p.state())
    denied = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert denied.returncode == 2, denied.stdout + denied.stderr
    assert (
        r.cli(
            "resume-publication", "--run", r.request["review_run_id"], ok=False
        ).returncode
        == 2
    )
    assert len(r.calls()) == count and r.p.state() == github
    assert (
        r.cli(
            "complete-evidence",
            "--run",
            r.request["review_run_id"],
            "--pr",
            "100",
            "--wait",
            "1",
            "--evidence",
            str(wait_proof),
            ok=False,
        ).returncode
        == 2
    )
    assert (
        r.cli(
            "resolve",
            "--run",
            r.request["review_run_id"],
            "--pr",
            "100",
            "--finding",
            "changes",
            "--fixed-head",
            r.request["prs"][0]["head_sha"],
            "--evidence",
            str(proof),
            ok=False,
        ).returncode
        == 2
    )
    adopted = recovery_request(r)
    assert recover(r, adopted, ok=False).returncode == 2
    assert len(r.calls()) == count


def save_failed_candidate(r: ReviewEnv, failure: str) -> dict[str, Any]:
    previous_pr = copy.deepcopy(r.p.state()["prs"][0])
    output = json.loads(r.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "候補の保存",
            "detail": "静的確認: 保存候補の指摘を引継ぎから落とさない。",
        }
    ]
    output["prs"][0]["evidence_waits"] = [
        {"question": "境界の動作", "required_evidence": "対象テストの結果"}
    ]
    r.output.write_text(json.dumps(output))
    result = json.loads(
        r.run(
            ok=False,
            extra={"REVIEW_AGENT_FINAL_EXIT": "3", "REVIEW_FINAL_ONLY": "1"}
            if failure == "runtime"
            else {"REVIEW_POSTCHECK_FAIL": "1"},
        ).stdout
    )["result"]
    assert result["status"] == "failed" and result["prs"][0]["findings"] == []
    state = r.p.state()
    if "statusCheckRollup" in previous_pr:
        state["prs"][0]["statusCheckRollup"] = previous_pr["statusCheckRollup"]
    else:
        state["prs"][0].pop("statusCheckRollup", None)
    r.p.save(state)
    saved: dict[str, Any] = json.loads((r.run_dir / "output.json").read_text())
    return saved


def handoff(r: ReviewEnv) -> ReviewEnv:
    return request(
        r,
        "--handoff-from-run",
        r.request["review_run_id"],
        "--handoff-scope-file",
        r.p.note("保存候補の指摘と証拠待ちを含めて再確認する。"),
    )


@pytest.mark.parametrize("failure", ["runtime", "postcheck"])
def test_handoff_freezes_failed_candidate_and_blocks_unadopted_findings(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    r = single_review_env
    saved = save_failed_candidate(r, failure)
    following = handoff(r)
    records = following.request["review_selection"]["handoff"]["candidates"]
    assert len(records) == 1 and records[0]["status"] == "candidate_only"
    candidate = records[0]
    assert candidate["source_run"] == r.request["review_run_id"]
    assert candidate["prs"][0]["findings"] == saved["prs"][0]["findings"]
    assert candidate["prs"][0]["evidence_waits"] == saved["prs"][0]["evidence_waits"]
    policy = importlib_policy(monkeypatch)
    assert candidate["validation"]["output_sha256"] == policy.digest(saved)
    assert all("coverage" not in pr for pr in candidate["prs"])
    output = json.loads(following.output.read_text())
    output["verdict"] = "approve"
    output["prs"][0].update(findings=[], evidence_waits=[])
    following.output.write_text(json.dumps(output))
    response = (
        following.run(ok=False, extra={"REVIEW_AGENT_FINAL_EXIT": "3"})
        if failure == "runtime"
        else following.run()
    )
    result = json.loads(response.stdout)["result"]
    assert following.calls()[-1]["context"]["review_handoff"]["candidates"] == records
    assert "candidate_only" in (following.run_dir / "prompt.md").read_text()
    if failure == "runtime":
        adopted = recovery_request(following)
        result = json.loads(recover(following, adopted).stdout)["result"]
        following = adopted
    assert result["prs"][0]["findings"] == []
    g = gate_for(following, monkeypatch)
    g.confirm()
    blocked = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]
    assert blocked["decision"] == "blocked"
    assert any("handoff_candidate" in reason for reason in blocked["reasons"])
    if failure == "runtime":
        assert following.request["review_selection"]["candidates"] == records
    proof = following.p.root / "candidate-confirmation.json"
    proof.write_text(
        json.dumps(
            {
                "ci": "同HEADのPython成功を親が確認",
                "regression_tests": "候補の指摘と証拠待ちに対応するテストが成功",
                "resolution": "親が固定された全候補の指摘と要求された証拠を確認した。",
                "reviewed_paths": [],
            }
        )
    )
    following.cli(
        "resolve",
        "--run",
        following.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "handoff-candidates",
        "--fixed-head",
        following.request["prs"][0]["head_sha"],
        "--evidence",
        str(proof),
    )
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0


def test_handoff_without_saved_candidate_does_not_invent_approval(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.output.write_text("{invalid final")
    r.run(ok=False)
    assert not (r.run_dir / "output.json").exists()
    following = handoff(r)
    assert following.request["review_selection"]["handoff"]["candidates"] == []
    assert not (following.run_dir / "result.json").exists()


def test_handoff_rejects_invalid_candidate_before_publishing_targets(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    saved = save_failed_candidate(r, "runtime")
    following = handoff(r)
    saved["prs"][0]["findings"][0]["detail"] = "保存finalに存在しない指摘"
    (r.run_dir / "output.json").write_text(json.dumps(saved))
    before = copy.deepcopy(r.p.state())
    denied = r.p.cli(
        "review-prepare",
        "--prs",
        "100",
        "--reason-file",
        r.p.note(),
        "--handoff-from-run",
        r.request["review_run_id"],
        "--handoff-scope-file",
        r.p.note("追加確認"),
        ok=False,
    )
    assert "saved output contradicts final" in denied.stderr
    assert following.run(ok=False).returncode == 2
    assert r.p.state() == before and len(r.calls()) == 1


def test_repeated_handoff_keeps_prior_candidate_when_next_run_has_no_candidate(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    save_failed_candidate(r, "runtime")
    following = handoff(r)
    frozen = following.request["review_selection"]["handoff"]["candidates"]
    following.run(ok=False, extra={"REVIEW_EFFECTIVE_UNAVAILABLE": "1"})
    assert not (following.run_dir / "output.json").exists()
    latest = handoff(following)
    assert latest.request["review_selection"]["handoff"]["candidates"] == frozen


def test_inspection_displays_failed_setting_observation_without_promoting_it(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    r.run(
        ok=False,
        extra={
            "REVIEW_AGENT_FINAL_EXIT": "7",
            "REVIEW_RUNTIME_REASON": "cli_error",
            "REVIEW_EFFECTIVE_UNAVAILABLE": "1",
        },
    )
    response, view = inspect(r, 100)
    assert response.returncode == 0, response.stdout
    latest = view["prs"][0]["history"][-1]
    assert latest["settings_check"] == {"status": "unconfirmed"}
    assert latest["runtime_settings"]["effective"]["model"]["status"] == "unavailable"
    assert latest["result"]["status"] == "failed"
    g = gate_for(r, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode != 0
    assert len(r.calls()) == 1


def test_autonomous_parent_uses_the_same_review_choice_path(
    parent_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_execution_basis import autonomous
    from tests.e2e.agents_ops.test_parent_session import make_review_env

    monkeypatch.setattr(parent_env, "agreed", lambda: autonomous(parent_env))
    r = make_review_env(parent_env, pr_count=1)
    config = select_config(r)
    chosen = request(
        r,
        "--review-config-file",
        str(config),
        "--review-model",
        "gpt-6.1-sol",
        "--model-reason-file",
        r.p.note("通常の確認"),
    )
    assert (
        chosen.request["authorization"]["execution_basis_kind"] == "autonomous_decision"
    )
    result = json.loads(run(chosen, config).stdout)["result"]
    assert result["request"]["review_selection"]["specified_by"] == "parent_choice"
    assert result["runtime_settings"]["effective"]["model"]["value"] == "gpt-6.1-sol"
    g = gate_for(chosen, monkeypatch)
    g.confirm()
    assert g.cli("merge", AGENT_AUTO_MERGE="0").returncode == 0
