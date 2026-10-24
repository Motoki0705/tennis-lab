"""Fail-closed contracts for the parent-run consumer and the future merge gate."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS / "review"))
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("parent_review_policy")


def request() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "review_run_id": "parent-1-" + "a" * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "reason": "親が1回の統合レビューを指示",
        "prs": [
            {
                "number": n,
                "head_sha": f"{n:040x}",
                "base_sha": "b" * 40,
                "base_ref": "main",
                "merge_base_sha": "b" * 40,
                "diff_sha256": f"{n:064x}",
                "branch": f"codex/child-{n}",
            }
            for n in (100, 101, 102)
        ],
    }


def output() -> dict[str, Any]:
    return {
        "verdict": "request_changes",
        "summary": "横断と個別の指摘あり",
        "cross_findings": [],
        "prs": [
            {
                "number": n,
                "coverage": "reviewed",
                "summary": "確認済み",
                "findings": [
                    {
                        "severity": "major",
                        "path": "a.py",
                        "line": 1,
                        "rule": "正しさ",
                        "detail": "境界条件",
                    }
                ]
                if n == 100
                else [],
            }
            for n in (100, 101, 102)
        ],
    }


def result(policy: Any) -> dict[str, Any]:
    manifest = request()
    parsed = policy.parse_output(json.dumps(output()), [100, 101, 102])
    return {
        "schema_version": 1,
        "kind": "result",
        "review_run_id": manifest["review_run_id"],
        "request_sha256": policy.digest(manifest),
        "request": manifest,
        "reviewer": "codex",
        "profile": "review",
        "started_at": "2026-10-18T00:00:00+00:00",
        "finished_at": "2026-10-18T00:01:00+00:00",
        "status": "succeeded",
        "error": None,
        "exception": None,
        **parsed,
        "prs": [{**p, "child_issues": [10 + i]} for i, p in enumerate(parsed["prs"])],
    }


def resolution(
    policy: Any, run: dict[str, Any], latest: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "resolution",
        "review_run_id": run["review_run_id"],
        "result_sha256": policy.digest(run),
        "parent_issue": 1,
        "pr": 100,
        "finding": "pr-100-1",
        "reviewed_head": run["request"]["prs"][0]["head_sha"],
        "fixed_snapshot": latest,
        "fix_diff_sha256": "f" * 64,
        "changed_paths": ["a.py"],
        "evidence": {
            "ci": "全check成功 headと一致",
            "regression_tests": "境界テスト成功",
            "resolution": "不具合解消",
            "reviewed_paths": ["a.py"],
        },
    }


@pytest.mark.parametrize(
    "case",
    [
        "prepared",
        "unknown_key",
        "duplicate",
        "wrong_run",
        "short_sha",
        "bad_diff",
        "wrong_state",
        "unordered",
    ],
)
def test_request_rejects_unknown_or_uninstructed_inputs(policy: Any, case: str) -> None:
    data = {**request(), "review_state": "instructed"}
    if case == "prepared":
        del data["review_state"]
    elif case == "unknown_key":
        data["automatic"] = True
    elif case == "duplicate":
        data["prs"].append(data["prs"][0])
    elif case == "wrong_run":
        data["review_run_id"] = "../../escape"
    elif case == "short_sha":
        data["prs"][0]["head_sha"] = "a"
    elif case == "bad_diff":
        data["prs"][0]["diff_sha256"] = "x" * 64
    elif case == "wrong_state":
        data["review_state"] = "targets_fixed"
    else:
        data["prs"].reverse()
    with pytest.raises(ValueError):
        policy.request_manifest(data)


@pytest.mark.parametrize(
    "case",
    [
        "minor",
        "nit",
        "missing_pr",
        "duplicate_pr",
        "foreign_cross",
        "extra_key",
        "contradiction",
        "coverage",
        "absolute_path",
        "parent_path",
        "line_bool",
    ],
)
def test_integrated_output_rejects_partial_or_invalid_schema(
    policy: Any, case: str
) -> None:
    data = output()
    finding = data["prs"][0]["findings"][0]
    if case in {"minor", "nit"}:
        finding["severity"] = case
    elif case == "missing_pr":
        data["prs"].pop()
    elif case == "duplicate_pr":
        data["prs"].append(data["prs"][0])
    elif case == "foreign_cross":
        data["cross_findings"] = [{**finding, "prs": [999]}]
    elif case == "extra_key":
        data["agent"] = "codex"
    elif case == "contradiction":
        data["verdict"] = "approve"
    elif case == "coverage":
        data["prs"][1]["coverage"] = "maybe"
    elif case == "absolute_path":
        finding["path"] = "/outside/a.py"
    elif case == "parent_path":
        finding["path"] = "../a.py"
    else:
        finding["line"] = True
    with pytest.raises(ValueError):
        policy.parse_output(json.dumps(data), [100, 101, 102])


def test_partial_output_is_explicit_and_cannot_cover(policy: Any) -> None:
    data = output()
    data["prs"][1]["coverage"] = "not_reviewed"
    data["verdict"] = "incomplete"
    assert (
        policy.parse_output("```json\n" + json.dumps(data) + "\n```", [100, 101, 102])[
            "verdict"
        ]
        == "incomplete"
    )
    run = result(policy)
    run["prs"][1]["coverage"] = "not_reviewed"
    run.update(status="failed", verdict="incomplete", error="context不足")
    coverage = policy.evaluate_coverage(
        run, request()["prs"], [], {}, fix_diff_digests={}
    )
    assert not coverage.covered
    assert "run_failed_or_incomplete" not in coverage.reasons
    assert "pr_101_not_reviewed" in coverage.reasons


def test_only_first_line_and_publisher_records_are_trusted(policy: Any) -> None:
    run = result(policy)
    line = policy.marker(run)
    comments = [
        {"user": {"login": login}, "body": body}
        for login, body in [
            ("attacker", line),
            ("bot", "quoted\n" + line),
            ("bot", "see " + line),
            ("bot", "<!-- agent-cross-review:v4 stale -->"),
            ("bot", line + "\nsummary\n" + line),
        ]
    ]
    assert policy.records(comments, "bot") == [run]
    for bad in ("<!-- parent-review:v1 broken -->", "<!-- parent-review:v1 {} -->"):
        with pytest.raises(ValueError):
            policy.records([{"user": {"login": "bot"}, "body": bad}], "bot")


@pytest.mark.parametrize(
    "case",
    [
        "digest",
        "profile",
        "reviewer",
        "time",
        "finding_id",
        "pr_set",
        "extra_key",
        "status",
    ],
)
def test_malformed_results_are_not_gate_evidence(policy: Any, case: str) -> None:
    run = result(policy)
    if case in {"digest", "profile", "reviewer"}:
        run[
            {"digest": "request_sha256", "profile": "profile", "reviewer": "reviewer"}[
                case
            ]
        ] = "wrong"
    elif case == "time":
        run["finished_at"] = "2026-10-17T00:00:00+00:00"
    elif case == "finding_id":
        run["prs"][0]["findings"][0]["id"] = "pr-100-999"
    elif case == "pr_set":
        run["prs"].pop()
    elif case == "extra_key":
        run["approved"] = True
    else:
        run["status"] = "failed"
    with pytest.raises(ValueError):
        policy.records([{"user": {"login": "bot"}, "body": policy.marker(run)}], "bot")


@pytest.mark.parametrize(
    "case",
    [
        "head",
        "base_sha",
        "base_ref",
        "merge_base_sha",
        "branch",
        "added",
        "removed",
        "all_removed",
    ],
)
def test_out_of_scope_snapshot_changes_invalidate_coverage(
    policy: Any, case: str
) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    if case in {"head", "base_sha", "base_ref", "merge_base_sha", "branch"}:
        current[0]["head_sha" if case == "head" else case] = (
            "d" * 40 if "sha" in case or case == "head" else "other"
        )
    elif case == "added":
        current.append({**current[0], "number": 103})
    elif case == "removed":
        current.pop()
    else:
        current = []
    coverage = policy.evaluate_coverage(run, current, [], {}, fix_diff_digests={})
    assert coverage.covered == (case == "removed")
    assert bool(coverage.reasons) == (case != "removed")


def test_coverage_and_unresolved_findings_are_distinct(policy: Any) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    before = policy.evaluate_coverage(run, current, [], {}, fix_diff_digests={})
    assert before.covered and before.unresolved == ((100, "pr-100-1"),)
    current[0].update(head_sha="d" * 40, diff_sha256="e" * 64)
    proof = resolution(policy, run, current[0])
    after = policy.evaluate_coverage(
        run, current, [proof], {100: ["a.py"]}, fix_diff_digests={100: "f" * 64}
    )
    assert after.covered and not after.unresolved
    assert policy.validate_resolution(proof) == proof
    assert policy.records(
        [{"user": {"login": "bot"}, "body": policy.marker(proof)}], "bot"
    ) == [proof]


@pytest.mark.parametrize(
    "case",
    [
        "outside_path",
        "missing_paths",
        "stale_head",
        "old_result",
        "old_run",
        "unknown_finding",
        "changed_base",
        "missing_diff_digest",
        "wrong_diff_digest",
    ],
)
def test_fix_evidence_does_not_cover_unexpected_changes(policy: Any, case: str) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    current[0].update(head_sha="d" * 40, diff_sha256="e" * 64)
    paths = {100: ["a.py"]}
    proof = resolution(policy, run, copy.deepcopy(current[0]))
    digests = {100: "f" * 64}
    if case == "outside_path":
        paths[100].append("unreviewed.py")
    elif case == "missing_paths":
        paths = {}
    elif case == "stale_head":
        proof["fixed_snapshot"]["head_sha"] = "f" * 40
    elif case == "old_result":
        proof["result_sha256"] = "f" * 64
    elif case == "old_run":
        proof["review_run_id"] = "parent-1-" + "f" * 32
    elif case == "unknown_finding":
        proof["finding"] = "pr-100-999"
    elif case == "changed_base":
        current[0]["base_sha"] = "f" * 40
        proof["fixed_snapshot"] = current[0]
    elif case == "missing_diff_digest":
        digests = {}
    elif case == "wrong_diff_digest":
        digests[100] = "e" * 64
    assert policy.evaluate_coverage(
        run, current, [proof], paths, fix_diff_digests=digests
    ).covered == (case == "changed_base")


def test_duplicate_json_keys_never_hide_findings(policy: Any) -> None:
    data = json.dumps(output())
    duplicate = '{"verdict":"approve",' + data[1:]
    with pytest.raises(ValueError, match="duplicate JSON"):
        policy.parse_output(duplicate, [100, 101, 102])


def test_error_reports_do_not_publish_private_session_or_checkout_paths(
    policy: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(OPS / "lib"))
    runner = importlib.import_module("parent_review")
    session = SimpleNamespace(
        store=SimpleNamespace(
            directory=tmp_path / "private-state",
            parents={
                "o/r#1": {
                    "session": "claude:secret-session",
                    "tasks": {"10": {"worktree": str(tmp_path / "worker")}},
                }
            },
        )
    )
    message = runner.public_error(
        session,
        ValueError(
            f"failed in {Path.cwd()}/.claude/worktrees/review-run: secret-session, {tmp_path / 'private-state'}, {tmp_path / 'worker'}"
        ),
    )
    assert "secret-session" not in message
    assert str(Path.cwd()) not in message
    assert str(tmp_path) not in message


def test_resolution_for_another_parent_cannot_cover_this_run(policy: Any) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    current[0].update(head_sha="d" * 40, diff_sha256="e" * 64)
    proof = resolution(policy, run, current[0])
    proof["parent_issue"] = 2
    assert not policy.evaluate_coverage(
        run, current, [proof], {100: ["a.py"]}, fix_diff_digests={100: "f" * 64}
    ).covered


@pytest.mark.parametrize("finding_path", ["a.py", None])
def test_parent_confirmed_regression_paths_are_covered(
    policy: Any, finding_path: str | None
) -> None:
    run = result(policy)
    run["prs"][0]["findings"][0]["path"] = finding_path
    current = copy.deepcopy(run["request"]["prs"])
    current[0].update(head_sha="d" * 40, diff_sha256="e" * 64)
    proof = resolution(policy, run, current[0])
    paths = ["a.py", "tests/test_a.py"]
    proof["changed_paths"] = paths
    proof["evidence"].update(
        reviewed_paths=paths,
        resolution="a.pyの境界条件修正とtests/test_a.pyの再現ケースを確認",
    )
    coverage = policy.evaluate_coverage(
        run, current, [proof], {100: paths}, fix_diff_digests={100: "f" * 64}
    )
    assert coverage.covered and not coverage.unresolved
    # Even an otherwise exact record cannot approve an omitted path.
    proof["evidence"]["reviewed_paths"] = ["a.py"]
    with pytest.raises(ValueError, match="parent's reviewed paths"):
        policy.validate_resolution(proof)


@pytest.mark.parametrize("bad_paths", [None, "a.py", ["../a.py"], ["a.py", "a.py"]])
def test_resolution_requires_explicit_valid_reviewed_paths(
    policy: Any, bad_paths: Any
) -> None:
    run = result(policy)
    proof = resolution(policy, run, run["request"]["prs"][0])
    proof["evidence"]["reviewed_paths"] = bad_paths
    with pytest.raises(ValueError):
        policy.validate_resolution(proof)
    del proof["evidence"]["reviewed_paths"]
    with pytest.raises(ValueError):
        policy.validate_resolution(proof)


@pytest.mark.parametrize("fixed", [False, True])
def test_peer_merge_preserves_reviewed_or_resolved_coverage(
    policy: Any, fixed: bool
) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    proofs, paths, digests = [], {}, {}
    if fixed:
        current[0].update(head_sha="d" * 40, diff_sha256="e" * 64)
        proofs = [resolution(policy, run, copy.deepcopy(current[0]))]
        paths, digests = {100: ["a.py"]}, {100: "f" * 64}
    # Only PR 101 was merged; retain its verified snapshot in the full run set.
    for n in (0, 2):
        current[n]["base_sha"] = "c" * 40
    advance = policy.BaseAdvance(101, "main", "b" * 40, "c" * 40)
    coverage = policy.evaluate_coverage(
        run,
        current,
        proofs,
        paths,
        fix_diff_digests=digests,
        base_advances=[advance],
    )
    assert coverage.covered
    assert coverage.unresolved == (() if fixed else ((100, "pr-100-1"),))
    assert not policy.evaluate_coverage(
        run, current, proofs, paths, fix_diff_digests=digests
    ).covered


def test_multiple_peer_merges_require_a_complete_base_chain(policy: Any) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    current[0]["base_sha"] = "e" * 40
    chain = [
        policy.BaseAdvance(101, "main", "b" * 40, "c" * 40),
        policy.BaseAdvance(102, "main", "c" * 40, "e" * 40),
    ]
    assert policy.evaluate_coverage(
        run, current, [], {}, fix_diff_digests={}, base_advances=chain
    ).covered
    assert not policy.evaluate_coverage(
        run, current, [], {}, fix_diff_digests={}, base_advances=chain[1:]
    ).covered


@pytest.mark.parametrize(
    "case",
    [
        "foreign_pr",
        "self_merge",
        "wrong_base",
        "retarget",
        "merge_main",
        "head",
        "diff",
    ],
)
def test_peer_merge_proof_never_covers_unreviewed_snapshot_changes(
    policy: Any, case: str
) -> None:
    run = result(policy)
    current = copy.deepcopy(run["request"]["prs"])
    current[0]["base_sha"] = "c" * 40
    advance = policy.BaseAdvance(
        999 if case == "foreign_pr" else 100 if case == "self_merge" else 101,
        "other" if case == "wrong_base" else "main",
        "b" * 40,
        "c" * 40,
    )
    if case == "retarget":
        current[0]["base_ref"] = "other"
    elif case == "merge_main":
        current[0]["merge_base_sha"] = "c" * 40
    elif case == "head":
        current[0]["head_sha"] = "d" * 40
    elif case == "diff":
        current[0]["diff_sha256"] = "d" * 64
    assert not policy.evaluate_coverage(
        run, current, [], {}, fix_diff_digests={}, base_advances=[advance]
    ).covered


def test_incomplete_peer_does_not_change_selected_pr_coverage(policy: Any) -> None:
    run = result(policy)
    run["prs"][1]["coverage"] = "not_reviewed"
    run.update(status="failed", verdict="incomplete", error="PR 101は未確認")
    current = [run["request"]["prs"][0]]
    coverage = policy.evaluate_coverage(run, current, [], {}, fix_diff_digests={})
    assert coverage.covered
    assert coverage.unresolved == ((100, "pr-100-1"),)


def test_rebase_inherits_resolved_findings_only_with_verified_fixed_snapshot(
    policy: Any,
) -> None:
    run = result(policy)
    original = run["request"]["prs"][0]
    fixed = {**original, "head_sha": "d" * 40, "diff_sha256": "e" * 64}
    proof = resolution(policy, run, fixed)
    rebased = {
        **fixed,
        "head_sha": "f" * 40,
        "base_sha": "c" * 40,
        "merge_base_sha": "c" * 40,
    }
    coverage = policy.evaluate_coverage(
        run,
        [rebased],
        [proof],
        {},
        fix_diff_digests={},
        inherited_snapshots=frozenset({(policy.digest(fixed), policy.digest(rebased))}),
    )
    assert coverage.covered and not coverage.unresolved
    absent = policy.evaluate_coverage(run, [rebased], [proof], {}, fix_diff_digests={})
    assert not absent.covered and absent.unresolved
