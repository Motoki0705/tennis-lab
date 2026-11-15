"""Historical input isolation, closed transport, budget admission and honest metrics."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.review_evaluation import historical_case, runner_evidence

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    for variable in ("DECISION_LOG_CAPTURE", "DECISION_LOG_RETRIEVAL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.syspath_prepend(str(OPS))
    monkeypatch.syspath_prepend(str(OPS / "lib"))
    # Other suites import a different top-level "review" package. Exercise
    # that collision even when this file runs alone, without evicting it.
    if "review" not in sys.modules:
        foreign = ModuleType("review")
        foreign.__dict__["__path__"] = []
        monkeypatch.setitem(sys.modules, "review", foreign)
    alias = "_tennis_lab_review_evaluation"
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            alias,
            OPS / "review/evaluation/__init__.py",
            submodule_search_locations=[str(OPS / "review/evaluation")],
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules[alias] = package
        spec.loader.exec_module(package)
    return SimpleNamespace(
        **{
            name: importlib.import_module(alias + "." + name)
            for name in (
                "catalog",
                "transport",
                "metrics",
                "execution",
                "restrictions",
                "report",
                "replay",
            )
        }
    )


@pytest.fixture
def small_case(tmp_path: Path, modules: Any) -> Any:
    return historical_case(tmp_path, modules.catalog)


def test_approved_catalog_has_no_answers_or_own_future_fixes_in_inputs(
    modules: Any,
) -> None:
    c = modules.catalog
    gold = c.read(c.DATA / "gold.json")
    assert sum(len(case["findings"]) for case in gold["cases"]) == 9
    answers = {case["case_id"]: case for case in gold["cases"]}
    for case in c.catalog()["cases"]:
        text = json.dumps(c.frozen_input(case), ensure_ascii=False)
        assert f"pr-{case['pr']}-1" not in text
        answer = answers[case["case_id"]]
        for proof in answer.get("verification", []):
            assert proof["fixed_head"] not in text
            assert proof["regression_tests"] and proof["confirmation"]
            assert not proof["reexecuted_in_this_preparation"]
        if answer["findings"]:
            assert {f["id"] for f in answer["findings"]} == {
                p["finding_id"] for p in answer["verification"]
            }
    negatives = [case for case in gold["cases"] if not case["findings"]]
    assert len(negatives) == 2
    assert all(
        case["negative_review"]["method"] == "independent_static_code_inspection"
        and case["negative_review"]["checks"]
        and case["negative_review"]["limitations"]
        for case in negatives
    )


@pytest.mark.parametrize(
    "fault",
    [
        "created_late",
        "edited_late",
        "answer_key",
        "github_edit_late",
        "context_edit_late",
    ],
)
def test_reviewer_input_rejects_later_comments_and_answer_key(
    small_case: Any, modules: Any, fault: str
) -> None:
    c = modules.catalog
    value = c.frozen_input(small_case.case, small_case.data)
    comment = {
        "id": 1,
        "body": "SPOILER",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "html_url": "https://github.com/o/r/issues/1#issuecomment-1",
    }
    if fault == "answer_key":
        value["gold_findings"] = []
    elif fault == "github_edit_late":
        comment["updated_at"] = "2099-01-01T00:00:00Z"
        value["github"]["repos/Motoki0705/tennis-lab/issues/1"]["value"]["comments"] = [
            comment
        ]
    elif fault == "context_edit_late":
        comment["updated_at"] = "2099-01-01T00:00:00Z"
        value["context"]["children"] = [{"comments": [comment]}]
    else:
        comment["created_at" if fault == "created_late" else "updated_at"] = (
            "2099-01-01T00:00:00Z"
        )
        value["context"]["parent_comments"] = [comment]
    with pytest.raises(ValueError):
        c.validate_input(value, small_case.case)


def test_preparation_is_pinned_and_never_copies_the_answer_key(
    small_case: Any, modules: Any
) -> None:
    c = modules.catalog
    bundle = c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)
    path = small_case.state / "bundles/pr-1699"
    assert sorted(p.name for p in path.iterdir()) == ["bundle.json", "review.diff"]
    assert "SPOILER" not in json.dumps(bundle)
    assert (
        c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)
        == bundle
    )
    (path / "review.diff").write_text("modified")
    with pytest.raises(ValueError, match="differs"):
        c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)


def test_transport_denies_future_git_and_unregistered_github_without_fallback(
    small_case: Any, modules: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = modules.catalog
    c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)
    transport = modules.transport.FrozenRetrieval(
        small_case.repo,
        small_case.state / "bundles/pr-1699/bundle.json",
        small_case.state / "receipts",
        data=small_case.data,
    )
    head = small_case.case["snapshot"]["head_sha"]
    assert (
        transport.call("code_file", {"revision": head, "path": "change.py"})["data"][
            "text"
        ]
        == "answer = 2\n"
    )
    assert not transport.call(
        "code_file", {"revision": small_case.future, "path": "future-answer.txt"}
    )["ok"]
    assert not transport.call(
        "commit_diff", {"revision": head, "commit_sha": small_case.future}
    )["ok"]
    assert "future fix" not in json.dumps(
        transport.call("git_history", {"revision": head})
    )

    def no_process(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a frozen GitHub operation attempted an external process")

    monkeypatch.setattr("subprocess.Popen", no_process)
    assert transport.call(
        "github_get", {"endpoint": "repos/Motoki0705/tennis-lab/issues/1"}
    )["ok"]
    missing = transport.call(
        "github_get", {"endpoint": "repos/Motoki0705/tennis-lab/issues/1699/comments"}
    )
    assert not missing["ok"] and missing["error"]["code"] == "UNREGISTERED_ENDPOINT"
    assert not transport.call("ci_log", {"job_id": 1})["ok"]
    assert set(transport.tools) == {
        "code_paths",
        "code_search",
        "code_file",
        "git_history",
        "commit_diff",
        "github_get",
        "review_context",
        "review_diff",
    }
    written = transport.frozen.write(
        "POST",
        "repos/Motoki0705/tennis-lab/issues/1699/comments",
        {"body": "local receipt"},
    )
    assert written["local_only"]
    with pytest.raises(Exception, match="local evaluation"):
        transport.frozen.write(
            "PUT", "repos/Motoki0705/tennis-lab/pulls/1699/merge", {}
        )


def observation(key: str = "first", **changes: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": key,
        "case_id": "pr-1699",
        "condition": "astra_fixed",
        "model": "gpt-6-astra",
        "role": "paired",
        "episode_id": "pr-1699",
        "agent_invocations": 1,
        "runtime_status": "completed",
        "output_status": "valid",
        "output": {
            "prs": [{"coverage": "reviewed", "findings": [], "evidence_waits": []}],
            "cross_findings": [],
        },
        "output_sha256": "pinned-output",
        "counter": {
            "key": key,
            "basis": "turn",
            "input_tokens": 10,
            "cached_input_tokens": 7,
            "output_tokens": 3,
        },
        "elapsed_seconds": 2,
        **changes,
    }


def test_usage_counts_cache_and_shared_source_once_and_keeps_missing_data(
    modules: Any,
) -> None:
    m = modules.metrics
    original = observation()
    repeated = observation("second", counter=original["counter"])
    totals = m.usage([original, repeated, original])
    assert [totals[name]["observed_sum"] for name in m.TOKENS] == [10, 7, 3]
    missing = observation("missing", counter=None)
    cumulative = observation(
        "cumulative",
        counter={
            "key": "cumulative",
            "basis": "cumulative",
            "input_tokens": 999,
            "cached_input_tokens": 900,
            "output_tokens": 99,
        },
    )
    partial = m.usage([original, missing, cumulative])
    assert partial["status"] == "partial" and partial["input_tokens"] == {
        "observed_sum": 10,
        "missing_counters": 2,
    }
    assert partial["excluded_non_turn_counters"] == ["cumulative"]
    conflicting = copy.deepcopy(repeated)
    conflicting["counter"]["input_tokens"] = 11
    with pytest.raises(ValueError, match="conflicting usage"):
        m.usage([original, conflicting])
    with pytest.raises(ValueError, match="no-AI"):
        m.usage([observation(agent_invocations=0)])


def test_detection_waits_for_parent_and_checks_pinned_adjudications(
    modules: Any,
) -> None:
    m, c = modules.metrics, modules.catalog
    gold = {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "pr-1699",
                "findings": [
                    {"id": "gold-1", "path": "guard.py", "severity": "major"},
                    {"id": "gold-2", "path": "guard.py", "severity": "major"},
                ],
            }
        ],
    }
    output = {
        "prs": [
            {
                "coverage": "reviewed",
                "findings": [
                    {"id": "finding-1", "path": "guard.py"},
                    {"id": "finding-2", "path": "elsewhere.py"},
                ],
                "evidence_waits": [],
            }
        ],
        "cross_findings": [],
    }
    row = observation(output=output)
    pending = m.summarize([row], gold)
    assert pending["detection"][0]["true_positives"] is None
    assert pending["candidate_matches"][0]["candidate_gold_ids"] == ["gold-1", "gold-2"]
    parent = {
        "schema_version": 1,
        "gold_sha256": c.digest(gold),
        "judgments": [
            {
                "run_id": row["run_id"],
                "output_sha256": row["output_sha256"],
                "adjudicator": "parent",
                "matches": [
                    {
                        "finding_id": "finding-1",
                        "gold_id": "gold-1",
                        "reason": "same trigger and effect",
                        "evidence": "reproduction record",
                    }
                ],
                "other_findings": [
                    {
                        "finding_id": "finding-2",
                        "classification": "false",
                        "reason": "the guard rejects this input",
                        "evidence": "independent test",
                    }
                ],
            }
        ],
    }
    scored = m.summarize([row], gold, parent)["detection"][0]
    assert (
        scored["true_positives"],
        scored["false_negatives"],
        scored["false_positives"],
    ) == (1, 1, 1)
    parent["judgments"][0]["output_sha256"] = "changed"
    with pytest.raises(ValueError, match="exact output"):
        m.summarize([row], gold, parent)
    assert not m.summarize([], gold)["default_policy_change_authorized"]


def test_serial_pilot_admission_and_collected_usage_reforecast(
    small_case: Any, modules: Any
) -> None:
    e = modules.execution
    e.initialize(small_case.state, data=small_case.data)
    with pytest.raises(ValueError, match="pilot"):
        e.reserve(
            small_case.state,
            1766,
            "gpt-6-astra",
            "parent instruction",
            data=small_case.data,
        )
    first = e.reserve(
        small_case.state,
        1699,
        "gpt-6-astra",
        "parent instruction",
        data=small_case.data,
    )
    assert (
        e.reserve(
            small_case.state,
            1699,
            "gpt-6-astra",
            "parent instruction",
            data=small_case.data,
        )
        == first
    )
    with pytest.raises(ValueError, match="unconfirmed"):
        e.reserve(
            small_case.state,
            1699,
            "gpt-6.1-sol",
            "parent instruction",
            data=small_case.data,
        )
    runner_evidence(small_case.state / "runs" / first["id"], "gpt-6-astra")
    observed = e.collect(small_case.state, first["id"], data=small_case.data)
    assert (
        observed["output_status"] == "valid"
        and observed["counter"]["input_tokens"] == 100
    )
    second = e.reserve(
        small_case.state,
        1699,
        "gpt-6.1-sol",
        "parent instruction",
        data=small_case.data,
    )
    runner_evidence(small_case.state / "runs" / second["id"], "gpt-6.1-sol")
    e.collect(small_case.state, second["id"], data=small_case.data)
    value = e.load(small_case.state, data=small_case.data)
    budget = e.forecast(
        value, e.observations(small_case.state, value), data=small_case.data
    )
    assert budget["status"] == "within_estimate" and budget["remaining_pairs"] == 12
    assert not budget["hard_token_limit"]
    e.reserve(
        small_case.state,
        1766,
        "gpt-6.1-sol",
        "parent instruction",
        data=small_case.data,
    )


@pytest.mark.parametrize("fault", ["format", "cleanup", "budget"])
def test_failed_or_expensive_pilot_blocks_progress(
    small_case: Any, modules: Any, fault: str
) -> None:
    e = modules.execution
    e.initialize(small_case.state, data=small_case.data)
    for model in modules.catalog.MODELS:
        slot = e.reserve(
            small_case.state, 1699, model, "parent instruction", data=small_case.data
        )
        runner_evidence(
            small_case.state / "runs" / slot["id"],
            model,
            output="not JSON" if fault == "format" else None,
            cleanup=fault != "cleanup",
            usage={
                "input_tokens": 55_000_000,
                "cached_input_tokens": 50_000_000,
                "output_tokens": 220_000,
            }
            if fault == "budget"
            else None,
        )
        if fault == "cleanup":
            with pytest.raises(ValueError, match="cleanup"):
                e.collect(small_case.state, slot["id"], data=small_case.data)
            break
        e.collect(small_case.state, slot["id"], data=small_case.data)
        if fault in {"format", "budget"}:
            break
    with pytest.raises(ValueError, match="unconfirmed|blocks further"):
        e.reserve(
            small_case.state,
            1766,
            "gpt-6-astra",
            "parent instruction",
            data=small_case.data,
        )


def test_launcher_disables_unpinned_tools_and_execution_is_explicit(
    small_case: Any, modules: Any
) -> None:
    args = modules.restrictions.restricted_arguments(
        [
            "exec",
            "--sandbox",
            "read-only",
            "--model",
            "gpt-6.1-sol",
            "-c",
            'mcp_servers.review_evaluation.command="python"',
            "-c",
            'web_search="disabled"',
            "-",
        ],
        ["live_github"],
    )
    assert 'web_search="disabled"' in args
    assert 'mcp_servers."live_github".enabled=false' in args
    for feature in (
        "shell_tool",
        "apps",
        "browser_use",
        "computer_use",
        "view_image",
        "multi_agent",
    ):
        assert args[args.index(feature) - 1] == "--disable"
    for overrides in (
        ["-c", 'web_search="live"'],
        ["-c", 'web_search="cached"'],
        ["--enable", "apps"],
        ['--config=web_search="live"'],
        ["-c", 'mcp_servers.review_evaluation.url="https://example.invalid/live"'],
    ):
        with pytest.raises(ValueError, match="unregistered"):
            modules.restrictions.restricted_arguments(
                ["exec", "--sandbox", "read-only", *overrides], []
            )
    with pytest.raises(ValueError, match="opt-in"):
        modules.execution.execute(
            small_case.repo,
            small_case.state,
            1699,
            "gpt-6-astra",
            "this turn does not execute",
        )


def test_capacity_projection_and_parent_selection_are_not_post_hoc(
    small_case: Any, modules: Any
) -> None:
    e, c, reporting = modules.execution, modules.catalog, modules.report
    e.initialize(small_case.state, data=small_case.data)
    selection = {
        "schema_version": 1,
        "source": "Synthetic parent choice made before any results",
        "choices": [
            {
                "case_id": case["case_id"],
                "model": "gpt-6.1-sol",
                "reason": "Fixture choice",
            }
            for case in c.catalog(small_case.data / "manifest.json")["cases"]
        ],
    }
    reporting.configure_selection(small_case.state, selection, data=small_case.data)
    for number in c.APPROVED_CASES:
        for model in c.MODELS:
            slot = e.reserve(
                small_case.state, number, model, "fixture only", data=small_case.data
            )
            tokens = 100 if model == "gpt-6-astra" else 60
            runner_evidence(
                small_case.state / "runs" / slot["id"],
                model,
                number=number,
                usage={
                    "input_tokens": tokens,
                    "cached_input_tokens": tokens // 2,
                    "output_tokens": tokens // 10,
                },
            )
            e.collect(small_case.state, slot["id"], data=small_case.data)
    paired = reporting.report(small_case.state, data=small_case.data)
    assert paired["adoption"]["paired_matrix_complete"]
    assert paired["adoption"]["selection_improvements"][
        "processed_input_plus_output"
    ] == pytest.approx(0.4)
    for number in (1699, 1766):
        extra = e.reserve(
            small_case.state,
            number,
            "gpt-6-astra",
            "fixture additional instruction",
            additional=True,
            reason="Independent fixture confirmation",
            data=small_case.data,
        )
        runner_evidence(
            small_case.state / "runs" / extra["id"], "gpt-6-astra", number=number
        )
        e.collect(small_case.state, extra["id"], data=small_case.data)
    with pytest.raises(ValueError, match="capacity exhausted"):
        e.reserve(
            small_case.state,
            1867,
            "gpt-6-astra",
            "fixture",
            additional=True,
            reason="would exceed approval",
            data=small_case.data,
        )
    changed = {**selection, "source": "Choice after results"}
    with pytest.raises(ValueError, match="post-hoc"):
        reporting.configure_selection(small_case.state, changed, data=small_case.data)
    # Idempotent submission is not a seventeenth invocation.
    e.reserve(small_case.state, 1699, "gpt-6-astra", "fixture", data=small_case.data)
    result = reporting.report(small_case.state, data=small_case.data)
    assert result["qualification"]["physical_usage"]["unique_counters"] == 16
    assert len(result["execution_evidence"]) == 16
    assert all(
        row["runtime_settings"]["validation"] == "verified"
        for row in result["execution_evidence"]
    )
    assert (
        result["qualification"]["physical_usage"]["input_tokens"]["observed_sum"]
        == 1320
    )
    projected = result["conditions"]["reuse_select"]
    costs = projected["results"]["conditions"]["reuse_select"]
    assert costs["additional_invocations"] == 2
    assert costs["additional_launch_rate"] == pytest.approx(2 / 7)
    assert result["adoption"]["selection_improvements"][
        "processed_input_plus_output"
    ] == pytest.approx(1 - 682 / 770)
    assert projected["requires_parent_joint_adjudication"]
    assert result["adoption"]["decision"] == "parent_required"
    assert not result["adoption"]["default_policy_change_authorized"]
    first = e.load(small_case.state, data=small_case.data)["runs"][0]
    path = small_case.state / "runs" / first["id"] / "observation.json"
    modified = c.read(path)
    modified["counter"]["input_tokens"] += 1
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="observation changed"):
        reporting.report(small_case.state, data=small_case.data)


def test_reuse_audit_preserves_findings_and_incomplete_coverage(modules: Any) -> None:
    candidate: dict[str, Any] = {
        "verdict": "incomplete",
        "cross_findings": [],
        "prs": [
            {
                "number": 1790,
                "coverage": "not_reviewed",
                "findings": [],
                "evidence_waits": [],
            }
        ],
    }
    assert not modules.replay.assess_candidate(candidate)["can_avoid_full_ai_rereview"]
    candidate["verdict"] = "request_changes"
    candidate["prs"][0].update(coverage="reviewed", findings=[{"id": "must-retain"}])
    result = modules.replay.assess_candidate(candidate)
    assert result["can_avoid_full_ai_rereview"] and result["findings_retained"] == 1
    assert not result["current_formal_recovery_authorized"]
    assert not candidate["prs"][0]["evidence_waits"]


def test_reuse_audit_rejects_duplicate_physical_repeat_before_reading_state(
    small_case: Any, modules: Any
) -> None:
    pair = {"source_run": "source", "repeat_run": "repeat"}
    (small_case.data / "reuse_pairs.json").write_text(
        json.dumps({"pairs": [pair, pair]})
    )
    with pytest.raises(ValueError, match="charged twice"):
        modules.replay.historical_reuse(small_case.state, data=small_case.data)


def test_private_execution_builds_the_shared_runner_contract_without_a_live_agent(
    small_case: Any, modules: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    opsproc: Any = importlib.import_module("lib.opsproc")
    command: Any = importlib.import_module("lib.agent_settings").command

    e, c = modules.execution, modules.catalog
    e.initialize(small_case.state, data=small_case.data)
    with pytest.raises(ValueError, match="selection before the first launch"):
        e.execute(
            small_case.repo,
            small_case.state,
            1699,
            "gpt-6.1-sol",
            "fixture only",
            execute_approved=True,
            data=small_case.data,
        )
    selection = {
        "schema_version": 1,
        "source": "Synthetic parent choice before execution",
        "choices": [
            {
                "case_id": case["case_id"],
                "model": "gpt-6.1-sol",
                "reason": "Fixture choice",
            }
            for case in c.catalog(small_case.data / "manifest.json")["cases"]
        ],
    }
    modules.report.configure_selection(
        small_case.state, selection, data=small_case.data
    )
    monkeypatch.setattr(e.shutil, "which", lambda _: "/fixture/codex")
    monkeypatch.setattr(e, "inspect_cli", lambda _: (["live_github"], "codex fixture"))
    monkeypatch.setenv("DECISION_LOG_CAPTURE", "must-not-be-inherited")
    seen = []

    def fake_runner(
        argv: list[str], *, cwd: Path, timeout: int, env: dict[str, str]
    ) -> None:
        seen.append(argv)
        assert timeout == 1860 and "DECISION_LOG_CAPTURE" not in env
        directory = cwd.parent
        assert sorted(path.name for path in cwd.iterdir()) == [".git"]
        assert "--task-id" not in argv and "--session" not in argv
        assert argv[argv.index("--timeout") + 1] == "1800"
        connection = Path(argv[argv.index("--mcp-config") + 1])
        assert set(c.read(connection)) == {"review_evaluation"}
        selection = c.read(directory / "review-selection.json")["selected"]
        raw = command(
            {
                "agent": "codex",
                "profile": "review",
                "selected": selection,
                "web_search": "disabled",
            },
            "read-only",
            cwd,
            directory / "agent_output.md",
            None,
            [],
            OPS / "review/result_schema.json",
            connection,
        )
        guarded = modules.restrictions.restricted_arguments(raw[1:], ["live_github"])
        assert "project_doc_max_bytes=0" in guarded
        assert 'mcp_servers."live_github".enabled=false' in guarded
        runner_evidence(directory, selection["model"])

    monkeypatch.setattr(opsproc, "run_process_group", fake_runner)
    row = e.execute(
        small_case.repo,
        small_case.state,
        1699,
        "gpt-6.1-sol",
        "fixture opt-in only",
        execute_approved=True,
        data=small_case.data,
    )
    assert row["output_status"] == "valid" and row["qualification_only"]
    assert row["model"] == "gpt-6.1-sol" and row["counter"]["input_tokens"] == 100
    assert (
        e.execute(
            small_case.repo,
            small_case.state,
            1699,
            "gpt-6.1-sol",
            "same fixture",
            execute_approved=True,
            data=small_case.data,
        )
        == row
    )
    assert len(seen) == 1
    # Two in-memory callers of the same reserved slot must still start only
    # one fabricated runner. No real CLI/model or extra pytest process runs.
    concurrent_state = small_case.state.with_name("concurrent-evaluation")
    e.initialize(concurrent_state, data=small_case.data)
    modules.report.configure_selection(
        concurrent_state, selection, data=small_case.data
    )
    c.prepare(small_case.repo, concurrent_state, 1699, data=small_case.data)
    reserve = e.reserve
    barrier = threading.Barrier(2)

    def simultaneous_reservation(*args: Any, **kwargs: Any) -> Any:
        slot = reserve(*args, **kwargs)
        barrier.wait(timeout=5)
        return slot

    monkeypatch.setattr(e, "reserve", simultaneous_reservation)

    def submit() -> Any:
        try:
            return e.execute(
                small_case.repo,
                concurrent_state,
                1699,
                "gpt-6.1-sol",
                "fixture only",
                execute_approved=True,
                data=small_case.data,
            )
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as callers:
        results = list(callers.map(lambda _: submit(), range(2)))
    assert len(seen) == 2  # one serial call and one simultaneous call
    assert sum(isinstance(result, dict) for result in results) == 1
    assert (
        sum(
            isinstance(result, str) and "already has files" in result
            for result in results
        )
        == 1
    )


def scoring_case() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    output = {
        "prs": [
            {
                "coverage": "reviewed",
                "findings": [
                    {"id": "f-1", "path": "guard.py"},
                    {"id": "f-2", "path": "other.py"},
                ],
                "evidence_waits": [],
            }
        ],
        "cross_findings": [],
    }
    row = observation(output=output)
    gold = {
        "case_id": "pr-1699",
        "findings": [
            {"id": "g-1", "path": "guard.py", "severity": "major"},
            {"id": "g-2", "path": "guard.py", "severity": "major"},
        ],
    }
    judgment = {
        "run_id": row["run_id"],
        "output_sha256": row["output_sha256"],
        "adjudicator": "parent",
        "matches": [
            {
                "finding_id": "f-1",
                "gold_id": "g-1",
                "reason": "same defect",
                "evidence": "independent test",
            }
        ],
        "other_findings": [
            {
                "finding_id": "f-2",
                "classification": "unconfirmed",
                "reason": "not established",
                "evidence": "static evidence only",
            }
        ],
    }
    return row, gold, judgment


@pytest.mark.parametrize("fault", ["missing", "overlap", "duplicate", "multiple_gold"])
def test_parent_classification_must_be_complete_and_unambiguous(
    modules: Any,
    fault: str,
) -> None:
    row, gold, judgment = scoring_case()
    if fault == "missing":
        judgment["other_findings"] = []
    elif fault == "overlap":
        judgment["other_findings"][0]["finding_id"] = "f-1"
    else:
        extra = copy.deepcopy(judgment["matches"][0])
        if fault == "multiple_gold":
            extra["gold_id"] = "g-2"
        judgment["matches"].append(extra)
    with pytest.raises(ValueError):
        modules.metrics.judged(row, gold["findings"], judgment)


def test_runtime_failure_candidate_is_retained_but_excluded_from_quality(
    modules: Any,
) -> None:
    row, gold, judgment = scoring_case()
    row["runtime_status"] = "failed"
    result = modules.metrics.summarize(
        [row],
        {"schema_version": 1, "cases": [gold]},
        {
            "schema_version": 1,
            "gold_sha256": modules.catalog.digest(
                {"schema_version": 1, "cases": [gold]}
            ),
            "judgments": [judgment],
        },
    )
    assert modules.metrics.findings(row)[0]["id"] == "f-1"
    cost = result["conditions"]["astra_fixed"]
    assert cost["runtime_failures"] == 1
    assert cost["detection"]["true_positives"]["confirmed_sum"] == 0
    assert cost["unconfirmed_finding_rate"] is None
    assert result["detection"][0]["status"] == "runtime_failure_excluded"
    assert result["status"] != "scored"


def test_runtime_failure_before_agent_start_is_counted(modules: Any) -> None:
    row, gold, _ = scoring_case()
    row.update(
        agent_invocations=0,
        counter=None,
        runtime_status="failed",
        output_status="missing",
        output=None,
        output_sha256=None,
    )
    result = modules.metrics.summarize(
        [row], {"schema_version": 1, "cases": [gold]}
    )
    cost = result["conditions"]["astra_fixed"]
    assert cost["runtime_failures"] == 1
    assert cost["agent_invocations"] == 0
    assert cost["quality_eligible_observations"] == 0
    assert cost["detection"]["true_positives"]["confirmed_sum"] == 0
    assert result["status"] == "incomplete"


@pytest.mark.parametrize(
    "capture_variable", ["DECISION_LOG_RETRIEVAL", "DECISION_LOG_CAPTURE"]
)
def test_direct_serve_rejects_formal_capture_without_writing(
    small_case: Any,
    modules: Any,
    monkeypatch: pytest.MonkeyPatch,
    capture_variable: str,
) -> None:
    import io

    capture = importlib.import_module("decision_logs.capture")
    owner = capture.Operation.begin(
        small_case.state / "formal-parent",
        "o/r",
        1,
        "review.run",
        "review",
    )
    c = modules.catalog
    c.prepare(
        small_case.repo, small_case.state / "evaluation", 1699, data=small_case.data
    )
    cli = importlib.import_module(c.__package__ + ".__main__")
    actual = modules.transport.FrozenRetrieval
    monkeypatch.setattr(
        modules.transport,
        "FrozenRetrieval",
        lambda repo, bundle, journal: actual(
            repo, bundle, journal, data=small_case.data
        ),
    )
    monkeypatch.setenv(capture_variable, str(owner.path))
    before = {
        p.relative_to(small_case.state): p.read_bytes()
        for p in (small_case.state / "formal-parent").rglob("*")
        if p.is_file()
    }
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "review_context", "arguments": {}},
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(
            sys,
            "stdin",
            SimpleNamespace(buffer=io.BytesIO((json.dumps(request) + "\n").encode())),
        )
        scoped.setattr(sys, "stdout", SimpleNamespace(buffer=io.BytesIO()))
        code = cli.main(
            [
                "serve",
                "--repo",
                str(small_case.repo),
                "--bundle",
                str(small_case.state / "evaluation/bundles/pr-1699/bundle.json"),
                "--journal",
                str(small_case.state / "evaluation/retrieval"),
            ]
        )
    after = {
        p.relative_to(small_case.state): p.read_bytes()
        for p in (small_case.state / "formal-parent").rglob("*")
        if p.is_file()
    }
    assert after == before
    assert code == 2


def test_public_frozen_input_rejects_worktree_path(
    small_case: Any, modules: Any
) -> None:
    c = modules.catalog
    case = copy.deepcopy(small_case.case)
    value = c.frozen_input(case, small_case.data)
    value["context"]["parent"]["body"] = str(
        Path(".claude") / "worktrees" / "private-task"
    )
    case["input_sha256"] = c.digest(value)
    (small_case.data / case["input_file"]).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="worktree"):
        c.frozen_input(case, small_case.data)


@pytest.mark.parametrize("reason", ["cli_error", "runtime_error"])
def test_collect_keeps_verified_failed_final_as_candidate(
    small_case: Any, modules: Any, reason: str
) -> None:
    e = modules.execution
    e.initialize(small_case.state, data=small_case.data)
    slot = e.reserve(
        small_case.state, 1699, "gpt-6-astra", "fixture", data=small_case.data
    )
    directory = small_case.state / "runs" / slot["id"]
    runtime = runner_evidence(directory, "gpt-6-astra")
    runtime.update(
        result="failed", exit_code=1, cli_exit_code=1, turn_end_reason=reason
    )
    if reason == "runtime_error":
        event = {"type": "error", "message": "fixture error after final"}
        with (directory / "agent-runtime/events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        runtime["cli_exit_code"] = 0
        runtime["runtime"]["runtime_errors"] = [event]
    (directory / "agent-runtime/result.json").write_text(json.dumps(runtime))
    row = e.collect(small_case.state, slot["id"], data=small_case.data)
    assert row["proof_error"] is None
    assert row["runtime_status"] == "failed"
    assert row["output_status"] == "candidate_only"
    assert row["output"] is not None and row["output_sha256"]


def test_recorded_report_is_bound_and_read_only(
    small_case: Any,
    modules: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e, c = modules.execution, modules.catalog
    e.initialize(small_case.state, data=small_case.data)
    c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)
    slot = e.reserve(
        small_case.state, 1699, "gpt-6-astra", "fixture", data=small_case.data
    )
    directory = small_case.state / "runs" / slot["id"]
    runner_evidence(directory, "gpt-6-astra")
    e.collect(small_case.state, slot["id"], data=small_case.data)
    state = e.load(small_case.state, data=small_case.data)
    proof = {
        k: state[k]
        for k in (
            "code_sha256",
            "catalog_sha256",
            "gold_sha256",
            "historical_usage_sha256",
            "protocol",
        )
    }
    proof["ledger_canonical_sha256"] = c.digest(state)
    before = {
        str(p.relative_to(small_case.state)): p.read_bytes()
        for p in small_case.state.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(e, "code_digest", lambda: "changed-for-new-validation")
    with pytest.raises(ValueError, match="frozen"):
        e.load(small_case.state, data=small_case.data)
    report = modules.report.report(
        small_case.state,
        data=small_case.data,
        recorded_provenance=proof,
        repo=small_case.repo,
    )
    assert (
        report["qualification"]["physical_usage"]["input_tokens"]["observed_sum"] == 100
    )
    assert before == {
        str(p.relative_to(small_case.state)): p.read_bytes()
        for p in small_case.state.rglob("*")
        if p.is_file()
    }
    bad = {**proof, "ledger_canonical_sha256": "changed"}
    with pytest.raises(ValueError, match="ledger"):
        modules.report.report(
            small_case.state,
            data=small_case.data,
            recorded_provenance=bad,
            repo=small_case.repo,
        )
    (directory / "agent-runtime/final.md").write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        modules.report.report(
            small_case.state,
            data=small_case.data,
            recorded_provenance=proof,
            repo=small_case.repo,
        )


def test_input_publication_replays_original_and_rejects_forgery(
    small_case: Any, modules: Any
) -> None:
    publication = importlib.import_module(modules.catalog.__package__ + ".publication")
    c = modules.catalog
    case = copy.deepcopy(small_case.case)
    original = c.frozen_input(case, small_case.data)
    original["context"]["parent"]["body"] = str(
        Path(".claude") / "worktrees" / "private-task"
    )
    case["input_sha256"] = c.digest(original)
    raw = small_case.repo / "recorded-input.json"
    raw.write_text(json.dumps(original))
    c.git(small_case.repo, "add", "recorded-input.json")
    c.git(small_case.repo, "commit", "-qm", "record input")
    commit = c.git(small_case.repo, "rev-parse", "HEAD").decode().strip()
    public, rules = publication.project_input(original)
    case["input_publication"] = {
        "version": 1,
        "source_commit": commit,
        "source_path": "recorded-input.json",
        "source_sha256": case["input_sha256"],
        "published_sha256": c.digest(public),
        "rules": rules,
    }
    path = small_case.data / case["input_file"]
    path.write_text(json.dumps(public))
    assert c.digest(public) != case["input_sha256"] and rules["worktree_paths"] == 1
    c.verify_input(small_case.repo, case, small_case.data)
    manifest = c.catalog(small_case.data / "manifest.json")
    manifest["cases"][0] = case
    (small_case.data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="analysis-only"):
        modules.execution.initialize(small_case.state, data=small_case.data)
    public["context"]["parent"]["body"] = "forged public input"
    case["input_publication"]["published_sha256"] = c.digest(public)
    path.write_text(json.dumps(public))
    with pytest.raises(ValueError, match="sanitization"):
        c.verify_input(small_case.repo, case, small_case.data)


def test_whole_publication_check_rejects_worktree_in_input(
    small_case: Any, modules: Any
) -> None:
    publication = importlib.import_module(modules.catalog.__package__ + ".publication")
    value = modules.catalog.frozen_input(small_case.case, small_case.data)
    value["context"]["parent"]["body"] = str(
        Path(".claude") / "worktrees" / "private-task"
    )
    path = small_case.data / small_case.case["input_file"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="worktree"):
        publication.check_files(small_case.data, [small_case.case["input_file"]])


def test_saved_parent_adjudications_are_still_accepted(modules: Any) -> None:
    root = OPS.parents[1]
    report = root / "knowledge/reports/review-model-comparison-20261111"
    rows = []
    for item in json.loads((report / "run-index.json").read_text()):
        directory = root / item["artifacts"]
        row = modules.catalog.read(directory / "measurement.json")
        row["output"] = modules.catalog.read(directory / "output.json")
        rows.append(row)
    result = modules.metrics.summarize(
        rows,
        modules.catalog.read(modules.catalog.DATA / "gold.json"),
        modules.catalog.read(report / "adjudications.json"),
    )
    assert result["status"] == "scored"
    assert len(result["detection"]) == 14
    assert all(x["status"] == "parent_adjudicated" for x in result["detection"])


def test_capture_added_after_server_start_is_rejected(
    small_case: Any,
    modules: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = modules.catalog
    c.prepare(small_case.repo, small_case.state, 1699, data=small_case.data)
    client = modules.transport.FrozenRetrieval(
        small_case.repo,
        small_case.state / "bundles/pr-1699/bundle.json",
        small_case.state / "retrieval",
        data=small_case.data,
    )
    before = {
        p.relative_to(small_case.state): p.read_bytes()
        for p in (small_case.state / "retrieval").rglob("*")
        if p.is_file()
    }
    monkeypatch.setenv("DECISION_LOG_RETRIEVAL", "explicit capture configuration")
    with pytest.raises(ValueError, match="formal decision capture"):
        client.call("review_context", {})
    assert before == {
        p.relative_to(small_case.state): p.read_bytes()
        for p in (small_case.state / "retrieval").rglob("*")
        if p.is_file()
    }


def test_pending_parent_is_not_an_explicit_unconfirmed_verdict(modules: Any) -> None:
    row, gold, judgment = scoring_case()
    pending = modules.metrics.judged(row, gold["findings"], None)
    assert pending["status"] == "awaiting_parent"
    assert pending["unconfirmed_findings"] is None
    explicit = modules.metrics.judged(row, gold["findings"], judgment)
    assert explicit["status"] == "parent_adjudicated"
    assert explicit["unconfirmed_findings"] == 1


def test_failed_candidates_do_not_dilute_successful_finding_rates(modules: Any) -> None:
    row, gold, judgment = scoring_case()
    failed = observation(
        "failed", output=copy.deepcopy(row["output"]), runtime_status="failed"
    )
    failed_judgment = copy.deepcopy(judgment)
    failed_judgment["run_id"] = failed["run_id"]
    failed_judgment["other_findings"][0]["classification"] = "false"
    truth = {"schema_version": 1, "cases": [gold]}
    result = modules.metrics.summarize(
        [row, failed],
        truth,
        {
            "schema_version": 1,
            "gold_sha256": modules.catalog.digest(truth),
            "judgments": [judgment, failed_judgment],
        },
    )
    cost = result["conditions"]["astra_fixed"]
    assert cost["quality_eligible_observations"] == 1
    assert cost["runtime_excluded_observations"] == 1
    assert cost["detection"]["true_positives"]["confirmed_sum"] == 1
    assert cost["detection"]["false_positives"]["confirmed_sum"] == 0
    assert cost["unconfirmed_finding_rate"] == 0.5
    assert cost["usage"]["input_tokens"]["observed_sum"] == 20


def test_invalid_output_is_excluded_even_with_a_parent_match(modules: Any) -> None:
    row, gold, judgment = scoring_case()
    row["output_status"] = "invalid"
    truth = {"schema_version": 1, "cases": [gold]}
    result = modules.metrics.summarize(
        [row],
        truth,
        {
            "schema_version": 1,
            "gold_sha256": modules.catalog.digest(truth),
            "judgments": [judgment],
        },
    )
    cost = result["conditions"]["astra_fixed"]
    assert cost["format_failures"] == 1 and cost["format_excluded_observations"] == 1
    assert cost["detection"]["true_positives"]["confirmed_sum"] == 0
    assert cost["quality_eligible_observations"] == 0
    assert result["status"] == "incomplete"
