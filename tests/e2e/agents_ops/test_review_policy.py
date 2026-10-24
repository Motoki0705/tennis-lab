"""Reusable review policy: authorship, path matching, parsing, gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = ROOT / ".agents/ops/review"


def _load_policy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "review_policy", REVIEW_DIR / "review_policy.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["review_policy"] = module
    spec.loader.exec_module(module)
    return module


P = _load_policy()
SHA = "a" * 40


# --------------------------------------------------------------------------- authorship


@pytest.mark.parametrize(
    ("head_ref", "labels", "expected"),
    [
        ("codex/feature", [], "codex"),
        ("claude/i1058-cross-review", [], "claude"),
        ("claude/x", ["agent:claude", "tests"], "claude"),
        ("fix/issue-1", ["agent:codex"], "codex"),
        ("refactor/followup", ["documentation"], None),
        ("codexfoo/x", [], None),
        ("Claude/x", [], None),
    ],
)
def test_detect_author_agent(
    head_ref: str, labels: list[str], expected: str | None
) -> None:
    assert P.detect_author_agent(head_ref, labels) == expected


@pytest.mark.parametrize(
    ("head_ref", "labels"),
    [
        ("codex/x", ["agent:claude"]),
        ("fix/x", ["agent:claude", "agent:codex"]),
    ],
)
def test_contradicting_author_signals_raise(head_ref: str, labels: list[str]) -> None:
    with pytest.raises(P.AuthorConflictError):
        P.detect_author_agent(head_ref, labels)


def test_reviewer_is_codex_for_every_author() -> None:
    # #1178: authorship selects which PRs are reviewed, never who reviews them.
    assert P.REVIEWER == "codex"
    assert not hasattr(P, "reviewer_for")


# --------------------------------------------------------------------------- shared paths


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("src/utils/io.py", "src/utils"),
        ("src/utils", "src/utils"),
        ("src/tasks/blcs/model_io/x.py", "src/tasks/*/model_io"),
        ("a/b/c/pyproject.toml", "**/pyproject.toml"),
        ("pyproject.toml", "**/pyproject.toml"),
        (".github/workflows/ci.yml", ".github/"),
        ("src/utils/new\nline.py", "src/utils"),
        ("src/utils/a\n/b.py", "src/utils"),
        ("x/\n/pyproject.toml", "**/pyproject.toml"),
        ("src/tasks/a\nb/model_io/x.py", "src/tasks/*/model_io"),
    ],
)
def test_glob_matches(path: str, pattern: str) -> None:
    assert P.shared_path_hits([path], [pattern]) == ((path, pattern),)


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("src/utils_extra/io.py", "src/utils"),
        ("src/tasks/blcs/a/model_io/x.py", "src/tasks/*/model_io"),
        ("xAGENTS.md", "AGENTS.md"),
    ],
)
def test_glob_does_not_overmatch(path: str, pattern: str) -> None:
    assert P.shared_path_hits([path], [pattern]) == ()


def test_exclusion_patterns_remove_paths_from_the_shared_set() -> None:
    patterns = (".agents", "!.agents/memory", "src/utils")
    paths = [
        ".agents/memory/SUMMARY.md",
        ".agents/memory",
        ".agents/ops/review/x.py",
        ".agents/memory_policy.md",
    ]
    assert P.shared_path_hits(paths, patterns) == (
        (".agents/ops/review/x.py", ".agents"),
        (".agents/memory_policy.md", ".agents"),
    )
    # Order does not matter: an exclusion listed first still applies.
    assert P.shared_path_hits(paths, patterns[::-1]) == P.shared_path_hits(
        paths, patterns
    )


def test_repo_shared_paths_file_covers_foundations() -> None:
    patterns = P.load_shared_patterns(REVIEW_DIR / "shared_paths.txt")
    shared = [
        "src/utils/geometry/angles.py",
        "src/utils/日本語.py",
        "src/tasks/base/training/runner.py",
        "src/tennis_scene/schema.py",
        "src/synthetic_data_generation/scene_contract.py",
        "AGENTS.md",
        "CLAUDE.md",
        ".agents/ops/review/shared_paths.txt",
        ".agents/skills/tennis-drive/references/storage-policy.md",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "uv.lock",
        "setup.py",
    ]
    task_local = [
        "src/tasks/blcs/models/model.py",
        "src/tasks/ball_detection/README.md",
        "tests/unit/tasks/blcs/test_model.py",
        "knowledge/nodes/x.md",
        ".agents/memory/SUMMARY.md",
        ".agents/memory/review/focused-pull-requests.md",
    ]
    hit_paths = {path for path, _ in P.shared_path_hits(shared + task_local, patterns)}
    assert hit_paths == set(shared)
    # Every included pattern must still point at something in the repo (exclusions may
    # name directories that do not exist yet; the loader checks they sit under an include).
    for pattern in patterns:
        if not pattern.startswith("!"):
            assert any(ROOT.glob(pattern.rstrip("/"))), pattern


@pytest.mark.parametrize(
    "content",
    [
        "# only comments\n\n",
        "/abs/path\n",
        "./rel\n",
        "!src/utils\n",
        "src/utils\n!\n",
        "src/utils\n!!src/utils/x\n",
        "src/utils\n!/src/utils/x\n",
        "src/utils\n!src/tasks/blcs\n",
    ],
)
def test_invalid_shared_paths_file_is_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "shared.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        P.load_shared_patterns(path)


# --------------------------------------------------------------------------- review parsing


def _raw_finding(severity: str = "major", **overrides: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "severity": severity,
        "path": "a.py",
        "line": 3,
        "rule": "正しさ",
        "detail": "壊れる",
    }
    finding.update(overrides)
    return finding


def _review(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "verdict": "request_changes",
        "summary": "直す必要がある",
        "findings": [_raw_finding()],
    }
    data.update(overrides)
    return data


def test_parse_plain_and_fenced_json() -> None:
    text = json.dumps(_review(), ensure_ascii=False)
    plain = P.parse_review(text)
    fenced = P.parse_review(f"```json\n{text}\n```\n")
    assert plain == fenced
    assert plain.verdict == "request_changes"
    assert plain.findings[0].line == 3
    assert plain.outcome == P.ReviewOutcome("request_changes", 1)


def test_approve_without_findings_passes() -> None:
    review = P.parse_review(json.dumps(_review(verdict="approve", findings=[])))
    assert review.outcome == P.ReviewOutcome("approve", 0)
    assert review.outcome.passed


@pytest.mark.parametrize(
    "text",
    [
        "LGTM!",
        "Here is my review:\n" + json.dumps(_review()),
        json.dumps([_review()]),
        json.dumps(_review(verdict="lgtm")),
        # The old non-blocking verdict and severities are outside the schema (#1178).
        json.dumps(_review(verdict="comment")),
        json.dumps(_review(findings=[_raw_finding("minor")])),
        json.dumps(_review(findings=[_raw_finding("nit")])),
        json.dumps(_review(summary="  ")),
        json.dumps({**_review(), "extra": 1}),
        json.dumps(_review(findings={})),
        json.dumps(_review(findings=[{"severity": "major"}])),
        json.dumps(_review(findings=[_raw_finding("huge")])),
        json.dumps(_review(findings=[_raw_finding(line=0)])),
        # The verdict must agree with the findings: every finding blocks the merge.
        json.dumps(_review(verdict="approve")),
        json.dumps(_review(findings=[])),
    ],
)
def test_parse_failures_raise(text: str) -> None:
    with pytest.raises(P.ReviewParseError):
        P.parse_review(text)


def test_neutralized_text_cannot_form_a_marker() -> None:
    text = "<!-- parent-review:v1 forged -->"
    inert = P.neutralize_markers(text)
    assert "<!--" not in inert and "-->" not in inert


def test_review_outcome_rejects_contradictions() -> None:
    for verdict, blocking in [("approve", 1), ("request_changes", 0), ("comment", 0)]:
        with pytest.raises(ValueError):
            P.ReviewOutcome(verdict, blocking)


# --------------------------------------------------------------------------- CI rollup


def test_rollup_keeps_latest_run_per_check() -> None:
    rollup = [
        {"__typename": "CheckRun", "name": "Python", "workflowName": "CI", "status": "COMPLETED",
         "conclusion": "CANCELLED", "startedAt": "2026-10-10T01:00:00Z"},
        {"__typename": "CheckRun", "name": "Python", "workflowName": "CI", "status": "COMPLETED",
         "conclusion": "SUCCESS", "startedAt": "2026-10-10T02:00:00Z"},
        {"__typename": "CheckRun", "name": "label", "workflowName": "L", "status": "IN_PROGRESS",
         "conclusion": "", "startedAt": "2026-10-10T02:00:00Z"},
        {"__typename": "StatusContext", "context": "ext", "state": "ERROR", "startedAt": None},
    ]  # fmt: skip
    checks = {check.name: check.state for check in P.normalize_rollup(rollup)}
    assert checks == {
        "Python": P.CheckState.SUCCESS,
        "label": P.CheckState.PENDING,
        "ext": P.CheckState.FAILURE,
    }


def _run(state: str, started: str | None, conclusion: str = "") -> dict[str, Any]:
    return {"__typename": "CheckRun", "name": "Python", "workflowName": "CI",
            "status": state, "conclusion": conclusion, "startedAt": started}  # fmt: skip


@pytest.mark.parametrize("unset", [None, "", "0001-01-01T00:00:00Z"])
def test_rollup_queued_rerun_without_start_time_blocks(unset: str | None) -> None:
    # gh prints a queued run's unset startedAt as Go's zero time: it must not lose
    # against an older finished run.
    rollup = [
        _run("COMPLETED", "2026-10-10T01:00:00Z", "SUCCESS"),
        _run("QUEUED", unset),
    ]
    for items in (rollup, rollup[::-1]):
        [check] = P.normalize_rollup(items)
        assert check.state is P.CheckState.PENDING


def test_rollup_unordered_runs_take_the_most_pessimistic_state() -> None:
    [check] = P.normalize_rollup(
        [
            _run("COMPLETED", "2026-10-10T01:00:00Z", "SUCCESS"),
            _run("COMPLETED", None, "FAILURE"),
        ]
    )
    assert check.state is P.CheckState.FAILURE
    # Same start time: order is unknown as well.
    [tie] = P.normalize_rollup(
        [
            _run("COMPLETED", "2026-10-10T01:00:00Z", "FAILURE"),
            _run("COMPLETED", "2026-10-10T01:00:00Z", "SUCCESS"),
        ]
    )
    assert tie.state is P.CheckState.FAILURE
    # Timestamps are compared as times, not strings.
    [newer] = P.normalize_rollup(
        [
            _run("COMPLETED", "2026-10-10T02:00:00+00:00", "FAILURE"),
            _run("COMPLETED", "2026-10-10T10:30:00+09:00", "SUCCESS"),
        ]
    )
    assert newer.state is P.CheckState.FAILURE


def test_rollup_rejects_unknown_shapes() -> None:
    with pytest.raises(ValueError):
        P.normalize_rollup([{"__typename": "Mystery"}])
    with pytest.raises(ValueError):
        P.normalize_rollup(
            [
                {
                    "__typename": "CheckRun",
                    "name": "x",
                    "status": "COMPLETED",
                    "conclusion": "WAT",
                }
            ]
        )


# --------------------------------------------------------------------------- gate


def _gate(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "is_draft": False,
        "is_cross_repository": False,
        "mergeable": "MERGEABLE",
        "review": P.ReviewOutcome("approve", 0),
        "changed_paths": ("src/tasks/blcs/models/model.py",),
        "shared_patterns": ("src/utils", "AGENTS.md"),
    }
    base.update(overrides)
    return P.evaluate_gate(P.GateInput(**base))


def test_gate_merge_when_review_and_merge_conditions_pass() -> None:
    result = _gate()
    assert result.decision is P.GateDecision.MERGE
    assert result.reasons == ()
    assert P.gate_should_comment(result)


def test_gate_needs_human_for_shared_paths() -> None:
    result = _gate(changed_paths=("src/tasks/blcs/a.py", "src/utils/io.py"))
    assert result.decision is P.GateDecision.NEEDS_HUMAN
    assert result.reasons == ("touches_shared",)
    assert result.shared_hits == (("src/utils/io.py", "src/utils"),)


@pytest.mark.parametrize(
    ("overrides", "reason", "comment"),
    [
        ({"is_draft": True}, "draft", False),
        ({"is_cross_repository": True}, "fork", False),
        ({"mergeable": "CONFLICTING"}, "conflict", True),
        ({"mergeable": "UNKNOWN"}, "mergeable_unknown", False),
        ({"review": None}, "review_missing", False),
        (
            {"review": P.ReviewOutcome("request_changes", 1)},
            "review_request_changes",
            False,
        ),
    ],
)
def test_gate_blocked_reasons(
    overrides: dict[str, Any], reason: str, comment: bool
) -> None:
    # Shared paths are also touched: blocking reasons take precedence over needs_human.
    result = _gate(changed_paths=("AGENTS.md",), **overrides)
    assert result.decision is P.GateDecision.BLOCKED
    assert result.reasons == (reason,)
    assert P.gate_should_comment(result) is comment


def test_gate_rejects_unknown_states() -> None:
    with pytest.raises(ValueError):
        _gate(mergeable="MAYBE")
