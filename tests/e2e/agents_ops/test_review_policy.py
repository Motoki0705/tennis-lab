"""Pure policy of the cross-review loop: authorship, path matching, parsing, gate."""

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
TARGET = P.ReviewTarget(head=SHA, merge_base="b" * 40, diff_sha256="c" * 64)


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
        ".agents/memory/INDEX.md",
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
        ".agents/memory/INDEX.md",
        ".agents/memory/wsl2-host-ram-silent-kill.md",
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


# --------------------------------------------------------------------------- markers


def _finding(severity: str = "major", detail: str = "d") -> Any:
    return P.Finding(
        severity=severity, path="a.py", line=3, rule="正しさ", detail=detail
    )


def _outcome(verdict: str, blocking: int = 0) -> Any:
    return P.ReviewOutcome(verdict, blocking)


def _comment(body: str, author: str = "bot", n: int = 1) -> Any:
    return P.Comment(author=author, body=body, url=f"https://x.invalid/c/{n}")


def _record(verdict: str, blocking: int = 0, *, target: Any = TARGET) -> Any:
    [record] = P.review_records(
        [_comment(P.review_marker(target, "codex", _outcome(verdict, blocking)))],
        trusted_login="bot",
    )
    return record


def test_review_records_only_trust_the_bot_login() -> None:
    comments = [
        _comment(P.review_marker(TARGET, "codex", _outcome("approve")), "someone-else"),
        _comment(
            f"{P.review_marker(TARGET, 'codex', _outcome('request_changes', 2))}\r\ntext"
        ),
    ]
    records = P.review_records(comments, trusted_login="bot")
    assert records == (
        P.ReviewRecord(
            target=TARGET,
            reviewer="codex",
            outcome=P.ReviewOutcome("request_changes", 2),
            url="https://x.invalid/c/1",
        ),
    )
    assert P.latest_review_for(records, target=TARGET, reviewer="claude") is None
    other_head = P.ReviewTarget("d" * 40, TARGET.merge_base, TARGET.diff_sha256)
    assert P.latest_review_for(records, target=other_head, reviewer="codex") is None


def test_review_of_the_same_head_against_another_base_does_not_count() -> None:
    # Retargeting a stacked PR keeps the head but changes the merge-base and the diff.
    records = (_record("approve"),)
    retargeted = P.ReviewTarget(SHA, "e" * 40, "f" * 64)
    same_base_other_diff = P.ReviewTarget(SHA, TARGET.merge_base, "f" * 64)
    assert P.latest_review_for(records, target=TARGET, reviewer="codex") is not None
    for target in (retargeted, same_base_other_diff):
        assert P.latest_review_for(records, target=target, reviewer="codex") is None


def test_reviews_under_the_old_v3_rules_are_not_read() -> None:
    # v3 reviews could pass with minor/nit findings; such PRs are reviewed again.
    v3 = (
        f"<!-- agent-cross-review:v3 {TARGET.marker_fields()} reviewer=codex "
        "verdict=comment blocking=0 -->\n<!-- agent-cross-review-light:v1 [] -->"
    )
    assert P.review_records([_comment(v3)], trusted_login="bot") == ()


@pytest.mark.parametrize(
    ("verdict", "blocking", "passed"),
    [("approve", 0, True), ("request_changes", 1, False)],
)
def test_review_outcome_passes(verdict: str, blocking: int, passed: bool) -> None:
    assert _outcome(verdict, blocking).passed is passed


def test_review_outcome_rejects_contradictions() -> None:
    for verdict, blocking in [
        ("approve", 1),
        ("request_changes", 0),
        ("comment", 0),
        ("lgtm", 0),
    ]:
        with pytest.raises(ValueError):
            P.ReviewOutcome(verdict, blocking)


def test_marker_quoted_in_the_body_does_not_override_the_header() -> None:
    quoted = P.review_marker(TARGET, "codex", _outcome("approve"))
    body = (
        f"{P.review_marker(TARGET, 'codex', _outcome('request_changes', 1))}\n"
        f"## review\n\nsummary quoting {quoted}\n- detail: {quoted}\n"
    )
    [record] = P.review_records([_comment(body)], trusted_login="bot")
    assert record.outcome.verdict == "request_changes"
    gate_quote = P.gate_marker(TARGET, "merge", [])
    assert P.gate_records([_comment(f"x\n{gate_quote}")], trusted_login="bot") == ()
    # A marker that is not alone on the first line is not a header either.
    assert P.review_records([_comment(f"see {quoted}")], trusted_login="bot") == ()


def test_neutralized_text_cannot_form_a_marker() -> None:
    quoted = P.review_marker(TARGET, "codex", _outcome("approve"))
    inert = P.neutralize_markers(quoted)
    assert "<!--" not in inert and "-->" not in inert
    assert P.review_records([_comment(inert)], trusted_login="bot") == ()


def test_gate_marker_round_trip() -> None:
    body = P.gate_marker(TARGET, "blocked", ["conflict", "ci_failed"])
    assert P.gate_records([_comment(body)], trusted_login="bot") == (
        P.GateRecord(
            target=TARGET, decision="blocked", reasons=("conflict", "ci_failed")
        ),
    )
    for bad in (
        P.ReviewTarget("short", "b" * 40, "c" * 64),
        P.ReviewTarget(SHA, "b" * 40, "c" * 40),
    ):
        with pytest.raises(ValueError):
            P.review_marker(bad, "codex", _outcome("approve"))


# --------------------------------------------------------------------------- convergence


def _t(i: int) -> Any:
    return P.ReviewTarget(f"{i:040x}", "b" * 40, "c" * 64)


def test_request_changes_streak_spans_heads_and_resets_on_approve() -> None:
    rc = [_record("request_changes", 1, target=_t(i)) for i in range(4)]
    passing = _record("approve", target=_t(10))
    other = P.ReviewRecord(
        _t(12), "claude", P.ReviewOutcome("approve", 0), "https://x.invalid"
    )

    def streak(*records: Any) -> list[str]:
        found = P.request_changes_streak(records, reviewer="codex")
        return [r.target.head for r in found]

    assert streak(rc[0], rc[1], rc[2]) == [_t(i).head for i in range(3)]
    assert streak(rc[0], passing, rc[1], rc[2]) == [_t(1).head, _t(2).head]
    # Reviews by another reviewer are ignored.
    assert len(streak(rc[0], rc[1], other, rc[2])) == 3
    assert streak(rc[0], rc[1], rc[2], passing) == []


def test_escalation_markers_round_trip() -> None:
    marker = P.escalation_marker(_t(1))
    comments = [_comment(marker), _comment(P.escalation_marker(_t(2)), "intruder")]
    assert P.escalated_streaks(comments, trusted_login="bot") == {_t(1)}


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
        "checks": (P.Check("Python", P.CheckState.SUCCESS),),
        "required_checks": ("Python",),
        "review": P.ReviewOutcome("approve", 0),
        "changed_paths": ("src/tasks/blcs/models/model.py",),
        "shared_patterns": ("src/utils", "AGENTS.md"),
    }
    base.update(overrides)
    return P.evaluate_gate(P.GateInput(**base))


def test_gate_merge_when_everything_is_green() -> None:
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
        ({"checks": (P.Check("Python", P.CheckState.FAILURE),)}, "ci_failed", True),
        ({"checks": (P.Check("Python", P.CheckState.PENDING),)}, "ci_pending", False),
        (
            {"checks": (P.Check("label", P.CheckState.SUCCESS),)},
            "ci_missing_required",
            False,
        ),
        ({"checks": ()}, "ci_missing_required", False),
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
