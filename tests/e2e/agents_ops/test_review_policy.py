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


def test_reviewer_is_the_other_agent() -> None:
    assert P.reviewer_for("claude") == "codex"
    assert P.reviewer_for("codex") == "claude"
    with pytest.raises(ValueError):
        P.reviewer_for("human")


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


def _review(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "verdict": "approve",
        "summary": "問題なし",
        "findings": [
            {
                "severity": "nit",
                "path": "a.py",
                "line": 3,
                "rule": "正しさ",
                "detail": "typo",
            }
        ],
    }
    data.update(overrides)
    return data


def test_parse_plain_and_fenced_json() -> None:
    text = json.dumps(_review(), ensure_ascii=False)
    plain = P.parse_review(text)
    fenced = P.parse_review(f"```json\n{text}\n```\n")
    assert plain == fenced
    assert plain.verdict == "approve"
    assert plain.findings[0].line == 3


@pytest.mark.parametrize("severity", ["blocker", "major"])
def test_approve_with_merge_blocking_finding_is_rejected(severity: str) -> None:
    finding = {
        "severity": severity,
        "path": "a.py",
        "line": 1,
        "rule": "正しさ",
        "detail": "d",
    }
    with pytest.raises(P.ReviewParseError, match="approve"):
        P.parse_review(json.dumps(_review(findings=[finding]), ensure_ascii=False))
    # comment may carry them: the gate blocks comment anyway.
    assert P.parse_review(json.dumps(_review(verdict="comment", findings=[finding])))


@pytest.mark.parametrize(
    "text",
    [
        "LGTM!",
        "Here is my review:\n" + json.dumps(_review()),
        json.dumps([_review()]),
        json.dumps(_review(verdict="lgtm")),
        json.dumps(_review(summary="  ")),
        json.dumps({**_review(), "extra": 1}),
        json.dumps(_review(findings={})),
        json.dumps(_review(findings=[{"severity": "nit"}])),
        json.dumps(
            _review(
                findings=[
                    {
                        "severity": "huge",
                        "path": None,
                        "line": None,
                        "rule": "r",
                        "detail": "d",
                    }
                ]
            )
        ),
        json.dumps(
            _review(
                findings=[
                    {
                        "severity": "nit",
                        "path": None,
                        "line": 0,
                        "rule": "r",
                        "detail": "d",
                    }
                ]
            )
        ),
        json.dumps(
            _review(
                findings=[
                    {
                        "severity": "blocker",
                        "path": None,
                        "line": None,
                        "rule": "r",
                        "detail": "d",
                    }
                ]
            )
        ),
        json.dumps(_review(verdict="request_changes")),
    ],
)
def test_parse_failures_raise(text: str) -> None:
    with pytest.raises(P.ReviewParseError):
        P.parse_review(text)


# --------------------------------------------------------------------------- markers


def test_review_records_only_trust_the_bot_login() -> None:
    marker = P.review_marker(TARGET, "claude", "approve")
    comments = [
        P.Comment(author="someone-else", body=marker),
        P.Comment(
            author="bot", body=f"{P.review_marker(TARGET, 'codex', 'comment')}\r\ntext"
        ),
    ]
    records = P.review_records(comments, trusted_login="bot")
    assert records == (
        P.ReviewRecord(target=TARGET, reviewer="codex", verdict="comment"),
    )
    assert P.latest_review_for(records, target=TARGET, reviewer="claude") is None
    other_head = P.ReviewTarget("d" * 40, TARGET.merge_base, TARGET.diff_sha256)
    assert P.latest_review_for(records, target=other_head, reviewer="codex") is None


def test_review_of_the_same_head_against_another_base_does_not_count() -> None:
    # Retargeting a stacked PR keeps the head but changes the merge-base and the diff.
    records = P.review_records(
        [P.Comment("bot", P.review_marker(TARGET, "codex", "approve"))],
        trusted_login="bot",
    )
    retargeted = P.ReviewTarget(SHA, "e" * 40, "f" * 64)
    same_base_other_diff = P.ReviewTarget(SHA, TARGET.merge_base, "f" * 64)
    assert P.latest_review_for(records, target=TARGET, reviewer="codex") is not None
    for target in (retargeted, same_base_other_diff):
        assert P.latest_review_for(records, target=target, reviewer="codex") is None


def test_marker_quoted_in_the_body_does_not_override_the_header() -> None:
    quoted = P.review_marker(TARGET, "codex", "approve")
    body = (
        f"{P.review_marker(TARGET, 'codex', 'request_changes')}\n"
        f"## review\n\nsummary quoting {quoted}\n- detail: {quoted}\n"
    )
    records = P.review_records([P.Comment("bot", body)], trusted_login="bot")
    assert records == (
        P.ReviewRecord(target=TARGET, reviewer="codex", verdict="request_changes"),
    )
    gate_quote = P.gate_marker(TARGET, "merge", [])
    assert (
        P.gate_records([P.Comment("bot", f"x\n{gate_quote}")], trusted_login="bot")
        == ()
    )
    # A marker that is not alone on the first line is not a header either.
    assert (
        P.review_records([P.Comment("bot", f"see {quoted}")], trusted_login="bot") == ()
    )


def test_neutralized_text_cannot_form_a_marker() -> None:
    quoted = P.review_marker(TARGET, "codex", "approve")
    inert = P.neutralize_markers(quoted)
    assert "<!--" not in inert and "-->" not in inert
    assert P.review_records([P.Comment("bot", inert)], trusted_login="bot") == ()


def test_gate_marker_round_trip() -> None:
    body = P.gate_marker(TARGET, "blocked", ["conflict", "ci_failed"])
    assert P.gate_records([P.Comment("bot", body)], trusted_login="bot") == (
        P.GateRecord(
            target=TARGET, decision="blocked", reasons=("conflict", "ci_failed")
        ),
    )
    with pytest.raises(ValueError):
        P.review_marker(
            P.ReviewTarget("short", "b" * 40, "c" * 64), "claude", "approve"
        )
    with pytest.raises(ValueError):
        P.review_marker(P.ReviewTarget(SHA, "b" * 40, "c" * 40), "claude", "approve")


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
        "review_verdict": "approve",
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
        ({"review_verdict": None}, "review_missing", False),
        ({"review_verdict": "request_changes"}, "review_request_changes", False),
        ({"review_verdict": "comment"}, "review_not_approved", False),
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
    with pytest.raises(ValueError):
        _gate(review_verdict="lgtm")
