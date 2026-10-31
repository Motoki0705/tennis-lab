"""Content contracts and the workflow CLI, using executable agy/gh fakes only."""

from __future__ import annotations

import functools
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / ".agents/ops"), str(ROOT / ".agents/ops/lib")]
cli: Any = importlib.import_module("issue_rewrite.cli")
pipeline: Any = importlib.import_module("issue_rewrite.pipeline")
invariants: Any = importlib.import_module("issue_rewrite.invariants")
github: Any = importlib.import_module("issue_rewrite.github")
publication: Any = importlib.import_module("issue_rewrite.publication")
issue_format: Any = importlib.import_module("issues.format")
policy: Any = importlib.import_module("parent.policy")
judgements: Any = importlib.import_module("parent.judgement_comments")


@pytest.fixture(scope="module")
def original() -> str:
    return str(
        issue_format.render_parent_body(
            {
                "目的・背景": (
                    "#17 を https://example.invalid/guide で確認する。@reviewer と 2.5 秒を比較する。\n\n"
                    "> 人間の引用はそのまま。\n> 続きも保つ。\n\n"
                    "<!-- metadata: a  b -->\n\n"
                    "```python\n## 見出しの例\nvalue = 17\n```\n\n"
                    "インラインは `model_v2` と `` `nested` ``。\n\n"
                    "1. 準備する。\n2. 確認する。"
                ),
                "全体のスコープ": "- [ ] 推論を確認する。\n- [x] 記録を保存する。",
                "全体方針・合意事項": "要件合意はまだない。",
                "全体の完了条件": "- [ ] 出力の内容を確認する。",
                "進捗概要": "未着手。",
                "依存関係・進め方": "先行依存なし。",
            }
        )
    )


@pytest.mark.parametrize(
    ("old", "new", "rule"),
    [
        ("## 全体のスコープ", "## 別のスコープ", "required_sections"),
        ("未着手。", "開始済み。", "managed_section"),
        ("> 続きも保つ。", "> 続きは変える。", "quotes"),
        ("metadata: a  b", "metadata: a b", "html_comments"),
        ("value = 17", "value=17", "fenced_code"),
        ("`model_v2`", "`model_v3`", "inline_code"),
        ("#17 を", "#18 を", "references"),
        ("/guide で", "/other で", "urls"),
        ("@reviewer と", "@other と", "mentions"),
        ("2.5 秒", "2.6 秒", "numbers"),
        ("- [ ] 推論", "- [x] 推論", "checkboxes"),
        ("- [ ] 推論", "- [ ] 新しい項目\n- [ ] 推論", "checkboxes"),
        ("確認する。@reviewer", "確認する。#17 @reviewer", "references"),
    ],
)
def test_each_invariant_rejects_a_concrete_violation(
    original: str, old: str, new: str, rule: str
) -> None:
    errors = invariants.check_invariants(original, original.replace(old, new, 1))
    assert any(error["rule"] == rule and error["detail"] for error in errors), errors


def test_required_order_and_body_validation_are_independent(original: str) -> None:
    swapped = (
        original.replace("## 全体のスコープ", "## temporary")
        .replace("## 全体方針・合意事項", "## 全体のスコープ")
        .replace("## temporary", "## 全体方針・合意事項")
    )
    assert any(
        x["rule"] == "required_sections"
        for x in invariants.check_invariants(original, swapped)
    )
    invalid = original.replace("未分割（子issue未作成）", "broken child row")
    errors = invariants.check_invariants(original, invalid)
    assert any(
        x["rule"] == "validate_body" and "child checklist" in x["detail"]
        for x in errors
    )


def test_rephrasing_ordered_list_renumbering_and_nonrequired_subheadings_pass(
    original: str,
) -> None:
    candidate = (
        original.replace("確認する。@reviewer", "分かりやすく確認する。@reviewer")
        .replace("1. 準備", "9. 準備")
        .replace("2. 確認", "10. 確認")
        .replace("先行依存なし。", "### 進め方\n\n先行依存はない。")
    )
    assert invariants.check_invariants(original, candidate) == []


def test_protected_fragments_and_managed_section_whitespace_are_exact(
    original: str,
) -> None:
    fragments = invariants.protected(original)
    assert fragments["inline_code"] == ["`model_v2`", "`` `nested` ``"]
    assert fragments["fenced_code"] == ["```python\n## 見出しの例\nvalue = 17\n```\n"]
    candidate = original.replace("\n\n未着手。", "\n未着手。")
    assert any(
        x["rule"] == "managed_section"
        for x in invariants.check_invariants(original, candidate)
    )


@pytest.mark.parametrize(
    "token",
    [
        "ghp_" + "a" * 36,
        "ghs_" + "a" * 36,
        "github_pat_" + "a" * 82,
        "ya29." + "a" * 60,
        "1//" + "a" * 50,
        "1/" + "a" * 40,
    ],
)
def test_github_and_google_token_shapes_are_rejected(token: str) -> None:
    assert "token_like_output" in publication.publication_errors("本文 " + token)


@pytest.mark.parametrize(
    "text",
    [
        "conversation_id=private-conversation",
        '"session_id": "private-conversation"',
        "session ID: private-conversation",
    ],
)
def test_public_diagnostics_redact_the_session_value_as_well_as_its_key(
    text: str,
) -> None:
    assert "private-conversation" not in publication.public_text(text)


class Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def sleep(self, seconds: float) -> None:
        self.seconds += seconds


@pytest.mark.parametrize("changing", [False, True])
def test_stability_observes_a_full_minute_or_fails_after_ten_minutes(
    changing: bool,
) -> None:
    clock = Clock()
    client = SimpleNamespace(
        issue=lambda _: {
            "body": str(
                int(clock.seconds // 10) if changing else int(clock.seconds >= 10)
            )
        }
    )
    if changing:
        with pytest.raises(ValueError, match="body_stability_timeout"):
            pipeline.stable_issue(client, 1, clock=clock.now, sleep=clock.sleep)
        assert clock.seconds == 600
    else:
        assert pipeline.stable_issue(client, 1, clock=clock.now, sleep=clock.sleep) == {
            "body": "1"
        }
        assert clock.seconds == 70


def test_stability_deadline_includes_time_spent_reading_github() -> None:
    clock = Clock()

    def issue(_: int) -> dict[str, str]:
        if clock.seconds:
            clock.seconds += 601
        return {"body": "unchanged"}

    with pytest.raises(ValueError, match="body_stability_timeout"):
        pipeline.stable_issue(
            SimpleNamespace(issue=issue), 1, clock=clock.now, sleep=clock.sleep
        )


@dataclass
class RewriteEnv:
    root: Path
    original: str
    rewritten: str
    agy: Path

    def state(self) -> dict[str, Any]:
        return json.loads((self.root / "github.json").read_text())  # type: ignore[no-any-return]

    def save(self, state: dict[str, Any]) -> None:
        (self.root / "github.json").write_text(json.dumps(state))

    def turns(self, *outputs: dict[str, Any]) -> None:
        (self.root / "turns.json").write_text(
            json.dumps([{"structured_output": output} for output in outputs])
        )

    def writer(self, body: str | None = None) -> dict[str, Any]:
        return {
            "body": body if body is not None else self.rewritten,
            "notes": "private-log conversation_id=private-conversation /home/private-person/token.json",
        }

    def run(self, command: str, *, attempt: int = 1, manual: bool = False) -> int:
        args = [
            command,
            "--repo",
            "o/r",
            "--enabled-since",
            self.state()["enabled_since"]
            if self.state().get("variable_status") != 404
            else "",
            "--issue-number",
            "1",
            "--attempt",
            str(attempt),
            "--bundle",
            str(self.root / f"handoff-{attempt}.json"),
            "--summary",
            str(self.root / "summary.md"),
        ]
        if command == "rewrite":
            args += [
                "--run-dir",
                str(self.root / f"private-{attempt}"),
                "--agy",
                str(self.agy),
            ]
        if manual:
            args += ["--manual"]
        return int(cli.main(args))

    def bundle(self, attempt: int = 1) -> dict[str, Any]:
        return json.loads((self.root / f"handoff-{attempt}.json").read_text())  # type: ignore[no-any-return]

    def calls(self, name: str) -> list[Any]:
        path = self.root / f"{name}-calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def mutations(self) -> list[Any]:
        return [
            call
            for call in self.calls("gh")
            if call["method"] != "GET" and call["endpoint"] != "graphql"
        ]


PASS = {"verdict": "pass", "differences": []}


@pytest.fixture
def rewrite_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, original: str
) -> RewriteEnv:
    for name, source in (("agy", "fake_rewrite_agy.py"), ("gh", "fake_rewrite_gh.py")):
        executable = tmp_path / name
        executable.write_text(
            f"#!{sys.executable}\n"
            + Path(__file__).with_name(source).read_text().split("\n", 1)[1]
        )
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("REWRITE_FAKE_ROOT", str(tmp_path))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    clock = Clock()
    monkeypatch.setattr(
        pipeline,
        "stable_issue",
        functools.partial(pipeline.stable_issue, clock=clock.now, sleep=clock.sleep),
    )
    env = RewriteEnv(
        tmp_path,
        original,
        original.replace("確認する。@reviewer", "分かりやすく確認する。@reviewer"),
        tmp_path / "agy",
    )
    env.save(
        {
            "enabled_since": "2026-10-30T00:00:00Z",
            "comments": [],
            "edits": [],
            "issue": {
                "id": 10,
                "number": 1,
                "repository_url": "https://api.github.com/repos/o/r",
                "title": "[書き直し] 本文を確認する",
                "body": original,
                "labels": [{"name": "issue:parent"}, {"name": "enhancement"}],
                "created_at": "2026-10-31T00:00:00Z",
                "user": {"login": "o"},
                "state": "open",
            },
        }
    )
    env.turns(env.writer(), PASS)
    return env


def labels(env: RewriteEnv) -> set[str]:
    return {item["name"] for item in env.state()["issue"]["labels"]}


def test_success_applies_only_verified_body_and_publishes_allowlisted_data(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    assert env.run("rewrite") == 0
    assert env.state()["issue"]["body"] == env.original and env.mutations() == []
    assert env.run("apply") == 0
    assert env.state()["issue"]["body"] == env.rewritten
    assert labels(env) == {"issue:parent", "enhancement", "rewrite:done"}
    assert len(env.calls("agy")) == 2
    assert all("--resume" not in call for call in env.calls("agy"))
    summary = (env.root / "summary.md").read_text()
    exported = (env.root / "handoff-1.json").read_text()
    for private in (
        "private-conversation",
        "private-log",
        "/home/private-person",
        "notes",
        "conversation_id",
        str(env.root),
    ):
        assert private not in summary + exported
    assert '"total_tokens": 120' in exported
    assert (env.root / "private-1/input.json").is_file()
    assert (env.root / "private-1").stat().st_mode & 0o777 == 0o700
    assert env.mutations()[0]["payload"] == {"body": env.rewritten}
    assert len(env.state()["edits"]) == 2  # creation version plus our first edit
    assert not any("actions/variables" in call["endpoint"] for call in env.calls("gh"))


@pytest.mark.parametrize("failure", ["invariant", "verifier"])
def test_a_single_repair_receives_the_actual_failure_and_can_succeed(
    rewrite_env: RewriteEnv, failure: str
) -> None:
    env = rewrite_env
    if failure == "invariant":
        env.turns(
            env.writer(env.rewritten.replace("#17 を", "#18 を")), env.writer(), PASS
        )
        expected = 3
    else:
        rejected = {
            "verdict": "fail",
            "differences": [
                {
                    "kind": "addition",
                    "original": "",
                    "rewritten": "学習も対象。",
                    "explanation": "学習が追加された。",
                }
            ],
        }
        env.turns(
            env.writer(
                env.rewritten.replace("先行依存なし。", "先行依存なし。学習も対象。")
            ),
            rejected,
            env.writer(),
            PASS,
        )
        expected = 4
    assert env.run("rewrite") == 0
    assert len(env.calls("agy")) == expected
    last_writer = (env.root / "private-1/writer-2.md").read_text()
    assert "previous_body" in last_writer and "violations" in last_writer
    assert (
        "references" if failure == "invariant" else "学習が追加された。"
    ) in last_writer
    assert env.run("apply") == 0 and "rewrite:done" in labels(env)


@pytest.mark.parametrize(
    "failure", ["invariant", "verifier", "contradictory_verdict", "empty_rejection"]
)
def test_exhausted_repairs_keep_original_body_and_record_failure(
    rewrite_env: RewriteEnv, failure: str
) -> None:
    env = rewrite_env
    if failure == "invariant":
        env.turns(
            env.writer(env.rewritten.replace("2.5 秒", "3.5 秒")),
            env.writer(env.rewritten.replace("2.5 秒", "4.5 秒")),
        )
        expected = 2
    else:
        difference = {
            "kind": "change",
            "original": "先行依存なし。",
            "rewritten": "先行依存なし。",
            "explanation": "private-log conversation_id=private-conversation",
        }
        rejected = {
            "verdict": "pass" if failure == "contradictory_verdict" else "fail",
            "differences": [] if failure == "empty_rejection" else [difference],
        }
        env.turns(env.writer(), rejected, env.writer(), rejected)
        expected = 4
    assert env.run("rewrite") == 1
    assert env.bundle()["status"] == "failed" and env.bundle()["body"] is None
    assert env.run("fail") == 0
    assert env.state()["issue"]["body"] == env.original and "rewrite:failed" in labels(
        env
    )
    assert all("body" not in call["payload"] for call in env.mutations())
    assert len(env.calls("agy")) == expected
    assert "repair_exhausted" in (env.root / "summary.md").read_text()
    assert "private-conversation" not in (env.root / "handoff-1.json").read_text()


@pytest.mark.parametrize(
    "bad_turn",
    [{"status": "SUCCESS"}, {"structured_output": {"body": "missing notes"}}],
)
def test_missing_structured_output_or_bad_schema_never_reaches_verifier(
    rewrite_env: RewriteEnv, bad_turn: dict[str, Any]
) -> None:
    env = rewrite_env
    (env.root / "turns.json").write_text(json.dumps([bad_turn]))
    assert env.run("rewrite") == 1
    assert len(env.calls("agy")) == 1
    assert env.run("fail") == 0 and "rewrite:failed" in labels(env)
    assert env.state()["issue"]["body"] == env.original


@pytest.mark.parametrize(
    "unsafe",
    [
        "ghs_" + "a" * 36,
        "ya29." + "a" * 60,
        "/home/private-person/token.json",
        "conversation_id=private-conversation",
    ],
)
def test_unsafe_candidate_is_never_exported_even_on_failure(
    rewrite_env: RewriteEnv, unsafe: str
) -> None:
    env = rewrite_env
    env.turns(
        env.writer(env.rewritten.replace("先行依存なし。", "先行依存なし。" + unsafe))
    )
    assert env.run("rewrite") == 1
    exported = (env.root / "handoff-1.json").read_text()
    assert unsafe not in exported and env.bundle()["source"] is None
    assert unsafe not in (env.root / "summary.md").read_text()
    assert env.run("fail") == 0 and "rewrite:failed" in labels(env)
    assert env.state()["issue"]["body"] == env.original


def comment(body: str, *, login: str = "o") -> dict[str, Any]:
    return {"id": 99, "user": {"login": login}, "body": body}


def agreement() -> str:
    value = {
        key: "合意した内容"
        for key in ("purpose", "scope", "behavior", "acceptance", "human_confirmation")
    }
    return str(
        policy.marker(
            {
                "parent": 1,
                "event": "agreed",
                "revision": 1,
                "agreement": value,
                "agreement_sha256": policy.digest(value),
            }
        )
    )


def execution_basis() -> str:
    inputs = {
        "schema_version": 1,
        "repo": "o/r",
        "parent_issue": 1,
        "issue": {},
        "comments": [],
        "context": {},
    }
    basis = {
        "schema_version": 1,
        "kind": "autonomous_decision",
        "purpose": "目的",
        "scope": "範囲",
        "behavior": "動作",
        "acceptance": "検証",
        "rationale": "着手可能",
        "sources": ["https://github.com/o/r/issues/1"],
        "input_sha256": policy.digest(inputs),
        "policy_version": "autonomous-start/v1",
    }
    return str(
        policy.marker(
            {
                "parent": 1,
                "event": "execution_basis",
                "basis": basis,
                "basis_sha256": policy.digest(basis),
                "generation": 1,
                "revision": 1,
                "inputs": inputs,
                "ignored_comments": [],
            }
        )
    )


def judgement(disposition: str = "startable") -> str:
    return str(
        judgements.comment_body(
            "o/r",
            1,
            {
                "disposition": disposition,
                "rationale": "根拠",
                "facts": ["確認済み"],
                "missing_information": [],
                "resume_condition": "確認済み",
                "next_action": "着手する",
                "sources": ["https://github.com/o/r/issues/1"],
                "target_parent": 1,
            },
        )
    )


@pytest.mark.parametrize("record", [agreement(), execution_basis(), judgement()])
def test_existing_execution_records_skip_without_agent_calls_or_body_writes(
    rewrite_env: RewriteEnv, record: str
) -> None:
    env = rewrite_env
    state = env.state()
    state["comments"] = [comment(record)]
    env.save(state)
    assert env.run("rewrite") == 0 and env.bundle()["status"] == "skipped"
    assert env.calls("agy") == []
    assert env.run("apply") == 0
    assert (
        "rewrite:skipped" in labels(env)
        and env.state()["issue"]["body"] == env.original
    )


@pytest.mark.parametrize(
    "record",
    [
        comment("引用\n" + agreement()),
        comment(agreement(), login="someone-else"),
        comment(judgement("human_hold")),
    ],
)
def test_marker_quotes_other_publishers_and_hold_are_not_execution_evidence(
    rewrite_env: RewriteEnv, record: dict[str, Any]
) -> None:
    env = rewrite_env
    state = env.state()
    state["comments"] = [record]
    env.save(state)
    assert env.run("rewrite") == 0 and env.bundle()["status"] == "ready"
    assert len(env.calls("agy")) == 2


def test_corrupt_owned_judgement_fails_closed_instead_of_using_substring_detection(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    state = env.state()
    state["comments"] = [comment(judgement() + "\n後から加えた指示")]
    env.save(state)
    assert env.run("rewrite") == 1 and env.calls("agy") == []
    assert env.mutations() == [] and env.state()["issue"]["body"] == env.original
    assert env.run("fail") == 0 and "rewrite:failed" in labels(env)
    assert "invalid_execution_record" in (env.root / "summary.md").read_text()


@pytest.mark.parametrize(
    "change",
    ["label", "created_before_activation", "disabled", "done", "failed", "skipped"],
)
def test_non_targets_and_already_final_automatic_runs_do_nothing(
    rewrite_env: RewriteEnv, change: str
) -> None:
    env = rewrite_env
    state = env.state()
    if change == "label":
        state["issue"]["labels"] = [{"name": "issue:child"}]
    elif change == "created_before_activation":
        state["issue"]["created_at"] = "2026-10-29T00:00:00Z"
    elif change == "disabled":
        state["variable_status"] = 404
    else:
        state["issue"]["labels"].append({"name": "rewrite:" + change})
    env.save(state)
    assert env.run("rewrite") == 0 and env.bundle()["status"] == "not_target"
    assert env.run("apply") == 0
    assert not env.calls("agy") and not env.mutations()


def test_manual_retry_of_failed_issue_replaces_only_the_final_label(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    state = env.state()
    state["issue"]["labels"].append({"name": "rewrite:failed"})
    env.save(state)
    assert env.run("rewrite", manual=True) == 0
    assert env.run("apply", manual=True) == 0
    assert labels(env) == {"issue:parent", "enhancement", "rewrite:done"}


def test_invalid_activation_is_never_treated_as_disabled(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    state = env.state()
    state["enabled_since"] = "invalid-activation-time"
    env.save(state)
    assert env.run("rewrite") == 1 and env.bundle()["status"] == "failed"
    assert not env.calls("agy") and not env.mutations()


def test_agreement_added_during_generation_is_rechecked_before_apply(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    assert env.run("rewrite") == 0
    state = env.state()
    state["comments"] = [comment(agreement())]
    env.save(state)
    assert env.run("apply") == 0 and "rewrite:skipped" in labels(env)
    assert env.state()["issue"]["body"] == env.original


def test_body_conflict_dispatches_latest_input_once_then_exhausts(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    assert env.run("rewrite") == 0
    state = env.state()
    latest = env.original.replace("先行依存なし。", "先行する依存はない。")
    state["issue"]["body"] = latest
    env.save(state)
    assert env.run("apply") == 0
    assert env.state()["dispatches"] == [
        {"ref": "main", "inputs": {"issue_number": "1", "attempt": "2"}}
    ]
    assert not any("body" in call["payload"] for call in env.mutations())
    assert labels(env) == {"issue:parent", "enhancement"}
    (env.root / "agy-calls.jsonl").unlink()
    env.turns(
        env.writer(
            latest.replace("確認する。@reviewer", "分かりやすく確認する。@reviewer")
        ),
        PASS,
    )
    assert env.run("rewrite", attempt=2, manual=True) == 0
    assert env.bundle(2)["source"]["body"] == latest
    state = env.state()
    state["issue"]["body"] += "\n人間がさらに編集した。"
    env.save(state)
    assert env.run("apply", attempt=2, manual=True) == 1
    assert env.run("fail", attempt=2) == 0
    assert len(env.state()["dispatches"]) == 1 and "rewrite:failed" in labels(env)
    assert "人間がさらに編集した。" in env.state()["issue"]["body"]
    assert "body_conflict_retry_exhausted" in (env.root / "summary.md").read_text()


@pytest.mark.parametrize(
    "interference", ["before", "after", "same_author", "missing_history"]
)
def test_interference_between_read_and_patch_is_failure_without_rollback(
    rewrite_env: RewriteEnv, interference: str
) -> None:
    env = rewrite_env
    assert env.run("rewrite") == 0
    state = env.state()
    state["interference"] = interference
    env.save(state)
    assert env.run("apply") == 1
    assert env.run("fail") == 0 and "rewrite:failed" in labels(env)
    assert "interrupted_during_write" in (env.root / "summary.md").read_text()
    assert len([call for call in env.mutations() if "body" in call["payload"]]) == 1
    if interference == "after":
        assert env.state()["issue"]["body"] == "human's latest body"


def test_history_detects_aba_even_when_the_body_matches_and_pages_all_edits(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    state = env.state()
    state["edits"] = [
        {
            "id": f"prior-{i}",
            "editedAt": "2026-10-31T00:00:00Z",
            "deletedAt": None,
            "editor": {"login": "o"},
        }
        for i in range(101)
    ]
    env.save(state)
    assert env.run("rewrite") == 0
    assert len(env.bundle()["source"]["history"]["edits"]) == 101
    state = env.state()
    state["edits"].append(
        {
            "id": "aba",
            "editedAt": "2026-10-31T00:00:01Z",
            "deletedAt": None,
            "editor": {"login": "o"},
        }
    )
    env.save(state)
    assert env.run("apply") == 0 and len(env.state()["dispatches"]) == 1
    assert not any("body" in call["payload"] for call in env.mutations())


def test_failure_job_can_finalize_when_the_generation_artifact_is_missing(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    assert env.run("fail") == 0
    assert (
        "rewrite:failed" in labels(env) and env.state()["issue"]["body"] == env.original
    )
    assert "workflow_job_failed" in (env.root / "summary.md").read_text()


def test_failure_job_rejects_a_corrupt_artifact_and_still_finalizes_the_original(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    unsafe = "ghs_" + "a" * 36
    (env.root / "handoff-1.json").write_text(unsafe)
    assert env.run("fail") == 0 and "rewrite:failed" in labels(env)
    report = (env.root / "summary.md").read_text()
    assert unsafe not in report and "invalid_failure_artifact" in report
    assert env.state()["issue"]["body"] == env.original


def test_record_added_during_stability_wait_prevents_any_model_usage(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    state = env.state()
    state["mutation"] = {"read": 3, "comments": [comment(agreement())]}
    env.save(state)
    assert env.run("rewrite") == 0 and env.bundle()["status"] == "skipped"
    assert not env.calls("agy") and not env.mutations()


@pytest.mark.parametrize("final", ["done", "skipped"])
def test_failure_job_preserves_a_concurrently_finalized_issue(
    rewrite_env: RewriteEnv, final: str
) -> None:
    env = rewrite_env
    state = env.state()
    state["issue"]["labels"].append({"name": "rewrite:" + final})
    env.save(state)
    assert env.run("fail") == 0
    assert "rewrite:failed" not in labels(env) and not env.mutations()


@pytest.mark.parametrize(
    "broken", ["author", "timestamp", "ambiguous", "old_edit_changed"]
)
def test_first_edit_creation_record_is_identified_strictly(broken: str) -> None:
    before: dict[str, Any] = {
        "body": "original",
        "viewer": "bot",
        "author": "owner",
        "created_at": "2026-10-31T00:00:00Z",
        "includes_created_edit": False,
        "edits": {},
    }
    creation = {
        "id": "created",
        "editedAt": "2026-10-31T00:00:00Z",
        "editor": {"login": "owner"},
        "deletedAt": None,
    }
    edit = {
        "id": "ours",
        "editedAt": "2026-10-31T00:01:00Z",
        "editor": {"login": "bot"},
        "deletedAt": None,
    }
    after = {
        **before,
        "body": "candidate",
        "includes_created_edit": True,
        "edits": {"created": creation, "ours": edit},
    }
    assert github.uninterrupted(before, after, "candidate")
    if broken == "author":
        creation["editor"] = {"login": "someone-else"}
    elif broken == "timestamp":
        creation["editedAt"] = "2026-10-31T00:00:01Z"
    elif broken == "ambiguous":
        after["edits"]["another-created"] = {**creation, "id": "another-created"}
    else:
        before["includes_created_edit"] = True
        before["edits"] = {"created": {**creation, "deletedAt": "2026-10-31T00:00:02Z"}}
    assert not github.uninterrupted(before, after, "candidate")


def test_apply_revalidates_the_artifact_instead_of_trusting_its_ready_status(
    rewrite_env: RewriteEnv,
) -> None:
    env = rewrite_env
    assert env.run("rewrite") == 0
    value = env.bundle()
    value["body"] = env.rewritten.replace("#17 を", "#18 を")
    (env.root / "handoff-1.json").write_text(json.dumps(value))
    assert env.run("apply") == 1 and env.mutations() == []
    assert env.state()["issue"]["body"] == env.original


def test_workflow_guards_and_permission_boundary_are_structural() -> None:
    # BaseLoader preserves the YAML key 'on' as a string.
    workflow = yaml.load(
        (ROOT / ".github/workflows/issue-rewrite.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"issues", "workflow_dispatch"}
    assert workflow["on"]["issues"]["types"] == ["opened", "labeled"]
    assert (
        workflow["run-name"]
        == "issue-rewrite #${{ github.event.issue.number || inputs.issue_number }}"
    )
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    jobs = workflow["jobs"]
    rewrite = jobs["rewrite"]
    assert "github.actor == github.repository_owner" in rewrite["if"]
    assert "vars.ISSUE_REWRITE_ENABLED_SINCE != ''" in rewrite["if"]
    assert "github.event.label.name == 'issue:parent'" in rewrite["if"]
    assert "github.ref == 'refs/heads/main'" in rewrite["if"]
    assert rewrite["runs-on"] == ["self-hosted", "linux", "x64", "tennis-lab"]
    assert rewrite["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }
    assert rewrite["env"]["HOME"] == "/var/lib/tennis-lab-actions/home"
    assert rewrite["env"]["AGY_CLI_DISABLE_AUTO_UPDATE"] == "true"
    assert (
        workflow["env"]["ISSUE_REWRITE_ENABLED_SINCE"]
        == "${{ vars.ISSUE_REWRITE_ENABLED_SINCE }}"
    )
    for name in ("apply", "failed"):
        assert jobs[name]["runs-on"] == "ubuntu-latest"
        assert (
            jobs[name]["permissions"]["issues"]
            == jobs[name]["permissions"]["actions"]
            == "write"
        )
        assert (
            "HOME" not in jobs[name]["env"]
            and "AGY_CLI_DISABLE_AUTO_UPDATE" not in jobs[name]["env"]
        )
        assert all(
            "run_headless" not in step.get("run", "")
            and "--agy" not in step.get("run", "")
            for step in jobs[name]["steps"]
        )
    for job in jobs.values():
        checkout = next(
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["persist-credentials"] == "false"
        assert checkout["with"]["ref"] == "${{ github.sha }}"
    upload = next(
        step
        for step in rewrite["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["path"].endswith("/public/handoff.json")
    assert "*" not in upload["with"]["path"]
