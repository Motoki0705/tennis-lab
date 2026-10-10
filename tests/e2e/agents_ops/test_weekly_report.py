"""Weekly ops report: rendering, stable IDs, agent contract, and the ``report`` CLI
run end-to-end against a fake gh / fake agent_exec."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / ".agents/ops/weekly-report")
)

# The ops tooling lives outside src/; import it like other .agents script tests.
agent: Any = importlib.import_module("opsreport.agent")
proposals: Any = importlib.import_module("opsreport.proposals")

# --- pure logic -----------------------------------------------------------------


def test_ids_are_sequential_in_rendered_order() -> None:
    props = [
        proposals.Proposal(section="その他の助言", title="z", detail="d", evidence="e"),
        proposals.Proposal(
            section=proposals.STALE_PR_SECTION,
            title="#7 pr",
            options=proposals.STALE_PR_OPTIONS,
        ),
        proposals.Proposal(
            section="コード負債", title="a", detail="d", evidence="e", priority="high"
        ),
    ]
    md, ids = proposals.render_proposals(props, "2026W42")

    assert ids == ["R-2026W42-01", "R-2026W42-02", "R-2026W42-03"]
    parsed = proposals.parse_proposals(md)
    # Stale PR first, then categories in CATEGORIES order.
    assert [(p.pid, p.title) for p in parsed] == [
        ("R-2026W42-01", "#7 pr"),
        ("R-2026W42-02", "a（優先度: 高）"),
        ("R-2026W42-03", "z"),
    ]
    assert [label for label, _ in parsed[0].options] == list(proposals.STALE_PR_OPTIONS)
    assert all(p.selection() == (False, None) for p in parsed)


def test_week_tag_uses_iso_year() -> None:
    # 2027-01-01 belongs to ISO week 2026-W53.
    assert proposals.week_tag(date(2027, 1, 1)) == "2026W53"


def test_checkboxes_in_free_text_cannot_become_options() -> None:
    prop = proposals.Proposal(
        section="コード負債",
        title="t",
        detail="line\n- [ ] not an option\n\n  - [x] nor this",
        evidence="e",
    )
    md, _ = proposals.render_proposals([prop], "2026W42")
    parsed = proposals.parse_proposals(md)

    assert parsed[0].options == []
    assert "[ ]" not in md.split("\n", 3)[3]


@pytest.mark.parametrize(
    ("head", "opts", "expected"),
    [
        ("[x] ", [], (True, None)),
        ("[ ] ", [], (False, None)),
        ("", [" ", "x", " "], (True, "b")),
        ("", [" ", " ", " "], (False, None)),
    ],
)
def test_selection(
    head: str, opts: list[str], expected: tuple[bool, str | None]
) -> None:
    lines = [f"- {head}**R-2026W42-01** title"]
    lines += [f"  - [{c}] {label}" for c, label in zip(opts, "abc", strict=False)]
    (parsed,) = proposals.parse_proposals("\n".join(lines))
    assert parsed.selection() == expected


@pytest.mark.parametrize(
    "body",
    [
        "- **R-2026W42-01** t\n  - [x] a\n  - [x] b",
        "- [x] **R-2026W42-01** t\n  - [ ] a\n  - [ ] b",
    ],
)
def test_ambiguous_selection_raises(body: str) -> None:
    (parsed,) = proposals.parse_proposals(body)
    with pytest.raises(proposals.SelectionError):
        parsed.selection()


def test_annotate_children_is_idempotent_and_refuses_relink() -> None:
    body = "intro\n- [x] **R-2026W42-01** a\n  detail\n- [ ] **R-2026W42-02** b\n"
    once = proposals.annotate_children(body, {"R-2026W42-01": 101})
    assert "- [x] **R-2026W42-01** a → #101\n" in once
    assert proposals.annotate_children(once, {"R-2026W42-01": 101}) == once
    assert proposals.parse_proposals(once)[0].child == 101
    with pytest.raises(ValueError, match="refusing to relink"):
        proposals.annotate_children(once, {"R-2026W42-01": 102})


def test_duplicate_ids_in_body_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        proposals.parse_proposals(
            "- [ ] **R-2026W42-01** a\n- [ ] **R-2026W42-01** b\n"
        )


@pytest.mark.parametrize(
    ("day", "expected"),
    [(date(2026, 10, 12), "claude"), (date(2026, 10, 5), "codex")],  # W42 even, W41 odd
)
def test_alternate_agent_by_iso_week(day: date, expected: str) -> None:
    assert agent.resolve_agent("alternate", day) == expected
    assert agent.resolve_agent("codex", day) == "codex"


def test_unknown_agent_setting_rejected() -> None:
    with pytest.raises(ValueError):
        agent.resolve_agent("gemini", date(2026, 10, 12))


def test_agent_output_accepts_single_json_fence(agent_output: dict[str, Any]) -> None:
    text = (
        "以下が結果です。\n```json\n"
        + json.dumps(agent_output, ensure_ascii=False)
        + "\n```\n"
    )
    parsed = agent.parse_agent_output(text, {7})
    assert [p.title for p in parsed.proposals] == [
        "big.py を分割する",
        "週次レポートのプロンプトを改善する",
    ]
    assert parsed.stale_pr_notes == {7: "rebaseして仕上げを推奨。差分が小さい。"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["proposals"][0].update(category="misc"), "category"),
        (lambda d: d["proposals"][0].update(priority="urgent"), "priority"),
        (lambda d: d["proposals"][0].update(options=["only one"]), "options"),
        (lambda d: d["proposals"][0].pop("evidence"), "evidence"),
        (lambda d: d.update(extra=1), "unexpected top-level"),
        (lambda d: d["stale_pr_notes"].update({"999": "x"}), "non-stale PR"),
    ],
)
def test_agent_output_contract_violations_fail(
    agent_output: dict[str, Any], mutate: Any, message: str
) -> None:
    mutate(agent_output)
    with pytest.raises(agent.AgentOutputError, match=message):
        agent.parse_agent_output(json.dumps(agent_output), {7})


def test_agent_output_must_be_json() -> None:
    with pytest.raises(agent.AgentOutputError):
        agent.parse_agent_output("提案はありません。", set())


# --- report CLI end-to-end ----------------------------------------------------------


def _mutating_calls(calls: list[list[str]]) -> list[list[str]]:
    return [
        c
        for c in calls
        if c[:2] in (["issue", "create"], ["issue", "edit"], ["label", "create"])
    ]


def test_dry_run_prints_body_and_creates_nothing(ops_env: Any) -> None:
    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert out.startswith("# title: 週次運用レポート 2026-W42\n")
    # Stale PR #7 (22 days idle) becomes proposal 01 with the three options; #8 is fresh.
    assert "- **R-2026W42-01** #7 放置されたPR" in out
    assert "  - [ ] rebaseして仕上げ" in out
    assert "推奨: rebaseして仕上げを推奨。差分が小さい。" in out
    assert "#8 新しいPR" not in out
    assert "- [ ] **R-2026W42-02** big.py を分割する（優先度: 高）" in out
    assert "- **R-2026W42-03** 週次レポートのプロンプトを改善する（優先度: 低）" in out
    # Missing sibling modules are stated explicitly, never silently omitted.
    assert "掃除モジュール未導入" in out
    assert "共有memory未導入" in out
    assert "CI（期間内 1 run）: failure 1" in out
    assert _mutating_calls(ops_env.calls()) == []
    # alternate + ISO week 42 (even) -> claude
    assert (ops_env.tmp / "agent_used.txt").read_text().strip() == "claude"

    (run_dir,) = ops_env.run_dirs()
    for name in ("collected.json", "prompt.md", "report.md", "run.log"):
        assert (run_dir / name).is_file(), name
    prompt = (ops_env.tmp / "prompt_seen.md").read_text(encoding="utf-8")
    assert "{{" not in prompt
    assert '"largest_code_files"' in prompt and "big.py" in prompt


def test_cleanup_module_output_is_passed_to_agent(ops_env: Any) -> None:
    cleanup = ops_env.repo / ".agents/ops/cleanup"
    cleanup.mkdir(parents=True)
    (cleanup / "cleanup.py").write_text(
        "import json, sys\n"
        "assert sys.argv[1:3] == ['scan', '--report-json'], sys.argv\n"
        "with open(sys.argv[3], 'w') as f:\n"
        "    json.dump({'candidates': [{'worktree': 'stale-wt'}]}, f)\n",
        encoding="utf-8",
    )

    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 0, result.stderr
    prompt = (ops_env.tmp / "prompt_seen.md").read_text(encoding="utf-8")
    assert "stale-wt" in prompt
    assert "掃除モジュール: `" in result.stdout


def test_cleanup_module_failure_fails_report(ops_env: Any) -> None:
    cleanup = ops_env.repo / ".agents/ops/cleanup"
    cleanup.mkdir(parents=True)
    (cleanup / "cleanup.py").write_text(
        "import sys\nprint('boom', file=sys.stderr)\nsys.exit(4)\n", encoding="utf-8"
    )

    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 1
    assert "boom" in result.stderr
    assert not (ops_env.tmp / "agent_used.txt").exists()


def test_cleanup_dir_without_entry_point_fails_report(ops_env: Any) -> None:
    (ops_env.repo / ".agents/ops/cleanup").mkdir(parents=True)

    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 1
    assert "no cleanup.py" in result.stderr


def test_report_creates_issue_and_label(ops_env: Any) -> None:
    result = ops_env.run("report", "--agent", "codex")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "https://github.com/o/r/issues/100"
    state = ops_env.state
    assert "ops-report" in state["labels"]
    issue = next(i for i in state["issues"] if i["number"] == 100)
    assert issue["title"] == "週次運用レポート 2026-W42"
    assert issue["labels"] == ["ops-report"]
    assert "agent: `codex`" in issue["body"]
    assert (ops_env.tmp / "agent_used.txt").read_text().strip() == "codex"


def test_second_report_in_same_week_fails_before_agent(ops_env: Any) -> None:
    assert ops_env.run("report").returncode == 0
    (ops_env.tmp / "agent_used.txt").unlink()

    result = ops_env.run("report")

    assert result.returncode == 1
    assert "already exists as #100" in result.stderr
    assert not (ops_env.tmp / "agent_used.txt").exists()
    assert sum(1 for i in ops_env.state["issues"] if "ops-report" in i["labels"]) == 1


def test_agent_failure_exits_nonzero_and_keeps_logs(ops_env: Any) -> None:
    result = ops_env.run("report", FAKE_AGENT_FAIL="1")

    assert result.returncode == 1
    assert "exited 3" in result.stderr and "quota exhausted" in result.stderr
    assert _mutating_calls(ops_env.calls()) == []
    (run_dir,) = ops_env.run_dirs()
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "report failed" in log
    assert (run_dir / "prompt.md").is_file()


def test_malformed_agent_output_exits_nonzero(ops_env: Any) -> None:
    ops_env.set_agent_output("提案: いろいろ")

    result = ops_env.run("report")

    assert result.returncode == 1
    assert "JSON" in result.stderr
    assert _mutating_calls(ops_env.calls()) == []


@pytest.mark.parametrize("failing", ["pr list", "issue create"])
def test_gh_failure_exits_nonzero(ops_env: Any, failing: str) -> None:
    result = ops_env.run("report", FAKE_GH_FAIL=failing)

    assert result.returncode == 1
    assert "injected failure" in result.stderr


def test_invalid_env_config_fails(ops_env: Any) -> None:
    result = ops_env.run("report", "--dry-run", WEEKLY_REPORT_STALE_PR_DAYS="soon")

    assert result.returncode == 1
    assert "WEEKLY_REPORT_STALE_PR_DAYS" in result.stderr
