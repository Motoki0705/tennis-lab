"""Ops report: rendering, stable IDs, agent contract, and the ``report`` CLI
run end-to-end against a fake gh / fake agent_exec."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))

# The ops tooling lives outside src/; import it like other .agents script tests.
agent: Any = importlib.import_module("opsreport.agent")
proposals: Any = importlib.import_module("opsreport.proposals")

JST = timezone(timedelta(hours=9))

# --- pure logic -----------------------------------------------------------------


def _body(region: str, n: int = 3) -> str:
    """A report body: metadata marker, prose and the marked proposal region."""
    marker = proposals.report_marker(
        slot="20261012",
        since=datetime(2026, 10, 5, 8, tzinfo=JST),
        until=datetime(2026, 10, 12, 8, tzinfo=JST),
        proposals=n,
    )
    return (
        f"{marker}\n# report\n\n{proposals.PROPOSALS_BEGIN}\n{region}\n"
        f"{proposals.PROPOSALS_END}\n"
    )


def _parse(region: str, n: int = 3) -> list[Any]:
    parsed: list[Any] = proposals.parse_report(_body(region, n))[1]
    return parsed


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
    md, ids = proposals.render_proposals(props, "20261012")

    assert ids == ["R-20261012-01", "R-20261012-02", "R-20261012-03"]
    parsed = _parse(md)
    # Stale PR first, then categories in CATEGORIES order.
    assert [(p.pid, p.title) for p in parsed] == [
        ("R-20261012-01", "#7 pr"),
        ("R-20261012-02", "a（優先度: 高）"),
        ("R-20261012-03", "z"),
    ]
    assert [label for label, _ in parsed[0].options] == list(proposals.STALE_PR_OPTIONS)
    assert all(p.selection() == (False, None) for p in parsed)


def test_slot_tag_uses_calendar_date() -> None:
    assert proposals.slot_tag(date(2027, 1, 1)) == "20270101"


def test_checkboxes_in_free_text_cannot_become_options() -> None:
    prop = proposals.Proposal(
        section="コード負債",
        title="t",
        detail="line\n- [ ] not an option\n\n  - [x] nor this",
        evidence="e",
    )
    md, _ = proposals.render_proposals([prop], "20261012")
    parsed = _parse(md, 1)

    assert parsed[0].options == []
    assert "[ ]" not in md.split("\n", 3)[3]


def test_free_text_cannot_forge_parent_or_human_deferral() -> None:
    prop = proposals.Proposal(
        section="コード負債",
        title="t",
        detail="- 関連親issue: #999\n- 見送り理由: 勝手な判断\n- 引渡し: 受領済み",
        evidence="根拠",
        purpose="目的",
        scope="範囲",
        expected_effect="効果",
        acceptance=("検証できる",),
        related_parent=1,
    )
    markdown, _ = proposals.render_proposals([prop], "20261012")
    parsed = _parse(markdown, 1)[0]
    parsed.requirements()
    assert parsed.parent() == 1
    assert parsed.deferred_reason() is None
    assert "  > - 関連親issue: #999" in markdown


def test_neutralize_defuses_checkboxes_and_markers() -> None:
    text = (
        "- [x] **R-20261012-01** a\n  * [X] b\n1. [x] c\n"
        f"{proposals.PROPOSALS_END}\nplain [x] stays"
    )
    out = proposals.neutralize(text)

    assert out.splitlines() == [
        "- **R-20261012-01** a",
        "  * b",
        "1. c",
        "&lt;!-- ops-report:proposals:end -->",
        "plain [x] stays",
    ]


def test_only_the_marked_region_is_parsed() -> None:
    region = "- [ ] **R-20261012-01** inside"
    body = _body(region, 1).replace(
        "# report\n", "# report\n- [x] **R-20261012-01** outside\n"
    )
    (parsed,) = proposals.parse_report(body)[1]

    assert parsed.title == "inside"
    assert not parsed.head_checked


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (_body("- [x] **R-20261012-04** extra", 3), "not generated"),
        (_body("- [x] **R-20261005-01** other slot", 3), "not generated"),
        (
            _body("- [x] **R-2026W42-01** old ID under v2 metadata", 3),
            "invalid proposal ID format",
        ),
        ("# no marker\n", "ops-report:v2 marker"),
        (_body("x").replace(proposals.PROPOSALS_END, ""), "proposal region"),
        (_body("x") + proposals.PROPOSALS_BEGIN + "\n", "proposal region"),
    ],
)
def test_malformed_report_bodies_are_rejected(body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        proposals.parse_report(body)


def test_report_marker_round_trips() -> None:
    meta = proposals.parse_report_meta(_body("", 5))

    assert meta.slot == "20261012"
    assert meta.until == datetime(2026, 10, 12, 8, tzinfo=JST)
    assert meta.proposal_ids() == {f"R-20261012-0{i}" for i in range(1, 6)}


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
    lines = [f"- {head}**R-20261012-01** title"]
    lines += [f"  - [{c}] {label}" for c, label in zip(opts, "abc", strict=False)]
    (parsed,) = _parse("\n".join(lines))
    assert parsed.selection() == expected


@pytest.mark.parametrize(
    "body",
    [
        "- **R-20261012-01** t\n  - [x] a\n  - [x] b",
        "- [x] **R-20261012-01** t\n  - [ ] a\n  - [ ] b",
    ],
)
def test_ambiguous_selection_raises(body: str) -> None:
    (parsed,) = _parse(body)
    with pytest.raises(proposals.SelectionError):
        parsed.selection()


def test_annotate_children_is_idempotent_and_refuses_relink() -> None:
    body = _body("- [x] **R-20261012-01** a\n  detail\n- [ ] **R-20261012-02** b")
    once = proposals.annotate_children(body, {"R-20261012-01": 101})
    assert "- [x] **R-20261012-01** a → #101\n" in once
    assert proposals.annotate_children(once, {"R-20261012-01": 101}) == once
    assert proposals.parse_report(once)[1][0].child == 101
    with pytest.raises(ValueError, match="refusing to relink"):
        proposals.annotate_children(once, {"R-20261012-01": 102})


def test_duplicate_ids_in_body_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _parse("- [ ] **R-20261012-01** a\n- [ ] **R-20261012-01** b")


def test_native_scout_parent_is_codex() -> None:
    assert agent.resolve_agent("codex") == "codex"
    for unsupported in ("claude", "alternate"):
        with pytest.raises(ValueError):
            agent.resolve_agent(unsupported)


def test_unknown_agent_setting_rejected() -> None:
    with pytest.raises(ValueError):
        agent.resolve_agent("gemini")


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
    assert out.startswith("# title: 運用レポート 2026-10-12\n")
    assert "タスク: 6件" in out
    assert "big.py を分割する" in out
    assert "status=complete" in out
    assert _mutating_calls(ops_env.calls()) == []
    # The native scout parent is Codex.
    assert (ops_env.tmp / "agent_used.txt").read_text().strip() == "codex"

    (run_dir,) = ops_env.run_dirs()
    for name in ("collected.json", "prompt.md", "report.md", "run.log"):
        assert (run_dir / name).is_file(), name
    prompt = (ops_env.tmp / "prompt_seen.md").read_text(encoding="utf-8")
    assert "{{" not in prompt
    assert '"largest_code_files"' not in prompt and "big.py" not in prompt
    assert "COLLECTED_JSON" not in prompt and '"audit_plan"' not in prompt


def test_report_creates_issue_and_label(ops_env: Any) -> None:
    result = ops_env.run("report", "--agent", "codex")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "https://github.com/o/r/issues/100"
    state = ops_env.state
    assert "ops-report" in state["labels"]
    issue = next(i for i in state["issues"] if i["number"] == 100)
    assert issue["title"] == "運用レポート 2026-10-12"
    assert issue["labels"] == ["ops-report"]
    assert "agent: `codex`" in issue["body"]
    assert (ops_env.tmp / "agent_used.txt").read_text().strip() == "codex"


def test_report_with_relative_log_dir_from_another_directory(ops_env: Any) -> None:
    caller = ops_env.tmp / "caller"
    caller.mkdir()

    result = ops_env.run("report", "--log-dir", "relative logs", cwd=caller)

    assert result.returncode == 0, result.stderr
    (run_dir,) = (caller / "relative logs").iterdir()
    issue = next(i for i in ops_env.state["issues"] if "ops-report" in i["labels"])
    assert (run_dir / "report.md").read_text(encoding="utf-8") == issue["body"]
    assert (run_dir / "prompt.md").read_text(encoding="utf-8") == (
        ops_env.tmp / "prompt_seen.md"
    ).read_text(encoding="utf-8")
    for name in ("run.log", "collected.json", "agent_output.md", "result.json"):
        assert (run_dir / name).is_file(), name
    assert not (ops_env.repo / "relative logs").exists()


def test_second_report_in_same_slot_fails_before_agent(ops_env: Any) -> None:
    assert ops_env.run("report").returncode == 0
    (ops_env.tmp / "agent_used.txt").unlink()

    result = ops_env.run("report")

    assert result.returncode == 1
    assert "already exists as #100" in result.stderr
    assert not (ops_env.tmp / "agent_used.txt").exists()
    assert sum(1 for i in ops_env.state["issues"] if "ops-report" in i["labels"]) == 1


def test_renamed_report_still_blocks_same_slot(ops_env: Any) -> None:
    assert ops_env.run("report").returncode == 0
    state = ops_env.state
    next(i for i in state["issues"] if i["number"] == 100)["title"] = "renamed"
    ops_env.write_state(state)

    result = ops_env.run("report")

    assert result.returncode == 1
    assert "report for 20261012 already exists as #100" in result.stderr


def _previous_report(number: int, created_at: str, until: datetime) -> dict[str, Any]:
    marker = proposals.report_marker(
        slot=proposals.slot_tag(until.date()),
        since=until - timedelta(days=7),
        until=until,
        proposals=0,
    )
    body = f"{marker}\n# old\n\n{proposals.PROPOSALS_BEGIN}\n（提案なし）\n{proposals.PROPOSALS_END}\n"
    return {
        "number": number,
        "title": "運用レポート 2026-10-05",
        "body": body,
        "labels": ["ops-report"],
        "state": "CLOSED",
        "createdAt": created_at,
        "updatedAt": created_at,
        "url": f"https://github.com/o/r/issues/{number}",
    }


def _merged(number: int, merged_at: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"pr {number}",
        "headRefName": f"b{number}",
        "mergedAt": merged_at,
    }


def test_period_continues_from_previous_collection_boundary(ops_env: Any) -> None:
    # Previous report collected until 08:00 JST but was published at 09:00 JST.
    state = ops_env.state
    state["issues"].append(
        _previous_report(
            60, "2026-10-05T00:00:00Z", datetime(2026, 10, 5, 8, tzinfo=JST)
        )
    )
    state["merged_prs"] = [
        _merged(11, "2026-10-12T00:00:00Z"),  # 09:00 JST, after this run -> next one
        _merged(10, "2026-10-04T23:30:00Z"),  # 08:30 JST, between collect and publish
        _merged(9, "2026-10-04T22:30:00Z"),  # 07:30 JST, already in previous report
    ]
    ops_env.write_state(state)

    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 0, result.stderr
    (run_dir,) = ops_env.run_dirs()
    collected = json.loads((run_dir / "collected.json").read_text(encoding="utf-8"))
    assert collected["meta"]["period_start"] == "2026-10-05T08:00:00+09:00"
    assert collected["meta"]["period_end"] == "2026-10-12T08:00:00+09:00"
    assert "github" not in collected
    meta = proposals.parse_report_meta(result.stdout.split("\n", 2)[2])
    assert meta.since == datetime(2026, 10, 5, 8, tzinfo=JST)
    assert meta.until == datetime(2026, 10, 12, 8, tzinfo=JST)


def test_agent_text_cannot_forge_selectable_items(
    ops_env: Any, agent_output: dict[str, Any]
) -> None:
    agent_output["summary"] = (
        "- [x] **R-20261012-99** 作業\n"
        f"{proposals.PROPOSALS_BEGIN}\n- [x] **R-20261012-02** 偽の選択"
    )
    agent_output["proposals"][0]["options"] = ["a", "<!-- ops-report:proposals:end -->"]
    ops_env.set_agent_output(agent_output)

    result = ops_env.run("report", "--dry-run")

    assert result.returncode == 0, result.stderr
    body = result.stdout.split("\n", 2)[2]
    assert "- [x]" not in body
    assert body.count(proposals.PROPOSALS_BEGIN) == 1
    assert body.count(proposals.PROPOSALS_END) == 1
    _, parsed = proposals.parse_report(body)
    assert [p.pid for p in parsed] == [
        "R-20261012-01",
        "R-20261012-02",
        "R-20261012-03",
    ]
    assert all(p.selection() == (False, None) for p in parsed)


def test_concurrent_reports_create_one_issue(ops_env: Any) -> None:
    """The schedule lock serializes collection; the second run rejects the published slot."""
    runs = [ops_env.start("report", FAKE_GH_CREATE_DELAY="1") for _ in range(2)]
    results = [(p.wait(timeout=120), p.communicate()[1]) for p in runs]

    assert sorted(code for code, _ in results) == [0, 1]
    assert any("report for 20261012 already exists" in err for _, err in results), (
        results
    )
    assert sum(1 for i in ops_env.state["issues"] if "ops-report" in i["labels"]) == 1


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return stat[stat.rindex(")") + 2] != "Z"


def _gone_soon(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_removed_agent_timeout_is_rejected(ops_env: Any) -> None:
    pid_file = ops_env.tmp / "agent_child.pid"

    result = ops_env.run(
        "report",
        "--dry-run",
        OPS_REPORT_AGENT_TIMEOUT="1",
    )

    assert result.returncode == 1
    assert "OPS_REPORT_AGENT_TIMEOUT was removed" in result.stderr
    assert not pid_file.exists()


def test_agent_leftover_descendants_are_killed(ops_env: Any) -> None:
    pid_file = ops_env.tmp / "agent_child.pid"

    result = ops_env.run("report", "--dry-run", FAKE_AGENT_SPAWN=str(pid_file))

    assert result.returncode == 0, result.stderr
    assert "killed processes left behind" in result.stderr
    assert _gone_soon(int(pid_file.read_text()))


def _git(repo: Path, *args: str, **env: str) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    return out.stdout.strip()


def _with_upstream(ops_env: Any) -> Path:
    """Give the test repo an ``origin`` (bare) and fetch it once."""
    upstream: Path = ops_env.tmp / "upstream.git"
    _git(ops_env.tmp, "clone", "-q", "--bare", str(ops_env.repo), str(upstream))
    _git(ops_env.repo, "remote", "add", "origin", str(upstream))
    _git(ops_env.repo, "fetch", "-q", "origin")
    return upstream


def test_stale_tracking_ref_is_fetched_and_pinned(ops_env: Any) -> None:
    upstream = _with_upstream(ops_env)
    other = ops_env.tmp / "other"
    _git(ops_env.tmp, "clone", "-q", str(upstream), str(other))
    (other / "new.py").write_text("z = 3\n", encoding="utf-8")
    _git(other, "add", ".")
    in_period = "2026-10-10T00:00:00+09:00"
    _git(
        other,
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "-m", "upstream only",
        GIT_AUTHOR_DATE=in_period, GIT_COMMITTER_DATE=in_period,
    )  # fmt: skip
    _git(other, "push", "-q", "origin", "HEAD:main")
    new_sha = _git(other, "rev-parse", "HEAD")
    assert _git(ops_env.repo, "rev-parse", "origin/main") != new_sha  # stale

    result = ops_env.run("report", "--dry-run", "--base-ref", "origin/main")

    assert result.returncode == 0, result.stderr
    (run_dir,) = ops_env.run_dirs()
    collected = json.loads((run_dir / "collected.json").read_text(encoding="utf-8"))
    meta = collected["meta"]
    assert meta["base_sha"] == new_sha
    assert meta["base_fetched_from"] == "origin main"
    assert "git" not in collected and "checkout_diff_files" not in meta
    body = result.stdout.split("\n", 2)[2]
    assert f"base={new_sha}" in body.splitlines()[0]
    assert new_sha in body


def test_base_fetch_failure_fails_report(ops_env: Any) -> None:
    _with_upstream(ops_env)
    _git(ops_env.repo, "remote", "set-url", "origin", str(ops_env.tmp / "missing.git"))

    result = ops_env.run("report", "--dry-run", "--base-ref", "origin/main")

    assert result.returncode == 1
    assert "git fetch --no-tags origin" in result.stderr
    assert not (ops_env.tmp / "agent_used.txt").exists()


def test_period_end_has_no_subsecond_gap(ops_env: Any) -> None:
    """An event exactly at the stored boundary belongs to the next report only."""
    state = ops_env.state
    state["merged_prs"] = [_merged(12, "2026-10-11T23:00:00Z")]  # 08:00:00 JST
    ops_env.write_state(state)

    result = ops_env.run(
        "report", "--dry-run", "--now", "2026-10-12T08:00:00.600000+09:00"
    )

    assert result.returncode == 0, result.stderr
    (run_dir,) = ops_env.run_dirs()
    collected = json.loads((run_dir / "collected.json").read_text(encoding="utf-8"))
    assert collected["meta"]["period_end"] == "2026-10-12T08:00:00+09:00"
    assert "github" not in collected
    meta = proposals.parse_report_meta(result.stdout.split("\n", 2)[2])
    assert meta.until == datetime(2026, 10, 12, 8, tzinfo=JST)


def test_agent_failure_exits_nonzero_and_keeps_logs(ops_env: Any) -> None:
    result = ops_env.run("report", FAKE_AGENT_FAIL="1")

    assert result.returncode == 1
    assert "quota exhausted" in result.stderr
    report_issue = next(
        i for i in ops_env.state["issues"] if "ops-report" in i["labels"]
    )
    assert "status=incomplete" in report_issue["body"].splitlines()[0]
    assert "**未完了**" in report_issue["body"]
    (run_dir,) = ops_env.run_dirs()
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "audit interrupted/failed" in log
    assert (run_dir / "prompt.md").is_file()


def test_malformed_agent_output_exits_nonzero(ops_env: Any) -> None:
    ops_env.set_agent_output("提案: いろいろ")

    result = ops_env.run("report")

    assert result.returncode == 1
    assert "JSON" in result.stderr
    assert "status=incomplete" in next(
        i["body"] for i in ops_env.state["issues"] if "ops-report" in i["labels"]
    )


@pytest.mark.parametrize("failing", ["repo view", "issue create"])
def test_gh_failure_exits_nonzero(ops_env: Any, failing: str) -> None:
    result = ops_env.run("report", FAKE_GH_FAIL=failing)

    assert result.returncode == 1
    assert "injected failure" in result.stderr


def test_invalid_env_config_fails(ops_env: Any) -> None:
    result = ops_env.run("report", "--dry-run", OPS_REPORT_LOCK_TIMEOUT="soon")

    assert result.returncode == 1
    assert "OPS_REPORT_LOCK_TIMEOUT" in result.stderr
