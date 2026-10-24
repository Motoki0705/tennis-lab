"""Adoption to issue publication, recovery and scheduled judgement, entirely offline."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(OPS / "ops-report"))
fmt: Any = importlib.import_module("issues")
org: Any = importlib.import_module("opsreport.organization")
apply: Any = importlib.import_module("opsreport.organization_apply")
props: Any = importlib.import_module("opsreport.proposals")
triage: Any = importlib.import_module("opsreport.triage")
state: Any = importlib.import_module("parent.state")
github: Any = importlib.import_module("parent.github")
policy: Any = importlib.import_module("parent.policy")


def raw_issue(n: int, **changes: Any) -> dict[str, Any]:
    return {
        "number": n,
        "id": n + 1000,
        "title": f"作業{n}",
        "body": "目的・範囲・完了条件。",
        "repository_url": "https://api.github.com/repos/o/r",
        "state": "open",
        "state_reason": None,
        "labels": [],
        "assignees": [],
        **changes,
    }


class FakeGitHub:
    repo = "o/r"
    rows = github.GitHub.rows
    issue_rows = github.GitHub.issue_rows
    issue = github.GitHub.issue
    comments = github.GitHub.comments
    pulls = github.GitHub.pulls
    identity = github.GitHub.identity
    family = github.GitHub.family
    fresh_reads = github.GitHub.fresh_reads

    def __init__(self) -> None:
        self.parent: int | None = None
        self._fresh = False
        self.issues: dict[int, dict[str, Any]] = {}
        self.parents: dict[int, int] = {}
        self.dependencies: dict[int, list[int]] = {}
        self.comment_rows: dict[int, list[dict[str, Any]]] = {}
        self.prs: list[dict[str, Any]] = []
        self.writes: list[tuple[str, str, dict[str, Any]]] = []
        self.fail = ""
        self.fail_after = ""
        self.on_write: Callable[[], None] | None = None
        self.on_get: Callable[[], None] | None = None

    def get(self, endpoint: str, *, missing_parent: bool = False) -> Any:
        if self.on_get:
            self.on_get()
        if self.fail and self.fail in endpoint:
            raise ValueError("injected GET failure")
        if endpoint == "user":
            return {"login": "operator"}
        path, _, query = endpoint.partition("?")
        params = parse_qs(query)
        page = int(params.get("page", ["1"])[0])

        def paged(rows: list[Any]) -> Any:
            return copy.deepcopy(rows[(page - 1) * 100 : page * 100])

        if path == "repos/o/r/issues":
            return paged(
                [
                    r
                    for _, r in sorted(self.issues.items())
                    if params.get("state") != ["open"] or r["state"] == "open"
                ]
            )
        if path == "repos/o/r/pulls":
            return paged(self.prs)
        if path.startswith("repos/o/r/pulls/"):
            return copy.deepcopy(
                next(p for p in self.prs if p["number"] == int(path.split("/")[-1]))
            )
        parts = path.removeprefix("repos/o/r/issues/").split("/")
        n = int(parts[0])
        if len(parts) == 1:
            return copy.deepcopy(self.issues[n])
        tail = "/".join(parts[1:])
        if tail == "timeline":
            return paged(
                [
                    {
                        "id": p["number"] + 10000,
                        "event": "cross-referenced",
                        "source": {
                            "issue": {
                                "id": p["number"] + 10000,
                                "number": p["number"],
                                "body": p.get("body", ""),
                                "repository_url": "https://api.github.com/repos/o/r",
                                "pull_request": {"url": "pr"},
                            }
                        },
                    }
                    for p in self.prs
                ]
            )
        if tail == "parent":
            assert missing_parent
            return (
                copy.deepcopy(self.issues[self.parents[n]])
                if n in self.parents
                else None
            )
        if tail == "sub_issues":
            return paged(
                [
                    self.issues[k]
                    for k, parent in sorted(self.parents.items())
                    if parent == n
                ]
            )
        if tail == "dependencies/blocked_by":
            return paged([self.issues[k] for k in self.dependencies.get(n, [])])
        if tail == "comments":
            return paged(self.comment_rows.get(n, []))
        raise AssertionError(endpoint)

    def write(self, method: str, endpoint: str, payload: dict[str, Any]) -> Any:
        self.writes.append((method, endpoint, copy.deepcopy(payload)))
        if method == "PATCH" and "/issues/comments/" in endpoint:
            cid = int(endpoint.split("/")[-1])
            result = next(
                c for rows in self.comment_rows.values() for c in rows if c["id"] == cid
            )
            result.update(payload)
        elif endpoint.endswith("/comments"):
            n = int(endpoint.split("/")[-2])
            comment = {
                "id": 1 + sum(len(rows) for rows in self.comment_rows.values()),
                "user": {"login": "operator"},
                "body": payload["body"],
            }
            self.comment_rows.setdefault(n, []).append(comment)
            result = comment
        elif method == "POST" and endpoint == "repos/o/r/issues":
            n = max(self.issues, default=0) + 1
            row = raw_issue(n, **payload)
            row["labels"] = [{"name": name} for name in payload["labels"]]
            self.issues[n] = row
            result = row
        else:
            match = re.fullmatch(r"repos/o/r/issues/(\d+)(/sub_issues)?", endpoint)
            assert match
            n = int(match[1])
            if match[2]:
                child = next(
                    k
                    for k, raw in self.issues.items()
                    if raw["id"] == payload["sub_issue_id"]
                )
                assert child not in self.parents
                self.parents[child] = n
                result = self.issues[child]
            else:
                self.issues[n].update(payload)
                result = self.issues[n]
        if self.on_write:
            self.on_write()
        if self.fail_after == endpoint:
            self.fail_after = ""
            raise ValueError("injected response loss after successful write")
        return copy.deepcopy(result)


def task_node(
    name: str = "split-module", role: str = "standalone", parent: Any = None
) -> dict[str, Any]:
    return {
        "key": name,
        "role": role,
        "issue": None,
        "parent": parent,
        "identifier": None,
        "summary": "責務を分けて検証する",
        "sections": {
            "このタスクの目的": "保守性を改善する。",
            "具体的な変更内容・対象範囲": "big.pyと回帰テスト。",
            "対象外": "timer・割当・mergeの起動。",
            "作業状況・停止事項": "未着手。",
            "Acceptance checklist": "- [ ] 既存の動作を維持する\n  文脈: 責務の分離。\n  下流への影響: 保守性。\n  検証方法・証拠: CPU回帰テスト。",
        },
        "dependencies": [],
    }


def parent_node(name: str = "improvements") -> dict[str, Any]:
    node = task_node(name, "parent")
    node["identifier"] = "保守性改善"
    node["sections"] = {
        s: f"{s}を提案の範囲で実現する。"
        for s in fmt.PARENT_REQUIRED
        if s != fmt.TASKS_HEADING
    }
    return node


def decision(
    pid: str = "100/R-20261019-01", target: str | None = "split-module"
) -> dict[str, Any]:
    return {
        "proposal": pid,
        "action": "issue" if target else "hold",
        "target": target,
        "reason": "単独の具体的な改善として扱う。"
        if target
        else "評価対象の選択が不足している。",
        "evidence": [f"issue:{pid.split('/')[0]}"],
        "resume_condition": "採用・評価対象・担当を再照合する。",
        "next_action": "予定回へ渡す。" if target else "レポートに評価対象を追記する。",
    }


def plan(
    nodes: list[Any] | None = None, decisions: list[Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input_sha256": "INPUT_DIGEST",
        "nodes": nodes if nodes is not None else [task_node()],
        "decisions": decisions if decisions is not None else [decision()],
    }


def start_decision(
    n: int,
    status: str = "startable",
    *,
    parent: int | None = None,
    reports: tuple[int, ...] = (),
) -> dict[str, Any]:
    ready = status == "startable"
    return {
        "issue": n,
        "disposition": status,
        "target_parent": None if status == "out_of_scope" else (parent or n),
        "priority": n if ready else None,
        "requirements": {
            "purpose": "保守性を改善",
            "scope": "big.pyとテスト",
            "behavior": "責務を分離",
            "acceptance": "CPUテスト成功",
        }
        if ready
        else None,
        "rationale": "固定した要件と出典を照合した。",
        "facts": ["採用状態と正式な親子関係を確認した。"],
        "missing_information": [],
        "resume_condition": "予定回で再照合",
        "next_action": "候補または管理先を返す",
        "sources": [
            f"issue:{k}" for k in sorted({n, *reports, *([parent] if parent else [])})
        ]
        + (["code:CODE_COMMIT:README.md"] if ready else []),
        "dependencies": [],
    }


class Harness:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp
        self.root = tmp / "repo"
        self.root.mkdir()
        self.root.joinpath("README.md").write_text("# fixture\n")
        for args in (
            ("init", "-q"),
            ("add", "."),
            (
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.com",
                "commit",
                "-qm",
                "fixture",
            ),
        ):
            subprocess.run(
                ["git", *args], cwd=self.root, capture_output=True, check=True
            )
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        codex = bin_dir / "codex"
        codex.write_text(
            f"#!{sys.executable}\n"
            + Path(__file__).with_name("fake_organization_cli.py").read_text()
        )
        codex.chmod(0o700)
        claude = bin_dir / "claude"
        claude.write_text(
            f"#!{sys.executable}\n"
            + Path(__file__).with_name("fake_judgement_cli.py").read_text()
        )
        claude.chmod(0o700)
        monkeypatch.setenv("PATH", str(bin_dir) + ":" + os.environ["PATH"])
        monkeypatch.setenv("CODEX_HOME", str(tmp / "codex-home"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp / "state"))
        monkeypatch.setenv("FAKE_ORGANIZATION_PLAN", str(tmp / "plan.json"))
        monkeypatch.setenv("FAKE_ORGANIZATION_CALLS", str(tmp / "calls.jsonl"))
        monkeypatch.setenv("FAKE_JUDGEMENT_RESPONSE", str(tmp / "judgement.json"))
        monkeypatch.setenv("FAKE_JUDGEMENT_CALLS", str(tmp / "judgement-calls.jsonl"))
        monkeypatch.delenv("OPS_REPORT_ORGANIZER_EXEC", raising=False)
        self.gh = FakeGitHub()
        self.run_number = 0
        self.response(plan())
        self.report()

    def response(self, value: Any) -> None:
        self.tmp.joinpath("plan.json").write_text(
            json.dumps(value, ensure_ascii=False)
            if not isinstance(value, str)
            else value
        )

    def report(
        self,
        n: int = 100,
        count: int = 1,
        selected: bool = True,
        status: str = "complete",
    ) -> None:
        proposals = [
            props.Proposal(
                "コード負債",
                f"責務を分割する{i}",
                detail="big.pyの責務を整理。",
                evidence="big.pyとテスト。",
                purpose="保守性を高める。",
                scope="big.pyとテスト。",
                expected_effect="変更しやすくなる。",
                acceptance=("回帰テスト成功。",),
            )
            for i in range(count)
        ]
        text, _ = props.render_proposals(proposals, "20261019")
        if selected:
            text = text.replace("- [ ] **", "- [x] **")
        marker = props.report_marker(
            slot="20261019",
            since=datetime.fromisoformat("2026-10-16T08:00:00+09:00"),
            until=datetime.fromisoformat("2026-10-19T08:00:00+09:00"),
            proposals=count,
            status=status,
        )
        self.gh.issues[n] = raw_issue(
            n,
            title="運用レポート",
            labels=[{"name": "ops-report"}],
            body=f"{marker}\n\n{props.PROPOSALS_BEGIN}\n{text}\n{props.PROPOSALS_END}\n",
        )

    def candidates(self) -> list[Any]:
        result: list[Any] = []
        for n, row in self.gh.issues.items():
            if {label["name"] for label in row["labels"]} != {"ops-report"} or row[
                "state"
            ] != "open":
                continue
            meta, parsed = props.parse_report(row["body"])
            if meta.status != "complete":
                continue
            result.extend((n, p) for p in parsed if p.selection()[0] or p.child)
        return result

    def run(self, *, dry_run: bool = False) -> Any:
        self.run_number += 1
        work_dir = self.tmp / f"run-{self.run_number}"
        work_dir.mkdir()
        result = triage.TriageResult()
        with state.Store(self.tmp / "state/parent").locked() as store:
            apply.run(
                gh=self.gh,
                store=store,
                root=self.root,
                work_dir=work_dir,
                candidates=self.candidates(),
                dry_run=dry_run,
                result=result,
                journal_path=self.tmp / "triage.json",
            )
        return result

    def calls(self) -> list[Any]:
        path = self.tmp / "calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def judge(self, decisions: list[dict[str, Any]], name: str) -> dict[str, Any]:
        self.tmp.joinpath("judgement.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_sha256": "INPUT_DIGEST",
                    "decisions": decisions,
                },
                ensure_ascii=False,
            )
        )
        judgement: Any = importlib.import_module("dispatcher.judgement")
        result: dict[str, Any] = judgement.run(
            self.gh,
            state.Store(self.tmp / "state/parent"),
            self.root,
            self.tmp / name,
            apply=True,
        )
        return result


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_parentless_adoption_creates_normal_standalone_once(harness: Harness) -> None:
    h = harness
    assert not h.run().errors
    task = h.gh.issues[101]
    assert {label["name"] for label in task["labels"]} == {"ai-proposed"}
    assert h.gh.parents == {}
    source = fmt.parse_proposal_source(task["body"])
    assert source == fmt.ProposalSource("R-20261019-01", 100)
    metadata = fmt.parse_task_metadata(task["body"])
    assert metadata["bindings"][0]["proposal_sha256"] == policy.digest(
        props.parse_report(h.gh.issues[100]["body"])[1][0].snapshot()
    )
    assert "→ #101" in h.gh.issues[100]["body"]
    assert "08:00・20:00" in h.gh.issues[100]["body"]
    assert "human_confirmation" not in task["body"]
    count = len(h.gh.writes)
    assert not h.run().errors
    assert len(h.calls()) == 1 and len(h.gh.writes) == count
    args = h.calls()[0]["args"]
    assert "gpt-6.1-sol" in args and 'model_reasoning_effort="max"' in args
    assert "read-only" in args and "--profile" not in args
    runtime = json.loads(
        h.tmp.joinpath("run-1/organization-runtime/result.json").read_text()
    )
    assert (
        runtime["task_success"] is None
        and runtime["runtime"]["effective"]["model"]["status"] == "observed"
    )


def test_multiple_proposals_share_parent_and_keep_distinct_index_anchors(
    harness: Harness,
) -> None:
    h = harness
    h.report(count=2)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="first"), decision("100/R-20261019-02", "second")],
        )
    )
    result = h.run()
    assert not result.errors
    assert h.gh.parents == {102: 101, 103: 101}
    assert not fmt.check_consistency(h.gh.family(101))
    assert fmt.index_proposals((n, r["body"]) for n, r in h.gh.issues.items()) == {
        (100, "R-20261019-01"): 102,
        (100, "R-20261019-02"): 103,
    }
    assert len(fmt.parse_task_metadata(h.gh.issues[101]["body"])["bindings"]) == 2
    assert not h.run().errors and len(h.calls()) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "repos/o/r/issues",
        "repos/o/r/issues/101/sub_issues",
        "repos/o/r/issues/101",
        "repos/o/r/issues/100",
    ],
)
def test_response_loss_at_every_publication_stage_recovers_without_duplicates(
    harness: Harness, endpoint: str
) -> None:
    h = harness
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="improvements")],
        )
    )
    h.gh.fail_after = endpoint
    try:
        first = h.run()
        errors = first.errors
    except ValueError as exc:
        errors = [str(exc)]
    assert any("response loss" in error for error in errors)
    assert not h.run().errors
    assert len(h.gh.issues) == 4 and len(h.calls()) == 1
    assert not fmt.check_consistency(h.gh.family(101))
    assert "→ #101" in h.gh.issues[100]["body"]


def test_information_hold_is_specific_and_unchanged_input_does_not_rerun_agent(
    harness: Harness,
) -> None:
    h = harness
    h.response(plan([], [decision(target=None)]))
    assert h.run().unassigned == ["R-20261019-01"]
    body = h.gh.issues[100]["body"]
    assert all(s in body for s in ("保留", "評価対象", "根拠", "再開条件", "次の対応"))
    writes = len(h.gh.writes)
    assert h.run().unassigned == ["R-20261019-01"]
    assert len(h.calls()) == 1 and len(h.gh.writes) == writes
    h.gh.issues[100]["body"] += "\n評価対象を追記した。\n"
    h.response(plan())
    assert not h.run().errors and 101 in h.gh.issues
    assert len(h.calls()) == 2


@pytest.mark.parametrize("status,selected", [("complete", False), ("incomplete", True)])
def test_unselected_or_incomplete_report_never_launches_agent(
    harness: Harness, status: str, selected: bool
) -> None:
    h = harness
    h.report(status=status, selected=selected)
    h.run()
    assert not h.calls() and not h.gh.writes


@pytest.mark.parametrize("scenario", ["wrong_model", "empty", "cli_error"])
def test_runtime_failure_has_no_github_side_effects(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("FAKE_ORGANIZATION_SCENARIO", scenario)
    with pytest.raises(ValueError, match="runner failed"):
        harness.run()
    assert len(harness.calls()) == 1 and not harness.gh.writes


@pytest.mark.parametrize(
    "change",
    ["digest", "unknown", "duplicate", "cycle", "missing", "evidence", "confirmation"],
)
def test_invalid_agent_plan_is_rejected_as_a_whole(
    harness: Harness, change: str
) -> None:
    value = plan()
    if change == "digest":
        value["input_sha256"] = "0" * 64
    elif change == "unknown":
        value["nodes"][0]["role"] = "launch-parent"
    elif change == "duplicate":
        value["decisions"].append(value["decisions"][0])
    elif change == "cycle":
        value["nodes"][0]["dependencies"] = ["split-module"]
    elif change == "missing":
        value["decisions"] = []
    elif change == "evidence":
        value["decisions"][0]["evidence"] = ["issue:999"]
    else:
        value["human_confirmation"] = "invented consent"
    harness.response(value)
    with pytest.raises(ValueError):
        harness.run()
    assert not harness.gh.writes


def test_dry_run_validates_plan_without_writing_issues_or_journal(
    harness: Harness,
) -> None:
    h = harness
    h.run(dry_run=True)
    assert (
        len(h.calls()) == 1
        and not h.gh.writes
        and not h.tmp.joinpath("triage.json").exists()
    )


def test_adds_child_to_unowned_existing_parent_and_preserves_old_requirements(
    harness: Harness,
) -> None:
    h = harness
    parent = parent_node()
    body = fmt.render_parent_body(parent["sections"])
    h.gh.issues[1] = raw_issue(
        1, title="[保守性改善] 既存の親", body=body, labels=[{"name": "issue:parent"}]
    )
    h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
        "関連親issue: 未指定", "関連親issue: #1"
    )
    h.response(plan([task_node(role="child", parent=1)]))
    assert not h.run().errors
    assert h.gh.parents == {101: 1}
    assert not fmt.check_consistency(h.gh.family(1))
    before = fmt.parse_sections(body)
    after = fmt.parse_sections(h.gh.issues[1]["body"])
    assert all(after[k] == v for k, v in before.items() if k != fmt.TASKS_HEADING)


@pytest.mark.parametrize("closed", [False, True])
def test_reuses_existing_work_without_reopening_renaming_or_reparenting(
    harness: Harness, closed: bool
) -> None:
    h = harness
    h.gh.issues[20] = raw_issue(
        20,
        title="既存の具体的な作業",
        state="closed" if closed else "open",
        state_reason="completed" if closed else None,
        assignees=[{"login": "historical-owner"}] if closed else [],
    )
    node = task_node(role="reuse")
    node.update(issue=20, summary="", sections={})
    h.response(plan([node]))
    assert not h.run().errors
    assert len(h.gh.issues) == 2
    assert h.gh.issues[20]["title"] == "既存の具体的な作業"
    assert h.gh.issues[20]["state"] == ("closed" if closed else "open")
    assert h.gh.issues[20]["labels"] == []
    assert h.gh.issues[20]["assignees"] == (
        [{"login": "historical-owner"}] if closed else []
    )
    h.gh.issues[20]["title"] = "改名後の作業"
    assert not h.run().errors and len(h.calls()) == 1
    assert "→ #20" in h.gh.issues[100]["body"]


@pytest.mark.parametrize("blocked", ["pr", "human"])
def test_completed_reuse_still_preserves_active_pr_and_explicit_hold_boundaries(
    harness: Harness, blocked: str
) -> None:
    h = harness
    h.gh.issues[20] = raw_issue(
        20,
        state="closed",
        state_reason="completed",
        assignees=[{"login": "historical-owner"}],
    )
    if blocked == "human":
        h.gh.issues[20]["labels"] = [{"name": "needs-human"}]
    else:
        h.gh.prs = [
            {
                "number": 2,
                "state": "open",
                "merged_at": None,
                "body": "Closes #20",
                "head": {"ref": "codex/existing", "sha": "a" * 40},
            }
        ]
    node = task_node(role="reuse")
    node.update(issue=20, summary="", sections={})
    h.response(plan([node]))
    with pytest.raises(ValueError, match="conflict"):
        h.run()
    assert not h.gh.writes


def test_created_issue_renamed_and_completed_after_lost_response_is_recovered(
    harness: Harness,
) -> None:
    h = harness
    h.gh.fail_after = "repos/o/r/issues"
    assert h.run().errors
    h.gh.issues[101].update(title="改名済み", state="closed", state_reason="completed")
    assert not h.run().errors
    assert len(h.gh.issues) == 2 and len(h.calls()) == 1
    assert h.gh.issues[101]["state"] == "closed"


def test_ambiguous_create_is_not_repeated_when_the_issue_cannot_be_found(
    harness: Harness,
) -> None:
    h = harness
    h.gh.fail_after = "repos/o/r/issues"
    assert h.run().errors
    created = h.gh.issues.pop(101)
    assert any("outcome is unknown" in error for error in h.run().errors)
    assert (
        len([w for w in h.gh.writes if w[0] == "POST" and w[1] == "repos/o/r/issues"])
        == 1
    )
    h.gh.issues[101] = created
    assert not h.run().errors and len(h.calls()) == 1


@pytest.mark.parametrize("change", ["adoption", "option", "body", "parent"])
def test_edit_while_agent_is_running_prevents_issue_creation(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    h = harness
    original = org.execute

    def execute(root: Path, run_dir: Path) -> Any:
        value = original(root, run_dir)
        body = h.gh.issues[100]["body"]
        if change == "adoption":
            body = body.replace("- [x] **", "- [ ] **")
        elif change == "option":
            body = body.replace("  - 根拠:", "  - [x] 新しい選択肢\n  - 根拠:")
        elif change == "body":
            body = body.replace("保守性を高める", "別の目的を採用する")
        else:
            body = body.replace("関連親issue: 未指定", "関連親issue: #1")
        h.gh.issues[100]["body"] = body
        return value

    monkeypatch.setattr(org, "execute", execute)
    with pytest.raises(ValueError, match="changed"):
        h.run()
    assert not h.gh.writes and len(h.gh.issues) == 1


@pytest.mark.parametrize("change", ["adoption", "body"])
def test_edit_after_create_records_reason_and_resumes_same_plan(
    harness: Harness, change: str
) -> None:
    h = harness
    original = h.gh.issues[100]["body"]

    def change_source() -> None:
        h.gh.on_write = None
        h.gh.issues[100]["body"] = (
            original.replace("- [x] **", "- [ ] **")
            if change == "adoption"
            else original.replace("保守性を高める", "別の目的を採用する")
        )

    h.gh.on_write = change_source
    assert h.run().errors
    assert "保留" in h.gh.issues[100]["body"]
    assert "再開条件" in h.gh.issues[100]["body"]
    assert "→ #101" not in h.gh.issues[100]["body"]
    assert len(h.gh.issues) == 2
    h.gh.issues[100]["body"] = original
    assert not h.run().errors and len(h.gh.issues) == 2
    assert len(h.calls()) == 1


def test_adoption_revocation_after_link_is_recorded_once_and_never_deletes_task(
    harness: Harness,
) -> None:
    h = harness
    h.run()
    h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace("- [x] **", "- [ ] **")
    assert h.run().errors
    writes = len(h.gh.writes)
    assert "保留" in h.gh.issues[100]["body"] and "→ #101" in h.gh.issues[100]["body"]
    assert h.run().errors
    assert len(h.gh.writes) == writes and len(h.gh.issues) == 2 and len(h.calls()) == 1


@pytest.mark.parametrize("conflict", ["assignee", "owner", "pr"])
def test_existing_owner_conflict_requires_reasoned_hold(
    harness: Harness, conflict: str
) -> None:
    h = harness
    h.gh.issues[1] = raw_issue(1)
    if conflict == "assignee":
        h.gh.issues[1]["assignees"] = [{"login": "person"}]
    elif conflict == "pr":
        h.gh.prs = [
            {
                "number": 2,
                "state": "open",
                "merged_at": None,
                "body": "Closes #1",
                "head": {"ref": "codex/one", "sha": "a" * 40},
            }
        ]
    else:
        with state.Store(h.tmp / "state/parent").locked() as store:
            store.parents["o/r#1"] = {
                "status": "active",
                "session": "claude:existing",
                "generation": 1,
                "tasks": {},
                "reviews": [],
            }
            store.save()
    node = task_node(role="reuse")
    node.update(issue=1, summary="", sections={})
    h.response(plan([node]))
    with pytest.raises(ValueError, match="conflict"):
        h.run()
    assert not h.gh.writes
    h.response(plan([], [decision(target=None)]))
    assert h.run().unassigned == ["R-20261019-01"]
    assert "再開条件" in h.gh.issues[100]["body"]


def test_all_pages_and_api_failures_are_not_replaced_by_empty_inputs(
    harness: Harness,
) -> None:
    h = harness
    for n in range(1, 205):
        if n != 100:
            h.gh.issues[n] = raw_issue(n, state="closed", state_reason="completed")
    h.run()
    fixed = json.loads(h.tmp.joinpath("run-1/organization-input.json").read_text())
    assert len(fixed["snapshot"]["issues"]) == 204
    assert h.gh.issues[205]["labels"] == [{"name": "ai-proposed"}]
    writes = len(h.gh.writes)
    h.gh.fail = "/dependencies/blocked_by"
    assert any("GET failure" in error for error in h.run().errors)
    assert len(h.calls()) == 1 and len(h.gh.writes) == writes


def test_corrupt_or_missing_journal_does_not_recreate_managed_issues(
    harness: Harness,
) -> None:
    h = harness
    h.run()
    journal = h.tmp / "triage.json"
    journal.write_text('{"schema_version":99}')
    with pytest.raises(ValueError, match="journal"):
        h.run()
    journal.unlink()
    with pytest.raises(ValueError, match="no journal"):
        h.run()
    assert len(h.gh.issues) == 2 and len(h.calls()) == 1


@pytest.fixture
def interrupted_shared_parent(harness: Harness) -> Harness:
    h = harness
    h.report(count=2)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="first"), decision("100/R-20261019-02", "second")],
        )
    )
    h.gh.fail_after = "repos/o/r/issues"
    assert any("response loss" in error for error in h.run().errors)
    assert set(h.gh.issues) == {100, 101} and not h.gh.parents
    assert fmt.parse_proposal_source(h.gh.issues[101]["body"]) is None
    assert len(fmt.parse_task_metadata(h.gh.issues[101]["body"])["bindings"]) == 2
    assert not fmt.index_proposals((n, r["body"]) for n, r in h.gh.issues.items())
    return h


@pytest.mark.parametrize("journal_state", ["missing", "empty"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_lost_shared_parent_journal_stops_before_agent_or_writes_and_can_be_restored(
    interrupted_shared_parent: Harness, journal_state: str, dry_run: bool
) -> None:
    h = interrupted_shared_parent
    journal = h.tmp / "triage.json"
    saved = journal.read_bytes()
    if journal_state == "missing":
        journal.unlink()
    else:
        empty = json.loads(saved)
        empty["transactions"] = []
        journal.write_text(json.dumps(empty))
    before = copy.deepcopy(h.gh.issues)
    writes = len(h.gh.writes)
    with pytest.raises(ValueError, match="no journal transaction") as error:
        h.run(dry_run=dry_run)
    assert "#101" in str(error.value)
    assert "restore" in str(error.value) and "ops-report-task" in str(error.value)
    assert h.gh.issues == before and len(h.gh.writes) == writes
    assert len(h.calls()) == 1
    if journal_state == "missing":
        assert not journal.exists()
    else:
        assert json.loads(journal.read_text()) == empty

    journal.write_bytes(saved)
    assert not h.run().errors
    assert set(h.gh.issues) == {100, 101, 102, 103}
    assert h.gh.parents == {102: 101, 103: 101}
    assert not fmt.check_consistency(h.gh.family(101))
    assert len(h.calls()) == 1


@pytest.mark.parametrize("change", ["plan", "node", "bindings"])
def test_shared_parent_metadata_must_match_journal_before_any_publication(
    interrupted_shared_parent: Harness, change: str
) -> None:
    h = interrupted_shared_parent
    body = h.gh.issues[101]["body"]
    metadata = fmt.parse_task_metadata(body)
    if change == "plan":
        metadata["plan_sha256"] = "0" * 64
    elif change == "node":
        metadata["node"] = "unknown-task"
    else:
        metadata["bindings"][0]["proposal_sha256"] = "0" * 64
    adoption: Any = importlib.import_module("issues.adoption")
    h.gh.issues[101]["body"] = body.replace(
        body.splitlines()[0],
        adoption.task_marker(
            metadata["plan_sha256"], metadata["node"], metadata["bindings"]
        ),
    )
    before = copy.deepcopy(h.gh.issues)
    writes = len(h.gh.writes)
    saved = h.tmp.joinpath("triage.json").read_bytes()
    with pytest.raises(ValueError):
        h.run()
    assert h.gh.issues == before and len(h.gh.writes) == writes
    assert h.tmp.joinpath("triage.json").read_bytes() == saved
    assert len(h.calls()) == 1


def test_newly_observed_shared_parent_requires_journal_before_organization(
    interrupted_shared_parent: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = interrupted_shared_parent
    h.tmp.joinpath("triage.json").unlink()
    published = h.gh.issues.pop(101)
    collect = org.collect
    observed = False

    def collect_then_publish(*args: Any) -> Any:
        nonlocal observed
        snapshot = collect(*args)
        if not observed:
            observed = True
            h.gh.issues[101] = published
        return snapshot

    monkeypatch.setattr(org, "collect", collect_then_publish)
    writes = len(h.gh.writes)
    with pytest.raises(ValueError, match="no journal transaction"):
        h.run()
    assert set(h.gh.issues) == {100, 101}
    assert len(h.calls()) == 1 and len(h.gh.writes) == writes
    assert not h.tmp.joinpath("triage.json").exists()


def test_cross_report_group_is_visible_through_unchanged_dispatcher_collection(
    harness: Harness,
) -> None:
    h = harness
    h.report(n=90)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision("90/R-20261019-01", "first"), decision(target="second")],
        )
    )
    h.run()
    collector: Any = importlib.import_module("dispatcher.collect")
    with state.Store(h.tmp / "state/parent").locked() as store:
        snapshot = collector.collect(h.gh, store, h.root)
    assert snapshot["issues"]["102"]["source_reports"] == [90]
    assert snapshot["issues"]["103"]["source_reports"] == [100]
    assert snapshot["issues"]["101"]["source_reports"] == [90, 100]
    assert {90, 100} <= collector.related(snapshot, 101)
    assert not collector.is_container(snapshot["issues"]["101"])
    assert not collector.is_container(snapshot["issues"]["102"])


def test_context_sources_remain_in_split_inputs_after_index_anchors_move(
    harness: Harness,
) -> None:
    h = harness
    h.report(n=90)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision("90/R-20261019-01", "first"), decision(target="second")],
        )
    )
    assert not h.run().errors
    parent = parent_node()
    h.gh.issues[1] = raw_issue(
        1,
        title="[移管先] 明示的に引き継いだ親",
        body=fmt.render_parent_body(
            parent["sections"],
            [fmt.ChildTask(n, "責務を分けて検証する") for n in (102, 103)],
        ),
        labels=[{"name": "issue:parent"}],
    )
    for n in (102, 103):
        h.gh.parents[n] = 1
        h.gh.issues[n]["title"] = "[移管先 #1] 責務を分けて検証する"
        h.gh.issues[n]["body"] = h.gh.issues[n]["body"].replace(
            "親issue: #101", "親issue: #1"
        )
    h.gh.issues[101]["body"] = policy.replace_section(
        h.gh.issues[101]["body"], fmt.TASKS_HEADING, fmt.NO_CHILDREN
    )
    assert not fmt.check_consistency(h.gh.family(1))
    collector: Any = importlib.import_module("dispatcher.collect")
    agent_input: Any = importlib.import_module("dispatcher.agent_input")
    with state.Store(h.tmp / "state/parent").locked() as store:
        fixed = collector.collect(h.gh, store, h.root, [101])
        fixed["input_sha256"] = policy.digest(collector.material(fixed))
        assert collector.related(fixed, 101) == {90, 100, 101}
        index_path = agent_input.write_agent_input(fixed, h.tmp / "split-input")
        index = json.loads(index_path.read_text())
        assert set(index["issues"]) == {"90", "100", "101"}
        assert index["issues"]["101"]["source_reports"] == [90, 100]
        h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
            "- [x] **", "- [ ] **"
        )
        with pytest.raises(collector.InputsChanged):
            collector.assert_fresh(h.gh, store, h.root, fixed, 101)


@pytest.mark.parametrize("stage", ["preflight", "publication", "final_check"])
def test_source_change_defers_only_its_subject_closure_in_v2(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    h = harness
    h.gh.issues[1] = raw_issue(1)
    h.report(n=90)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision("90/R-20261019-01", "first"), decision(target="second")],
        )
    )
    assert not h.run().errors
    judgement: Any = importlib.import_module("dispatcher.judgement")

    def revoke() -> None:
        h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
            "- [x] **", "- [ ] **"
        )

    if stage == "preflight":
        original = judgement.judge

        def judge(*args: Any) -> Any:
            output = original(*args)
            revoke()
            return output

        monkeypatch.setattr(judgement, "judge", judge)
    elif stage == "publication":

        def on_write() -> None:
            if h.gh.writes[-1][1] == "repos/o/r/issues/101/comments":
                h.gh.on_write = None
                revoke()

        h.gh.on_write = on_write
    else:
        original_check = judgement.retain_fresh_decisions

        def check(*args: Any) -> None:
            if args[-1] == "final_check":
                revoke()
            original_check(*args)

        monkeypatch.setattr(judgement, "retain_fresh_decisions", check)
    result = h.judge(
        [
            start_decision(1),
            start_decision(90, "out_of_scope"),
            start_decision(100, "out_of_scope"),
            start_decision(101, reports=(90, 100)),
            start_decision(102, "existing_parent", parent=101, reports=(90,)),
            start_decision(103, "existing_parent", parent=101, reports=(100,)),
        ],
        "changed-source",
    )
    assert result["schema_version"] == 2 and result["result"] == "recorded", result
    assert [d["issue"] for d in result["deferred"]] == [100, 101, 102, 103]
    assert next(d for d in result["deferred"] if d["issue"] == 101)["stage"] == stage
    assert [d["issue"] for d in result["decisions"]] == [1, 90]
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert len(h.tmp.joinpath("judgement-calls.jsonl").read_text().splitlines()) == 1
    fixed = json.loads(h.tmp.joinpath("changed-source/input.json").read_text())
    with state.Store(h.tmp / "state/parent").locked() as store:
        judgement.assert_candidate_fresh(
            h.gh, store, h.root, fixed, result["candidates"][0]
        )


@pytest.mark.parametrize("anchor", ["parent", "child"])
def test_autonomous_parent_notes_preserve_adopted_task_delegation(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, anchor: str
) -> None:
    h = harness
    if anchor == "child":
        h.report(count=2)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="improvements")]
            if anchor == "parent"
            else [decision(target="first"), decision("100/R-20261019-02", "second")],
        )
    )
    assert not h.run().errors
    result = h.judge(
        [
            start_decision(100, "out_of_scope"),
            start_decision(101, reports=(100,)),
            start_decision(102, "existing_parent", parent=101, reports=(100,)),
            start_decision(103, "existing_parent", parent=101, reports=(100,)),
        ],
        "before-assignment",
    )
    assert result["result"] == "recorded" and result["deferred"] == [], result
    candidate = result["candidates"][0]
    fixed = json.loads(h.tmp.joinpath("before-assignment/input.json").read_text())
    judgement: Any = importlib.import_module("dispatcher.judgement")
    execution: Any = importlib.import_module("parent.execution")
    sessions: Any = importlib.import_module("parent.session")
    tasks: Any = importlib.import_module("parent.adopted_tasks")
    worker = h.root / ".claude/worktrees/adopted-task"
    subprocess.run(
        ["git", "worktree", "add", "-b", "codex/adopted-task", str(worker)],
        cwd=h.root,
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(h.root)
    with state.Store(h.tmp / "state/parent").locked() as store:
        session = sessions.ParentSession(h.gh, store, 101, "claude:fixture")
        judgement.assert_candidate_fresh(h.gh, store, h.root, fixed, candidate)
        auth = execution.validate_candidate(
            session, candidate["basis_sha256"], candidate["generation"]
        )
        session.register()
        session.note("採用出典と実行根拠を照合し、既存の子へ委任する。")
        assert execution.authorization(session) == auth
        for n in (101, 102):
            assert tasks.check_task_adoptions(h.gh, h.gh.issues[n]["body"])
        delegated = session.delegate(
            102, "codex", "codex/adopted-task", worker, None, None
        )
        assert delegated["action"] == "launch"
        assert delegated["assignment"]["authorization"] == auth
        assert session.status()["proposal_receipts"] == []
        assert session.status()["proposal_agreements"] == []
        assert execution.current_basis(session)["kind"] == "autonomous_decision"
        adopted = h.gh.issues[100]["body"]
        h.gh.issues[100]["body"] = adopted.replace("- [x] **", "- [ ] **")
        for n in (101, 102):
            with pytest.raises(ValueError, match="adoption"):
                tasks.check_task_adoptions(h.gh, h.gh.issues[n]["body"])
        h.gh.issues[100]["body"] = adopted
        h.gh.write(
            "POST",
            "repos/o/r/issues/101/comments",
            {"body": "要件を追加してください。"},
        )
        with pytest.raises(ValueError, match="inputs changed"):
            session.agreed()
    history = policy.events(h.gh.comment_rows[101], "operator", 101)
    assert any(e["event"] == "note" for e in history)
    assert not any(
        e["event"] in {"agreed", "proposal_received", "proposal_agreed"}
        for e in history
    )


def test_task_provenance_requires_separate_autonomous_basis_and_current_adoption(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = harness
    h.report(count=2)
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="first"), decision("100/R-20261019-02", "second")],
        )
    )
    h.run()
    source = fmt.ProposalSource("R-20261019-01", 100)
    sessions: Any = importlib.import_module("parent.session")
    execution: Any = importlib.import_module("parent.execution")
    proposals_module: Any = importlib.import_module("parent.proposals")
    with state.Store(h.tmp / "state/parent").locked() as store:
        store.parents["o/r#101"] = {
            "status": "active",
            "session": "claude:fixture",
            "generation": 1,
            "tasks": {},
            "reviews": [],
        }
        session = sessions.ParentSession(h.gh, store, 101, "claude:fixture")
        with pytest.raises(ValueError, match="実行根拠"):
            proposals_module.require_source(session, source)
        monkeypatch.setattr(
            execution, "current_basis", lambda s: {"kind": "human_agreement"}
        )
        with pytest.raises(ValueError, match="autonomous"):
            proposals_module.require_source(session, source)
        monkeypatch.setattr(
            execution, "current_basis", lambda s: {"kind": "autonomous_decision"}
        )
        assert proposals_module.require_source(session, source).parent() is None
        h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
            "- [x] **", "- [ ] **"
        )
        with pytest.raises(ValueError, match="adoption"):
            proposals_module.require_source(session, source)


def test_concurrent_daily_cli_triage_uses_shared_lock_and_creates_one_issue(
    ops_env: Any,
) -> None:
    ops = ops_env
    generated = ops.run("report")
    assert generated.returncode == 0, generated.stderr
    data = ops.state
    report = next(row for row in data["issues"] if row["number"] == 100)
    report["body"] = report["body"].replace(
        "- [ ] **R-20261012-02**", "- [x] **R-20261012-02**"
    )
    ops.write_state(data)
    response = plan(decisions=[decision("100/R-20261012-02")])
    Path(ops.env["FAKE_ORGANIZATION_PLAN"]).write_text(
        json.dumps(response, ensure_ascii=False)
    )
    first = ops.start("triage")
    second = ops.start("triage")
    for process in (first, second):
        stdout, stderr = process.communicate(timeout=90)
        assert process.returncode == 0, stdout + stderr
    assert len(ops.state["issues"]) == len(data["issues"]) + 1
    assert len(Path(ops.env["FAKE_ORGANIZATION_CALLS"]).read_text().splitlines()) == 1
    assert not ops.state.get("parents")
    assert (
        not Path(ops.env["XDG_STATE_HOME"])
        .joinpath("tennis-lab-agents/parent-sessions/registry.json")
        .exists()
    )


def test_cancelled_mapped_proposal_does_not_block_independent_new_issue(
    harness: Harness,
) -> None:
    h = harness
    h.report(count=2, selected=False)
    h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
        "- [ ] **R-20261019-01**", "- [x] **R-20261019-01**"
    )
    h.run()
    h.gh.issues[100]["body"] = (
        h.gh.issues[100]["body"]
        .replace("- [x] **R-20261019-01**", "- [ ] **R-20261019-01**")
        .replace("- [ ] **R-20261019-02**", "- [x] **R-20261019-02**")
    )
    h.response(plan([task_node("second")], [decision("100/R-20261019-02", "second")]))
    result = h.run()
    assert result.errors and ("R-20261019-02", 102) in result.linked_existing
    assert len(h.gh.issues) == 3
    history = {
        "reports": [
            {
                "number": 100,
                "url": "https://github.com/o/r/issues/100",
                "body": h.gh.issues[100]["body"],
            }
        ]
    }
    decisions = triage.memory_decisions(history, list(h.gh.issues.values()))
    assert [row["proposal_id"] for row in decisions] == ["R-20261019-02"]


def test_context_only_split_tasks_recheck_all_adoptions(harness: Harness) -> None:
    h = harness
    h.response(
        plan(
            [
                parent_node(),
                task_node("first", "child", "improvements"),
                task_node("second", "child", "improvements"),
            ],
            [decision(target="improvements")],
        )
    )
    h.run()
    tasks: Any = importlib.import_module("parent.adopted_tasks")
    assert fmt.parse_proposal_source(h.gh.issues[102]["body"]) is None
    assert tasks.check_task_adoptions(h.gh, h.gh.issues[102]["body"])
    h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace(
        "保守性を高める", "目的を変える"
    )
    with pytest.raises(ValueError, match="digest changed"):
        tasks.check_task_adoptions(h.gh, h.gh.issues[102]["body"])


def test_selected_option_and_digest_stay_bound_after_the_human_changes_choice(
    harness: Harness,
) -> None:
    h = harness
    body = h.gh.issues[100]["body"]
    body = body.replace(
        "  - 根拠:", "  - [x] 継続 --> / 調査\n  - [ ] close\n  - 根拠:"
    )
    h.gh.issues[100]["body"] = body
    assert not h.run().errors
    source = fmt.parse_proposal_source(h.gh.issues[101]["body"])
    assert source.option == "継続 --> / 調査"
    assert (
        fmt.parse_task_metadata(h.gh.issues[101]["body"])["bindings"][0]["source"][
            "option"
        ]
        == source.option
    )
    h.gh.issues[100]["body"] = (
        h.gh.issues[100]["body"]
        .replace("  - [x] 継続 --> / 調査", "  - [ ] 継続 --> / 調査")
        .replace("  - [ ] close", "  - [x] close")
    )
    assert h.run().errors and len(h.gh.issues) == 2 and len(h.calls()) == 1


def test_fenced_task_metadata_example_is_not_an_adopted_task(harness: Harness) -> None:
    h = harness
    h.run()
    marker = h.gh.issues[101]["body"].splitlines()[1]
    assert (
        fmt.parse_task_metadata("## 書式例\n\n```markdown\n" + marker + "\n```\n")
        is None
    )


def test_public_source_check_works_with_only_the_documented_ops_import_root(
    harness: Harness,
) -> None:
    h = harness
    h.run()
    data = h.tmp / "source-input.json"
    data.write_text(json.dumps(h.gh.issues))
    script = """
import json, sys
from pathlib import Path
from parent.adopted_tasks import check_task_adoptions
rows = json.loads(Path(sys.argv[1]).read_text())
class GitHub:
    def issue(self, n):
        return rows[str(n)]
assert len(check_task_adoptions(GitHub(), rows['101']['body'])) == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(data)],
        cwd=h.root,
        env={**os.environ, "PYTHONPATH": str(OPS)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_incomplete_adopted_requirement_records_specific_resume_instructions(
    ops_env: Any,
) -> None:
    ops = ops_env
    ops.publish_proposal_fixture()
    data = ops.state
    report = next(row for row in data["issues"] if row["number"] == 100)
    report["body"] = (
        report["body"]
        .replace("- [ ] **R-20261012-02**", "- [x] **R-20261012-02**")
        .replace("  - 目的: 大きなファイルの変更を容易にする。", "  - 目的: ")
    )
    ops.write_state(data)
    result = ops.run("triage")
    assert result.returncode == 1
    fresh = next(row for row in ops.state["issues"] if row["number"] == 100)
    assert all(
        token in fresh["body"] for token in ("保留", "目的", "再開条件", "次の対応")
    )
    assert len(ops.state["issues"]) == len(data["issues"])
    assert not Path(ops.env["FAKE_ORGANIZATION_CALLS"]).exists()


@pytest.mark.parametrize(
    "mutation", ["unknown_version", "duplicate", "changed_digest", "wrong_position"]
)
def test_task_source_metadata_cannot_silently_fall_back_to_legacy(
    harness: Harness, mutation: str
) -> None:
    h = harness
    h.run()
    body = h.gh.issues[101]["body"]
    if mutation == "unknown_version":
        body = body.replace("ops-report-task:v1", "ops-report-task:v2")
    elif mutation == "duplicate":
        body += "\n" + body.splitlines()[1]
    elif mutation == "changed_digest":
        metadata = fmt.parse_task_metadata(body)
        body = body.replace(metadata["bindings"][0]["proposal_sha256"], "x" * 64)
    else:
        first, second, rest = body.split("\n", 2)
        body = first + "\n\n" + second + "\n" + rest
    with pytest.raises(ValueError):
        fmt.parse_task_metadata(body)
    h.gh.issues[101]["body"] = body
    writes = len(h.gh.writes)
    result = h.judge([], "invalid-task-source")
    assert result["result"] == "failed" and result["candidates"] == []
    assert len(h.gh.writes) == writes
    assert not h.tmp.joinpath("judgement-calls.jsonl").exists()


def test_triage_publishes_only_until_the_scheduled_judgement_entry_runs(
    harness: Harness,
) -> None:
    h = harness
    h.run()
    assert (
        h.gh.comment_rows == {}
    )  # No execution basis or development owner from triage.
    judgement: Any = importlib.import_module("dispatcher.judgement")
    results = []
    for hour in ("08", "20"):
        result = h.judge(
            [start_decision(100, "out_of_scope"), start_decision(101, reports=(100,))],
            f"scheduled-{hour}",
        )
        assert result["result"] == "recorded", result["diagnostics"]
        assert result["schema_version"] == 2 and result["deferred"] == []
        assert result["candidates"][0]["parent"] == 101
        run_dir = h.tmp / f"scheduled-{hour}"
        index = json.loads(run_dir.joinpath("agent-input/index.json").read_text())
        assert index["subjects"] == [100, 101]
        assert index["issues"]["101"]["source_reports"] == [100]
        assert index["issues"]["101"]["mechanical_guard"] is None
        issue_file = run_dir / "agent-input" / index["issues"]["101"]["file"]
        task = json.loads(issue_file.read_text())
        assert "".join(task["body"]["segments"]) == h.gh.issues[101]["body"]
        assert task["content_complete"] and task["omissions"] == []
        runtime = json.loads(run_dir.joinpath("agent-runtime/result.json").read_text())
        assert runtime["contract"]["read_dirs"] == [str(run_dir / "agent-input")]
        assert (
            runtime["contract"]["json_schema_sha256"]
            == hashlib.sha256(
                OPS.joinpath("dispatcher/judgement.schema.json").read_bytes()
            ).hexdigest()
        )
        results.append(result)
    calls = [
        json.loads(line)
        for line in h.tmp.joinpath("judgement-calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 2 and all("--json-schema" in c["args"] for c in calls)
    assert len(h.calls()) == 1  # Only the proposal organizer used a runtime here.
    comments = h.gh.comment_rows[101]
    basis = next(
        e["basis"]
        for e in reversed(policy.events(comments, "operator", 101))
        if e["event"] == "execution_basis"
    )
    assert basis["kind"] == "autonomous_decision" and "human_confirmation" not in basis
    assert all(
        "merge_permission" not in row["body"] and '"event": "agreed"' not in row["body"]
        for row in comments
    )
    h.gh.issues[100]["body"] = h.gh.issues[100]["body"].replace("- [x] **", "- [ ] **")
    snapshot = json.loads(h.tmp.joinpath("scheduled-20/input.json").read_text())
    with (
        state.Store(h.tmp / "state/parent").locked() as store,
        pytest.raises(ValueError, match="changed"),
    ):
        judgement.assert_candidate_fresh(
            h.gh, store, h.root, snapshot, results[-1]["candidates"][0]
        )
