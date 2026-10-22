"""Judgments with fake paginated GitHub, the real runner and an offline Claude CLI."""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs

import pytest

ROOT = Path(__file__).resolve().parents[3]
FAKE_CLI = Path(__file__).with_name("fake_judgement_cli.py")


def issue(
    n: int,
    *,
    body: str = "目的: parserを改善する。対象: README。完了条件: 境界を検証する。",
    state: str = "open",
    reason: str | None = None,
    labels: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "number": n,
        "repository_url": "https://api.github.com/repos/o/r",
        "title": f"作業 {n}",
        "body": body,
        "state": state,
        "state_reason": reason,
        "labels": [{"name": label} for label in labels],
        "assignees": [],
    }


def decision(
    n: int,
    status: str = "startable",
    *,
    target: int | None = None,
    priority: int | None = None,
    deps: tuple[int, ...] = (),
) -> dict[str, Any]:
    ready = status == "startable"
    target = None if status == "out_of_scope" else (n if target is None else target)
    sources = [f"issue:{k}" for k in sorted({n, *deps, *([target] if target else [])})]
    if ready:
        sources.append("code:CODE_COMMIT:README.md")
    return {
        "issue": n,
        "disposition": status,
        "target_parent": target,
        "priority": (priority or n) if ready else None,
        "requirements": {
            "purpose": "parserの境界を明確にする",
            "scope": "READMEとテスト。起動・mergeは対象外",
            "behavior": "不正入力を拒否する",
            "acceptance": "CPUのfake fixtureで境界を検証する",
        }
        if ready
        else None,
        "rationale": f"#{n} の要件と資料を照合した。{status}として扱う。",
        "facts": [f"#{n} の本文を確認した。"],
        "missing_information": ["期待する入力形式を選択する必要がある"]
        if status == "information_needed"
        else [],
        "resume_condition": "阻害条件の解消を確認し、予定回または明示実行で再判断する。",
        "next_action": "判定に従い割当側へ候補または待ち理由を渡す。",
        "sources": sources,
        "dependencies": list(deps),
    }


class FakeGitHub:
    repo = "o/r"

    def __init__(self) -> None:
        self.issues = {1: issue(1)}
        self.parents: dict[int, int] = {}
        self.dependencies: dict[int, list[int]] = {}
        self.comment_rows: dict[int, list[dict[str, Any]]] = {}
        self.prs: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.writes: list[dict[str, Any]] = []
        self.fail = ""
        self.on_get: Callable[[str], None] | None = None
        self.on_write: Callable[[], None] | None = None

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        self.calls.append(endpoint)
        if self.on_get:
            self.on_get(endpoint)
        if self.fail and self.fail in endpoint:
            raise ValueError("injected API failure: " + endpoint)
        if endpoint == "user":
            return {"login": "operator"}
        path, _, query = endpoint.partition("?")
        params = parse_qs(query)
        page = int(params.get("page", ["1"])[0])

        def paged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return copy.deepcopy(rows[(page - 1) * 100 : page * 100])

        if path == "repos/o/r/issues":
            return paged(
                [v for _, v in sorted(self.issues.items()) if v["state"] == "open"]
            )
        if path == "repos/o/r/pulls":
            return paged(self.prs)
        parts = path.removeprefix("repos/o/r/issues/").split("/")
        n = int(parts[0])
        if len(parts) == 1:
            return copy.deepcopy(self.issues[n])
        tail = "/".join(parts[1:])
        if tail == "parent":
            assert missing_parent
            return (
                copy.deepcopy(self.issues[self.parents[n]])
                if n in self.parents
                else None
            )
        if tail == "sub_issues":
            return paged(
                [self.issues[k] for k, v in sorted(self.parents.items()) if v == n]
            )
        if tail == "dependencies/blocked_by":
            return paged([self.issues[k] for k in self.dependencies.get(n, [])])
        if tail == "comments":
            return paged(self.comment_rows.get(n, []))
        raise AssertionError(endpoint)

    def rows(self, endpoint: str) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            importlib.import_module("parent.github").GitHub.rows(self, endpoint),
        )

    def identity(self, raw: object) -> int:
        return cast(
            int, importlib.import_module("parent.github").GitHub.identity(self, raw)
        )

    def issue(self, n: int) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            importlib.import_module("parent.github").GitHub.issue(self, n),
        )

    def comments(self, n: int) -> list[dict[str, Any]]:
        return self.rows(f"repos/o/r/issues/{n}/comments")

    def pulls(self) -> list[dict[str, Any]]:
        return self.rows("repos/o/r/pulls?state=all")

    def write(
        self, method: str, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if self.on_write:
            self.on_write()
        self.writes.append({"method": method, "endpoint": endpoint, "payload": payload})
        if method == "PATCH":
            cid = int(endpoint.rsplit("/", 1)[1])
            row = next(
                c for rows in self.comment_rows.values() for c in rows if c["id"] == cid
            )
            row["body"] = payload["body"]
            return copy.deepcopy(row)
        assert method == "POST" and endpoint.endswith("/comments")
        n = int(endpoint.split("/")[-2])
        cid = (
            max(
                (c["id"] for rows in self.comment_rows.values() for c in rows),
                default=0,
            )
            + 1
        )
        row = {"id": cid, "user": {"login": "operator"}, "body": payload["body"]}
        self.comment_rows.setdefault(n, []).append(row)
        return copy.deepcopy(row)


class Harness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
        self.mod = importlib.import_module("dispatcher.judgement")
        self.collector = importlib.import_module("dispatcher.collect")
        self.schema = importlib.import_module("dispatcher.schema")
        self.root, self.work = root, root / "work"
        self.work.mkdir()
        self.git("init", "-q", "-b", "test")
        (self.work / "README.md").write_text("# Parser\n\nStrict inputs.\n")
        self.git("add", "README.md")
        self.git(
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        )
        self.store = importlib.import_module("parent.state").Store(
            root / "private-state"
        )
        self.gh = FakeGitHub()
        self.output = root / "response.json"
        self.calls = root / "agent-calls.jsonl"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "claude"
        fake.write_bytes(FAKE_CLI.read_bytes())
        fake.chmod(0o700)
        monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
        monkeypatch.setenv("FAKE_JUDGEMENT_RESPONSE", str(self.output))
        monkeypatch.setenv("FAKE_JUDGEMENT_CALLS", str(self.calls))
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
        self.count = 0

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.work,
            env={
                **os.environ,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def response(self, *rows: dict[str, Any]) -> None:
        self.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_sha256": "INPUT_DIGEST",
                    "decisions": list(rows),
                },
                ensure_ascii=False,
            )
        )

    def run(self, *, apply: bool = True) -> dict[str, Any]:
        self.count += 1
        self.run_dir = self.root / f"run-{self.count}"
        return cast(
            dict[str, Any],
            self.mod.run(
                self.gh, self.store, self.work, self.run_dir, apply=apply, timeout=15
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        return cast(
            dict[str, Any], json.loads((self.run_dir / "input.json").read_text())
        )


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(tmp_path, monkeypatch)


def test_unagreed_standalone_and_existing_family_route_without_new_hierarchy(
    h: Harness,
) -> None:
    h.gh.issues.update(
        {
            2: issue(2, labels=("ai-proposed", "ready")),
            3: issue(3),
            4: issue(4, labels=("issue:parent",)),
            5: issue(5),
            6: issue(6, state="closed", reason="not_planned"),
            7: issue(7),
            8: issue(8),
            9: issue(9, labels=("needs-human",)),
            10: issue(10, labels=("ops-report",)),
        }
    )
    h.gh.parents[3] = 4
    h.gh.dependencies[5] = [6]
    h.gh.prs = [
        {
            "number": 100,
            "body": "Closes #7",
            "state": "closed",
            "merged_at": None,
            "head": {"ref": "codex/unfinished", "sha": "a" * 40},
        }
    ]
    h.gh.issues[8]["assignees"] = [{"login": "human"}]
    h.response(
        decision(1),
        decision(2, "information_needed"),
        decision(3, "existing_parent", target=4),
        decision(4),
        decision(5, "dependency_wait", deps=(6,)),
        decision(7, "existing_owner"),
        decision(8, "existing_owner"),
        decision(9, "human_hold"),
        decision(10, "out_of_scope"),
    )
    result = h.run()
    assert result["result"] == "recorded", result
    assert [(c["parent"], c["route"]) for c in result["candidates"]] == [
        (1, "standalone"),
        (4, "parent"),
    ]
    assert all(
        c["generation"] == 1 and c["requirements"]["acceptance"]
        for c in result["candidates"]
    )
    assert {d["issue"]: d["target_parent"] for d in result["decisions"]}[3] == 4
    assert not (h.store.directory / "registry.json").exists()
    assert all("/comments" in w["endpoint"] for w in h.gh.writes)
    assert 10 not in h.gh.comment_rows
    published = json.dumps(h.gh.comment_rows, ensure_ascii=False)
    assert '"event": "agreed"' not in published
    assert "human_confirmation" not in published and "allow_shared" not in published
    assert "private-judgement-session" not in published
    invocation = json.loads((h.run_dir / "agent-runtime/invocation.json").read_text())
    calls = [json.loads(line) for line in h.calls.read_text().splitlines()]
    args = calls[0]["args"]
    assert args[args.index("--model") + 1] == "claude-opus-5-5"
    assert args[args.index("--tools") + 1 : args.index("--permission-mode")] == [
        "Read",
        "Grep",
        "Glob",
    ]
    assert "dontAsk" in args and invocation
    assert (h.run_dir / "result.json").stat().st_mode & 0o077 == 0


def test_every_page_and_cross_parent_dependency_are_collected(h: Harness) -> None:
    h.gh.issues = {n: issue(n) for n in range(1, 107)}
    h.gh.parents = {n: 1 for n in range(2, 107)}
    h.gh.issues.update(
        {n: issue(n, state="closed", reason="completed") for n in range(200, 305)}
    )
    h.gh.issues[400] = issue(400, state="closed", reason="completed")
    h.gh.parents[200] = 400
    h.gh.dependencies[1] = list(range(200, 305))
    h.gh.comment_rows[1] = [
        {"id": n, "user": {"login": "human"}, "body": f"指示 {n}"}
        for n in range(1, 107)
    ]
    h.gh.prs = [
        {
            "number": n,
            "body": "",
            "state": "closed",
            "merged_at": None,
            "head": {"ref": f"branch-{n}", "sha": "a" * 40},
        }
        for n in range(500, 606)
    ]
    with h.store.locked():
        snapshot = h.collector.collect(h.gh, h.store, h.work)
    assert len(snapshot["subjects"]) == 106
    assert len(snapshot["issues"]["1"]["children"]) == 105
    assert len(snapshot["issues"]["1"]["dependencies"]) == 105
    assert len(snapshot["issues"]["1"]["comments"]) == 106
    assert len(snapshot["pulls"]) == 106
    assert snapshot["issues"]["200"]["parent"] == 400
    for path in (
        "issues?state=open&sort=created&direction=asc",
        "issues/1/sub_issues",
        "issues/1/dependencies/blocked_by",
        "issues/1/comments",
        "pulls?state=all",
    ):
        assert any(path in c and "page=2" in c for c in h.gh.calls)


@pytest.mark.parametrize(
    "change",
    [
        "json",
        "duplicate_keys",
        "unknown_version",
        "unknown_id",
        "missing_issue",
        "extra_key",
        "invented_source",
        "missing_code",
        "human_confirmation",
        "wrong_digest",
        "bool_priority",
    ],
)
def test_invalid_agent_output_never_publishes_or_emits_candidates(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    value = json.loads(h.output.read_text())
    if change == "json":
        h.output.write_text("not JSON")
    elif change == "duplicate_keys":
        h.output.write_text('{"schema_version":1,"schema_version":1}')
    else:
        d = value["decisions"][0]
        if change == "unknown_version":
            value["schema_version"] = 99
        elif change == "unknown_id":
            d["issue"] = 999
        elif change == "missing_issue":
            value["decisions"] = []
        elif change == "extra_key":
            d["launch"] = True
        elif change == "invented_source":
            d["sources"].append("issue:999")
        elif change == "missing_code":
            d["sources"] = ["issue:1"]
        elif change == "human_confirmation":
            d["requirements"]["human_confirmation"] = "fabricated"
        elif change == "wrong_digest":
            value["input_sha256"] = "a" * 64
        else:
            d["priority"] = True
        h.output.write_text(json.dumps(value))
    result = h.run()
    assert (
        result["result"] == "failed"
        and result["candidates"] == []
        and result["diagnostics"]
    )
    assert h.gh.writes == []


@pytest.mark.parametrize(
    "path",
    [
        "issues?state=open&sort=created&direction=asc&per_page=100&page=2",
        "issues/1/comments",
        "issues/1/parent",
        "issues/1/dependencies/blocked_by",
        "pulls?state=all",
    ],
)
def test_api_failure_never_uses_partial_inputs(h: Harness, path: str) -> None:
    h.gh.issues.update({n: issue(n) for n in range(2, 102)})
    h.gh.fail = path
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert result["diagnostics"] and h.gh.writes == []
    assert not h.calls.exists()


@pytest.mark.parametrize("scenario", ["wrong_model", "cli_error", "empty"])
def test_runtime_failure_is_not_task_success(
    h: Harness, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("FAKE_JUDGEMENT_SCENARIO", scenario)
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == []


def test_same_inputs_still_run_agent_and_update_only_changed_comment(
    h: Harness,
) -> None:
    h.gh.issues[2] = issue(2)
    h.response(decision(1), decision(2, "information_needed"))
    first = h.run()
    assert first["result"] == "recorded", first
    writes = len(h.gh.writes)
    second = h.run()
    assert second["result"] == "recorded", second
    assert first["input_sha256"] == second["input_sha256"]
    assert first["candidates"] == second["candidates"]
    assert len(h.gh.writes) == writes
    assert len(h.calls.read_text().splitlines()) == 2
    changed = decision(2, "information_needed")
    changed["rationale"] = "新しい未決事項の理由を具体化した。"
    h.response(decision(1), changed)
    third = h.run()
    assert third["result"] == "recorded", third
    assert third["input_sha256"] == first["input_sha256"]
    assert len(h.gh.writes) == writes + 1
    assert h.gh.writes[-1]["method"] == "PATCH"
    assert len(h.gh.comment_rows[2]) == 1


@pytest.mark.parametrize(
    "change", ["requirements", "human_comment", "dependency", "new_pr", "owner"]
)
def test_changes_immediately_before_publication_reject_all_candidates(
    h: Harness, change: str
) -> None:
    h.gh.issues[200] = issue(200, state="closed", reason="completed")
    h.gh.dependencies[1] = [200]
    h.response(decision(1, deps=(200,)))

    def concurrent_edit(endpoint: str) -> None:
        if not h.calls.exists():
            return
        h.gh.on_get = None
        if change == "requirements":
            h.gh.issues[1]["body"] += "\n新しい要件"
        elif change == "human_comment":
            h.gh.comment_rows[1] = [
                {
                    "id": 999,
                    "user": {"login": "human"},
                    "body": "実装を停止してください。",
                }
            ]
        elif change == "dependency":
            h.gh.issues[200]["state_reason"] = "not_planned"
        elif change == "new_pr":
            h.gh.prs.append(
                {
                    "number": 300,
                    "body": "Closes #1",
                    "state": "open",
                    "merged_at": None,
                    "head": {"ref": "worker", "sha": "a" * 40},
                }
            )
        else:
            h.store.parents["o/r#1"] = {
                "status": "active",
                "generation": 1,
                "session": "claude:private-owner",
                "tasks": {},
                "reviews": [],
            }
            h.store.save()

    h.gh.on_get = concurrent_edit
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == []


def test_change_during_comment_write_cannot_record_a_basis(h: Harness) -> None:
    h.gh.issues[200] = issue(200, state="closed", reason="completed")
    h.gh.dependencies[1] = [200]
    h.response(decision(1, deps=(200,)))

    def edit() -> None:
        h.gh.on_write = None
        h.gh.issues[200]["state_reason"] = "not_planned"

    h.gh.on_write = edit
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert len(h.gh.writes) == 1
    assert not any(
        '"event": "execution_basis"' in c["body"] for c in h.gh.comment_rows[1]
    )


@pytest.mark.parametrize(
    "blocker",
    [
        "cancelled_dependency",
        "closed_unmerged_pr",
        "needs_human",
        "checking",
        "revoked",
        "stopped",
        "existing_owner",
        "child",
    ],
)
def test_agent_cannot_override_mechanical_vetoes(h: Harness, blocker: str) -> None:
    d = decision(1)
    policy = importlib.import_module("parent.policy")
    if blocker == "cancelled_dependency":
        h.gh.issues[200] = issue(200, state="closed", reason="not_planned")
        h.gh.dependencies[1] = [200]
        d = decision(1, deps=(200,))
    elif blocker == "closed_unmerged_pr":
        h.gh.prs = [
            {
                "number": 100,
                "body": "Closes #1",
                "state": "closed",
                "merged_at": None,
                "head": {"ref": "worker", "sha": "a" * 40},
            }
        ]
    elif blocker == "needs_human":
        h.gh.issues[1]["labels"] = [{"name": "needs-human"}]
    elif blocker in {"checking", "revoked"}:
        event = (
            {"parent": 1, "event": "checking"}
            if blocker == "checking"
            else {
                "parent": 1,
                "event": "basis_revoked",
                "revision": 1,
                "generation": 1,
                "reason": "人間の停止指示",
            }
        )
        h.gh.comment_rows[1] = [
            {"id": 9, "user": {"login": "operator"}, "body": policy.marker(event)}
        ]
    elif blocker == "stopped":
        h.gh.issues[1]["body"] += (
            "\n\n## 作業状況・停止事項\n\n停止\n<!-- parent-task-status:v1 state=stopped detail="
            + "a" * 64
            + " -->"
        )
    elif blocker == "existing_owner":
        with h.store.locked():
            h.store.parents["o/r#1"] = {
                "session": "claude:private-owner",
                "status": "active",
                "generation": 1,
                "tasks": {},
                "reviews": [],
            }
            h.store.save()
    else:
        h.gh.issues[2] = issue(2, state="closed", reason="completed")
        h.gh.parents[1] = 2
        d["target_parent"] = 2
        d["sources"].append("issue:2")
    h.response(d)
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == []


def test_assignment_contract_checks_external_context_then_parent_basis_under_one_lock(
    h: Harness,
) -> None:
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "recorded", result
    candidate, snapshot = result["candidates"][0], h.snapshot()
    execution = importlib.import_module("parent.execution")
    session_class = importlib.import_module("parent.session").ParentSession
    with h.store.locked():
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
        auth = execution.validate_candidate(
            session_class(h.gh, h.store, 1, None),
            candidate["basis_sha256"],
            candidate["generation"],
        )
    assert auth["execution_basis_kind"] == "autonomous_decision"
    assert auth["execution_basis_sha256"] == candidate["basis_sha256"]
    h.gh.issues[1]["body"] += "\n\n## 進捗概要\n\n表示だけを更新"
    with h.store.locked():
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)
    h.gh.issues[1]["body"] += "\n\n## 重要な変更\n\n仕様変更"
    with h.store.locked(), pytest.raises(ValueError, match="inputs changed"):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)


def test_own_comment_is_not_an_instruction_but_same_marker_by_other_author_is(
    h: Harness,
) -> None:
    h.response(decision(1))
    first = h.run()
    assert first["result"] == "recorded", first
    hold = h.gh.comment_rows[1][0]
    h.gh.comment_rows[1].append({**hold, "id": 500, "user": {"login": "human"}})
    second = h.run()
    assert second["result"] == "recorded", second
    assert first["input_sha256"] != second["input_sha256"]
    assert (
        len(
            [
                c
                for c in h.gh.comment_rows[1]
                if c["body"].startswith("<!-- issue-start-judgement:")
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "change",
    [
        "requirements",
        "source",
        "unknown_key",
        "route",
        "bool_priority",
        "unknown_parent",
        "snapshot_version",
    ],
)
def test_modified_candidate_is_rejected_before_reservation(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    result = h.run()
    assert result["result"] == "recorded", result
    candidate = copy.deepcopy(result["candidates"][0])
    snapshot = h.snapshot()
    if change == "requirements":
        candidate["requirements"]["scope"] = "変更された別の作業範囲"
    elif change == "source":
        candidate["sources"].append("https://uncollected.invalid/evidence")
    elif change == "unknown_key":
        candidate["human_confirmation"] = "fabricated"
    elif change == "route":
        candidate["route"] = "parent"
    elif change == "bool_priority":
        candidate["priority"] = True
    elif change == "unknown_parent":
        candidate["parent"] = 9999
    else:
        snapshot["schema_version"] = 99
    with h.store.locked(), pytest.raises(ValueError):
        h.mod.assert_candidate_fresh(h.gh, h.store, h.work, snapshot, candidate)


def test_report_proposal_work_issue_uses_normal_collection_without_starting_extra_agents(
    h: Harness,
) -> None:
    h.gh.issues[1]["body"] = (
        "<!-- ops-report-proposal: R-01 report=#9 option=plan -->\n\n"
        + h.gh.issues[1]["body"]
    )
    h.gh.issues[9] = issue(9, labels=("ops-report",))
    adopted = decision(1)
    adopted["sources"].append("issue:9")
    h.response(adopted, decision(9, "out_of_scope"))
    result = h.run()
    assert result["result"] == "recorded", result
    assert [c["parent"] for c in result["candidates"]] == [1]
    assert len(h.calls.read_text().splitlines()) == 1
    assert h.snapshot()["issues"]["1"]["source_reports"] == [9]
    h.gh.issues[9]["body"] += "\n採用を取り消す。"
    with h.store.locked(), pytest.raises(ValueError, match="inputs changed"):
        h.mod.assert_candidate_fresh(
            h.gh, h.store, h.work, h.snapshot(), result["candidates"][0]
        )


def test_current_own_display_is_never_an_independent_source(h: Harness) -> None:
    h.response(decision(1))
    assert h.run()["result"] == "recorded"
    d = decision(1)
    d["sources"].append(f"comment:1:{h.gh.comment_rows[1][0]['id']}")
    h.response(d)
    writes = len(h.gh.writes)
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert len(h.gh.writes) == writes


def test_human_explicit_stop_is_returned_with_facts_and_resume_condition(
    h: Harness,
) -> None:
    h.gh.comment_rows[1] = [
        {
            "id": 999,
            "user": {"login": "human"},
            "body": "この案は見送る。次の素材を決めるまで着手しない。",
        }
    ]
    held = decision(1, "human_hold")
    held["rationale"] = "人間が素材決定まで見送りを明示した。"
    held["sources"].append("comment:1:999")
    held["resume_condition"] = "人間が素材を決め、見送りを解除したときに再判断する。"
    h.response(held)
    result = h.run()
    assert result["result"] == "recorded" and result["candidates"] == []
    body = h.gh.comment_rows[1][-1]["body"]
    assert "人間が素材決定まで見送りを明示した。" in body
    assert "再開条件:" in body and "次の対応:" in body and "確認事実:" in body
    assert "#issuecomment-999" in body


def test_empty_input_still_launches_one_judgement_turn(h: Harness) -> None:
    h.gh.issues = {}
    h.response()
    result = h.run()
    assert result["result"] == "recorded" and result["candidates"] == []
    assert len(h.calls.read_text().splitlines()) == 1
    assert h.gh.writes == []


@pytest.mark.parametrize(
    "change", ["duplicate_comment", "cycle", "foreign_dependency", "foreign_subissue"]
)
def test_broken_or_unsupported_input_fails_before_agent(
    h: Harness, change: str
) -> None:
    h.response(decision(1))
    if change == "duplicate_comment":
        c = {"id": 1, "user": {"login": "human"}, "body": "指示"}
        h.gh.comment_rows[1] = [c, c]
    elif change == "cycle":
        h.gh.issues[2] = issue(2)
        h.gh.parents = {1: 2, 2: 1}
    else:
        h.gh.issues[2] = issue(2)
        h.gh.issues[2]["repository_url"] = "https://api.github.com/repos/another/repo"
        if change == "foreign_dependency":
            h.gh.dependencies[1] = [2]
        else:
            h.gh.parents[2] = 1
    result = h.run()
    assert result["result"] == "failed" and result["candidates"] == []
    assert h.gh.writes == [] and not h.calls.exists()


def test_preview_and_cli_print_no_candidates_or_local_private_information(
    h: Harness, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    h.response(decision(1))
    monkeypatch.setattr(h.mod, "GitHub", lambda repo: h.gh)
    code = h.mod.main(
        [
            "--repo",
            "o/r",
            "--cwd",
            str(h.work),
            "--run-dir",
            str(h.root / "cli"),
            "--parent-state-dir",
            str(h.store.directory),
            "--timeout",
            "15",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0 and json.loads(output)["result"] == "preview"
    assert json.loads(output)["candidate_count"] == 0
    assert str(h.root) not in output and "session" not in output
    assert h.gh.writes == []


def test_dirty_code_and_broken_registry_fail_before_agent(h: Harness) -> None:
    h.response(decision(1))
    (h.work / "README.md").write_text("uncommitted change")
    result = h.run()
    assert result["result"] == "failed" and not h.calls.exists()
    h.git("restore", "README.md")
    h.store.directory.mkdir(exist_ok=True)
    h.store.path.write_text('{"version":1,"parents":')
    result = h.run()
    assert result["result"] == "failed" and not h.calls.exists()
    assert h.gh.writes == []
