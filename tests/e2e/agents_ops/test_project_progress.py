"""Parent-owned Projects updates against fake gh and a private registry."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"


def base_state() -> dict[str, Any]:
    children = {1204: [1205, 1206, 1207, 1210], 1210: [1211, 1212]}
    parents = {child: parent for parent, rows in children.items() for child in rows}
    issues: dict[str, dict[str, Any]] = {}
    for n in (1204, 1205, 1206, 1207, 1210, 1211, 1212):
        state = "CLOSED" if n in {1205, 1207, 1211, 1212} else "OPEN"
        parent = parents.get(n)
        issues[str(n)] = {
            "id": f"I{n}",
            "number": n,
            "title": f"正式タイトル {n}",
            "url": f"https://github.com/o/r/issues/{n}",
            "body": "",
            "state": state,
            "stateReason": "NOT_PLANNED"
            if n in {1207, 1212}
            else "COMPLETED"
            if state == "CLOSED"
            else None,
            "parent": None
            if parent is None
            else {
                "id": f"I{parent}",
                "number": parent,
                "repository": {"nameWithOwner": "o/r"},
            },
            "repository": {"nameWithOwner": "o/r"},
            "children": children.get(n, []),
            "assignees": ["researcher" if n == 1204 else "worker"],
            "labels": ["issue:parent"]
            if n == 1204
            else ["issue:child", *(["issue:parent"] if n == 1210 else [])],
            "subIssuesSummary": {
                "total": len(children.get(n, [])),
                "completed": 0,
                "percentCompleted": 0,
            },
        }
    issues["1204"]["subIssuesSummary"].update(completed=2, percentCompleted=50)
    issues["1210"]["subIssuesSummary"].update(completed=2, percentCompleted=100)
    issues["1204"]["body"] = (
        "## 子タスクのチェックリスト\n\n- [ ] #1205 完了済だがチェック漏れ\n- [ ] #1206 作業中\n- [ ] #1207 取消（取消）\n- [x] #1999 本文だけの参照\n"
    )
    fields = [
        {
            "id": f"F{i}",
            "databaseId": 1000 + i,
            "name": name,
            "dataType": "SINGLE_SELECT" if name == "Status" else "TEXT",
            **(
                {"options": [{"id": "OLD", "name": "Backlog"}]}
                if name == "Status"
                else {}
            ),
        }
        for i, name in enumerate(
            (
                "Title",
                "Assignees",
                "Parent issue",
                "Sub-issues progress",
                "Linked pull requests",
                "Priority",
                "Status",
            )
        )
    ]
    return {
        "project": {
            "id": "P",
            "number": 3,
            "title": "prj-tennis-lab",
            "url": "https://github.com/users/Motoki0705/projects/3",
            "closed": False,
            "public": False,
            "viewerCanUpdate": True,
            "owner": {"login": "Motoki0705", "databaseId": 42},
        },
        "fields": fields,
        "views": [
            {
                "id": "V1",
                "number": 1,
                "name": "Backlog",
                "layout": "BOARD_LAYOUT",
                "filter": None,
                "visible": ["F0", "F6"],
                "groups": [],
                "columns": ["Status"],
            }
        ],
        "items": [],
        "workflows": [
            {"id": "W1", "name": "Item closed", "enabled": True, "number": 1}
        ],
        "issues": issues,
        "pulls": [
            {
                "number": 99,
                "body": "Closes #1206",
                "state": "open",
                "merged_at": None,
                "html_url": "https://github.com/o/r/pull/99",
            },
            {
                "number": 98,
                "body": "Closes #1206",
                "state": "closed",
                "merged_at": None,
                "html_url": "https://github.com/o/r/pull/98",
            },
        ],
    }


@dataclass
class ProjectEnv:
    tmp: Path
    env: dict[str, str]

    @property
    def state_file(self) -> Path:
        return self.tmp / "github.json"

    def state(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.state_file.read_text())
        return value

    def save(self, value: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(value))

    def spec(self, value: dict[str, Any]) -> str:
        path = self.tmp / "input.json"
        path.write_text(json.dumps(value, ensure_ascii=False))
        return str(path)

    def initial(self) -> dict[str, Any]:
        rows = []
        for raw in self.state()["issues"].values():
            parent = "issue:parent" in raw["labels"]
            cancelled = raw["stateReason"] == "NOT_PLANNED"
            rows.append(
                {
                    "issue": raw["number"],
                    "stage": None
                    if cancelled
                    else "作業中"
                    if parent
                    else "完了"
                    if raw["state"] == "CLOSED"
                    else "未着手",
                    "wait_reason": "取消: 後継へ移行" if cancelled else "",
                    "next_action": "親が次の作業を確認する",
                    **({"acceptance": "未確認"} if parent else {}),
                }
            )
        return {"schema_version": 1, "items": rows}

    def calls(self) -> list[dict[str, Any]]:
        path = Path(self.env["PROGRESS_GH_CALLS"])
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def mutations(self) -> list[dict[str, Any]]:
        return [
            row["payload"]
            for row in self.calls()
            if row["payload"]
            and (
                row["payload"].get("query", "").startswith("mutation")
                or "--method" in row["args"]
                and row["args"][row["args"].index("--method") + 1] == "POST"
            )
        ]

    def run(
        self, *args: str, ok: bool = True, **env: str
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "parent",
                "--repo",
                "o/r",
                "--parent",
                "1204",
                *args,
            ],
            cwd=ROOT,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == (0 if ok else 2), result.stdout + result.stderr
        return result

    def sync(
        self, event: str, spec: dict[str, Any], *args: str, ok: bool = True, **env: str
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            "project-sync",
            "--event",
            event,
            "--spec",
            self.spec(spec),
            *args,
            ok=ok,
            **env,
        )

    def inspect(self, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "project_progress",
                "inspect",
                "--repo",
                "o/r",
                "--parent",
                "1204",
            ],
            cwd=ROOT,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def initialize(self) -> None:
        self.run("project-setup", "--apply")
        self.sync("initial", self.initial(), "--apply")

    def item(self, n: int) -> dict[str, Any]:
        return next(
            row for row in self.state()["items"] if row["content"]["number"] == n
        )


@pytest.fixture
def project_env(tmp_path: Path) -> ProjectEnv:
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "gh"
    fake.write_text(
        f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(Path(__file__).with_name('fake_project_gh.py'))!r}, run_name='__main__')\n"
    )
    fake.chmod(0o700)
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"CLAUDE_CODE_SESSION_ID", "CODEX_SESSION_ID", "CODEX_THREAD_ID"}
        and not key.startswith("PROGRESS_")
    }
    env.update(
        PATH=f"{binary}:{os.environ['PATH']}",
        PYTHONPATH=str(OPS),
        CODEX_SESSION_ID="progress-owner",
        XDG_STATE_HOME=str(tmp_path / "state"),
        PROGRESS_GH_STATE=str(tmp_path / "github.json"),
        PROGRESS_GH_CALLS=str(tmp_path / "calls.jsonl"),
    )
    state = tmp_path / "state/tennis-lab-agents/parent-sessions"
    state.mkdir(parents=True)
    (state / "registry.json").write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "o/r#1204": {
                        "session": "codex:progress-owner",
                        "status": "active",
                        "generation": 1,
                        "tasks": {},
                        "reviews": [],
                    }
                },
            }
        )
    )
    project = ProjectEnv(tmp_path, env)
    project.save(base_state())
    return project


def update(n: int = 1206, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "items": [
            {
                "issue": n,
                "stage": "作業中",
                "wait_reason": "",
                "next_action": "検証を進める",
                **values,
            }
        ],
    }


def test_setup_plan_and_apply_preserve_existing_project_and_workflows(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    before = p.state()
    dry = json.loads(p.run("project-setup", "--dry-run").stdout)
    assert len(dry["plan"]) == 7 and dry["dry_run"]
    assert not p.mutations() and p.state() == before
    result = json.loads(p.run("project-setup", "--apply").stdout)
    after = p.state()
    assert after["project"] == before["project"]
    assert after["fields"][:7] == before["fields"]
    assert after["views"][0] == before["views"][0]
    assert after["workflows"] == before["workflows"]
    assert result["manual_steps"] == [] and len(result["views"]) == 3
    board = next(row for row in after["views"] if row["name"] == "子タスクのカンバン")
    assert board["columns"] == ["段階"] and board["groups"] == ["Parent issue"]
    assert {row["name"] for row in after["fields"]} >= {
        "段階",
        "待ち理由",
        "次の作業",
        "親の受入状態",
    }
    count = len(p.mutations())
    assert not json.loads(p.run("project-setup", "--apply").stdout)["plan"]
    assert len(p.mutations()) == count


def test_initial_registration_and_honest_denominators(project_env: ProjectEnv) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    spec = p.initial()
    before = len(p.mutations())
    result = json.loads(p.sync("initial", spec, "--dry-run").stdout)
    assert len(p.mutations()) == before
    assert sum(row["operation"] == "add_item" for row in result["plan"]) == 7
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    original_registry = registry.read_bytes()
    result = json.loads(p.sync("initial", spec, "--apply").stdout)
    assert registry.read_bytes() == original_registry
    root = next(row for row in result["issues"] if row["number"] == 1204)
    assert root["title"] == "正式タイトル 1204" and root["assignees"] == ["researcher"]
    assert [row["number"] for row in root["children"]] == [1205, 1206, 1207, 1210]
    assert root["progress"]["direct"] == {
        "completed": 1,
        "total": 3,
        "cancelled": [1207],
    }
    assert (
        root["progress"]["leaves"]["completed"] == 2
        and root["progress"]["leaves"]["total"] == 3
    )
    assert root["progress"]["leaves"]["excluded_subtrees"] == [1207, 1212]
    assert root["progress"]["standard"]["total"] == 4
    assert root["progress"]["checklist"]["completion_disagreements"] == [1205]
    assert root["progress"]["checklist"]["missing"] == [1210]
    assert root["progress"]["checklist"]["not_direct"] == [1999]
    text = p.item(1204)["values"]["次の作業"]
    assert "直属 1/3（取消1除外）" in text and "末端 2/3" in text and "標準 2/4" in text
    assert p.item(1204)["values"]["親の受入状態"] == "未確認"
    assert p.item(1207)["values"].get("段階") is None
    count = len(p.mutations())
    assert not json.loads(p.sync("initial", spec, "--apply").stdout)["plan"]
    assert len(p.mutations()) == count


def test_milestones_start_pr_stop_resume_and_verified_merge(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.initialize()
    p.sync("start", update(), "--apply")
    p.sync("pr", update(stage="レビュー中", pr=99), "--apply")
    p.sync(
        "stop",
        update(stage="レビュー中", wait_reason="CI失敗: 修正後に再実行"),
        "--apply",
    )
    assert p.item(1206)["values"]["段階"] == "レビュー中"
    assert p.item(1206)["values"]["待ち理由"].startswith("CI失敗")
    p.sync("resume", update(stage="レビュー中"), "--apply")
    assert p.item(1206)["values"]["待ち理由"] is None
    state = p.state()
    state["issues"]["1206"].update(state="CLOSED", stateReason="COMPLETED")
    p.save(state)
    before = len(p.mutations())
    result = p.sync("merge", update(stage="完了", pr=98), "--apply", ok=False)
    assert "merged" in result.stderr and len(p.mutations()) == before
    state = p.state()
    state["issues"]["1206"].update(state="OPEN", stateReason=None)
    state["pulls"][0].update(state="closed", merged_at="2026-10-22T03:00:00Z")
    p.save(state)
    p.sync("merge", update(stage="完了", pr=99), "--apply", ok=False)
    state["issues"]["1206"].update(state="CLOSED", stateReason="COMPLETED")
    p.save(state)
    p.sync("merge", update(stage="完了", pr=99), "--apply")
    assert p.item(1206)["values"]["段階"] == "完了"
    assert "直属 2/3" in p.item(1204)["values"]["次の作業"]
    assert p.item(1204)["values"]["親の受入状態"] == "未確認"


def test_all_children_closed_do_not_accept_parent_automatically(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    state = p.state()
    for n in ("1206", "1210"):
        state["issues"][n].update(state="CLOSED", stateReason="COMPLETED")
    p.save(state)
    p.initialize()
    assert p.item(1204)["values"]["親の受入状態"] == "未確認"
    spec = update(1204, stage="完了", acceptance="受入済み")
    p.sync("refresh", spec, "--apply", ok=False)
    p.sync("accept", spec, "--apply")
    assert p.item(1204)["values"]["親の受入状態"] == "受入済み"
    assert p.state()["issues"]["1204"]["state"] == "OPEN"


def test_parent_pr_merge_is_acceptance_pending(project_env: ProjectEnv) -> None:
    p = project_env
    state = p.state()
    state["pulls"].append(
        {
            "number": 101,
            "body": "Closes #1204",
            "state": "closed",
            "merged_at": "2026-10-22T03:00:00Z",
            "html_url": "https://github.com/o/r/pull/101",
        }
    )
    p.save(state)
    p.initialize()
    p.sync(
        "merge",
        update(1204, stage="受入待ち", acceptance="受入待ち", pr=101),
        "--apply",
    )
    assert p.item(1204)["values"]["親の受入状態"] == "受入待ち"
    assert p.item(1204)["values"]["段階"] == "受入待ち"


def test_acceptance_checks_unfinished_descendants_of_a_closed_parent(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    state = p.state()
    for n in ("1206", "1210"):
        state["issues"][n].update(state="CLOSED", stateReason="COMPLETED")
    state["issues"]["1211"].update(state="OPEN", stateReason=None)
    p.save(state)
    p.initialize()
    result = p.sync(
        "accept", update(1204, stage="完了", acceptance="受入済み"), "--apply", ok=False
    )
    assert "unfinished descendant" in result.stderr
    assert p.item(1204)["values"]["親の受入状態"] == "未確認"


def test_an_undivided_nested_parent_still_requires_explicit_acceptance(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    state = p.state()
    state["issues"]["1210"].update(
        children=[],
        state="CLOSED",
        stateReason="COMPLETED",
        subIssuesSummary={"total": 0, "completed": 0, "percentCompleted": 0},
    )
    del state["issues"]["1211"]
    del state["issues"]["1212"]
    p.save(state)
    p.initialize()
    assert p.item(1210)["values"]["親の受入状態"] == "未確認"
    assert p.item(1210)["values"]["段階"] == "作業中"


def test_cancelled_branch_excludes_its_completed_descendants(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    state = p.state()
    state["issues"]["1210"].update(state="CLOSED", stateReason="NOT_PLANNED")
    p.save(state)
    result = p.inspect()
    assert result.returncode == 0, result.stderr
    root = next(
        row for row in json.loads(result.stdout)["issues"] if row["number"] == 1204
    )
    assert root["progress"]["direct"] == {
        "completed": 1,
        "total": 2,
        "cancelled": [1207, 1210],
    }
    assert root["progress"]["leaves"]["completed"] == 1
    assert root["progress"]["leaves"]["total"] == 2


@pytest.mark.parametrize(
    "operation",
    [
        "createProjectV2Field",
        "createProjectV2View",
        "updateProjectV2View",
        "createProjectView",
    ],
)
def test_setup_recovers_after_lost_mutation_response(
    project_env: ProjectEnv, operation: str
) -> None:
    p = project_env
    p.run("project-setup", "--apply", ok=False, PROGRESS_LOSE_RESPONSE=operation)
    p.run("project-setup", "--apply", PROGRESS_LOSE_RESPONSE=operation)
    state = p.state()
    assert len(state["fields"]) == 11 and len(state["views"]) == 4
    assert not json.loads(p.run("project-setup").stdout)["plan"]


@pytest.mark.parametrize("missing", ["owner", "field"])
def test_rest_creation_requires_numeric_ids_before_any_write(
    project_env: ProjectEnv, missing: str
) -> None:
    p = project_env
    state = p.state()
    if missing == "owner":
        state["project"]["owner"].pop("databaseId")
    else:
        state["fields"][0].pop("databaseId")
    p.save(state)
    p.run("project-setup", "--apply", ok=False)
    assert not p.mutations()


def test_rest_creation_failure_does_not_create_an_unconfigured_board(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.run("project-setup", "--apply", ok=False, PROGRESS_FAIL_WRITE="createProjectView")
    assert not any(row["name"] == "子タスクのカンバン" for row in p.state()["views"])
    assert not any(
        row["payload"]
        and row["payload"].get("variables", {}).get("input", {}).get("name")
        == "子タスクのカンバン"
        for row in p.calls()
    )
    p.run("project-setup", "--apply")
    assert len(p.state()["fields"]) == 11 and len(p.state()["views"]) == 4


def test_same_host_parent_lock_serializes_concurrent_setup(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: p.run("project-setup", "--apply"), range(2)))
    assert all(result.returncode == 0 for result in results)
    assert len(p.state()["fields"]) == 11 and len(p.state()["views"]) == 4


@pytest.mark.parametrize(
    "operation", ["addProjectV2ItemById", "updateProjectV2ItemFieldValue"]
)
def test_sync_recovers_partial_registration_without_duplicates(
    project_env: ProjectEnv, operation: str
) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    spec = p.initial()
    p.sync("initial", spec, "--apply", ok=False, PROGRESS_LOSE_RESPONSE=operation)
    p.sync("initial", spec, "--apply", PROGRESS_LOSE_RESPONSE=operation)
    assert len(p.state()["items"]) == 7
    assert not json.loads(p.sync("initial", spec).stdout)["plan"]


@pytest.mark.parametrize(
    "env",
    [
        {"PROGRESS_FAIL_READ": "ProgressItems"},
        {"PROGRESS_PARTIAL_ERROR": "ProgressIssue"},
        {"PROGRESS_FAIL_PULLS": "1"},
        {"PROGRESS_BAD_PAGE": "1"},
    ],
)
def test_read_failures_do_not_write_or_report_success(
    project_env: ProjectEnv, env: dict[str, str]
) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    before = len(p.mutations())
    p.sync("initial", p.initial(), "--apply", ok=False, **env)
    assert len(p.mutations()) == before


def test_stale_issue_snapshot_aborts_before_first_write(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    before = len(p.mutations())
    result = p.sync("initial", p.initial(), "--apply", ok=False, PROGRESS_RACE="1")
    assert "changed before writing" in result.stderr and len(p.mutations()) == before


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "wrong_type",
        "missing_option",
        "duplicate",
        "permission",
        "archived",
        "identity",
        "hierarchy",
    ],
)
def test_schema_identity_and_permissions_fail_before_writes(
    project_env: ProjectEnv, kind: str
) -> None:
    p = project_env
    p.initialize()
    state = p.state()
    if kind == "missing":
        state["fields"] = [row for row in state["fields"] if row["name"] != "待ち理由"]
    elif kind == "wrong_type":
        next(row for row in state["fields"] if row["name"] == "待ち理由")[
            "dataType"
        ] = "NUMBER"
    elif kind == "missing_option":
        next(row for row in state["fields"] if row["name"] == "段階")["options"].pop()
    elif kind == "duplicate":
        duplicate = copy.deepcopy(state["fields"][-1])
        duplicate["id"] = "DUPLICATE"
        state["fields"].append(duplicate)
    elif kind == "permission":
        state["project"]["viewerCanUpdate"] = False
    elif kind == "archived":
        next(row for row in state["items"] if row["content"]["number"] == 1206)[
            "isArchived"
        ] = True
    elif kind == "identity":
        state["project"]["number"] = 4
    else:
        state["issues"]["1206"]["parent"]["number"] = 1210
    p.save(state)
    count = len(p.mutations())
    p.sync("start", update(), "--apply", ok=False)
    assert len(p.mutations()) == count


@pytest.mark.parametrize(
    "spec",
    [
        {"schema_version": 2, "items": []},
        update(9999),
        update(1206, typo="ignored?"),
        update(1207, stage="完了"),
        update(1206, stage="完了"),
        update(1204, acceptance="受入済み", stage="完了"),
        update(1206, next_action="/private/worktree"),
    ],
)
def test_invalid_specs_and_unaccepted_work_are_rejected(
    project_env: ProjectEnv, spec: dict[str, Any]
) -> None:
    p = project_env
    p.initialize()
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    value = json.loads(registry.read_text())
    value["parents"]["o/r#1204"]["tasks"] = {"1206": {"worktree": "/private/worktree"}}
    registry.write_text(json.dumps(value))
    count = len(p.mutations())
    p.sync("refresh", spec, "--apply", ok=False)
    assert len(p.mutations()) == count


def test_only_owner_can_write_and_pm_inspection_has_no_state_or_mutation(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.run("project-setup", "--apply", CODEX_SESSION_ID="another-session", ok=False)
    assert not p.mutations()
    result = p.inspect(XDG_STATE_HOME=str(p.tmp / "never-created"), CODEX_SESSION_ID="")
    assert result.returncode == 0, result.stderr
    assert not (p.tmp / "never-created").exists() and not p.mutations()
    data = json.loads(result.stdout)
    prs = next(row for row in data["issues"] if row["number"] == 1206)["pull_requests"]
    assert [row["state"] for row in prs] == ["closed_unmerged", "open"]


def test_complete_project_connections_beyond_one_hundred_items(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    state = p.state()
    state["fields"] += [
        {"id": f"EXTRA{i}", "name": f"extra{i}", "dataType": "TEXT"} for i in range(105)
    ]
    state["items"] = [
        {
            "id": f"OTHER{i}",
            "isArchived": False,
            "content": {"id": f"PR{i}", "__typename": "PullRequest", "number": i},
            "values": {},
        }
        for i in range(205)
    ]
    p.save(state)
    result = p.inspect()
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["project_item_count"] == 205 and len(report["fields"]) == 112
    assert not p.mutations()


def test_ui_configuration_is_reported_until_columns_and_groups_are_set(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    state = p.state()
    board = next(row for row in state["views"] if row["name"] == "子タスクのカンバン")
    board.update(columns=["Status"], groups=[])
    p.save(state)
    assert len(json.loads(p.run("project-setup").stdout)["manual_steps"]) == 2
    board.update(columns=["段階"], groups=["Parent issue"])
    p.save(state)
    result = json.loads(p.run("project-setup").stdout)
    assert result["manual_steps"] == [] and result["plan"] == []


def test_initial_requires_every_issue_and_dropped_writes_do_not_succeed(
    project_env: ProjectEnv,
) -> None:
    p = project_env
    p.run("project-setup", "--apply")
    partial = p.initial()
    partial["items"].pop()
    before = len(p.mutations())
    p.sync("initial", partial, "--apply", ok=False)
    assert len(p.mutations()) == before
    p.sync("initial", p.initial(), "--apply")
    result = p.sync("start", update(), "--apply", ok=False, PROGRESS_DROP_WRITE="1")
    assert "did not converge" in result.stderr
