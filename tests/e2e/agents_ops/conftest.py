"""Fixtures for .agents/ops tests: a fake ``gh`` with on-disk state, a fake
``agent_exec.sh`` and a throwaway git repository."""

from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS_REPORT = ROOT / ".agents/ops/ops-report/ops_report.py"
NOW = "2026-10-12T08:00:00+09:00"  # Monday of ISO week 2026-W42

FAKE_GH = r"""
import json, os, re, subprocess, sys, time

args = sys.argv[1:]
with open(os.environ["FAKE_GH_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(args) + "\n")
fail = os.environ.get("FAKE_GH_FAIL")
if fail and " ".join(args).startswith(fail):
    print(f"fake gh: injected failure for {fail!r}", file=sys.stderr)
    sys.exit(1)
path = os.environ["FAKE_GH_STATE"]
with open(path, encoding="utf-8") as fh:
    state = json.load(fh)


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def opts(name):
    return [args[i + 1] for i, a in enumerate(args) if a == name]


def emit(rows):
    fields = opt("--json").split(",")
    rows = rows[: int(opt("--limit", "30"))]
    print(json.dumps([{k: r[k] for k in fields} for r in rows]))


def gh_issue(r):
    return {**r, "labels": [{"name": n} for n in r["labels"]], "author": r.get("author", {"login": "bot"})}


def save():
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


head = args[:2]
if head == ["repo", "view"]:
    print(json.dumps({"nameWithOwner": "o/r"}))
elif head == ["api", "graphql"]:
    request = json.load(sys.stdin)
    assert "userContentEdits" in request["query"]
    variables = request["variables"]
    number = variables["number"]
    row = next(r for r in state["issues"] if r["number"] == number)
    history = list(reversed(state.get("edit_history", {}).get(str(number), [])))
    offset = int(variables.get("after") or "0")
    print(json.dumps({"data": {"repository": {"issue": {
        "number": number, "body": row["body"], "createdAt": row["createdAt"], "includesCreatedEdit": bool(history),
        "userContentEdits": {"totalCount": len(history), "nodes": history[offset:offset + 100],
            "pageInfo": {"hasNextPage": offset + 100 < len(history), "endCursor": str(offset + 100)}},
    }}}}))
elif args[0] == "api" and "--method" in args:
    method = opt("--method")
    endpoint = next(a for a in args if a == "user" or a.startswith("repos/"))
    payload = json.load(sys.stdin) if method != "GET" else None
    api_path, _, query = endpoint.partition("?")
    def api_emit(value, code=200):
        if "--include" in args:
            print(f"HTTP/1.1 {code} {'OK' if code < 300 else 'Not Found'}\n")
        print(json.dumps(value))
        sys.exit(0 if code < 300 else 1)
    def api_issue(row):
        return {**gh_issue(row), "id": row["number"] + 10000, "state": row["state"].lower(),
                "created_at": row["createdAt"],
                "state_reason": row.get("state_reason"), "assignees": row.get("assignees", []),
                "repository_url": "https://api.github.com/repos/o/r"}
    def api_pr(row):
        return {**row, "state": row.get("state", "open").lower(), "merged_at": row.get("mergedAt"),
                "body": state.get("pr_details", {}).get(str(row["number"]), {}).get("body", ""),
                "head": {"ref": row["headRefName"], "sha": "a" * 40}}
    def page(rows):
        p = int(re.search(r"(?:^|&)page=(\d+)", query)[1]) if "page=" in query else 1
        return rows[(p-1)*100:p*100]
    def api_saved(value):
        # A concurrent human edit after an irreversible write, including a lost response.
        for old, new in json.loads(os.environ.get("FAKE_GH_API_EDIT", "[]")):
            for issue in state["issues"]:
                if "ops-report" in issue["labels"]:
                    issue["body"] = issue["body"].replace(old, new)
        save()
        if os.environ.get("FAKE_GH_API_FAIL_AFTER_SAVE") == api_path:
            print("injected failure after save", file=sys.stderr)
            sys.exit(1)
        api_emit(value, 201)
    if method != "GET" and os.environ.get("FAKE_GH_API_FAIL") == api_path:
        print("injected API failure", file=sys.stderr)
        sys.exit(1)
    if api_path == "user":
        api_emit({"login": "bot"})
    if api_path == "repos/o/r/actions/variables/ISSUE_REWRITE_ENABLED_SINCE":
        api_emit({"message": "Not Found"}, 404)
    if api_path == "repos/o/r/pulls":
        api_emit(page([api_pr(r) for r in state["prs"]]))
    if api_path.startswith("repos/o/r/pulls/"):
        api_emit(api_pr(next(r for r in state["prs"] if r["number"] == int(api_path.rsplit("/", 1)[1]))))
    timeline = re.fullmatch(r"repos/o/r/issues/(\d+)/timeline", api_path)
    if timeline:
        rows = [api_pr(r) for r in state["prs"]]
        events = state.get("timeline", {}).get(timeline[1], [])
        api_emit(page(events + [{"id": r["number"], "event": "cross-referenced", "source": {"issue": {
            **r, "repository_url": "https://api.github.com/repos/o/r", "pull_request": {"url": "fixture-pr"}}}}
            for r in rows if re.search(r"#" + timeline[1] + r"\b", r["body"])]))
    m = re.fullmatch(r"repos/o/r/issues(?:/(\d+))?(?:/(parent|sub_issues|comments|dependencies/blocked_by))?", api_path)
    assert m, args
    n, tail = m.groups()
    if n is None:
        if method == "GET":
            api_emit(page([api_issue(i) for i in state["issues"]]))
        number = state["next_number"]
        state["next_number"] += 1
        row = {"number": number, **payload, "state": "OPEN", "state_reason": None,
               "createdAt": "2026-10-12T00:00:00Z", "updatedAt": "2026-10-12T00:00:00Z",
               "url": f"https://github.com/o/r/issues/{number}"}
        state["issues"].append(row)
        api_saved(api_issue(row))
    row = next((r for r in state["issues"] if r["number"] == int(n)), None)
    if row is None:
        api_emit({"message": "Not Found"}, 404)
    if tail == "parent":
        parent = state.get("parents", {}).get(n)
        api_emit(api_issue(next(i for i in state["issues"] if i["number"] == parent)) if parent else {"message": "No parent issue found"}, 200 if parent else 404)
    if tail == "dependencies/blocked_by":
        api_emit(page([api_issue(i) for i in state["issues"] if i["number"] in state.get("dependencies", {}).get(n, [])]))
    if tail == "sub_issues":
        if method == "GET":
            api_emit(page([api_issue(i) for i in state["issues"] if state.get("parents", {}).get(str(i["number"])) == int(n)]))
        child = next(i for i in state["issues"] if i["number"] + 10000 == payload["sub_issue_id"])
        assert str(child["number"]) not in state.get("parents", {})
        state.setdefault("parents", {})[str(child["number"])] = int(n)
        api_saved(api_issue(child))
    if tail == "comments":
        if method == "GET":
            api_emit(page(state.get("comments", {}).get(n, [])))
        time.sleep(float(os.environ.get("FAKE_GH_CREATE_DELAY", "0")))
        comment_id = state["next_comment_id"]
        state["next_comment_id"] += 1
        comment = {"id": comment_id, "user": {"login": "bot"}, "body": payload["body"],
                   "created_at": "2026-10-12T00:00:00Z", "updated_at": "2026-10-12T00:00:00Z",
                   "html_url": f"https://github.com/o/r/issues/{n}#issuecomment-{comment_id}"}
        state.setdefault("comments", {}).setdefault(n, []).append(comment)
        api_saved(comment)
    if method == "PATCH":
        if "body" in payload and row["body"] != payload["body"]:
            history = state.setdefault("edit_history", {}).setdefault(n, [])
            if not history:
                history.append({"id": f"{n}-created", "diff": row["body"], "editedAt": row["createdAt"], "deletedAt": None})
            history.append({"id": f"{n}-edit-{len(history)}", "diff": payload["body"], "editedAt": "2026-10-31T00:00:01Z", "deletedAt": None})
        row.update(payload)
        api_saved(api_issue(row))
    api_emit(api_issue(row))
elif args == ["api", "user"]:
    print(json.dumps({"login": "bot"}))
elif head == ["issue", "list"]:
    rows = state["issues"]
    st = opt("--state", "open")
    if st != "all":
        rows = [r for r in rows if r["state"].lower() == st]
    if opt("--label"):
        rows = [r for r in rows if opt("--label") in r["labels"]]
    emit([gh_issue(r) for r in sorted(rows, key=lambda r: -r["number"])])
elif head == ["issue", "view"]:
    issue = next(r for r in state["issues"] if r["number"] == int(args[2]))
    print(json.dumps({k: gh_issue(issue)[k] for k in opt("--json").split(",")}))
elif head == ["issue", "create"]:
    # Widen the check-then-create window so unlocked concurrent runs would race.
    time.sleep(float(os.environ.get("FAKE_GH_CREATE_DELAY", "0")))
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    number = state["next_number"]
    state["next_number"] += 1
    with open(opt("--body-file"), encoding="utf-8") as fh:
        body = fh.read()
    state["issues"].append({
        "number": number, "title": opt("--title"), "body": body, "labels": opts("--label"),
        "state": "OPEN", "createdAt": "2026-10-12T00:00:00Z", "updatedAt": "2026-10-12T00:00:00Z",
        "url": f"https://github.com/o/r/issues/{number}",
        "author": {"login": os.environ.get("FAKE_GH_CREATE_AUTHOR", "bot")},
    })
    # Simulate a human editing report checkboxes while triage is filing children.
    for old, new in json.loads(os.environ.get("FAKE_GH_CREATE_EDIT", "[]")):
        for issue in state["issues"]:
            if "ops-report" in issue["labels"]:
                issue["body"] = issue["body"].replace(old, new)
    save()
    print(os.environ.get("FAKE_GH_CREATE_URL", f"https://github.com/o/r/issues/{number}"))
elif head == ["issue", "edit"]:
    issue = next(r for r in state["issues"] if r["number"] == int(args[2]))
    if opt("--title") is not None:
        issue["title"] = opt("--title")
    with open(opt("--body-file"), encoding="utf-8") as fh:
        issue["body"] = fh.read()
    save()
    if os.environ.get("FAKE_GH_EDIT_FAIL_AFTER_SAVE"):
        print("fake gh: edit response lost after publication", file=sys.stderr)
        sys.exit(1)
elif head == ["pr", "list"]:
    if opt("--head"):
        rows = [r for r in state.get("memory_prs", []) if r["headRefName"] == opt("--head")]
        if opt("--state") != "all":
            rows = [r for r in rows if r["state"].lower() == opt("--state")]
        emit(rows)
    elif opt("--state") == "open":
        emit(state["prs"])
    else:
        rows = state["merged_prs"]
        search = opt("--search")
        if search:
            assert search.startswith("merged:>="), search
            rows = [r for r in rows if r["mergedAt"][:10] >= search[len("merged:>="):]]
        emit(rows)
elif head == ["pr", "view"]:
    memory_pr = next((r for r in state.get("memory_prs", []) if args[2] in [r["url"], str(r["number"])]), None)
    if memory_pr:
        detail = memory_pr
        if detail["state"] == "OPEN":
            detail["baseRefOid"] = subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip()
            ci = os.environ.get("FAKE_MEMORY_CI", "SUCCESS")
            detail["statusCheckRollup"] = [] if ci == "MISSING" else [{
                "__typename": "CheckRun", "name": "Python", "workflowName": "CI",
                "status": "IN_PROGRESS" if ci == "PENDING" else "COMPLETED",
                "conclusion": ci if ci != "PENDING" else "",
                "startedAt": "2026-10-12T00:00:00Z",
            }]
            if os.environ.get("FAKE_MEMORY_CHANGE_HEAD"):
                detail["views"] = detail.get("views", 0) + 1
                if detail["views"] >= 2:
                    detail["headRefOid"] = "a" * 40
            save()
    else:
        detail = state["pr_details"][args[2]]
    print(json.dumps({k: detail[k] for k in opt("--json").split(",")}))
elif head == ["pr", "create"]:
    branch = opt("--head")
    sha = subprocess.check_output(["git", "rev-parse", branch], text=True).strip()
    base = subprocess.check_output(["git", "rev-parse", "origin/main"], text=True).strip()
    number = state["next_number"]
    state["next_number"] += 1
    with open(opt("--body-file"), encoding="utf-8") as fh:
        body = fh.read()
    url = f"https://github.com/o/r/pull/{number}"
    state.setdefault("memory_prs", []).append({
        "number": number, "url": url, "body": body, "state": "OPEN",
        "headRefName": branch, "headRefOid": sha, "baseRefName": "main", "baseRefOid": base,
        "isDraft": False, "isCrossRepository": False, "mergeable": "MERGEABLE", "statusCheckRollup": [],
    })
    save()
    print(url)
elif head == ["pr", "merge"]:
    pr = next(r for r in state["memory_prs"] if r["url"] == args[2])
    assert opt("--match-head-commit") == pr["headRefOid"]
    upstream = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    subprocess.run(["git", "--git-dir", upstream, "update-ref", "refs/heads/main", pr["headRefOid"]], check=True)
    pr["state"] = "MERGED"
    save()
elif head == ["run", "list"]:
    emit(state["runs"])
elif head == ["label", "list"]:
    emit([{"name": n} for n in state["labels"]])
elif head == ["label", "create"]:
    state["labels"].append(args[2])
    state.setdefault("label_specs", {})[args[2]] = {
        "color": opt("--color"), "description": opt("--description")
    }
    save()
else:
    print(f"fake gh: unsupported {args}", file=sys.stderr)
    sys.exit(2)
"""

FAKE_AGENT_EXEC = r"""#!/usr/bin/env bash
set -euo pipefail
out="" prompt="" agent=""
while (($#)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --mcp-config) export FAKE_AUDIT_MCP_CONFIG="$2"; shift 2 ;;
    --prompt-file) prompt="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    *) shift ;;
  esac
done
# tmp + rename: concurrent runs in one test must not race on the copy.
cp "$prompt" "$FAKE_AGENT_PROMPT_COPY.$$" && mv -f "$FAKE_AGENT_PROMPT_COPY.$$" "$FAKE_AGENT_PROMPT_COPY"
echo "$agent" > "$FAKE_AGENT_USED"
if [[ -n "${FAKE_AGENT_SPAWN:-}" ]]; then
  # A descendant like the tools a real agent CLI starts; it must not outlive the run.
  sleep 300 >/dev/null 2>&1 &
  echo $! > "$FAKE_AGENT_SPAWN"
fi
"$FAKE_AUDIT_PYTHON" "$FAKE_AUDIT_SCRIPT" "$prompt" "$FAKE_AGENT_OUTPUT" "$out"
"""

AGENT_OUTPUT: dict[str, Any] = {
    "memory_update": None,
    "summary": "- 今回は小さな修正が中心。\n- 放置PRが1件ある。",
    "proposals": [
        {
            "category": "code-debt",
            "priority": "high",
            "title": "big.py を分割する",
            "detail": "big.py が肥大化している。\n- [ ] これはチェックボックスではなく本文",
            "evidence": "big.py 300行",
            "purpose": "大きなファイルの変更を容易にする。",
            "scope": "big.py の責務を分割する。",
            "expected_effect": "影響範囲が小さくなる。",
            "acceptance": ["既存のテストが通り、責務を分けられる。"],
            "related_parent": None,
            "related_issues": [],
        },
        {
            "category": "control-rules",
            "priority": "low",
            "title": "運用レポートのプロンプトを改善する",
            "detail": "提案数の目安を見直す。",
            "evidence": ".agents/ops/ops-report/prompt.md",
            "options": ["5件に絞る", "上限なし"],
            "purpose": "提案の使いやすさを改善する。",
            "scope": "運用レポートのプロンプト。",
            "expected_effect": "人間が提案を判断しやすくなる。",
            "acceptance": ["採用する提案の範囲を判断できる。"],
            "related_parent": None,
            "related_issues": [],
        },
    ],
    "stale_pr_notes": {
        "7": {"recommendation": "rebaseして仕上げ", "reason": "差分が小さい。"}
    },
}


def base_state() -> dict[str, Any]:
    return {
        "next_number": 100,
        "next_comment_id": 1,
        "labels": ["ai-proposed"],
        "issues": [
            {
                "number": 50,
                "title": "古い提案",
                "body": "x",
                "labels": ["ai-proposed"],
                "state": "OPEN",
                "createdAt": "2026-08-01T00:00:00Z",
                "updatedAt": "2026-08-02T00:00:00Z",
                "url": "https://github.com/o/r/issues/50",
            },
        ],
        "prs": [
            {
                "number": 7,
                "title": "放置されたPR",
                "headRefName": "claude/old",
                "author": {"login": "me"},
                "createdAt": "2026-09-01T00:00:00Z",
                "updatedAt": "2026-09-20T00:00:00Z",
                "isDraft": False,
                "labels": [],
                "url": "https://github.com/o/r/pull/7",
            },
            {
                "number": 8,
                "title": "新しいPR",
                "headRefName": "codex/new",
                "author": {"login": "me"},
                "createdAt": "2026-10-11T00:00:00Z",
                "updatedAt": "2026-10-11T00:00:00Z",
                "isDraft": True,
                "labels": [{"name": "needs-human"}],
                "url": "https://github.com/o/r/pull/8",
            },
        ],
        "pr_details": {
            "7": {
                "additions": 40,
                "deletions": 3,
                "changedFiles": 2,
                "files": [{"path": "src/a.py"}, {"path": "tests/test_a.py"}],
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "statusCheckRollup": [
                    {"conclusion": "FAILURE", "status": "COMPLETED"},
                    {"state": "SUCCESS"},
                ],
                "closingIssuesReferences": [{"number": 42}],
                "body": "Fixes #42",
            }
        },
        "merged_prs": [
            {
                "number": 6,
                "title": "merged",
                "headRefName": "x",
                "mergedAt": "2026-10-10T00:00:00Z",
            },
            {
                "number": 5,
                "title": "old",
                "headRefName": "y",
                "mergedAt": "2026-09-01T00:00:00Z",
            },
        ],
        "runs": [
            {
                "databaseId": 1,
                "workflowName": "CI",
                "conclusion": "failure",
                "status": "completed",
                "headBranch": "main",
                "event": "push",
                "createdAt": "2026-10-11T00:00:00Z",
                "url": "https://github.com/o/r/actions/runs/1",
            },
            {
                "databaseId": 2,
                "workflowName": "CI",
                "conclusion": "success",
                "status": "completed",
                "headBranch": "main",
                "event": "push",
                "createdAt": "2026-09-01T00:00:00Z",
                "url": "https://github.com/o/r/actions/runs/2",
            },
        ],
    }


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def make_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "big.py").write_text("x = 1\n" * 300, encoding="utf-8")
    (path / "small.py").write_text("# TODO: tidy\ny = 2\n", encoding="utf-8")
    (path / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
    (path / "knowledge").mkdir()
    (path / "knowledge/summary.md").write_text("# summary\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(
        path,
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "-m", "init", "--date", "2026-10-09T00:00:00+09:00",
    )  # fmt: skip
    return path


def _install(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@dataclass
class OpsEnv:
    tmp: Path
    repo: Path
    env: dict[str, str]
    state_file: Path
    calls_file: Path

    @property
    def state(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
        return data

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        return [
            json.loads(ln)
            for ln in self.calls_file.read_text(encoding="utf-8").splitlines()
        ]

    def set_agent_output(self, payload: dict[str, Any] | str) -> None:
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        Path(self.env["FAKE_AGENT_OUTPUT"]).write_text(text, encoding="utf-8")

    def publish_proposal_fixture(self) -> None:
        """Feed triage's published-proposal contract independently of audit routing."""
        sys.path.insert(0, str(ROOT / ".agents/ops/ops-report"))
        proposals = importlib.import_module("opsreport.proposals")
        payload = json.loads(Path(self.env["FAKE_AGENT_OUTPUT"]).read_text())
        props = [
            proposals.Proposal(
                section=proposals.STALE_PR_SECTION,
                title="#7 放置されたPR",
                detail="推奨: rebaseして仕上げ",
                evidence="#7",
                options=proposals.STALE_PR_OPTIONS,
                purpose="PRの対応を判断する。",
                scope="PR #7",
                expected_effect="放置状態を解消する。",
                acceptance=("対応結果を記録する。",),
            )
        ]
        for raw in payload["proposals"]:
            args = {k: v for k, v in raw.items() if k != "category"}
            props.append(
                proposals.Proposal(
                    section=proposals.CATEGORIES[raw["category"]], **args
                )
            )
        markdown, ids = proposals.render_proposals(props, "20261012")
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.repo, text=True
        ).strip()
        body = proposals.report_marker(
            slot="20261012",
            since=datetime.fromisoformat("2026-10-09T08:00:00+09:00"),
            until=datetime.fromisoformat(NOW),
            proposals=len(ids),
            base=base,
        )
        body += (
            "\n# 運用レポート\n"
            + proposals.PROPOSALS_BEGIN
            + "\n"
            + markdown
            + "\n"
            + proposals.PROPOSALS_END
            + "\n"
        )
        state = self.state
        number = state["next_number"]
        state["next_number"] += 1
        state["issues"].append(
            {
                "number": number,
                "title": "運用レポート 2026-10-12",
                "body": body,
                "labels": ["ops-report"],
                "state": "OPEN",
                "createdAt": "2026-10-12T00:00:00Z",
                "updatedAt": "2026-10-12T00:00:00Z",
                "url": f"https://github.com/o/r/issues/{number}",
                "author": {"login": "bot"},
            }
        )
        self.write_state(state)

    def _cmd(self, args: tuple[str, ...]) -> list[str]:
        cmd = [sys.executable, str(OPS_REPORT), *args]
        cmd += ["--repo-root", str(self.repo)]
        if "--log-dir" not in args:
            cmd += ["--log-dir", str(self.tmp / "logs")]
        if args[0] == "schedule" and "--now" not in args:
            cmd += ["--now", NOW]
        if args[0] == "report":
            # Defaults for tests; a test may pass its own --base-ref / --now.
            if "--base-ref" not in args:
                cmd += ["--base-ref", "main"]
            if "--now" not in args:
                cmd += ["--now", NOW]
        return cmd

    def run(
        self, *args: str, cwd: Path | None = None, **extra_env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._cmd(args),
            cwd=cwd,
            env={**self.env, **extra_env},
            text=True,
            capture_output=True,
            check=False,
        )

    def start(self, *args: str, **extra_env: str) -> subprocess.Popen[str]:
        """Start a run in the background (for concurrency tests)."""
        return subprocess.Popen(
            self._cmd(args),
            env={**self.env, **extra_env},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_dirs(self) -> list[Path]:
        return sorted((self.tmp / "logs").iterdir())


@pytest.fixture
def agent_output() -> dict[str, Any]:
    """A valid agent final message (fresh deep copy per test)."""
    copied: dict[str, Any] = json.loads(json.dumps(AGENT_OUTPUT))
    return copied


@pytest.fixture
def ops_env(tmp_path: Path) -> Iterator[OpsEnv]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir / "gh", f"#!{sys.executable}\n{FAKE_GH}")
    _install(
        bin_dir / "codex",
        f"#!{sys.executable}\n"
        + Path(__file__).with_name("fake_organization_cli.py").read_text(),
    )
    fake_exec = tmp_path / "fake_agent_exec.sh"
    _install(fake_exec, FAKE_AGENT_EXEC)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OPS_REPORT_", "WEEKLY_REPORT_", "FAKE_"))
    }
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_GH_STATE": str(tmp_path / "gh_state.json"),
            "FAKE_GH_CALLS": str(tmp_path / "gh_calls.jsonl"),
            "FAKE_AGENT_OUTPUT": str(tmp_path / "agent_output.json"),
            "FAKE_AGENT_PROMPT_COPY": str(tmp_path / "prompt_seen.md"),
            "FAKE_AGENT_USED": str(tmp_path / "agent_used.txt"),
            "FAKE_AUDIT_SCRIPT": str(Path(__file__).with_name("fake_audit_agent.py")),
            "FAKE_AUDIT_PYTHON": sys.executable,
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "OPS_REPORT_AGENT_EXEC": str(fake_exec),
            "OPS_REPORT_ORGANIZER_EXEC": str(ROOT / ".agents/ops/lib/agent_exec.sh"),
            "CODEX_HOME": str(tmp_path / "codex-home"),
            "FAKE_ORGANIZATION_PLAN": str(tmp_path / "organization_plan.json"),
            "FAKE_ORGANIZATION_CALLS": str(tmp_path / "organization_calls.jsonl"),
            "OPS_REPORT_LOCK_FILE": str(tmp_path / "state" / "ops-report.lock"),
        }
    )
    ops = OpsEnv(
        tmp=tmp_path,
        repo=make_repo(tmp_path / "repo"),
        env=env,
        state_file=tmp_path / "gh_state.json",
        calls_file=tmp_path / "gh_calls.jsonl",
    )
    ops.write_state(base_state())
    ops.set_agent_output(AGENT_OUTPUT)
    init = ops.run(
        "schedule",
        "init",
        "--now",
        "2026-10-11T12:00:00+09:00",
        "--log-dir",
        str(tmp_path / "setup-logs"),
    )
    assert init.returncode == 0, init.stderr
    yield ops
