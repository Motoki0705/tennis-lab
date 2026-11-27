"""Parent workflow through fake gh, private temporary state and real throwaway git."""

from __future__ import annotations

import copy
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from tests.support.agents_ops.directory_seed import DirectorySeed
from tests.support.agents_ops.model_pins import REVIEW_FIXED_MODEL
from tests.support.agents_ops.worker_preflight import completed_implementation

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"


def untracked_legacy_fixture(p: ParentEnv) -> None:
    """Construct pre-journal test data; never use this to repair live evidence."""
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    assert directory.is_relative_to(p.root)
    for name in ("comment-outbox", "comment-updates"):
        path = directory / name
        if path.exists():
            shutil.rmtree(path)


FAKE_GH = r"""
import json, os, re, sys
from urllib.parse import parse_qs
args = sys.argv[1:]
with open(os.environ["PARENT_GH_CALLS"], "a") as f:
    f.write(json.dumps(args) + "\n")
state_path = os.environ["PARENT_GH_STATE"]
with open(state_path) as f:
    state = json.load(f)
def response(value, code=200):
    if "--include" in args:
        print(f"HTTP/1.1 {code} {'OK' if code == 200 else 'Not Found'}\n")
    print(json.dumps(value))
    sys.exit(0 if code < 300 else 1)
def save():
    with open(state_path, "w") as f:
        json.dump(state, f)
def fail():
    print("injected GitHub failure", file=sys.stderr)
    sys.exit(1)
if args[:2] == ["repo", "view"]:
    response({"nameWithOwner": "o/r"})
if args == ["--version"]:
    print("gh version 2.97.0")
    sys.exit(0)
if args[:2] == ["run", "watch"]:
    assert "--repo" in args and args[args.index("--repo") + 1] == "o/r"
    if os.environ.get("PARENT_EXPECT_UNLOCKED_WATCH"):
        import fcntl
        with open(".git/ops-code.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state.get("watch_label"):
        state["issues"]["1"]["labels"].append({"name": state["watch_label"]})
        save()
    sys.exit(state.get("watch_returncode", 0))
if args[:2] == ["pr", "view"]:
    if os.environ.get("PARENT_FAIL_CI_VIEW"):
        fail()
    pr = next(p for p in state["prs"] if p["number"] == int(args[2]))
    values = {"statusCheckRollup": pr.get("statusCheckRollup", []),
              "headRefOid": pr["head"]["sha"]}
    if os.environ.get("PARENT_VIEW_HEAD_RACE"):
        values["headRefOid"] = os.environ["PARENT_VIEW_HEAD_RACE"]
    response({k: values[k] for k in args[args.index("--json") + 1].split(",")})
if args[:2] == ["pr", "merge"]:
    pr = next(p for p in state["prs"] if p["number"] == int(args[2]))
    assert "--merge" in args and "--match-head-commit" in args
    expected = args[args.index("--match-head-commit") + 1]
    if os.environ.get("PARENT_MERGE_FAIL") or pr["state"] != "open" or pr["head"]["sha"] != expected:
        fail()
    if os.environ.get("PARENT_MERGE_HEAD_RACE"):
        pr["head"]["sha"] = os.environ["PARENT_MERGE_HEAD_RACE"]
        save()
        fail()
    pr["state"] = "closed"
    pr["merged"] = True
    pr["merged_at"] = "2026-10-18T03:00:00Z"
    pr["merge_commit_sha"] = pr["head"]["sha"]
    for child in re.findall(r"Closes #(\d+)", pr["body"]):
        if not os.environ.get("PARENT_MERGE_SKIP_CLOSE"):
            state["issues"][child].update(state="closed", state_reason="completed")
    if os.environ.get("PARENT_MERGE_ADVANCE_BASE"):
        import subprocess
        subprocess.run(["git", "merge", "--no-edit", "--no-ff", expected], check=True, capture_output=True)
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        pr["merge_commit_sha"] = base
        if os.environ.get("PARENT_MERGE_UNEXPLAINED_COMMIT"):
            subprocess.run(["git", "commit", "--allow-empty", "-m", "unexplained base advance"], check=True, capture_output=True)
            base = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        for other in state["prs"]:
            other["base"]["sha"] = base
            if other["state"] == "open" and str(other["number"]) == os.environ.get("PARENT_MERGE_FOLLOW_MAIN"):
                subprocess.run(["git", "-C", os.environ["PARENT_MERGE_FOLLOW_WORKTREE"], "merge", "--no-edit", base], check=True, capture_output=True)
                other["head"]["sha"] = subprocess.check_output(["git", "-C", os.environ["PARENT_MERGE_FOLLOW_WORKTREE"], "rev-parse", "HEAD"], text=True).strip()
    if os.environ.get("PARENT_MERGE_CI_PENDING"):
        target = int(os.environ["PARENT_MERGE_CI_PENDING"])
        next(p for p in state["prs"] if p["number"] == target)["statusCheckRollup"] = [
            {"__typename": "CheckRun", "name": "Python", "workflowName": "CI",
             "status": "QUEUED", "conclusion": ""}
        ]
    save()
    if os.environ.get("PARENT_MERGE_RESPONSE_LOST"):
        fail()
    response({"merged": True})
assert args[0] == "api", args
endpoint = next(a for a in args if a in ("user", "graphql") or a.startswith("repos/"))
method = "POST" if endpoint == "graphql" else args[args.index("--method") + 1]
if os.environ.get("PARENT_FAIL_PATH") == endpoint and method != "GET" and not (os.environ.get("PARENT_FAIL_AFTER_CHILD_RELATION") and state["parents"].get("10") != 1):
    fail()
if os.environ.get("PARENT_FAIL_GET_PATH") == endpoint and method == "GET":
    fail()
payload = json.load(sys.stdin) if method != "GET" else None
path, _, query = endpoint.partition("?")
path = path.casefold()
page = int(re.search(r"(?:^|&)page=(\d+)", query)[1]) if "page=" in query else 1
def paged(rows):
    return rows[(page-1)*100:page*100]
if endpoint == "user":
    response({"login": "operator"})
def knowledge_closing(pr):
    return [
        [m[1] or m[3] or "o/r", int(m[2] or m[4])]
        for m in re.finditer(
            r"\b(?:clos(?:e|es|ed)|fix(?:es|ed)?|resolv(?:e|es|ed))[: ]+"
            r"(?:<?https://github.com/([\w.-]+/[\w.-]+)/issues/(\d+)|"
            r"([\w.-]+/[\w.-]+)?#(\d+))",
            pr["body"], re.I,
        )
    ]
if endpoint == "graphql":
    assert method == "POST" and "query KnowledgeClosingIssues" in payload["query"]
    variables = payload["variables"]
    assert variables["after"] is None
    assert (variables["owner"], variables["name"]) == ("o", "r")
    pr = next(p for p in state["prs"] if p["number"] == variables["number"])
    refs = pr.get("closing_refs", knowledge_closing(pr))
    nodes = [{"number": n, "repository": {"nameWithOwner": repo}} for repo, n in refs]
    observation = {"data": {"repository": {"nameWithOwner": "o/r", "pullRequest": {
        "number": pr["number"], "state": pr["state"].upper(), "isDraft": pr["draft"],
        "headRefOid": pr["head"]["sha"], "baseRefOid": pr["base"]["sha"],
        "baseRefName": pr["base"]["ref"], "title": pr.get("title"), "body": pr["body"],
        "closingIssuesReferences": {"nodes": nodes, "totalCount": len(nodes),
                                    "pageInfo": {"hasNextPage": False, "endCursor": None}},
    }}}}
    if "after_first_closing_read" in pr:
        pr.update(pr.pop("after_first_closing_read"))
        save()
    response(observation)
if path == "repos/o/r/labels/parent-evidence":
    response({"name": "parent-evidence"})
if path == "repos/o/r/actions/variables/issue_supplement_enabled_since":
    supplement = state.pop("complete_supplement_on_activation_read", None)
    if supplement:
        row = state["issues"][str(supplement["number"])]
        row["body"] = supplement["body"]
        row["labels"].append({"name": "supplement:done"})
        save()
    code = state.get("supplement_variable_status", 200 if "supplement_enabled_since" in state else 404)
    response({"name": "ISSUE_SUPPLEMENT_ENABLED_SINCE", "value": state.get("supplement_enabled_since")}, code)
if path == "repos/o/r/actions/workflows/issue-supplement.yml/runs":
    response({"workflow_runs": paged(state.get("supplement_runs", []))}, state.get("supplement_workflow_status", 200))
if path == "repos/o/r":
    response({"default_branch": state["default_branch"], "owner": {"login": "operator"}})
if path == "repos/o/r/branches/main":
    import subprocess
    tip = subprocess.check_output(["git", "rev-parse", "main"], text=True).strip()
    response({"name": "main", "commit": {"sha": tip}})
if path == "repos/o/r/actions/workflows/knowledge.yml/runs":
    head = parse_qs(query)["head_sha"][0]
    assert parse_qs(query)["event"] == ["pull_request"]
    rows = [run for run in state["knowledge_runs"] if run["head_sha"] == head]
    response({"total_count": len(rows), "workflow_runs": paged(rows)})
artifacts = re.fullmatch(r"repos/o/r/actions/runs/(\d+)/artifacts", path)
if artifacts:
    rows = state["knowledge_artifacts"].get(artifacts[1], [])
    response({"total_count": len(rows), "artifacts": paged(rows)})
artifact_zip = re.fullmatch(r"repos/o/r/actions/artifacts/(\d+)/zip", path)
if artifact_zip:
    from pathlib import Path
    sys.stdout.buffer.write(Path(state["artifact_zips"][artifact_zip[1]]).read_bytes())
    sys.exit(0)
knowledge_merge = re.fullmatch(r"repos/o/r/pulls/(\d+)/merge", path)
if knowledge_merge:
    import subprocess
    assert method == "PUT" and payload["merge_method"] == "merge"
    pr = next(p for p in state["prs"] if p["number"] == int(knowledge_merge[1]))
    if os.environ.get("PARENT_KNOWLEDGE_SEND_FAILURE"):
        fail()
    if pr["state"] != "open" or pr["head"]["sha"] != payload["sha"]:
        response({"message": "HEAD changed"}, 409)
    subprocess.run(["git", "merge", "--no-edit", "--no-ff", payload["sha"]], check=True, capture_output=True)
    merge = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    pr.update(state="closed", merged=True, merged_at="2026-11-22T00:00:00Z", merge_commit_sha=merge)
    for repo, child in pr.get("closing_refs", knowledge_closing(pr)):
        assert repo.casefold() == "o/r"
        state["issues"][str(child)].update(state="closed", state_reason="completed")
    state.setdefault("knowledge_merges", []).append({"number": pr["number"], "payload": payload})
    save()
    if os.environ.get("PARENT_KNOWLEDGE_LOST_RESPONSE"):
        fail()
    response({"sha": merge, "merged": True})
commit = re.fullmatch(r"repos/o/r/commits/([0-9a-f]{40})", path)
if commit:
    if state.get("knowledge_merges"):
        import subprocess
        parents = subprocess.check_output(["git", "show", "-s", "--format=%P", commit[1]], text=True).split()
        response({"sha": commit[1], "parents": [{"sha": sha} for sha in parents]})
    response({"sha": commit[1]})
checks = re.fullmatch(r"repos/o/r/commits/([0-9a-f]{40})/check-runs", path)
if checks:
    assert method == "GET" and "check_name=Python&filter=latest" in query
    pr = next(p for p in state["prs"] if checks[1] in {p["head"]["sha"], p.get("merge_commit_sha")})
    if "ci_response" in pr:
        response(pr["ci_response"])
    runs = pr.get("ci_check_runs")
    if runs is None:
        runs = [
            {"id": pr["number"] * 1000 + i + 1, "name": c["name"],
             "head_sha": checks[1], "check_suite": {"id": pr["number"]}, "status": c["status"].lower(),
             "conclusion": (c["conclusion"] or "").lower() or None}
            for i, c in enumerate(pr.get("statusCheckRollup") or [])
            if c.get("__typename") == "CheckRun" and c.get("name") == "Python"
        ]
    response({"total_count": len(runs), "check_runs": paged(runs)})
if path == "repos/o/r/actions/runs":
    head = parse_qs(query).get("head_sha", [None])[0]
    prs = [p for p in state["prs"] if head in {p["head"]["sha"], p.get("merge_commit_sha")}]
    rows = [
        {"id": p["number"], "check_suite_id": p["number"], "head_sha": head,
         "head_branch": p["base"]["ref"] if p.get("merged") else p["head"]["ref"],
         "event": "push" if p.get("merged") else "pull_request",
         "path": ".github/workflows/ci.yml", "run_number": p["number"], "run_attempt": 1,
         "status": "completed", "conclusion": None}
        for p in prs
    ]
    response({"total_count": len(rows), "workflow_runs": rows})
ci_jobs = re.fullmatch(r"repos/o/r/actions/runs/(\d+)/attempts/1/jobs", path)
if ci_jobs:
    rows = state.get("failed_ci_jobs", {}).get(ci_jobs[1], [])
    response({"total_count": len(rows), "jobs": paged(rows)})
ci_log = re.fullmatch(r"repos/o/r/actions/jobs/(\d+)/logs", path)
if ci_log and ci_log[1] in state.get("fake_ci_log_ids", []):
    assert "--include" in args
    print("HTTP/1.1 302 Found\nLocation: https://ci.invalid/fake-log\n")
    sys.exit(0)
pull_comments = re.fullmatch(r"repos/o/r/pulls/(\d+)/(comments|reviews)", path)
if pull_comments:
    assert method == "GET"
    response(paged(state.get("pull_" + pull_comments[2], {}).get(pull_comments[1], [])))
pull = re.fullmatch(r"repos/o/r/pulls/(\d+)", path)
if pull:
    if os.environ.get("PARENT_READ_HEAD_RACE") and int(pull[1]) == 100:
        state["detail_reads"] = state.get("detail_reads", 0) + 1
        if state["detail_reads"] == 2:
            next(p for p in state["prs"] if p["number"] == 100)["head"]["sha"] = os.environ["PARENT_READ_HEAD_RACE"]
        save()
    if int(pull[1]) == state["prs"][0]["number"]:
        if os.environ.get("PARENT_CHANGE_CLOSING_ON_READ"):
            state["closing_reads"] = state.get("closing_reads", 0) + 1
            if state["closing_reads"] == 2:
                state["prs"][0]["body"] = os.environ["PARENT_CHANGE_CLOSING_ON_READ"]
            save()
        if os.environ.get("PARENT_CHANGE_PULLS_ON_READ"):
            state["pull_reads"] = state.get("pull_reads", 0) + 1
            if state["pull_reads"] == 2:
                state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
            save()
    pr = next(p for p in state["prs"] if p["number"] == int(pull[1]))
    if pr.get("merged") and os.environ.get("PARENT_KNOWLEDGE_LOST_RESPONSE"):
        fail()
    response(pr)
comment_edit = re.fullmatch(r"repos/o/r/issues/comments/(\d+)", path)
if comment_edit:
    issue, comment = next((n, c) for n, rows in state["comments"].items() for c in rows if c["id"] == int(comment_edit[1]))
    if method == "GET":
        response({**comment, "html_url": f"https://github.com/o/r/issues/{issue}#issuecomment-{comment['id']}"})
    assert method == "PATCH"
    comment["body"] = payload["body"]
    save()
    response(comment)
timeline = re.fullmatch(r"repos/o/r/issues/(\d+)/timeline", path)
if timeline:
    rows = [{"event": "cross-referenced", "created_at": "2026-10-24T00:00:00Z", "source": {"issue": {"id": p["number"] + 10000, "number": p["number"], "body": p.get("body", ""), "repository_url": "https://api.github.com/repos/o/r", "pull_request": {"url": "pr"}}}} for p in state["prs"] if re.search(r"#" + timeline[1] + r"\b", p.get("body", ""))]
    references_parent = re.compile(r"#" + timeline[1] + r"\b")
    overrides = state.get("issue_listing_overrides", {})
    for issue in state["issues"].values():
        source = {**issue, **overrides.get(str(issue["number"]), {})}
        body = source.get("body") or ""
        if body.startswith("<!-- parent-evidence:") and references_parent.search(body):
            source.setdefault("id", source["number"] + 10000)
            rows.append({
                "event": "cross-referenced", "created_at": source["created_at"],
                "source": {"type": "issue", "issue": source},
            })
    response(paged(rows))
if path == "repos/o/r/pulls":
    if os.environ.get("PARENT_CHANGE_CLOSING_ON_READ"):
        state["closing_reads"] = state.get("closing_reads", 0) + 1
        if state["closing_reads"] == 2:
            state["prs"][0]["body"] = os.environ["PARENT_CHANGE_CLOSING_ON_READ"]
        save()
    if os.environ.get("PARENT_CHANGE_PULLS_ON_READ"):
        state["pull_reads"] = state.get("pull_reads", 0) + 1
        if state["pull_reads"] == 2:
            state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
        save()
    response(paged(state["prs"]))
match = re.fullmatch(r"repos/o/r/issues(?:/(\d+))?(?:/(parent|sub_issues|comments|dependencies/blocked_by))?", path)
assert match, endpoint
n, tail = match.groups()
if n is None:
    if method == "GET":
        rows = list(state["issues"].values())
        params = parse_qs(query)
        if "labels" in params:
            labels = set(params["labels"][0].split(","))
            rows = [r for r in rows if labels <= {label["name"] for label in r["labels"]}]
            if "parent-evidence" in labels:
                rows = [r for r in rows if r["number"] not in state.get("unlisted_evidence", [])]
        overrides = state.get("issue_listing_overrides", {})
        response(paged([{**r, **overrides.get(str(r["number"]), {})} for r in rows]))
    assert method == "POST", method
    if payload["labels"] == ["parent-evidence"]:
        if os.environ.get("PARENT_REJECT_EVIDENCE_CREATE"):
            print("HTTP/1.1 403 Forbidden\n")
            response({"message": "injected evidence creation rejection"}, 403)
        n = str(state.get("next_evidence", 9000))
        state["next_evidence"] = int(n) + 1
    else:
        n = str(state["next"])
        state["next"] += 1
    row = {"id": int(n)+10000, "number": int(n), "repository_url": "https://api.github.com/repos/o/r",
           "title": payload["title"], "body": payload["body"], "labels": [{"name": x} for x in payload["labels"]],
           "user": {"login": "operator"},
           "state": "open", "state_reason": None, "created_at": os.environ.get("PARENT_CREATED_AT", "2026-10-31T00:00:00Z")}
    state["issues"][n] = row
    save()
    if os.environ.get("PARENT_FAIL_AFTER_CREATE"):
        fail()
    response(row, 201)
if tail != "comments" and n not in state["issues"]:
    response({"message": "Not Found"}, 404)
row = state["issues"].get(n)
if tail == "parent":
    parent = state["parents"].get(n)
    response(state["issues"][str(parent)] if parent else {"message": "No parent issue found"}, 200 if parent else 404)
if tail == "dependencies/blocked_by":
    response(paged([state["issues"][str(d)] for d in state.get("dependencies", {}).get(n, [])]))
if tail == "sub_issues":
    if method == "GET":
        response(paged([i for key, i in state["issues"].items() if state["parents"].get(key) == int(n)]))
    child = next(i for i in state["issues"].values() if i["id"] == payload["sub_issue_id"])
    assert str(child["number"]) not in state["parents"]
    state["parents"][str(child["number"])] = int(n)
    save()
    response(child, 201)
if tail == "comments":
    if method == "GET":
        response(paged(state["comments"].get(n, [])))
    if os.environ.get("PARENT_REJECT_COMMENT") == n:
        print("HTTP/1.1 403 Forbidden\n")
        response({"message": "injected comment rejection"}, 403)
    comment = {"id": 1 + max((c["id"] for rows in state["comments"].values() for c in rows), default=0), "user": {"login": "operator"}, "body": payload["body"]}
    state["comments"].setdefault(n, []).append(comment)
    save()
    if os.environ.get("PARENT_FAIL_AFTER_COMMENT") in (n, "any"):
        fail()
    if os.environ.get("PARENT_FAIL_AFTER_COMMENT_CONTAINING") and os.environ["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] in payload["body"]:
        fail()
    response(comment, 201)
if method == "PATCH":
    if os.environ.get("PARENT_FAIL_PATCH_ISSUE") == n:
        fail()
    for k, v in payload.items():
        row[k] = [{"name": x} for x in v] if k == "labels" else v
    save()
response(row)
"""


def run_git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=directory, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def evidence_number(state: dict[str, Any], parent: int = 1) -> int:
    matches = [
        int(n)
        for n, row in state["issues"].items()
        if row["labels"] == [{"name": "parent-evidence"}]
        and json.loads(
            row["body"].splitlines()[0][len("<!-- parent-evidence:v1 ") : -4]
        )["parent"]
        == parent
    ]
    assert len(matches) == 1, "fixture requires exactly one evidence issue"
    return matches[0]


def evidence_comments(state: dict[str, Any], parent: int = 1) -> list[dict[str, Any]]:
    return cast(
        list[dict[str, Any]],
        state["comments"].setdefault(str(evidence_number(state, parent)), []),
    )


def work_issues(state: dict[str, Any]) -> dict[str, Any]:
    return {
        n: r
        for n, r in state["issues"].items()
        if {x["name"] for x in r["labels"]} != {"parent-evidence"}
    }


@dataclass
class ParentEnv:
    root: Path
    repo: Path
    env: dict[str, str]
    state_path: Path
    calls_path: Path
    formats: Any

    def state(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.state_path.read_text()))

    def save(self, state: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, ensure_ascii=False))

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def comments(self, n: int = 1) -> list[dict[str, Any]]:
        state = self.state()
        rows = list(state["comments"].get(str(n), []))
        evidence = [
            r
            for r in state["issues"].values()
            if r["labels"] == [{"name": "parent-evidence"}]
            and json.loads(
                r["body"].splitlines()[0][len("<!-- parent-evidence:v1 ") : -4]
            )["parent"]
            == n
        ]
        assert len(evidence) <= 1
        if evidence:
            rows += state["comments"].get(str(evidence[0]["number"]), [])
        return sorted(rows, key=lambda c: c["id"])

    def spec(self, value: Any) -> str:
        path = self.root / "input.json"
        path.write_text(json.dumps(value, ensure_ascii=False))
        return str(path)

    def note(self, text: str = "確認済みの証拠・次の操作。") -> str:
        path = self.root / "note.md"
        path.write_text(text)
        return str(path)

    def review_flags(self) -> tuple[str, ...]:
        """Explicit parent choice for workflow fixtures that cross state/gate boundaries."""
        path = self.root / "review-model-reason.md"
        path.write_text("永続状態・復旧・merge gateの境界を確認するためAstraを選ぶ。")
        return (
            "--review-model",
            REVIEW_FIXED_MODEL,
            "--review-effort",
            "xhigh",
            "--model-reason-file",
            str(path),
        )

    def announce_completion(self) -> dict[str, Any]:
        """Act as the parent: explicitly select and record the verified overall outcome."""
        source = json.loads(
            self.cli(
                "note",
                "--body-file",
                self.note("親が全体の受入・実merge/closeと残件を照合した。"),
            ).stdout
        )["url"]
        artifacts = [
            f"https://github.com/o/r/pull/{pr['number']}"
            for pr in self.state()["prs"]
            if pr.get("merged_at")
        ]
        assert artifacts, "completion fixture requires actual merged deliverables"
        return cast(
            dict[str, Any],
            json.loads(
                self.cli(
                    "notify",
                    "--spec",
                    self.spec(
                        {
                            "schema_version": 1,
                            "kind": "completion",
                            "key": "completion",
                            "summary": "全体の実装と受入を確認した。",
                            "evidence": [source],
                            "resolves": [],
                            "details": {
                                "achievements": [
                                    "全体の受入条件と実merge/closeを照合した。"
                                ],
                                "remaining": [],
                                "artifacts": artifacts,
                            },
                        }
                    ),
                ).stdout
            ),
        )

    def cli(
        self,
        *args: str,
        parent: str | None = "1",
        repo: str = "o/r",
        ok: bool | None = True,
        extra: dict[str, str] | None = None,
        summary: str | None = "テスト親の要約",
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        launcher = (
            [
                sys.executable,
                str(OPS / "checkout_job.py"),
                "--repo",
                str(self.repo),
                "--",
            ]
            if args and args[0] == "improvement-expire"
            else []
        )
        # Most callers exercise other parent contracts. Tests of missing/saved
        # summaries pass summary=None to send the command exactly as written.
        if (
            args[:1] == ("register",)
            and "--summary" not in args
            and summary is not None
        ):
            args = (*args, "--summary", summary)
        result = subprocess.run(
            [
                *launcher,
                sys.executable,
                "-m",
                "parent",
                "--repo",
                repo,
                *(("--parent", parent) if parent is not None else ()),
                *args,
            ],
            cwd=self.repo,
            env={**self.env, **(extra or {})},
            input=stdin,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if ok is True:
            assert result.returncode == 0, result.stderr + result.stdout
        elif ok is False:
            assert result.returncode != 0, result.stdout
        return result

    def agreed(self) -> None:
        self.cli("register")
        self.cli("checking")
        self.cli(
            "agree",
            "--spec",
            self.spec(
                {
                    "purpose": "目的を固定する",
                    "scope": "親と子",
                    "behavior": "合意後に実装する",
                    "acceptance": "fakeで検証する",
                    "human_confirmation": "この目的・範囲・動作・受入条件で進めてください。",
                }
            ),
        )

    def child(
        self,
        key: str,
        deps: list[int] | None = None,
        source: bool = False,
        ok: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        sections = {
            name: name + "について具体的に実装する。"
            for name in self.formats.CHILD_REQUIRED
        }
        sections["依存関係"] = "なし。"
        sections["Acceptance checklist"] = (
            "- [ ] 動作する\n  文脈: 親の目的。\n  下流への影響: 誤委任を防ぐ。\n  検証方法・証拠: fake gh。"
        )
        spec = {
            "role": "child",
            "summary": key,
            "parent": {"identifier": "開発", "number": 1},
            "sections": sections,
            "labels": ["enhancement", "ci", "tests"],
        }
        if source:
            # Report adoption is not the parent's existing requirements agreement.
            if "90" not in self.state()["issues"]:
                report_formats = importlib.import_module("opsreport.proposals")
                proposal = report_formats.Proposal(
                    section="その他の助言",
                    title="合意対象の提案",
                    evidence="fake gh",
                    purpose="目的",
                    scope="変更範囲",
                    expected_effect="効果",
                    acceptance=("検証できる",),
                    related_parent=1,
                )
                markdown, _ = report_formats.render_proposals([proposal], "20261012")
                marker = "<!-- ops-report:v2 slot=20261012 since=2026-10-09T00:00:00+00:00 until=2026-10-12T00:00:00+00:00 proposals=1 status=complete -->"
                state = self.state()
                state["issues"]["90"] = {
                    **state["issues"]["1"],
                    "number": 90,
                    "id": 10090,
                    "title": "運用レポート",
                    "labels": [{"name": "ops-report"}],
                    "body": marker
                    + "\n"
                    + report_formats.PROPOSALS_BEGIN
                    + "\n"
                    + markdown.replace("- [ ] **", "- [x] **")
                    + report_formats.PROPOSALS_END
                    + "\n",
                }
                self.save(state)
                self.cli(
                    "receive-proposal", "--report", "90", "--proposal", "R-20261012-01"
                )
                self.cli(
                    "agree",
                    "--report",
                    "90",
                    "--proposal",
                    "R-20261012-01",
                    "--spec",
                    self.spec(
                        {
                            "purpose": "提案の目的を固定する",
                            "scope": "提案由来の子",
                            "behavior": "合意後に実装する",
                            "acceptance": "fakeで検証する",
                            "human_confirmation": "この提案の要件で合意します。",
                        }
                    ),
                )
            spec["source"] = {
                "proposal_id": "R-20261012-01",
                "report_number": 90,
                "option": None,
            }
        manifest = {"key": key, "spec": spec, "dependencies": deps or []}
        return self.cli("create-child", "--spec", self.spec(manifest), ok=ok)

    def detail(self) -> str:
        return self.spec(
            {
                "reason": "仕様の確認待ち",
                "facts": "再現ログを確認済み",
                "impact": "このタスクと依存先",
                "resume_condition": "判断の合意を記録",
                "next_action": "確認を得て再開する",
            }
        )

    def worktree(self, name: str) -> Path:
        worktree = self.repo / ".claude/worktrees" / name
        run_git(self.repo, "worktree", "add", "-b", "codex/" + name, str(worktree))
        return worktree

    def pr(
        self, n: int, child: int, head: str | None = None, branch: str | None = None
    ) -> dict[str, Any]:
        return {
            "number": n,
            "title": f"子issue #{child} の実装",
            "state": "open",
            "draft": False,
            "body": f"Closes #{child}",
            "merged_at": None,
            "head": {
                "sha": head or run_git(self.repo, "rev-parse", "HEAD"),
                "ref": branch or f"codex/child-{child}",
                "repo": {"full_name": "o/r"},
            },
            "base": {
                "sha": run_git(self.repo, "rev-parse", "main"),
                "ref": "main",
                "repo": {"full_name": "o/r", "default_branch": "main"},
            },
        }


@pytest.fixture
def parent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParentEnv:
    monkeypatch.syspath_prepend(str(OPS))
    monkeypatch.syspath_prepend(str(OPS / "ops-report"))
    formats = importlib.import_module("issues")
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "--initial-branch=main")
    # Seed capture/restore may copy or delete this repo immediately after Git
    # returns. Keep automatic maintenance inside the command's lifetime.
    run_git(repo, "config", "maintenance.autoDetach", "false")
    run_git(repo, "config", "gc.autoDetach", "false")
    run_git(repo, "config", "user.name", "Test Operator")
    run_git(repo, "config", "user.email", "test@example.invalid")
    (repo / "input.txt").write_text("base\n")
    run_git(repo, "add", "input.txt")
    run_git(repo, "commit", "-m", "baseline")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + FAKE_GH)
    gh.chmod(0o755)
    for agent in ("claude", "codex"):
        path = bin_dir / agent
        path.write_text(
            "#!/bin/sh\necho 'a test must never launch an agent' >&2\nexit 97\n"
        )
        path.chmod(0o755)
    sections = {
        name: "未合意。"
        for name in formats.PARENT_REQUIRED
        if name != formats.TASKS_HEADING
    }
    body = formats.render_parent_body(sections, [])
    issues = {
        str(n): {
            "id": 10000 + n,
            "number": n,
            "repository_url": "https://api.github.com/repos/o/r",
            "title": f"[開発] 親{n}",
            "body": body,
            "labels": [{"name": "issue:parent"}],
            "state": "open",
            "state_reason": None,
            "created_at": "2026-10-31T00:00:00Z",
        }
        for n in (1, 2)
    }
    state_path = tmp_path / "github.json"
    state_path.write_text(
        json.dumps(
            {
                "issues": issues,
                "parents": {},
                "comments": {"1": []},
                "prs": [],
                "next": 10,
                "default_branch": "main",
            }
        )
    )
    calls_path = tmp_path / "calls.jsonl"
    calls_path.touch()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "CODEX_THREAD_ID",
            "CODEX_SESSION_ID",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_PARENT_SESSION_ID",
            "CLAUDE_CODE_AGENT_ID",
        }
        and not k.startswith("HERDR_")
    }
    env.update(
        PATH=str(bin_dir) + os.pathsep + env["PATH"],
        PYTHONPATH=str(OPS),
        PYTHONDONTWRITEBYTECODE="1",
        XDG_STATE_HOME=str(tmp_path / "private-state"),
        CLAUDE_CODE_SESSION_ID="private-session-a-unique",
        PARENT_GH_STATE=str(state_path),
        PARENT_GH_CALLS=str(calls_path),
    )
    # Parent operations must never reach the host Herdr executable. Display
    # integration tests explicitly replace this sentinel with their fake CLI.
    herdr = bin_dir / "herdr"
    herdr.write_text(
        "#!/bin/sh\necho 'unexpected Herdr access from parent fixture' >&2\nexit 99\n"
    )
    herdr.chmod(0o755)
    return ParentEnv(tmp_path, repo, env, state_path, calls_path, formats)


@pytest.mark.parametrize(
    "final",
    [None, "supplement:done", "supplement:failed", "supplement:skipped", "old_parent"],
)
def test_supplement_gates_checking_and_agree_before_writes(
    parent_env: ParentEnv, final: str | None
) -> None:
    p = parent_env
    p.cli("register")
    p.cli("checking")
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    if final == "old_parent":
        state["issues"]["1"]["created_at"] = "2026-10-29T23:59:59Z"
    elif final:
        state["issues"]["1"]["labels"].append({"name": final})
    p.save(state)
    offset = len(p.calls())
    checked = p.cli("checking", ok=final is not None)
    agreed = p.cli(
        "agree",
        "--spec",
        p.spec(
            {
                "purpose": "本文を固定する",
                "scope": "親issue",
                "behavior": "補足確定後に合意する",
                "acceptance": "書込み順を守る",
                "human_confirmation": "この内容で合意する。",
            }
        ),
        ok=final is not None,
    )
    if final is None:
        for result in (checked, agreed):
            assert "pending" in result.stderr
            assert (
                "wait-supplement" in result.stderr
                and "supplement:skipped" in result.stderr
            )
        assert not any(
            "--method" in call and call[call.index("--method") + 1] != "GET"
            for call in p.calls()[offset:]
        )
        assert p.state() == state
    else:
        assert json.loads(agreed.stdout)["phase"] == "要件合意済み"


@pytest.mark.parametrize(
    "final", ["supplement:done", "supplement:failed", "supplement:skipped"]
)
def test_wait_supplement_watches_latest_exact_title_and_refreshes_labels(
    parent_env: ParentEnv, final: str
) -> None:
    p = parent_env
    state = p.state()
    state.update(
        supplement_enabled_since="2026-10-30T00:00:00Z",
        supplement_runs=[
            {
                "id": 104,
                "display_title": "issue-supplement #11",
                "created_at": "2026-10-31T04:00:00Z",
            },
            {
                "id": 102,
                "display_title": "issue-supplement #1",
                "created_at": "2026-10-31T02:00:00Z",
            },
            {
                "id": 101,
                "display_title": "issue-supplement #1",
                "created_at": "2026-10-31T01:00:00Z",
            },
        ],
        watch_label=final,
    )
    p.save(state)
    env = {"CLAUDE_CODE_SESSION_ID": ""}
    if final == "supplement:done":
        ops = p.repo / ".agents/ops"
        shutil.copytree(OPS, ops, ignore=shutil.ignore_patterns("__pycache__"))
        run_git(p.repo, "add", ".agents/ops")
        run_git(p.repo, "commit", "-m", "install native parent for supplement wait")
        env.update(PYTHONPATH=str(ops), PARENT_EXPECT_UNLOCKED_WATCH="1")
    result = p.cli("wait-supplement", extra=env)
    assert json.loads(result.stdout) == {
        "parent": 1,
        "supplement_state": final.removeprefix("supplement:"),
        "run_id": 102,
    }
    assert [c for c in p.calls() if c[:2] == ["run", "watch"]] == [
        ["run", "watch", "102", "--repo", "o/r"]
    ]
    assert not any(
        "--method" in c and c[c.index("--method") + 1] != "GET" for c in p.calls()
    )


@pytest.mark.parametrize("workflow_status", [200, 404])
def test_wait_supplement_without_run_guides_dispatch_or_human_skip(
    parent_env: ParentEnv, workflow_status: int
) -> None:
    p = parent_env
    state = p.state()
    state.update(
        supplement_enabled_since="2026-10-30T00:00:00Z",
        supplement_workflow_status=workflow_status,
    )
    p.save(state)
    result = p.cli("wait-supplement", ok=False)
    assert all(
        word in result.stderr
        for word in ("issue-supplement.yml", "workflow_dispatch", "supplement:skipped")
    )
    assert not any(c[:2] == ["run", "watch"] for c in p.calls())


@pytest.mark.parametrize("watch_code", [0, 1])
def test_wait_supplement_does_not_claim_success_when_watch_or_final_state_fails(
    parent_env: ParentEnv, watch_code: int
) -> None:
    p = parent_env
    state = p.state()
    state.update(
        supplement_enabled_since="2026-10-30T00:00:00Z",
        supplement_runs=[
            {
                "id": 101,
                "display_title": "issue-supplement #1",
                "created_at": "2026-10-31T01:00:00Z",
            }
        ],
        watch_returncode=watch_code,
    )
    p.save(state)
    result = p.cli("wait-supplement", ok=False)
    assert ("still pending" if watch_code == 0 else "watch 101 failed") in result.stderr
    assert len([c for c in p.calls() if c[:2] == ["run", "watch"]]) == 1


def test_wait_supplement_pages_runs_once_and_skips_wait_for_final_state(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state.update(
        supplement_enabled_since="2026-10-30T00:00:00Z",
        supplement_runs=[
            {
                "id": 200 + n,
                "display_title": "issue-supplement #2",
                "created_at": "2026-10-31T02:00:00Z",
            }
            for n in range(100)
        ]
        + [
            {
                "id": 101,
                "display_title": "issue-supplement #1",
                "created_at": "2026-10-31T01:00:00Z",
            }
        ],
        watch_label="supplement:done",
    )
    p.save(state)
    assert json.loads(p.cli("wait-supplement").stdout)["run_id"] == 101
    assert any("page=2" in a for c in p.calls() for a in c)
    offset = len(p.calls())
    assert json.loads(p.cli("wait-supplement").stdout) == {
        "parent": 1,
        "supplement_state": "done",
        "run_id": None,
    }
    assert not any(c[:2] == ["run", "watch"] for c in p.calls()[offset:])


@pytest.mark.parametrize(
    "state_label", [None, "supplement:done", "supplement:failed", "supplement:skipped"]
)
def test_sync_cannot_patch_parent_during_supplement_even_without_agreement(
    parent_env: ParentEnv, state_label: str | None
) -> None:
    p = parent_env
    p.cli("register")
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    if state_label:
        state["issues"]["1"]["labels"].append({"name": state_label})
    p.save(state)
    offset = len(p.calls())
    result = p.cli("sync", ok=state_label is not None)
    if state_label is None:
        assert "pending" in result.stderr and "wait-supplement" in result.stderr
        assert not any("PATCH" in call for call in p.calls()[offset:])
        assert p.state()["issues"]["1"]["body"] == state["issues"]["1"]["body"]
    else:
        assert sum("PATCH" in call for call in p.calls()[offset:]) == 1
        assert "parent-progress:v1" in p.state()["issues"]["1"]["body"]


def test_update_body_gates_target_issue_instead_of_session_parent(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    state["issues"]["1"]["labels"].append({"name": "supplement:done"})
    p.save(state)
    store_module = importlib.import_module("parent.state")
    github_module = importlib.import_module("parent.github")
    session_module = importlib.import_module("parent.session")
    policy_module = importlib.import_module("parent.policy")
    with store_module.Store(p.root / "target-issue-state").locked() as store:
        gh = github_module.GitHub("o/r", store=store)
        session = session_module.ParentSession(gh, store, 1, None)
        old = gh.issue(2)["body"]
        with pytest.raises(ValueError, match="#2: issue supplement is pending"):
            session.update_body(
                2,
                policy_module.replace_section(
                    old, "進捗概要", "別の対象親の本文更新。"
                ),
                old,
            )
    assert not any("PATCH" in call for call in p.calls())


def body_with_supplement(body: str) -> str:
    anchor = "## 全体のスコープ"
    assert body.count(anchor) == 1
    return body.replace(
        anchor,
        "<details>\n<summary>Geminiによる補足・背景情報</summary>\n\n"
        "既存コードに基づく背景の事実。\n\n</details>\n\n" + anchor,
        1,
    )


def test_parent_body_update_detects_supplement_despite_cached_issue(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    store_module = importlib.import_module("parent.state")
    github_module = importlib.import_module("parent.github")
    session_module = importlib.import_module("parent.session")
    policy_module = importlib.import_module("parent.policy")
    with store_module.Store(p.root / "cache-test-state").locked() as store:
        gh = github_module.GitHub("o/r", store=store)
        session = session_module.ParentSession(gh, store, 1, None)
        old = gh.issue(1)["body"]
        proposed = policy_module.replace_section(
            old, "進捗概要", "子タスクを照合した。"
        )
        state = p.state()
        supplemented = body_with_supplement(old)
        state["issues"]["1"]["body"] = supplemented
        p.save(state)
        with pytest.raises(ValueError, match="body changed"):
            session.update_body(1, proposed, old)
    assert p.state()["issues"]["1"]["body"] == supplemented
    assert not any("PATCH" in c for c in p.calls())


def test_update_body_rechecks_body_with_final_supplement_state(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    p.save(state)
    store_module = importlib.import_module("parent.state")
    github_module = importlib.import_module("parent.github")
    session_module = importlib.import_module("parent.session")
    policy_module = importlib.import_module("parent.policy")
    with store_module.Store(p.root / "supplement-race-state").locked() as store:
        gh = github_module.GitHub("o/r", store=store)
        session = session_module.ParentSession(gh, store, 1, None)
        old = gh.issue(1)["body"]
        proposed = policy_module.replace_section(
            old, "進捗概要", "子タスクを照合した。"
        )
        supplemented = body_with_supplement(old)
        state = p.state()
        # The old implementation compares a fresh body before this activation
        # read, then accepts done from a second issue GET without comparing it.
        state["complete_supplement_on_activation_read"] = {
            "number": 1,
            "body": supplemented,
        }
        p.save(state)
        with pytest.raises(ValueError, match="#1: body changed"):
            session.update_body(1, proposed, old)
    final = p.state()
    assert "complete_supplement_on_activation_read" not in final
    assert final["issues"]["1"]["body"] == supplemented
    assert {label["name"] for label in final["issues"]["1"]["labels"]} >= {
        "supplement:done"
    }
    assert not any("PATCH" in call for call in p.calls())


def peer_board_request(p: ParentEnv, body: str) -> dict[str, Any]:
    """A peer posts and the recipient reads/acks, with no GitHub or parent write."""
    peer = {**p.env, "CLAUDE_CODE_SESSION_ID": "board-peer"}
    p.cli("register", parent="2", extra={"CLAUDE_CODE_SESSION_ID": "board-peer"})
    service = importlib.import_module("board.service")
    messages = importlib.import_module("board.messages")
    delivery = importlib.import_module("board.delivery")
    target = service.whoami(p.env)["name"]
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    github_before, registry_before, calls_before = (
        p.state(),
        registry.read_bytes(),
        p.calls(),
    )
    fake = delivery.FakeNotifier()
    created = messages.post(
        peer,
        to=[target],
        kind="request",
        subject="親への連絡",
        body=body,
        notifiers={"claude": fake, "codex": fake},
    )["post"]
    assert created["recipients"][0]["delivery_state"] == "notified"
    assert messages.read(p.env, created["id"])["post"]["body"] == body
    acknowledged = messages.ack(p.env, post_id=created["id"], state="done")["post"]
    assert acknowledged["recipients"][0]["response_state"] == "done"
    # No parent-session/parent-review marker can have been published anywhere.
    assert p.state() == github_before
    assert p.calls() == calls_before
    assert registry.read_bytes() == registry_before
    return cast(dict[str, Any], acknowledged)


@pytest.mark.parametrize("wrapper", ["exact", "quoted", "fenced"])
def test_board_marker_and_ack_cannot_supply_human_agreement(
    parent_env: ParentEnv,
    wrapper: str,
) -> None:
    p = parent_env
    p.agreed()
    state = p.state()
    # Move a genuine, otherwise valid agreement from fake GitHub to the board.
    # Keep the checking event so no other agreement can authorize the task.
    agreement = evidence_comments(state).pop()["body"]
    p.save(state)
    untracked_legacy_fixture(p)
    body = {
        "exact": agreement,
        "quoted": "引用\n" + agreement,
        "fenced": "```text\n" + agreement + "\n```",
    }[wrapper]
    peer_board_request(p, body)
    before = p.state()
    status = json.loads(p.cli("status").stdout)
    assert status["phase"] == "要件確認中"
    assert status["execution_basis"] is None
    assert status["review"] == "未指示"
    rejected = p.child("board-is-not-consent", ok=False)
    assert "要件合意記録または有効な実行根拠がありません" in rejected.stderr
    assert p.state() == before


def test_assignment_is_not_agreement_and_both_inputs(parent_env: ParentEnv) -> None:
    from tests.support.agents_ops.decision_logs import journal, result

    p = parent_env
    assert json.loads(p.cli("status").stdout)["assignment"] == "unassigned"
    # Late captured facts remain visible without registering their parent.
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    log = journal.Journal(directory, "o/r", 1)
    log.capture("historical-unassigned", result())
    before = log.path.read_bytes()
    unassigned = json.loads(p.cli("status").stdout)
    assert unassigned["remaining_publications"][0]["kind"] == "decision_log"
    assert unassigned["remaining_publications"][0]["unpublished_count"] == 1
    assert unassigned["decision_resume_command"] == (
        "parent --repo o/r --parent RECOVERY_PARENT publish-decision-log --unassigned-parent 1"
    )
    assert log.path.read_bytes() == before
    p.cli("register", parent="https://github.com/o/r/issues/1")
    assigned = json.loads(p.cli("status").stdout)
    assert assigned["phase"] == "担当指定のみ"
    assert (
        assigned["decision_resume_command"]
        == "parent --repo o/r --parent 1 publish-decision-log"
    )
    p.child("forbidden", ok=False)
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.root),
        ok=False,
    )
    assert len(work_issues(p.state())) == 2
    assert (
        sum(
            "POST" in call and call[-3:] == ["repos/o/r/issues", "--input", "-"]
            for call in p.calls()
        )
        == 1
    )  # The evidence container is allowed; no work issue was created.
    p.cli("checking")
    assert json.loads(p.cli("status").stdout)["phase"] == "要件確認中"
    p.child("still-forbidden", ok=False)
    p.agreed()
    assert json.loads(p.cli("status").stdout)["phase"] == "要件合意済み"
    assert (
        evidence_comments(p.state())[-1]["body"]
        .splitlines()[0]
        .startswith("<!-- parent-session:v1 ")
    )
    assert (
        "> この目的・範囲・動作・受入条件で進めてください。"
        in evidence_comments(p.state())[-1]["body"]
    )
    # Machine-private IDs never enter comments or issue bodies.
    assert "private-session-a-unique" not in json.dumps(p.state())
    p.cli("register", parent="https://github.com/other/repo/issues/1", ok=False)


def test_identity_errors_precede_io_and_codex_uses_environment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    for env in (
        {"CLAUDE_CODE_SESSION_ID": ""},
        {"CODEX_SESSION_ID": "also-present"},
        {"CLAUDE_CODE_SESSION_ID": "", "CODEX_THREAD_ID": "thread-only"},
        {
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "parent",
            "CODEX_THREAD_ID": "child",
        },
    ):
        p.cli("register", ok=False, extra=env)
    assert p.calls() == []
    p.cli(
        "register",
        extra={
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "codex-session-private",
            "CODEX_THREAD_ID": "codex-session-private",
        },
    )
    assert "codex-session-private" not in json.dumps(p.state())


def test_exclusive_assignment_release_ended_and_handoff(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    assert "o/r#1" in p.cli("register", parent="2", ok=False).stderr
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.cli("register", ok=False, extra=second)
    p.cli("handoff", "--accept", "--note-file", p.note(), ok=False, extra=second)
    p.cli("handoff", "--note-file", p.note())
    assert json.loads(p.cli("status").stdout)["assignment"] == "handoff_pending"
    p.child("pending-rejected", ok=False)
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    p.cli("checking", ok=False)  # previous session lost ownership
    p.cli("release", "--ended", "--note-file", p.note(), extra=second)
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"
    p.cli("register", ok=False)
    p.cli("handoff", "--accept", "--note-file", p.note())
    p.cli("release", "--note-file", p.note())
    assert json.loads(p.cli("status").stdout)["assignment"] == "released"
    p.cli("register", parent="2")


@pytest.mark.parametrize(
    ("first_repo", "second_repo"), [("O/R", "o/r"), ("o/r", "O/R")]
)
def test_repository_case_cannot_bypass_assignment(
    parent_env: ParentEnv, first_repo: str, second_repo: str
) -> None:
    p = parent_env
    p.cli("register", repo=first_repo)
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    before = registry.read_bytes()
    calls_before = len(p.calls())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    denied = p.cli("register", repo=second_repo, extra=second, ok=False)
    assert "別セッション担当中" in denied.stderr
    assert set(json.loads(before)["parents"]) == {"o/r#1"}
    assert registry.read_bytes() == before
    assert all("GET" in call for call in p.calls()[calls_before:])


@pytest.mark.parametrize("legacy_key", [None, "O/R#1"])
def test_repository_case_preserves_delegation_reservation(
    parent_env: ParentEnv, legacy_key: str | None
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    args = (
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.worktree("a")),
    )
    assert json.loads(p.cli(*args).stdout)["action"] == "launch"
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    if legacy_key:
        data = json.loads(registry.read_text())
        data["parents"][legacy_key] = data["parents"].pop("o/r#1")
        registry.write_text(json.dumps(data))
    before = registry.read_bytes()
    calls_before = len(p.calls())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    for repo in ("O/R", "o/r"):
        denied = p.cli("register", repo=repo, extra=second, ok=False)
        assert "別セッション担当中" in denied.stderr
        denied = p.cli(*args, repo=repo, extra=second, ok=False)
        assert "別セッション" in denied.stderr
        denied = p.cli(*args, repo=repo, ok=False)
        assert "already delegated" in denied.stderr
        status = json.loads(p.cli("status", repo=repo).stdout)
        assert status["current_session_is_owner"]
        assert status["tasks"]["10"]["delegated"]
    assert registry.read_bytes() == before
    assert all("GET" in call for call in p.calls()[calls_before:])


@pytest.mark.parametrize("alias_status", ["active", "released"])
def test_repository_alias_collision_fails_closed(
    parent_env: ParentEnv, alias_status: str
) -> None:
    p = parent_env
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    registry.parent.mkdir(parents=True)
    row = {
        "session": "claude:private-session-a-unique",
        "status": "active",
        "generation": 1,
        "tasks": {"10": {"delegated": True}},
        "reviews": [],
    }
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "O/R#1": row,
                    "o/r#1": {
                        **row,
                        "session": "claude:private-session-b-unique",
                        "status": alias_status,
                    },
                },
            }
        )
    )
    before = registry.read_bytes()
    for args in (
        ("status",),
        ("register",),
        (
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/a",
            "--worktree",
            str(p.root),
        ),
    ):
        denied = p.cli(*args, ok=False)
        assert "repository alias collision" in denied.stderr
        assert registry.read_bytes() == before
    assert p.calls() == []


@pytest.mark.parametrize("failure", ["after-create", "relation", "progress"])
def test_child_creation_recovers_partial_failure_without_duplicates(
    parent_env: ParentEnv, failure: str
) -> None:
    p = parent_env
    p.agreed()
    if failure == "after-create":
        p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    else:
        p.env["PARENT_FAIL_PATH"] = (
            "repos/o/r/issues/1/sub_issues"
            if failure == "relation"
            else "repos/o/r/issues/1"
        )
        if failure == "progress":
            p.env["PARENT_FAIL_AFTER_CHILD_RELATION"] = "1"
    p.child("idempotent", source=True, ok=False)
    p.env.pop("PARENT_FAIL_PATH", None)
    p.env.pop("PARENT_FAIL_AFTER_CREATE", None)
    p.env.pop("PARENT_FAIL_AFTER_CHILD_RELATION", None)
    if failure == "after-create":
        state = p.state()
        state["issues"]["10"]["labels"].append({"name": p.formats.STANDALONE_LABEL})
        p.save(state)
    result = json.loads(p.child("idempotent", source=True).stdout)
    assert result["reused"] and result["child"] == 10
    p.child("idempotent", source=True)
    state = p.state()
    assert len(work_issues(state)) == 4 and state["parents"]["10"] == 1
    child = state["issues"]["10"]
    assert child["body"].splitlines()[0].startswith("<!-- ops-report-proposal:")
    assert {x["name"] for x in child["labels"]} == {
        "enhancement",
        "ci",
        "tests",
        "issue:child",
    }
    assert "- [ ] #10 idempotent" in state["issues"]["1"]["body"]
    assert "private-session-a-unique" not in json.dumps(state)


def test_child_creation_only_adds_role_to_explicit_work_type_and_area_labels(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    # Parent #1 and child #20 must not share a cache-key prefix.
    state = p.state()
    state["next"] = 20
    p.save(state)
    p.child("explicit-labels")
    labels = {x["name"] for x in p.state()["issues"]["20"]["labels"]}
    assert labels == {"enhancement", "ci", "tests", "issue:child"}

    def label_patches() -> int:
        return sum(
            "PATCH" in call and "repos/o/r/issues/20" in call for call in p.calls()
        )

    assert label_patches() == 0
    state = p.state()
    state["issues"]["20"]["labels"].extend(
        [{"name": p.formats.STANDALONE_LABEL}, {"name": "agent:codex"}]
    )
    p.save(state)
    p.child("explicit-labels")
    labels.add("agent:codex")
    assert {x["name"] for x in p.state()["issues"]["20"]["labels"]} == labels
    assert label_patches() == 1
    p.child("explicit-labels")
    assert {x["name"] for x in p.state()["issues"]["20"]["labels"]} == labels
    assert label_patches() == 1
    assert (
        sum(
            "POST" in call and "repos/o/r/issues/1/sub_issues" in call
            for call in p.calls()
        )
        == 1
    )


def test_child_creation_requires_explicit_standalone_parent_migration(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    state = p.state()
    state["issues"]["1"]["labels"].append({"name": p.formats.STANDALONE_LABEL})
    p.save(state)
    before = len(p.calls())
    result = p.child("cannot-split", ok=False)
    assert "migrate the parent" in result.stderr
    assert p.formats.STANDALONE_LABEL in result.stderr
    assert p.state() == state
    assert not any(
        method in call for call in p.calls()[before:] for method in ("POST", "PATCH")
    )


def test_child_key_conflict_and_duplicate_existing_fail_closed(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("one")
    p.child("two")
    p.child("one", deps=[11], ok=False)
    state = p.state()
    state["issues"]["99"] = {**state["issues"]["10"], "number": 99, "id": 10099}
    p.save(state)
    p.child("one", ok=False)
    assert len(work_issues(p.state())) == 5


def test_concurrent_registration_never_owns_two_parents(parent_env: ParentEnv) -> None:
    p = parent_env
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda n: p.cli("register", parent=str(n), ok=None), (1, 2))
        )
    assert sorted(r.returncode for r in results) == [0, 2], [
        (r.stdout, r.stderr) for r in results
    ]
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    data = json.loads(registry.read_text())
    assert len(data["parents"]) == 1


def test_rechecking_revokes_agreement_and_missing_confirmation_fails(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("before-change")
    p.cli("checking")
    assert json.loads(p.cli("status").stdout)["phase"] == "要件合意済み"
    p.child("independent-work")
    agreement = {
        "purpose": "目的",
        "scope": "範囲",
        "behavior": "動作",
        "acceptance": "条件",
        "human_confirmation": "",
    }
    p.cli("agree", "--spec", p.spec(agreement), ok=False)
    agreement["human_confirmation"] = "この変更した要件で合意します。"
    p.cli("agree", "--spec", p.spec(agreement))
    p.child("after-change")
    assert len(work_issues(p.state())) == 5


def test_report_and_human_assignment_share_entry_and_never_imply_consent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["2"]["body"] = (
        "<!-- ops-report-proposal: R-opaque-2 report=#90 option=none -->\n"
        + state["issues"]["2"]["body"]
    )
    state["issues"]["2"]["labels"].append({"name": "ai-proposed"})
    p.save(state)
    for n in (1, 2):
        result = json.loads(p.cli("register", parent=str(n)).stdout)
        assert result["phase"] == "担当指定のみ"
        p.cli("release", "--note-file", p.note(), parent=str(n))
    assert len(work_issues(p.state())) == 2


def test_proposal_identity_prevents_duplicate_under_new_task_key(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("original", source=True)
    assert json.loads(p.child("new-key", source=True).stdout)["reused"]
    assert len(work_issues(p.state())) == 4


def test_local_identifiers_are_rejected_from_public_notes(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    p.cli("release", "--note-file", p.note(p.env["CLAUDE_CODE_SESSION_ID"]), ok=False)
    assert "private-session-a-unique" not in json.dumps(p.state())
    assert json.loads(p.cli("status").stdout)["assignment"] == "active"


def test_stop_resume_retries_and_repeated_cycles_keep_distinct_notices(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    for _ in range(2):
        for action in ("stop", "resume"):
            p.env["PARENT_FAIL_AFTER_COMMENT"] = str(evidence_number(p.state()))
            p.cli(action, "--child", "10", "--spec", p.detail(), ok=False)
            p.env.pop("PARENT_FAIL_AFTER_COMMENT")
            p.cli(action, "--child", "10", "--spec", p.detail())
    comments = evidence_comments(p.state())
    for event in ("child_stopped", "child_resumed"):
        assert (
            sum(f'"event": "{event}"' in c["body"].splitlines()[0] for c in comments)
            == 2
        )
    assert p.state()["issues"]["10"]["body"].count("再現ログを確認済み") == 4


def test_handoff_retries_do_not_lose_claim_or_repeat_notice(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.cli("handoff", "--note-file", p.note())
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.env["PARENT_FAIL_AFTER_COMMENT"] = str(evidence_number(p.state()))
    failed = p.cli(
        "handoff", "--accept", "--note-file", p.note(), ok=None, extra=second
    )
    assert failed.returncode == 1
    receipt = json.loads(failed.stdout)
    assert receipt["parent_action_applied"] is True
    assert receipt["assignment_evidence"]["status"] == "error"
    assert "uncertain" in receipt["assignment_evidence"]["error"]
    p.env.pop("PARENT_FAIL_AFTER_COMMENT")
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    assert json.loads(p.cli("status", extra=second).stdout)["current_session_is_owner"]
    assert (
        sum(
            '"event": "handoff"' in c["body"].splitlines()[0]
            for c in evidence_comments(p.state())
        )
        == 1
    )


def test_partial_stop_dependency_resume_and_improvement_preserve_scope(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.child("c")
    p.child("d", deps=[11])
    stopped = json.loads(p.cli("stop", "--child", "10", "--spec", p.detail()).stdout)
    assert stopped["progress"] == {
        "10": "stopped",
        "11": "held",
        "12": "continue",
        "13": "held",
    }
    state = p.state()
    body = state["issues"]["10"]["body"]
    assert all(
        word in body
        for word in ("仕様の確認待ち", "再現ログを確認済み", "判断の合意を記録")
    )
    notification = evidence_comments(state)[-1]["body"]
    assert "#10" in notification and "#11" in notification and "#12" in notification
    assert "再現ログを確認済み" not in notification
    assert "#11: 依存保留" in state["issues"]["1"]["body"]
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(p.root),
        ok=False,
    )
    before = p.formats.parse_sections(state["issues"]["12"]["body"])[
        "Acceptance checklist"
    ]
    improvement = {
        "purpose": "将来の改善",
        "evidence": "今回観測した事実",
        "scope": "次回の変更だけ",
    }
    p.cli("improvement", "--child", "12", "--spec", p.spec(improvement))
    assert (
        p.formats.parse_sections(p.state()["issues"]["12"]["body"])[
            "Acceptance checklist"
        ]
        == before
    )
    p.cli("improvement", "--child", "12", "--spec", p.spec(improvement))
    assert p.state()["issues"]["12"]["body"].count("将来の改善") == 1
    resumed = json.loads(p.cli("resume", "--child", "10", "--spec", p.detail()).stdout)
    assert set(resumed["progress"].values()) == {"continue"}


def test_delegation_reconcile_and_completed_tasks_do_not_relaunch(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")

    def titles() -> tuple[str, str]:
        # Local task names feed read-only status views (worker_status).
        tasks = json.loads(p.cli("status").stdout)["tasks"]
        return tasks["10"]["title"], tasks["11"]["title"]

    assert titles() == ("a", "b")
    a, b = p.worktree("a"), p.worktree("b")
    for child, name, path in ((10, "a", a), (11, "b", b)):
        response = p.cli(
            "delegate",
            "--child",
            str(child),
            "--agent",
            "codex",
            "--branch",
            "codex/" + name,
            "--worktree",
            str(path),
        )
        assert json.loads(response.stdout)["action"] == "launch"
        assert "worker_runs" not in response.stdout
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        ok=False,
    )
    assert titles() == ("a", "b")
    # A renamed child issue updates the recorded name on the next sync.
    state = p.state()
    assert state["issues"]["11"]["title"].endswith("] b")
    state["issues"]["11"]["title"] += "改名"
    p.save(state)
    p.cli("sync")
    assert titles() == ("a", "b改名")
    assert "- [ ] #11 b改名" in p.state()["issues"]["1"]["body"]
    result = json.loads(p.cli("reconcile").stdout)
    assert {t["action"] for t in result["tasks"]} == {"do_not_launch"}
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/a")]
    p.save(state)
    completed_implementation(p)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("関連テストが成功した。"),
    )
    assert "成果PR: #100" in p.state()["issues"]["10"]["body"]
    assert "関連テストが成功した。" in p.state()["issues"]["10"]["body"]
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        "--restart-evidence-file",
        p.note(),
        ok=False,
    )
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="completed")
    p.save(state)
    p.cli("sync")
    result = json.loads(p.cli("reconcile").stdout)
    assert result["tasks"][0]["state"] == "completed"
    assert result["tasks"][0]["prs"][0]["number"] == 100
    assert "- [x] #10 a" in p.state()["issues"]["1"]["body"]
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(a),
        ok=False,
    )


def test_invalid_worktree_and_reused_worker_branch_are_refused(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(p.repo),
        ok=False,
    )
    worktree = p.worktree("a")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
    )
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_issue_body_metadata_and_existing_evidence_survive_updates(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("keep", source=True)
    state = p.state()
    before = state["issues"]["10"]["body"]
    body = before.replace(
        "## 対象外",
        "<!-- retain-this-comment -->\n\n```sh\n## example heading\necho keep\n```\n\n## 対象外",
    )
    state["issues"]["10"]["body"] = body
    p.save(state)
    p.cli("stop", "--child", "10", "--spec", p.detail())
    p.cli("resume", "--child", "10", "--spec", p.detail())
    actual = p.state()["issues"]["10"]["body"]
    assert (
        "<!-- retain-this-comment -->\n\n```sh\n## example heading\necho keep\n```"
        in actual
    )
    assert actual.splitlines()[0] == before.splitlines()[0]
    assert "<!-- parent-task:v1" in actual


def review_setup(
    p: ParentEnv,
    *,
    independent: bool = False,
    standalone: bool = False,
    parent_self: bool = False,
    pr_count: int = 3,
    shared: bool = False,
) -> list[int]:
    if standalone:
        state = p.state()
        state["issues"]["1"].update(
            title="人間が作成した単独タスク",
            labels=[{"name": "bug"}],
            body="## 目的\n単独タスクを実装する。\n\n## Acceptance checklist\n- [ ] 回帰テストが通る。\n",
        )
        p.save(state)
    else:
        policy = importlib.import_module("parent.policy")
        state = p.state()
        state["issues"]["1"]["body"] = policy.replace_section(
            state["issues"]["1"]["body"],
            "全体の完了条件",
            "- [ ] 各成果の受入と統合結果を確認する。",
        )
        p.save(state)
    p.agreed()
    state = p.state()
    state["prs"] = []
    for i in range(1 if standalone or parent_self else pr_count):
        if not (standalone or parent_self):
            p.child(f"child-{i}")
        worktree = p.worktree(f"pr-{i}")
        filename = f"change-{i}.txt" if independent else "change.txt"
        (worktree / filename).write_text(f"new {i}\n")
        if shared and i == 0:
            (worktree / ".agents").mkdir()
            (worktree / ".agents/shared.txt").write_text("shared foundation\n")
            run_git(worktree, "add", ".agents/shared.txt")
        run_git(worktree, "add", filename)
        run_git(worktree, "commit", "-m", f"child {i}")
        state["prs"].append(
            p.pr(
                100 + i,
                1 if standalone or parent_self else 10 + i,
                run_git(worktree, "rev-parse", "HEAD"),
            )
        )
    # Read the current issue/comments state, then replace PRs only.
    current = p.state()
    current["prs"] = state["prs"]
    p.save(current)
    return [pr["number"] for pr in state["prs"]]


def test_nested_manual_parent_keeps_status_reconcile_and_result_commands(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("nested")
    state = p.state()
    render = importlib.import_module("issues.cli").render_spec
    sections = p.formats.parse_sections(state["issues"]["10"]["body"])
    sections.update(
        {
            heading: "内側の親として管理する。"
            for heading in p.formats.PARENT_REQUIRED
            if heading != p.formats.TASKS_HEADING
        }
    )
    child_sections = {
        heading: "具体的に実装する。" for heading in p.formats.CHILD_REQUIRED
    }
    child_sections["依存関係"] = "なし。"
    child_sections["Acceptance checklist"] = (
        "- [ ] 動作する\n  文脈: 入れ子。\n  下流への影響: 管理の継続。\n  検証方法・証拠: fake。"
    )
    leaf = render(
        {
            "role": "child",
            "summary": "leaf",
            "sections": child_sections,
            "parent": {"number": 10, "identifier": "内側"},
        }
    )
    nested = render(
        {
            "role": "parent",
            "identifier": "内側",
            "summary": "nested",
            "parent": {"number": 1, "identifier": "開発"},
            "sections": sections,
            "children": [{"number": 20, "title": leaf["title"]}],
        }
    )
    state["issues"]["10"].update({k: nested[k] for k in ("title", "body")})
    state["issues"]["10"]["labels"] = [{"name": n} for n in nested["labels"]]
    state["issues"]["20"] = {
        **state["issues"]["10"],
        "number": 20,
        "id": 10020,
        **{k: leaf[k] for k in ("title", "body")},
        "labels": [{"name": n} for n in leaf["labels"]],
    }
    state["parents"]["20"] = 10
    state["prs"].append(p.pr(50, 20))
    state["prs"].append(p.pr(51, 10))  # The upper parent's artifact is independent.
    p.save(state)
    identity = {"CLAUDE_CODE_SESSION_ID": "private-nested-parent"}
    p.cli("register", parent="10", extra=identity)
    status = json.loads(p.cli("status", parent="10", extra=identity).stdout)
    assert status["parent"] == 10 and status["assignment"] == "active"
    assert status["launch_plan"]["tasks"][0]["child"] == 20
    tasks = json.loads(p.cli("reconcile", parent="10", extra=identity).stdout)["tasks"]
    assert len(tasks) == 1 and tasks[0]["child"] == 20
    result = p.cli(
        "record-result",
        "--child",
        "20",
        "--evidence-file",
        p.note("内側の親が差分とテストを確認した。"),
        parent="10",
        extra=identity,
    )
    assert json.loads(result.stdout)["prs"] == [50]
    assert "検証結果:" in p.state()["issues"]["20"]["body"]
    assert json.loads(p.cli("status", parent="20").stdout)["launch_plan"]["tasks"] == []
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    registry = directory / "registry.json"
    data = json.loads(registry.read_text())
    data["parents"]["o/r#10"]["dispatch"] = {}
    registry.write_text(json.dumps(data))
    refused = p.cli("status", parent="10", extra=identity, ok=False)
    assert "formal parent" in refused.stderr


def test_cancelled_is_not_success_and_missing_pr_verification_is_rejected(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.cli("record-result", "--child", "10", "--evidence-file", p.note(), ok=False)
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="not_planned")
    p.save(state)
    statuses = json.loads(p.cli("sync").stdout)["progress"]
    assert statuses == {"10": "cancelled", "11": "held"}
    assert "- [ ] #10 a（取消）" in p.state()["issues"]["1"]["body"]
    state["issues"]["10"]["state_reason"] = None
    p.save(state)
    p.cli("sync", ok=False)


def test_fake_or_quoted_agreement_cannot_unlock_split(parent_env: ParentEnv) -> None:
    p = parent_env
    p.cli("register")
    state = p.state()
    text = (
        '<!-- parent-session:v1 {"parent": 1, "event": "agreed", "agreement_sha256": "'
        + "a" * 64
        + '"} -->'
    )
    state["comments"]["1"].extend(
        [
            {"user": {"login": "stranger"}, "body": text},
            {"user": {"login": "operator"}, "body": "Quoted example\n" + text},
        ]
    )
    p.save(state)
    p.child("forbidden", ok=False)


def test_dependency_cycle_and_unknown_fail_closed(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    policy = importlib.import_module("parent.policy")
    state = p.state()
    original = state["issues"]["10"]["body"]
    state["issues"]["10"]["body"] = policy.replace_section(
        original, "依存関係", "- #11"
    )
    p.save(state)
    assert "cyclic" in p.cli("reconcile", ok=False).stderr
    state["issues"]["10"]["body"] = policy.replace_section(
        original, "依存関係", "- #999"
    )
    p.save(state)
    assert "outside this parent" in p.cli("reconcile", ok=False).stderr


def test_unexpected_interruption_requires_explicit_recovery(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    worktree = p.worktree("a")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/a",
        "--worktree",
        str(worktree),
    )
    second = {"CLAUDE_CODE_SESSION_ID": "private-session-b-unique"}
    p.cli("handoff", "--accept", "--note-file", p.note(), ok=False, extra=second)
    result = json.loads(
        p.cli(
            "handoff",
            "--accept",
            "--recover",
            "--note-file",
            p.note("旧worker終了と既存worktreeを確認した。"),
            extra=second,
        ).stdout
    )
    assert result["tasks"][0]["action"] == "do_not_launch"
    p.cli("checking", ok=False)


def test_cli_dependencies_are_published_and_latest_body_controls_resume(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    state = p.state()
    state["prs"] = [p.pr(100, 10)]
    p.save(state)
    worktree = p.worktree("b")
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(worktree),
        "--dependencies",
        "10",
    )
    assert (
        p.formats.parse_sections(p.state()["issues"]["11"]["body"])["依存関係"]
        == "- #10"
    )
    p.cli("checking")
    p.cli("stop", "--child", "10", "--spec", p.detail())
    assert json.loads(p.cli("reconcile").stdout)["tasks"][1]["state"] == "held"
    # Human correction of dependencies takes precedence over the old local cache.
    state = p.state()
    policy = importlib.import_module("parent.policy")
    state["issues"]["11"]["body"] = policy.replace_section(
        state["issues"]["11"]["body"], "依存関係", "なし。"
    )
    p.save(state)
    result = json.loads(p.cli("reconcile").stdout)
    assert result["tasks"][1]["state"] == "continue"
    assert result["tasks"][1]["assignment"]["dependencies"] == []


# The consumer is tested against the producer's actual public/local contract above.
FAKE_PARENT_REVIEWER = r"""
import hashlib, json, os, pathlib, sys, time
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
if args[0] in {"status", "stop"}:
    record = json.loads((pathlib.Path(opt("--run-dir")) / "result.json").read_text())
    print(json.dumps({"schema_version": 1, "run_id": record["run_id"], "runner_alive": False, "agent_alive": False, "agent_group_alive": False}))
    sys.exit(0)
cwd = pathlib.Path(opt("--cwd"))
context = json.loads((cwd / "context.json").read_text())
request = json.loads((cwd / "request.json").read_text())
call = {"args": args, "cwd": str(cwd), "context": context,
        "prompt": pathlib.Path(opt("--prompt-file")).read_text(),
        "diffs": {str(p["number"]): (cwd / f'pr-{p["number"]}.diff').read_text() for p in request["prs"]},
        "files": {str(p["number"]): sorted(x.name for x in (cwd / f'pr-{p["number"]}').iterdir()) for p in request["prs"]}}
with open(os.environ["REVIEW_AGENT_CALLS"], "a") as f:
    f.write(json.dumps(call) + "\n")
if os.environ.get("REVIEW_AGENT_FAIL"):
    sys.exit(3)
pathlib.Path(opt("--out")).write_text(pathlib.Path(os.environ["REVIEW_AGENT_OUTPUT"]).read_text())
runtime = pathlib.Path(opt("--run-dir"))
runtime.mkdir()
(runtime / "final.md").write_bytes(pathlib.Path(opt("--out")).read_bytes())
schema = pathlib.Path(opt("--json-schema")).read_bytes()
(runtime / "output-schema.json").write_bytes(schema)
exit_code = int(os.environ.get("REVIEW_AGENT_FINAL_EXIT", "0"))
reason = os.environ.get("REVIEW_RUNTIME_REASON", "runtime_error" if exit_code else "completed")
session_id, turn_id = "review-session", "review-turn"
event_text = pathlib.Path(opt("--out")).read_text()
events = [
    {"type":"thread.started", "thread_id":session_id},
    {"type":"turn.started", "turn_id":turn_id},
    {"type":"item.completed", "item":{"type":"agent_message", "text":event_text}},
    {"type":"turn.completed", "usage":{"input_tokens":1,"output_tokens":1}},
]
error = os.environ.get("REVIEW_RUNTIME_ERROR_MESSAGE")
errors = []
if exit_code and reason != "cli_error":
    errors = [{"type":"error", "message":error or "transport failed after completed review"}]
    events.extend(errors)
(runtime / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
choice = request.get("review_selection")
selected = choice["selected"] if choice is not None else {"model":"review-fixture", "effort":"max", "service_tier":"priority"}
effective = {k:{"status":"observed", "value":v, "source":"fixture"} for k,v in selected.items()}
if os.environ.get("REVIEW_EFFECTIVE_UNAVAILABLE"):
    effective["model"] = {"status":"unavailable", "value":None, "reason":"model not available"}
(runtime / "result.json").write_text(json.dumps({
    "schema_version": 1, "run_id": "review-" + request["review_run_id"],
    "result": "failed" if exit_code else "completed", "exit_code": exit_code, "cli_exit_code": exit_code,
    "turn_end_reason": reason, "error": error,
    "agent_spawned": True, "cleanup_complete": not bool(os.environ.get("REVIEW_UNCONFIRMED_CLEANUP")), "finished_at": 1,
    "prompt_sha256": hashlib.sha256(pathlib.Path(opt("--prompt-file")).read_bytes()).hexdigest(),
    "run_dir": str(runtime), "output_path": opt("--out"), "final_path": str(runtime / "final.md"),
    "contract": {"json_schema_sha256": hashlib.sha256(schema).hexdigest(), "cwd": str(cwd),
                 "mode":"read-only", "settings":{"agent":"codex", "profile":"review", "selected":selected},
                 **({"review_selection":choice} if choice is not None else {})},
    "runtime": {"session_id":session_id, "turn_id":turn_id, "requested_session_id":None,
                "runtime_errors":errors, "effective":effective}
}))
if os.environ.get("REVIEW_BROKEN_RUNTIME"):
    (runtime / "result.json").write_text("{invalid runtime")
sys.exit(exit_code)
"""


@dataclass
class ReviewEnv:
    p: ParentEnv
    request: dict[str, Any]
    request_path: Path
    fake: Path
    output: Path
    calls_path: Path
    clock_script: Path | None = None

    @property
    def run_dir(self) -> Path:
        return self.request_path.parent / "review-1"

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.calls_path.read_text().splitlines()]

    def comments(self, n: int) -> list[dict[str, Any]]:
        return self.p.comments(n)


def make_review_env(
    p: ParentEnv,
    *,
    independent: bool = False,
    standalone: bool = False,
    parent_self: bool = False,
    pr_count: int = 3,
    shared: bool = False,
    before_review: Callable[[ParentEnv], None] | None = None,
) -> ReviewEnv:
    """Prepare an approved progression for tests of later parent operations.

    Public lifecycle coverage lives in test_progression_cli. This fixture calls
    the real reviewer adapter once, then persists the accepted result without
    merging; it never recreates the removed production review-only CLI.
    """
    (p.repo / "AGENTS.md").write_text("# Fixture rules\n")
    run_git(p.repo, "add", "AGENTS.md")
    run_git(p.repo, "commit", "-m", "base review rules")
    selected = review_setup(
        p,
        independent=independent,
        standalone=standalone,
        parent_self=parent_self,
        pr_count=pr_count,
        shared=shared,
    )
    if before_review:
        before_review(p)
    remote = p.state()
    for pr in remote["prs"]:
        pr.update(
            labels=pr.get("labels", []),
            mergeable=True,
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(remote)
    fake = p.root / "reviewer"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_PARENT_REVIEWER)
    fake.chmod(0o755)
    output = p.root / "review-output.json"
    output.write_text(
        json.dumps({"verdict": "approve", "summary": "Reviewed fixture", "major": []})
    )
    calls = p.root / "review-calls.jsonl"
    calls.touch()
    p.env.update(
        REVIEW_AGENT_CALLS=str(calls),
        REVIEW_AGENT_OUTPUT=str(output),
        AGENT_AUTO_MERGE="1",
    )
    response = p.cli(
        "review-request",
        "--prs",
        *map(str, selected),
        *p.review_flags(),
        "--instruction-file",
        p.note("Review this fixture"),
        "--evidence-file",
        p.note("Accepted implementation fixture"),
        "--agent-exec",
        str(fake),
        "--wait",
        extra={"AGENT_AUTO_MERGE": "0"},
    )
    identifier = json.loads(response.stdout)["review_run_id"]
    path = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/progressions"
        / identifier
        / "state.json"
    )
    driver = p.root / "prepare-approved-fixture.py"
    driver.write_text(
        f"""\
from pathlib import Path
from lib.ops_review.progression import read, save
from lib.ops_review.progression_backend import LiveBackend
from parent.state import Store

path = Path({str(path)!r})
state = read(path)
backend = LiveBackend(state, Store(path.parents[2]))
state["review"].update(result=backend.review(state, path.parent), stage="accepted")
state["status"] = "parent_wait"
# Event IDs are shared with the registry outbox and must remain append-only.
save(path, state)
"""
    )
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed = subprocess.run(
            [sys.executable, str(driver)],
            cwd=p.repo,
            env=p.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
    assert completed.returncode == 0, completed.stderr
    request = json.loads(path.read_text())["request"]
    return ReviewEnv(p, request, path, fake, output, calls)


@dataclass(frozen=True)
class ReviewSeed:
    directory: DirectorySeed
    template: ReviewEnv

    @classmethod
    def capture(cls, review: ReviewEnv, snapshot: Path) -> ReviewSeed:
        template = replace(
            review,
            p=replace(review.p, env=review.p.env.copy()),
            request=copy.deepcopy(review.request),
        )
        return cls(DirectorySeed.capture(review.p.root, snapshot), template)

    def restore(self, p: ParentEnv) -> ReviewEnv:
        self.directory.restore(p.root)
        run_git(p.repo, "worktree", "repair")
        # A copied preparation journal is not an operation in this test.
        p.calls_path.write_text("")

        def relocated(path: Path) -> Path:
            return p.root / path.relative_to(self.directory.original)

        return replace(
            self.template,
            p=p,
            request=copy.deepcopy(self.template.request),
            request_path=relocated(self.template.request_path),
            fake=relocated(self.template.fake),
            output=relocated(self.template.output),
            calls_path=relocated(self.template.calls_path),
            clock_script=relocated(self.template.clock_script)
            if self.template.clock_script
            else None,
        )


REVIEW_SEEDS = pytest.StashKey[dict[tuple[int, bool], ReviewSeed]]()


def prepared_review(
    p: ParentEnv,
    request: pytest.FixtureRequest,
    *,
    pr_count: int,
    independent: bool = False,
) -> ReviewEnv:
    # Seed only untouched, instructed reviews. Each test gets private git,
    # registry, GitHub state and review files; real review execution stays local.
    seeds: dict[tuple[int, bool], ReviewSeed] = request.config.stash.setdefault(
        REVIEW_SEEDS, {}
    )
    key = (pr_count, independent)
    if key not in seeds:
        review = make_review_env(p, pr_count=pr_count, independent=independent)
        cache = request.getfixturevalue("tmp_path_factory").mktemp("parent-review-seed")
        seeds[key] = ReviewSeed.capture(review, cache / "snapshot")
        return review
    return seeds[key].restore(p)


@pytest.fixture
def review_env(parent_env: ParentEnv, request: pytest.FixtureRequest) -> ReviewEnv:
    # Keep three targets for multi-PR assertions and downstream consumers.
    return prepared_review(parent_env, request, pr_count=3)


@pytest.fixture
def single_review_env(
    parent_env: ParentEnv, request: pytest.FixtureRequest
) -> ReviewEnv:
    return prepared_review(parent_env, request, pr_count=1)


@pytest.fixture
def pair_review_env(parent_env: ParentEnv, request: pytest.FixtureRequest) -> ReviewEnv:
    return prepared_review(parent_env, request, pr_count=2)


@pytest.fixture
def independent_pair_review_env(
    parent_env: ParentEnv, request: pytest.FixtureRequest
) -> ReviewEnv:
    return prepared_review(parent_env, request, pr_count=2, independent=True)


def review_policy_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS / "review"))
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.ops_review.parent_review_policy")
