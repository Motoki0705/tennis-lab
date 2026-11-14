"""Complete GitHub inputs and pinned code; no assignments or GitHub writes."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from issues import parse_proposal_source, parse_sections, parse_task_metadata
from issues.evidence import is_evidence_issue, is_proposal_issue, require_work_issue
from issues.improvement import parse_source as improvement_source
from issues.supplement_state import read_enabled_since, supplement_state
from parent.execution import comment_snapshot, requirement_body
from parent.github import GitHub
from parent.history import History, from_projection, instruction_comments
from parent.improvement_receipts import bound_extraction
from parent.policy import closing_issues, digest, nonempty, number, object_value
from parent.state import Store

from .state import client_for, repository_scope


class InputsChanged(ValueError):
    """A collected subject changed; collection/validation failures remain fatal."""


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"code snapshot git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout


def code_snapshot(root: Path) -> dict[str, Any]:
    if git(root, "status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("judgement requires a clean tracked code checkout")
    commit = git(root, "rev-parse", "HEAD").strip()
    files = []
    for entry in git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        mode, kind, oid = metadata.split()
        files.append({"path": path, "mode": mode, "kind": kind, "oid": oid})
    return {"commit": commit, "files": files}


def unique_numbers(rows: Iterable[Mapping[str, Any]], label: str) -> list[int]:
    values = [number(r.get("number")) for r in rows]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} within/across pages")
    return values


def is_container(raw: Mapping[str, Any]) -> bool:
    return (
        is_evidence_issue(raw)
        or is_proposal_issue(raw)
        or "ops-report" in {x["name"] for x in raw["labels"]}
        or (raw["body"].splitlines()[0] if raw["body"] else "").startswith(
            "<!-- ops-report:v"
        )
    )


def body_dependencies(body: str, repo: str) -> list[int]:
    section = parse_sections(body).get("依存関係", "")
    for url_repo in re.findall(r"https://github\.com/([^/]+/[^/]+)/issues/", section):
        if url_repo.casefold() != repo.casefold():
            raise ValueError("cross-repository dependency requires explicit inspection")
    # A qualified foreign shorthand must never turn into a local #number.
    if re.search(r"[\w.-]+/[\w.-]+#[1-9][0-9]*", section):
        raise ValueError("use same-repository #number or an issue URL for dependencies")
    return sorted(
        {int(n) for n in re.findall(r"#([1-9][0-9]*)", section)}
        | {
            int(n)
            for n in re.findall(
                r"https://github\.com/[^/]+/[^/]+/issues/([1-9][0-9]*)", section
            )
        }
    )


def collect(
    gh: GitHub, store: Store, root: Path, seeds: Iterable[int] | None = None
) -> dict[str, Any]:
    gh = client_for(gh, store)
    with repository_scope(store), gh.fresh_reads(), gh.history_read_scope():
        return _collect(gh, store, root, seeds)


def _collect(
    gh: GitHub, store: Store, root: Path, seeds: Iterable[int] | None = None
) -> dict[str, Any]:
    """Caller holds Store.locked(). Every page/failure is significant."""
    login = nonempty(object_value(gh.get("user")).get("login"), "gh login")
    extractions = gh.improvement_extractions(store)
    with gh.fresh_reads():
        enabled_since = read_enabled_since(gh)
    if seeds is None:
        listed = [
            r
            for r in gh.rows(
                f"repos/{gh.repo}/issues?state=open&sort=created&direction=asc"
            )
            if "pull_request" not in r
            and not is_evidence_issue(r)
            and not is_proposal_issue(r)
        ]
        subjects = sorted(unique_numbers(listed, "open issues"))
        for raw in listed:
            gh.identity(raw)
            if raw.get("state") != "open":
                raise ValueError("open issue listing returned a non-open issue")
    else:
        subjects = sorted(set(number(n) for n in seeds))
    pulls = gh.pulls()
    unique_numbers(pulls, "PRs")
    prs = []
    for p in pulls:
        if p.get("state") not in {"open", "closed"} or "merged_at" not in p:
            raise ValueError("PR state/merged_at missing or invalid")
        if p.get("body") is not None and not isinstance(p["body"], str):
            raise ValueError("PR body must be text")
        if p["merged_at"] is not None and not isinstance(p["merged_at"], str):
            raise ValueError("PR merged_at must be a timestamp or null")
        head = object_value(p.get("head"))
        prs.append(
            {
                "number": number(p["number"]),
                "body": p.get("body") or "",
                "state": p["state"],
                "merged_at": p["merged_at"],
                "head": {
                    "ref": nonempty(head.get("ref"), "PR branch"),
                    "sha": nonempty(head.get("sha"), "PR head SHA"),
                },
                "issues": sorted(closing_issues(p.get("body") or "", gh.repo)),
            }
        )
    issues: dict[str, Any] = {}
    pending = list(subjects)
    while pending:
        n = pending.pop()
        if str(n) in issues:
            continue
        base = f"repos/{gh.repo}/issues/{n}"
        raw = gh.issue(n)
        if raw.get("state") not in {"open", "closed"}:
            raise ValueError("unknown issue state")
        if raw.get("state_reason") not in {
            None,
            "completed",
            "not_planned",
            "reopened",
            "duplicate",
        }:
            raise ValueError("unknown issue state reason")
        if raw.get("body") is not None and not isinstance(raw["body"], str):
            raise ValueError("issue body must be text")
        labels = raw.get("labels")
        assignees = raw.get("assignees")
        if not isinstance(labels, list) or not isinstance(assignees, list):
            raise ValueError("issue labels/assignees missing")
        parent_raw = gh.get(base + "/parent", missing_parent=True)
        if parent_raw is not None:
            require_work_issue(object_value(parent_raw))
        parent = gh.identity(parent_raw) if parent_raw is not None else None
        children_raw = gh.rows(base + "/sub_issues")
        children = sorted(unique_numbers(children_raw, "children"))
        formal_deps = gh.rows(base + "/dependencies/blocked_by")
        deps = sorted(
            set(unique_numbers(formal_deps, "dependencies"))
            | set(body_dependencies(raw.get("body") or "", gh.repo))
        )
        for row in [*children_raw, *formal_deps]:
            require_work_issue(row)
            gh.identity(row)
        comments = gh.parent_comments(n, login)
        ids = [comment_snapshot(c)["id"] for c in comments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate comments within/across pages")
        normalized_comments = []
        for c in comments:
            if c.get("body") is not None and not isinstance(c["body"], str):
                raise ValueError("comment body must be text")
            item = comment_snapshot(c)
            if not isinstance(item["body"], str):
                raise ValueError("comment body must be text")
            normalized_comments.append(
                {
                    "id": item["id"],
                    "user": {"login": item["login"]},
                    "body": item["body"],
                    **(
                        {
                            "_parent_record_issue": c["_parent_record_issue"],
                            "_parent_record_source": "evidence_issue",
                        }
                        if c.get("_parent_record_source") == "evidence_issue"
                        else {}
                    ),
                }
            )
        source_reports: set[int] = set()
        extraction = bound_extraction(raw.get("body") or "", n, extractions)
        improvement = extraction or improvement_source(raw.get("body") or "")
        improvement_sources = [improvement.parent] if improvement is not None else []
        if not is_container({"labels": labels, "body": raw.get("body") or ""}):
            source = parse_proposal_source(raw.get("body") or "")
            if source is not None:
                source_reports.add(source.report_number)
            metadata = parse_task_metadata(raw.get("body") or "")
            if metadata is not None:
                # Context-only parents/split tasks must retain source freshness
                # even when their proposal's index anchor is moved or closed.
                source_reports.update(
                    binding["source"]["report_number"]
                    for binding in metadata["bindings"]
                )
        issues[str(n)] = {
            "number": n,
            "title": nonempty(raw.get("title"), "issue title"),
            "body": raw.get("body") or "",
            "state": raw["state"],
            "state_reason": raw.get("state_reason"),
            "created_at": nonempty(raw.get("created_at"), "issue created_at"),
            "labels": sorted(
                [
                    {"name": nonempty(object_value(x).get("name"), "label")}
                    for x in labels
                ],
                key=lambda x: x["name"],
            ),
            "assignees": sorted(
                nonempty(object_value(x).get("login"), "assignee") for x in assignees
            ),
            "parent": parent,
            "children": children,
            "dependencies": deps,
            "source_reports": sorted(source_reports),
            **(
                {"improvement_sources": improvement_sources}
                if improvement_sources
                else {}
            ),
            "comments": sorted(normalized_comments, key=lambda x: x["id"]),
        }
        supplement_state(
            [label["name"] for label in issues[str(n)]["labels"]],
            issues[str(n)]["created_at"],
            enabled_since,
            body=raw.get("body") or "",
        )
        pending.extend(
            [
                *children,
                *deps,
                *source_reports,
                *improvement_sources,
                *([parent] if parent else []),
            ]
        )
    if seeds is None and any(issues[str(n)]["state"] != "open" for n in subjects):
        raise ValueError("issue listing changed during collection")
    for issue_key, raw in issues.items():
        if (
            raw["parent"] is not None
            and int(issue_key) not in issues[str(raw["parent"])]["children"]
        ):
            raise ValueError("formal parent/child endpoints disagree")
        for child in raw["children"]:
            if issues[str(child)]["parent"] != int(issue_key):
                raise ValueError("formal child/parent endpoints disagree")
        route(issues, int(issue_key))  # Detect cycles before the agent runs.
    owners = {}
    for key, row in store.parents.items():
        repo, _, n = key.rpartition("#")
        if repo == gh.repo.casefold():
            number(int(n))
            owners[n] = {
                "status": row["status"],
                "generation": row["generation"],
                "record_sha256": digest(row),
                "tasks": {
                    k: {"delegated": bool(v.get("delegated")), "sha256": digest(v)}
                    for k, v in row["tasks"].items()
                },
            }
    return {
        "schema_version": 1,
        "repo": gh.repo.casefold(),
        "publisher": login,
        "supplement_enabled_since": enabled_since,
        "code": unlocked_code_snapshot(gh, root),
        "subjects": subjects,
        "issues": issues,
        "pulls": sorted(prs, key=lambda x: x["number"]),
        "owners": owners,
        "improvement_extractions": extractions,
    }


def unlocked_code_snapshot(gh: GitHub, root: Path) -> dict[str, Any]:
    with gh.io_scope():
        return code_snapshot(root)


def route(issues: Mapping[str, Any], n: int) -> int:
    seen = set()
    while True:
        if n in seen:
            raise ValueError("cyclic formal issue ancestry")
        seen.add(n)
        parent = issues[str(n)]["parent"]
        if parent is None:
            return n
        n = parent


def related(snapshot: Mapping[str, Any], issue: int) -> set[int]:
    pending, found = [issue], set()
    while pending:
        n = pending.pop()
        if n in found:
            continue
        raw = snapshot["issues"][str(n)]
        found.add(n)
        pending.extend(
            [
                *raw["children"],
                *raw["dependencies"],
                *raw["source_reports"],
                *raw.get("improvement_sources", []),
                *([raw["parent"]] if raw["parent"] else []),
            ]
        )
    return found


def source_references(
    repo: str, sources: Sequence[str]
) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    """Read both agent references and the URLs stored in published candidates."""
    issues: set[int] = set()
    pulls: set[int] = set()
    comments: set[tuple[int, int]] = set()
    for source in sources:
        if source.startswith("code:"):
            continue  # Every target already includes the pinned code tree.
        url = re.fullmatch(
            re.escape(f"https://github.com/{repo}")
            + r"/(issues|pull)/([1-9][0-9]*)(?:#issuecomment-([1-9][0-9]*))?",
            source,
        )
        if url:
            kind, n, comment = url.groups()
            if kind == "pull" and comment:
                raise ValueError("PR comment sources were not collected")
            source = (
                f"comment:{n}:{comment}"
                if comment
                else f"{'pr' if kind == 'pull' else 'issue'}:{n}"
            )
        reference = re.fullmatch(
            r"(issue|comment|pr):([1-9][0-9]*)(?::([1-9][0-9]*))?", source
        )
        if reference is None or (reference[1] == "comment") != bool(reference[3]):
            raise ValueError(f"unknown source reference: {source}")
        (pulls if reference[1] == "pr" else issues).add(int(reference[2]))
        if reference[1] == "comment":
            comments.add((int(reference[2]), int(reference[3])))
    return issues, pulls, comments


def require_current_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Old saved inputs must be recollected, never relabelled under their digest."""
    if (
        "rewrite_enabled_since" in snapshot
        or "supplement_enabled_since" not in snapshot
    ):
        raise ValueError(
            "unsupported dispatcher activation key; recollect the judgement input "
            "with supplement_enabled_since before resuming"
        )
    if not isinstance(snapshot.get("improvement_extractions"), dict):
        raise ValueError(
            "dispatcher extraction index is missing; recollect the judgement input"
        )


def history_owner(snapshot: Mapping[str, Any], issue: int) -> int:
    if str(issue) in snapshot["issues"]:
        return issue
    owners = {
        int(n)
        for n, raw in snapshot["issues"].items()
        if any(
            c.get("_parent_record_source") == "evidence_issue"
            and c.get("_parent_record_issue") == issue
            for c in raw["comments"]
        )
    }
    if len(owners) != 1:
        raise ValueError(
            f"evidence source issue #{issue} has no unique collected parent"
        )
    return owners.pop()


def material(
    snapshot: Mapping[str, Any],
    issue: int | None = None,
    sources: Sequence[str] = (),
) -> dict[str, Any]:
    require_current_snapshot(snapshot)
    cited_issues, cited_pulls, cited_comments = source_references(
        snapshot["repo"], sources
    )
    cited_issues = {history_owner(snapshot, n) for n in cited_issues}
    if issue is None:
        selected = {int(n) for n in snapshot["issues"]}
    else:
        selected = set()
        for n in {issue, *cited_issues}:
            selected.update(related(snapshot, n))
    issues = {}
    for n in sorted(selected):
        raw = snapshot["issues"][str(n)]
        prose = instruction_comments(
            snapshot["repo"], n, snapshot["publisher"], raw["comments"]
        )
        history = from_projection(
            snapshot["repo"], n, snapshot["publisher"], raw["comments"]
        ).session_events()
        controls = [
            {k: v for k, v in e.items() if k != "body"}
            for e in history
            if e["event"] in {"checking", "agreed", "basis_revoked"}
        ]
        issues[str(n)] = {
            **{k: v for k, v in raw.items() if k not in {"body", "comments"}},
            "requirements": requirement_body(raw["body"]),
            "comments": prose,
            "controls": controls,
        }
    result = {
        "schema_version": 1,
        "repo": snapshot["repo"],
        "publisher": snapshot["publisher"],
        "supplement_enabled_since": snapshot["supplement_enabled_since"],
        "code": snapshot["code"],
        "subjects": snapshot["subjects"] if issue is None else [issue],
        "issues": issues,
        "pulls": [
            p
            for p in snapshot["pulls"]
            if p["number"] in cited_pulls or set(p["issues"]) & selected
        ],
        "owners": {
            n: r
            for n, r in snapshot["owners"].items()
            if int(n) in selected or set(r["tasks"]) & {str(s) for s in selected}
        },
        "improvement_extractions": {
            n: r
            for n, r in snapshot["improvement_extractions"].items()
            if int(n) in selected
        },
    }
    if cited_comments:
        # Explicit citations remain evidence even when the usual projection
        # omits their system progress markers. Missing comments change this map.
        result["source_comments"] = {
            f"{n}:{c['id']}": comment_snapshot(c)
            for n in sorted({history_owner(snapshot, n) for n, _ in cited_comments})
            for c in snapshot["issues"][str(n)]["comments"]
            if (c.get("_parent_record_issue", n), c["id"]) in cited_comments
            or (n, c["id"]) in cited_comments
        }
    return result


class SnapshotGitHub(GitHub):
    """Give execution_input exactly the inputs reviewed by the read-only agent."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.repo = snapshot["repo"]

    def get(self, endpoint: str, *, missing_parent: bool = False) -> object:
        if endpoint == "user":
            return {"login": self.snapshot["publisher"]}
        raise ValueError(f"unexpected snapshot request: {endpoint}")

    def improvement_extractions(self, store: Store) -> dict[str, Any]:
        return deepcopy(object_value(self.snapshot["improvement_extractions"]))

    def issue(self, number: int) -> dict[str, Any]:
        return deepcopy(object_value(self.snapshot["issues"][str(number)]))

    def comments(self, parent: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = deepcopy(
            self.snapshot["issues"][str(parent)]["comments"]
        )
        return result

    def record_history(
        self, parent: int, login: str | None = None, **kwargs: Any
    ) -> History:
        return from_projection(
            self.repo,
            parent,
            login or self.snapshot["publisher"],
            self.comments(parent),
        )


def assert_material_fresh(
    snapshot: dict[str, Any],
    current: dict[str, Any],
    issue: int,
    sources: Sequence[str] = (),
) -> None:
    if material(current, issue, sources) != material(snapshot, issue, sources):
        raise InputsChanged(
            "judgement inputs changed; wait for the next slot or explicitly judge again"
        )


def assert_fresh(
    gh: GitHub,
    store: Store,
    root: Path,
    snapshot: dict[str, Any],
    issue: int,
    sources: Sequence[str] = (),
) -> dict[str, Any]:
    """Caller holds the registry lock; no agent is launched on a mismatch."""
    cited_issues, _, _ = source_references(snapshot["repo"], sources)
    current = collect(
        gh, store, root, [issue, *(history_owner(snapshot, n) for n in cited_issues)]
    )
    assert_material_fresh(snapshot, current, issue, sources)
    return current
