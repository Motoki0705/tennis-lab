"""Observe the Git comparison and Actions result for a knowledge records PR."""

from __future__ import annotations

import io
import json
import re
import subprocess
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from issues.format import visible_lines
from lib import knowledge_ci
from parent.policy import closing_references, nonempty, number, object_value
from parent.snapshot import git

if TYPE_CHECKING:
    from lib.ops_review.progression_backend import LiveBackend
    from parent.github import GitHub

WORKFLOW = "Knowledge library"
RESULT_SIZE_LIMIT = 65536
_CLOSING_ISSUES_QUERY = """
query KnowledgeClosingIssues($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    nameWithOwner
    pullRequest(number: $number) {
      number state isDraft title body headRefOid baseRefName baseRefOid
      closingIssuesReferences(first: 100, after: $after) {
        totalCount
        nodes { number repository { nameWithOwner } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


def commit_sha(value: object, what: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{what} is unavailable")
    return value


@dataclass(frozen=True)
class Comparison:
    head: str
    base: str
    merge: str

    @classmethod
    def from_pr(cls, pr: dict[str, Any]) -> Comparison:
        return cls(
            commit_sha(object_value(pr.get("head")).get("sha"), "PR HEAD"),
            commit_sha(object_value(pr.get("base")).get("sha"), "PR base"),
            commit_sha(pr.get("merge_commit_sha"), "PR test merge commit"),
        )


def assess(
    root: Path, repo: str, pr: dict[str, Any]
) -> tuple[knowledge_ci.Scope, Comparison | None]:
    try:
        pr_number = number(pr.get("number"))
        if pr.get("state") != "open":
            raise ValueError("PR is not open")
        if any(
            str(
                object_value(object_value(pr.get(side)).get("repo")).get("full_name")
            ).casefold()
            != repo.casefold()
            for side in ("head", "base")
        ):
            raise ValueError("knowledge records require a same-repository PR")
        comparison = Comparison.from_pr(pr)
        # Fetch only the pinned comparison. The test merge brings both parents
        # with it. No checkout, remote-tracking ref, or FETCH_HEAD is changed.
        try:
            git("cat-file", "-e", comparison.merge + "^{commit}", cwd=root)
        except ValueError:
            git(
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                "--refmap=",
                "origin",
                f"refs/pull/{pr_number}/merge",
                cwd=root,
            )
        scope = knowledge_ci.classify_pr(
            root,
            head_sha=comparison.head,
            base_sha=comparison.base,
            merge_sha=comparison.merge,
        )
        return scope, comparison if scope.records_only else None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return knowledge_ci.Scope(False, str(exc)), None


def references(body: str, repo: str) -> set[int]:
    pattern = (
        rf"https://github\.com/{re.escape(repo)}/(?:issues|pull)/([1-9][0-9]*)\b"
        rf"|(?<![\w./#-])(?:{re.escape(repo)})?#([1-9][0-9]*)\b"
    )
    return {
        int(m[1] or m[2])
        for m in re.finditer(pattern, "\n".join(visible_lines(body)), re.I)
    }


def closing_targets(
    gh: GitHub, root: Path, pr: dict[str, Any], comparison: Comparison
) -> set[int]:
    """Require GitHub and the PR description to name the same closing issues."""
    owner, name = gh.repo.split("/")
    repo = gh.repo.casefold()
    body = pr.get("body") or ""
    declared = closing_references(body, repo)
    if any(name != repo for name, _ in declared):
        raise ValueError("PR body declares a closing target in another repository")
    expected = {
        "number": pr["number"],
        "state": "OPEN",
        "isDraft": False,
        "title": pr.get("title"),
        "body": body,
        "headRefOid": comparison.head,
        "baseRefName": "main",
        "baseRefOid": comparison.base,
    }
    after = None
    seen_cursors = set()
    actual: set[tuple[str, int]] = set()
    total = None
    while True:
        data = gh.graphql(
            _CLOSING_ISSUES_QUERY,
            {"owner": owner, "name": name, "number": pr["number"], "after": after},
        )
        repository = object_value(data.get("repository"))
        pull = object_value(repository.get("pullRequest"))
        if nonempty(
            repository.get("nameWithOwner"), "repository"
        ).casefold() != repo or any(
            pull.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                "GitHub closing issues were read for a different PR snapshot"
            )
        connection = object_value(pull.get("closingIssuesReferences"))
        count = connection.get("totalCount")
        nodes = connection.get("nodes")
        if (
            type(count) is not int
            or count < 0
            or (total is not None and count != total)
            or not isinstance(nodes, list)
        ):
            raise ValueError("GitHub closing issue list is incomplete or changed")
        total = count
        for raw in nodes:
            node = object_value(raw)
            identifier = (
                nonempty(
                    object_value(node.get("repository")).get("nameWithOwner"),
                    "closing issue repository",
                ).casefold(),
                number(node.get("number")),
            )
            if identifier[0] != repo:
                raise ValueError(
                    "GitHub reports a closing target in another repository"
                )
            if identifier in actual:
                raise ValueError("GitHub closing issue list changed during pagination")
            actual.add(identifier)
        page = object_value(connection.get("pageInfo"))
        if page.get("hasNextPage") is False and len(actual) == total:
            break
        if page.get("hasNextPage") is not True or not nodes or len(actual) >= total:
            raise ValueError("GitHub closing issue list is incomplete or changed")
        after = nonempty(page.get("endCursor"), "closing issue cursor")
        if after in seen_cursors:
            raise ValueError("GitHub repeated a closing issue cursor")
        seen_cursors.add(after)
    if actual != declared:
        raise ValueError(
            "GitHub closing targets differ from PR body "
            f"(GitHub only: {sorted(n for _, n in actual - declared)}, "
            f"body only: {sorted(n for _, n in declared - actual)})"
        )
    # Commit messages can close an issue on the default branch independently
    # of the PR's closingIssuesReferences connection.
    messages = git(
        "--no-replace-objects",
        "log",
        "-z",
        "--format=%B",
        f"{comparison.base}..{comparison.head}",
        cwd=root,
    ).decode(errors="replace")
    extra = closing_references(messages, repo, markdown=False) | closing_references(
        pr.get("title") or "", repo, markdown=False
    )
    if extra - declared:
        raise ValueError(
            "commit or PR title declares a closing target missing from the PR body"
        )
    return {n for _, n in actual}


def base_tip(gh: GitHub, branch: str) -> str:
    result = object_value(gh.get(f"repos/{gh.repo}/branches/{quote(branch, safe='')}"))
    if result.get("name") != branch:
        raise ValueError("unexpected base branch observation")
    return commit_sha(object_value(result.get("commit")).get("sha"), "base tip")


def latest_run(
    backend: LiveBackend, pr: dict[str, Any], head: str
) -> dict[str, Any] | None:
    runs = backend.paged(
        f"repos/{backend.repo}/actions/workflows/knowledge.yml/runs"
        f"?event=pull_request&head_sha={head}",
        "workflow_runs",
    )
    relevant = []
    for run in runs:
        if (
            run.get("event") != "pull_request"
            or run.get("path") != knowledge_ci.WORKFLOW_PATH
            or run.get("head_sha") != head
        ):
            raise ValueError("unexpected Knowledge library run in Actions response")
        if run.get("head_branch") != pr["head"]["ref"]:
            continue
        linked = run.get("pull_requests")
        if linked and pr["number"] not in {p["number"] for p in linked}:
            continue
        relevant.append(run)
    # The workflow run number increases for each new run. Do not fall back
    # to an older successful run or attempt while the latest one is pending.
    return max(
        relevant,
        key=lambda r: (number(r.get("run_number")), number(r.get("run_attempt"))),
        default=None,
    )


def result_bytes(backend: LiveBackend, run: dict[str, Any]) -> bytes:
    run_id = number(run.get("id"))
    name = knowledge_ci.artifact_name(run_id, number(run.get("run_attempt")))
    artifacts = backend.paged(
        f"repos/{backend.repo}/actions/runs/{run_id}/artifacts?name={name}",
        "artifacts",
    )
    matches = [a for a in artifacts if a.get("name") == name]
    if len(matches) != 1 or matches[0].get("expired") is not False:
        raise ValueError(f"expected one live Actions artifact named {name}")
    artifact = matches[0]
    if number(artifact.get("size_in_bytes")) > RESULT_SIZE_LIMIT:
        raise ValueError("knowledge records artifact is too large")
    endpoint = (
        f"repos/{backend.repo}/actions/artifacts/{number(artifact.get('id'))}/zip"
    )
    download = subprocess.run(
        ["gh", "api", "--hostname", "github.com", "--method", "GET", endpoint],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if download.returncode:
        raise ValueError(f"download failed for Actions artifact {name}")
    if len(download.stdout) > RESULT_SIZE_LIMIT:
        raise ValueError("knowledge records artifact is too large")
    try:
        with zipfile.ZipFile(io.BytesIO(download.stdout)) as archive:
            entries = archive.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != knowledge_ci.RESULT_FILENAME
                or entries[0].file_size > RESULT_SIZE_LIMIT
            ):
                raise ValueError("expected one knowledge records result in artifact")
            return archive.read(entries[0])
    except (
        zipfile.BadZipFile,
        OSError,
        RuntimeError,
        NotImplementedError,
        zlib.error,
    ) as exc:
        raise ValueError("invalid knowledge records artifact ZIP") from exc


def require_current_comparison(
    root: Path,
    comparison: Comparison,
    *,
    checked_base: str,
    checked_merge: str,
    current_main: str,
) -> None:
    if checked_base == comparison.base and checked_merge != comparison.merge:
        raise ValueError(
            "knowledge records result merge_sha does not match the PR test merge at the same base"
        )
    try:
        knowledge_ci.require_unchanged_base(root, checked_base, current_main)
        if comparison.base != checked_base:
            # Prove both bases are ancestors with the same watched paths as main.
            # This also proves the watched paths match between the two bases.
            knowledge_ci.require_unchanged_base(root, comparison.base, current_main)
    except ValueError as exc:
        raise ValueError(
            f"{exc}. Merge main into the PR branch and refresh evidence"
        ) from exc


def observe_ci(
    backend: LiveBackend, pr: dict[str, Any], comparison: Comparison, *, root: Path
) -> dict[str, Any]:
    current_main = base_tip(backend.gh, pr["base"]["ref"])
    result: dict[str, Any] = {
        "workflow": WORKFLOW,
        "status": "pending",
        "head": comparison.head,
        "base_sha": comparison.base,
        "merge_sha": comparison.merge,
        "observed_main_sha": current_main,
    }
    run = latest_run(backend, pr, comparison.head)
    if run is None:
        return {**result, "reasons": ["Knowledge library PR run is missing"]}
    result["run"] = {
        key: run[key] for key in ("id", "run_attempt", "status", "conclusion")
    }
    if run["status"] != "completed":
        return {**result, "reasons": ["Knowledge library PR run is pending"]}
    if run["conclusion"] != "success":
        return {
            **result,
            "status": "failure",
            "reasons": [f"Knowledge library conclusion: {run['conclusion']}"],
        }
    try:
        data = result_bytes(backend, run)
        artifact = object_value(json.loads(data))
        checked_base = commit_sha(artifact.get("base_sha"), "result base")
        checked_merge = commit_sha(artifact.get("merge_sha"), "result test merge")
        knowledge_ci.verify_result(
            data,
            run,
            pr_number=pr["number"],
            head_sha=comparison.head,
            base_sha=checked_base,
            merge_sha=checked_merge,
        )
        knowledge_ci.ensure_commits(root, checked_base, comparison.head, checked_merge)
        scope = knowledge_ci.classify_pr(
            root,
            head_sha=comparison.head,
            base_sha=checked_base,
            merge_sha=checked_merge,
        )
        if not scope.records_only:
            raise ValueError(
                f"knowledge records result comparison is invalid: {scope.reason}"
            )
        require_current_comparison(
            root,
            comparison,
            checked_base=checked_base,
            checked_merge=checked_merge,
            current_main=current_main,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {**result, "status": "failure", "reasons": [str(exc)]}
    return {
        **result,
        "status": "success",
        "reasons": [],
        "verified_base_sha": checked_base,
        "verified_merge_sha": checked_merge,
    }
