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
from parent.policy import number, object_value
from parent.snapshot import git

if TYPE_CHECKING:
    from lib.ops_review.progression_backend import LiveBackend
    from parent.github import GitHub

WORKFLOW = "Knowledge library"
RESULT_SIZE_LIMIT = 65536


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
        r"|(?<![\w./#-])#([1-9][0-9]*)\b"
    )
    return {
        int(m[1] or m[2])
        for m in re.finditer(pattern, "\n".join(visible_lines(body)), re.I)
    }


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


def observe_ci(
    backend: LiveBackend, pr: dict[str, Any], comparison: Comparison
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workflow": WORKFLOW,
        "status": "pending",
        "head": comparison.head,
        "base_sha": comparison.base,
        "merge_sha": comparison.merge,
    }
    if base_tip(backend.gh, pr["base"]["ref"]) != comparison.base:
        return {**result, "reasons": ["PR comparison is behind the current base"]}
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
        # A successful run on an earlier comparison can coexist with a new
        # test merge. Keep waiting for a PR event against the current base.
        artifact = object_value(json.loads(data))
        checked_base = commit_sha(artifact.get("base_sha"), "result base")
        if checked_base != comparison.base:
            return {
                **result,
                "reasons": ["Knowledge library result is for an earlier comparison"],
            }
        knowledge_ci.verify_result(
            data,
            run,
            pr_number=pr["number"],
            head_sha=comparison.head,
            base_sha=comparison.base,
            merge_sha=comparison.merge,
        )
    except (ValueError, subprocess.TimeoutExpired) as exc:
        return {**result, "status": "failure", "reasons": [str(exc)]}
    return {**result, "status": "success", "reasons": []}
