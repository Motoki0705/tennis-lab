"""Capture and validate final GitHub pull-request evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from issue_task_candidate import (
    compute_candidate_fingerprint,
    compute_revision_fingerprint,
    current_revision,
    revision_changed_paths,
)
from issue_task_issue import canonical_json_bytes, sha256_bytes
from issue_task_state import load_state, validate_state, write_state

PR_EVIDENCE_PATH = "05-packaging/pr-evidence.json"
_REPO_RE = re.compile(r"^https://github\.com/([^/]+/[^/]+)/issues/\d+(?:$|[/?#])")
_PASS_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED"}


def _repo_from_issue_url(url: str) -> str:
    match = _REPO_RE.match(url)
    if match is None:
        raise ValueError("state.toml issue_url is not a supported GitHub Issue URL")
    return match.group(1)


def _gh_json(*args: str) -> object:
    if shutil.which("gh") is None:
        raise ValueError("gh CLI is required for capture-pr")
    completed = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or f"gh {' '.join(args)} failed")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gh {' '.join(args)} returned invalid JSON: {exc}") from exc


def _normalize_checks(raw: object) -> tuple[list[dict[str, str]], str]:
    if not isinstance(raw, list):
        raise ValueError("GitHub statusCheckRollup must be a list")
    checks: list[dict[str, str]] = []
    all_pass = bool(raw)
    for item in raw:
        if not isinstance(item, dict):
            all_pass = False
            continue
        typename = str(item.get("__typename", ""))
        if typename == "CheckRun":
            name = str(item.get("name", ""))
            status = str(item.get("status", ""))
            conclusion = str(item.get("conclusion", ""))
            passed = status == "COMPLETED" and conclusion in _PASS_CONCLUSIONS
            checks.append(
                {
                    "kind": typename,
                    "name": name,
                    "status": status,
                    "conclusion": conclusion,
                    "verdict": "PASS" if passed else "FAIL",
                }
            )
        elif typename == "StatusContext":
            name = str(item.get("context", ""))
            state = str(item.get("state", ""))
            passed = state == "SUCCESS"
            checks.append(
                {
                    "kind": typename,
                    "name": name,
                    "status": state,
                    "conclusion": state,
                    "verdict": "PASS" if passed else "FAIL",
                }
            )
        else:
            passed = False
            checks.append(
                {
                    "kind": typename or "UNKNOWN",
                    "name": str(item.get("name", item.get("context", ""))),
                    "status": str(item.get("status", item.get("state", ""))),
                    "conclusion": str(item.get("conclusion", item.get("state", ""))),
                    "verdict": "FAIL",
                }
            )
        all_pass = all_pass and passed
    checks.sort(key=lambda item: (item["kind"], item["name"]))
    return checks, "PASS" if all_pass else "FAIL"


def _flatten_files(raw: object) -> list[str]:
    pages = raw
    if not isinstance(pages, list):
        raise ValueError("GitHub PR files response must be a list")
    if pages and all(isinstance(page, list) for page in pages):
        items = [item for page in pages for item in page]
    else:
        items = pages
    files: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError("GitHub PR files response contains an invalid entry")
        files.add(str(item["filename"]))
        if item.get("status") == "renamed":
            previous_filename = item.get("previous_filename")
            if not isinstance(previous_filename, str) or not previous_filename:
                raise ValueError(
                    "GitHub PR files response contains a renamed entry without a valid "
                    "previous_filename"
                )
            files.add(previous_filename)
    return sorted(files)


def evidence_digest(payload: dict[str, Any]) -> str:
    return "sha256:" + sha256_bytes(canonical_json_bytes(payload))


def capture_pr_evidence(task_dir: Path, *, pr_number: int) -> None:
    """Query the real PR head, all changed-file pages, and remote checks."""
    state = load_state(task_dir)
    if state.get("candidate_binding_mode") != "ENFORCED":
        raise ValueError("capture-pr is available only for schema-v5-or-newer tasks")
    if state.get("phase") != "packaging" or state.get("status") != "validated":
        raise ValueError("capture-pr requires packaging/validated state")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))
    if pr_number < 1:
        raise ValueError("pr_number must be positive")

    repo = _repo_from_issue_url(str(state["issue_url"]))
    metadata = _gh_json(
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "number,url,headRefOid,isDraft,state,statusCheckRollup",
    )
    if not isinstance(metadata, dict):
        raise ValueError("gh pr view returned an unexpected payload")
    remote_number = metadata.get("number")
    head_sha = metadata.get("headRefOid")
    if remote_number != pr_number:
        raise ValueError("remote PR number does not match capture-pr")
    if not isinstance(head_sha, str) or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None:
        raise ValueError("remote PR head SHA is invalid")
    if metadata.get("isDraft") is True:
        raise ValueError("final PR must not be draft")
    if metadata.get("state") not in {"OPEN", "MERGED"}:
        raise ValueError("final PR must be open or merged")

    checks, remote_verdict = _normalize_checks(metadata.get("statusCheckRollup"))
    files_payload = _gh_json(
        "api",
        "--paginate",
        "--slurp",
        f"repos/{repo}/pulls/{pr_number}/files?per_page=100",
    )
    files = _flatten_files(files_payload)

    local_head = current_revision(task_dir)
    if local_head != head_sha:
        raise ValueError("checked-out local HEAD does not match the remote PR head")
    current_candidate = compute_candidate_fingerprint(task_dir, state)
    validated_candidate = str(state.get("validation_candidate_sha256", ""))
    if current_candidate != validated_candidate:
        raise ValueError("current content differs from the validated candidate")
    revision_candidate = compute_revision_fingerprint(task_dir, state, head_sha)
    if revision_candidate != validated_candidate:
        raise ValueError("remote PR head content differs from the validated candidate")
    local_files = revision_changed_paths(task_dir, state, head_sha)
    if files != local_files:
        raise ValueError(
            "complete paginated PR file list differs from the validated revision"
        )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "repository": repo,
        "pr_number": pr_number,
        "url": str(metadata.get("url", "")),
        "state": str(metadata.get("state", "")),
        "head_sha": head_sha,
        "candidate_sha256": validated_candidate,
        "files": files,
        "checks": checks,
        "remote_checks_verdict": remote_verdict,
    }
    path = task_dir / PR_EVIDENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

    next_state = dict(state)
    next_state["pr_number"] = pr_number
    next_state["pr_head_sha"] = head_sha
    next_state["remote_checks_verdict"] = remote_verdict
    next_state["pr_evidence_sha256"] = evidence_digest(evidence)
    errors = validate_state(task_dir, next_state)
    if errors:
        raise ValueError("; ".join(errors))
    write_state(task_dir, next_state)


def load_pr_evidence(task_dir: Path) -> dict[str, Any]:
    path = task_dir / PR_EVIDENCE_PATH
    if not path.is_file():
        raise ValueError(f"missing required file: {PR_EVIDENCE_PATH}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PR_EVIDENCE_PATH} is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{PR_EVIDENCE_PATH} schema_version must be 1")
    return payload


def pr_evidence_errors(task_dir: Path, state: dict[str, Any]) -> list[str]:
    try:
        payload = load_pr_evidence(task_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    if evidence_digest(payload) != state.get("pr_evidence_sha256"):
        errors.append("PR evidence digest does not match state")
    if payload.get("pr_number") != state.get("pr_number"):
        errors.append("PR evidence number does not match state")
    if payload.get("head_sha") != state.get("pr_head_sha"):
        errors.append("PR evidence head SHA does not match state")
    if payload.get("candidate_sha256") != state.get("validation_candidate_sha256"):
        errors.append("PR evidence candidate does not match Validator candidate")
    if payload.get("remote_checks_verdict") != "PASS":
        errors.append("PR evidence remote checks are not PASS")
    if state.get("remote_checks_verdict") != payload.get("remote_checks_verdict"):
        errors.append("remote check verdict differs between PR evidence and state")
    files = payload.get("files")
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        errors.append("PR evidence files must be a string array")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("PR evidence must contain remote checks")
    return errors
