"""Machine-readable canonical verification commands."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from issue_task_candidate import compute_candidate_fingerprint, repository_root
from issue_task_issue import canonical_json_bytes, sha256_bytes
from issue_task_state import load_state

CHECK_MANIFEST_PATH = "02-planning/checks.json"
RESULT_PATHS = {
    "preflight": "03-implementation/preflight-checks.json",
    "test": "03-implementation/test-checks.json",
    "seal": "03-implementation/seal-checks.json",
}
CHECK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
AC_ID_RE = re.compile(r"^AC-\d{3}$")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _check_digest(check: dict[str, Any]) -> str:
    normalized = {
        "id": check["id"],
        "argv": check["argv"],
        "cwd": check["cwd"],
        "env": check["env"],
        "stages": check["stages"],
        "required": check["required"],
        "authority": check["authority"],
    }
    return "sha256:" + sha256_bytes(canonical_json_bytes(normalized))


def load_check_manifest(task_dir: Path) -> dict[str, Any]:
    path = task_dir / CHECK_MANIFEST_PATH
    if not path.is_file():
        raise ValueError(f"missing required file: {CHECK_MANIFEST_PATH}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{CHECK_MANIFEST_PATH} is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{CHECK_MANIFEST_PATH} schema_version must be 1")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{CHECK_MANIFEST_PATH} must define at least one check")

    ids: set[str] = set()
    has_test = False
    has_seal = False
    for index, check in enumerate(checks):
        prefix = f"{CHECK_MANIFEST_PATH} checks[{index}]"
        if not isinstance(check, dict):
            raise ValueError(f"{prefix} must be an object")
        check_id = check.get("id")
        if not isinstance(check_id, str) or not CHECK_ID_RE.fullmatch(check_id):
            raise ValueError(f"{prefix}.id is invalid")
        if check_id in ids:
            raise ValueError(f"duplicate canonical check id: {check_id}")
        ids.add(check_id)
        argv = check.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ValueError(f"{prefix}.argv must be a non-empty string array")
        cwd = check.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            raise ValueError(f"{prefix}.cwd must be a non-empty string")
        cwd_path = Path(cwd)
        if cwd_path.is_absolute() or ".." in cwd_path.parts:
            raise ValueError(f"{prefix}.cwd must stay inside the repository")
        env = check.get("env")
        if not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError(f"{prefix}.env must map strings to strings")
        stages = check.get("stages")
        if (
            not isinstance(stages, list)
            or not stages
            or any(stage not in RESULT_PATHS for stage in stages)
            or len(stages) != len(set(stages))
        ):
            raise ValueError(f"{prefix}.stages is invalid")
        required = check.get("required")
        if not isinstance(required, bool):
            raise ValueError(f"{prefix}.required must be a boolean")
        authority = check.get("authority")
        if (
            not isinstance(authority, list)
            or not authority
            or any(not isinstance(item, str) or not AC_ID_RE.fullmatch(item) for item in authority)
        ):
            raise ValueError(f"{prefix}.authority must contain AC IDs")
        if required and "test" in stages:
            has_test = True
        if required and "seal" in stages:
            has_seal = True
    if not has_test:
        raise ValueError("canonical checks must include a required test-stage command")
    if not has_seal:
        raise ValueError("canonical checks must include a required seal-stage command")
    return payload


def manifest_errors(task_dir: Path) -> list[str]:
    try:
        load_check_manifest(task_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    return []


def _results_payload(task_dir: Path, stage: str, candidate: str) -> dict[str, Any]:
    path = task_dir / RESULT_PATHS[stage]
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and payload.get("stage") == stage
            and payload.get("candidate_sha256") == candidate
            and isinstance(payload.get("results"), list)
        ):
            return payload
    return {
        "schema_version": 1,
        "stage": stage,
        "candidate_sha256": candidate,
        "results": [],
    }


def run_check(task_dir: Path, stage: str, check_id: str) -> int:
    if stage not in RESULT_PATHS:
        raise ValueError(f"unknown verification stage: {stage}")
    manifest = load_check_manifest(task_dir)
    checks = {item["id"]: item for item in manifest["checks"]}
    if check_id not in checks:
        raise ValueError(f"unknown canonical check id: {check_id}")
    check = checks[check_id]
    if stage not in check["stages"]:
        raise ValueError(f"canonical check {check_id} is not authorized for {stage}")

    root = repository_root(task_dir)
    cwd = (root / check["cwd"]).resolve()
    if cwd != root and root not in cwd.parents:
        raise ValueError(f"canonical check {check_id} cwd escapes the repository")
    environment = os.environ.copy()
    environment.update(check["env"])
    candidate = compute_candidate_fingerprint(task_dir, load_state(task_dir))
    completed = subprocess.run(
        check["argv"],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
    )

    logs = task_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"canonical-{stage}-{check_id}.log"
    log_path.write_bytes(
        b"$ "
        + b" ".join(arg.encode(errors="backslashreplace") for arg in check["argv"])
        + b"\n\n[stdout]\n"
        + completed.stdout
        + b"\n[stderr]\n"
        + completed.stderr
    )

    payload = _results_payload(task_dir, stage, candidate)
    result = {
        "id": check_id,
        "invocation_sha256": _check_digest(check),
        "candidate_sha256": candidate,
        "exit_code": completed.returncode,
        "verdict": "PASS" if completed.returncode == 0 else "FAIL",
        "log_path": log_path.relative_to(task_dir).as_posix(),
    }
    payload["results"] = [
        item for item in payload["results"] if item.get("id") != check_id
    ] + [result]
    payload["results"].sort(key=lambda item: str(item.get("id", "")))
    _atomic_json(task_dir / RESULT_PATHS[stage], payload)
    return completed.returncode


def stage_result_errors(task_dir: Path, stage: str, candidate: str) -> list[str]:
    if stage not in RESULT_PATHS:
        return [f"unknown verification stage: {stage}"]
    try:
        manifest = load_check_manifest(task_dir)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    path = task_dir / RESULT_PATHS[stage]
    if not path.is_file():
        return [f"missing canonical check results: {RESULT_PATHS[stage]}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{RESULT_PATHS[stage]} is invalid JSON: {exc}"]
    errors: list[str] = []
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return [f"{RESULT_PATHS[stage]} schema_version must be 1"]
    if payload.get("stage") != stage:
        errors.append(f"{RESULT_PATHS[stage]} stage does not match {stage}")
    if payload.get("candidate_sha256") != candidate:
        errors.append(f"{RESULT_PATHS[stage]} candidate fingerprint is stale")
    results = payload.get("results")
    if not isinstance(results, list):
        return errors + [f"{RESULT_PATHS[stage]} results must be a list"]
    by_id = {
        item.get("id"): item for item in results if isinstance(item, dict)
    }
    for check in manifest["checks"]:
        if not check["required"] or stage not in check["stages"]:
            continue
        check_id = check["id"]
        result = by_id.get(check_id)
        if not isinstance(result, dict):
            errors.append(f"required canonical check was not run for {stage}: {check_id}")
            continue
        if result.get("invocation_sha256") != _check_digest(check):
            errors.append(f"canonical invocation mismatch for {stage}: {check_id}")
        if result.get("candidate_sha256") != candidate:
            errors.append(f"canonical result candidate mismatch for {stage}: {check_id}")
        if result.get("verdict") != "PASS" or result.get("exit_code") != 0:
            errors.append(f"required canonical check did not PASS for {stage}: {check_id}")
    return errors
