"""File-level exceptions to the default CPU selection and immutable run plans."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

REGISTRY = "scripts/ci_metrics/lanes.json"
SHARED_PATHS = (
    "src/utils/",
    "src/tasks/base/",
    "tests/support/",
    ".spin/",
    ".github/workflows/",
    "scripts/ci_metrics/",
)
SHARED_FILES = {"pyproject.toml", "uv.lock", "setup.py", "setup.cfg"}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_test_files(root: Path) -> list[str]:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = config["tool"]["pytest"]["ini_options"]["python_files"]
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "tests").rglob("*.py")
        if path.is_file() and any(fnmatch.fnmatchcase(path.name, p) for p in patterns)
    )


def _relative_path(value: Any, *, pattern: bool = False) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.startswith("/")
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in value.split("/"))
        and (pattern or not any(c in value for c in "*?["))
    )


def validate_registry(data: Any, files: list[str]) -> dict[str, Any]:
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "suites"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["suites"], list)
    ):
        raise ValueError("invalid lane registry schema")
    ids: set[str] = set()
    registered: set[str] = set()
    for suite in data["suites"]:
        if not isinstance(suite, dict) or set(suite) != {
            "id",
            "files",
            "paths",
            "reason",
            "owner",
        }:
            raise ValueError("lane suite requires id/files/paths/reason/owner")
        name = suite["id"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-z][a-z0-9-]*", name)
            or name in ids
        ):
            raise ValueError("lane suite ids must be valid and unique")
        ids.add(name)
        for key in ("reason", "owner"):
            if not isinstance(suite[key], str) or not suite[key].strip():
                raise ValueError(f"lane suite {name} requires a nonempty {key}")
        selected = suite["files"]
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"lane suite {name} requires explicit test files")
        for filename in selected:
            if not _relative_path(filename) or filename not in files:
                raise ValueError(
                    f"lane suite {name}: nonexistent test file {filename!r}"
                )
            if filename in registered:
                raise ValueError(f"duplicate lane registration: {filename}")
            registered.add(filename)
        paths = suite["paths"]
        if (
            not isinstance(paths, list)
            or not paths
            or any(not _relative_path(path, pattern=True) for path in paths)
            or len(set(paths)) != len(paths)
        ):
            raise ValueError(
                f"lane suite {name} requires unique relative path patterns"
            )
    return data


def load_registry(root: Path) -> tuple[dict[str, Any], list[str]]:
    files = source_test_files(root)
    return validate_registry(json.loads((root / REGISTRY).read_text()), files), files


def _selection(
    registry: dict[str, Any],
    event: str,
    ref: str,
    changes: list[dict[str, Any]],
    diff_error: str | None,
) -> tuple[str, list[str], list[str]]:
    reasons = []
    if event == "push" and ref == "refs/heads/main":
        reasons.append("main_push")
    elif event == "workflow_dispatch":
        reasons.append("manual_full")
    elif event == "local":
        reasons.append("local_full")
    elif event != "pull_request":
        raise ValueError("unsupported lane plan event/ref")
    if diff_error is not None:
        reasons.append(f"diff_unavailable: {diff_error}")
    for change in changes:
        status = change["status"]
        if status.startswith("R") or status == "D":
            reasons.append(f"rename_or_deletion: {', '.join(change['paths'])}")
        for path in change["paths"]:
            if (
                path in SHARED_FILES
                or PurePosixPath(path).name == "conftest.py"
                or path.startswith(SHARED_PATHS)
            ):
                reasons.append(f"shared_foundation: {path}")
    suites = registry["suites"]
    if reasons:
        return "full", sorted(set(reasons)), sorted(suite["id"] for suite in suites)
    paths = {path for change in changes for path in change["paths"]}
    requested = [
        suite["id"]
        for suite in suites
        if paths.intersection(suite["files"])
        or any(
            fnmatch.fnmatchcase(path, pattern)
            for path in paths
            for pattern in suite["paths"]
        )
    ]
    return "pr", ["pr_changed_paths"], sorted(requested)


def make_plan(
    registry: dict[str, Any],
    files: list[str],
    *,
    source_sha: str | None,
    event: str,
    ref: str = "",
    run_id: str | None = None,
    attempt: str | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
    changes: list[dict[str, Any]] | None = None,
    diff_error: str | None = None,
    main_successes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_registry(registry, files)
    changes = changes if changes is not None else []
    if (
        event == "pull_request"
        and (not base_sha or not head_sha)
        and diff_error is None
    ):
        diff_error = "missing PR base/head"
    scope, reasons, requested = _selection(registry, event, ref, changes, diff_error)
    plan = {
        "schema_version": 1,
        "registry": registry,
        "policy_sha256": digest(registry),
        "source_files": files,
        "source_sha": source_sha,
        "event": event,
        "ref": ref,
        "run_id": run_id,
        "attempt": attempt,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "changes": changes,
        "diff_error": diff_error,
        "scope": scope,
        "reasons": reasons,
        "requested_suites": requested,
        "deferred_suites": sorted(
            suite["id"] for suite in registry["suites"] if suite["id"] not in requested
        ),
        "main_successes": main_successes or {},
    }
    return {**plan, "plan_id": digest(plan)}


def validate_plan(plan: Any, *, root: Path | None = None) -> dict[str, Any]:
    if (
        not isinstance(plan, dict)
        or type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
    ):
        raise ValueError("invalid lane plan schema")
    unsigned = {k: v for k, v in plan.items() if k != "plan_id"}
    if digest(unsigned) != plan.get("plan_id"):
        raise ValueError("lane plan identity mismatch")
    files = plan["source_files"]
    if (
        not isinstance(files, list)
        or any(not _relative_path(f) for f in files)
        or files != sorted(set(files))
    ):
        raise ValueError("invalid lane plan source file inventory")
    validate_registry(plan["registry"], files)
    for change in plan["changes"]:
        if (
            not isinstance(change, dict)
            or set(change) != {"status", "paths"}
            or not isinstance(change["status"], str)
            or not re.fullmatch(r"[AMDTUXB]|[RC]\d{1,3}", change["status"])
            or not isinstance(change["paths"], list)
            or len(change["paths"])
            != (2 if change["status"].startswith(("R", "C")) else 1)
            or any(not _relative_path(p, pattern=True) for p in change["paths"])
        ):
            raise ValueError("invalid lane plan diff evidence")
    expected = make_plan(
        plan["registry"],
        files,
        source_sha=plan["source_sha"],
        event=plan["event"],
        ref=plan["ref"],
        run_id=plan["run_id"],
        attempt=plan["attempt"],
        base_sha=plan["base_sha"],
        head_sha=plan["head_sha"],
        changes=plan["changes"],
        diff_error=plan["diff_error"],
        main_successes=plan["main_successes"],
    )
    if expected != plan:
        raise ValueError("lane plan does not match the registry/diff decision")
    if root is not None:
        registry, current_files = load_registry(root)
        if registry != plan["registry"] or current_files != files:
            raise ValueError("lane plan differs from current source/registry")
    return plan


def read_plan(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return validate_plan(json.loads(path.read_text()), root=root)


def partition_nodeids(
    nodeids: list[str], plan: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise ValueError("CPU universe must be nonempty and unique")
    source_files = set(plan["source_files"])
    if any(node.split("::", 1)[0] not in source_files for node in nodeids):
        raise ValueError("CPU universe has files outside the source inventory")
    deferred_files = {
        filename: suite["id"]
        for suite in plan["registry"]["suites"]
        if suite["id"] in plan["deferred_suites"]
        for filename in suite["files"]
    }
    deferred = {
        node: deferred_files[node.split("::", 1)[0]]
        for node in nodeids
        if node.split("::", 1)[0] in deferred_files
    }
    return [node for node in nodeids if node not in deferred], deferred


def audit_partition(
    manifest: dict[str, Any], universe: list[str], plan: dict[str, Any]
) -> None:
    requested, deferred = partition_nodeids(universe, plan)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("plan_id") != plan["plan_id"]
        or manifest.get("policy_sha256") != plan["policy_sha256"]
        or manifest.get("scope") != plan["scope"]
        or manifest.get("universe_nodeids") != universe
        or manifest.get("requested_nodeids") != requested
        or manifest.get("deferred") != deferred
    ):
        raise ValueError(
            "lane partition differs from pre-selection CPU universe/plan (U = R disjoint-union D)"
        )
    splits = manifest["split_files"]
    groups = {
        node if node.split("::", 1)[0] in splits else node.split("::", 1)[0]
        for node in requested
    }
    assignments = manifest["assignments"]
    if (
        set(assignments) != groups
        or any(
            type(index) is not int or not 0 <= index < manifest["count"]
            for index in assignments.values()
        )
        or set(assignments.values()) != set(range(manifest["count"]))
    ):
        raise ValueError("requested CPU groups must cover every shard exactly once")
    selected = [
        node
        for node in requested
        if assignments[
            node if node.split("::", 1)[0] in splits else node.split("::", 1)[0]
        ]
        == manifest["index"]
    ]
    if selected != manifest["selected_nodeids"]:
        raise ValueError(
            "selected inventory does not match the requested CPU assignment"
        )
