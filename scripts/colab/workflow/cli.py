"""Terminal interface for reproducible tennis-lab training on Google Colab."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

from .common import (
    RCLONE_REMOTE_RE,
    SCHEMA_VERSION,
    WorkflowError,
    atomic_write_json,
    digest_json,
    ensure_within,
    read_json,
    sha256_file,
    strict_relative_path,
    validate_run_id,
)
from .jobs import OVERRIDE_KEY_RE, Job, load_registry
from .snapshot import Snapshot, create_snapshot

DEFAULT_DRIVE_ROOT = "tennis_lab"
DEFAULT_STATE_ROOT = (
    Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
    / "tennis-lab/colab-runs"
)
GPU_CHOICES = ("T4", "L4", "G4", "H100", "A100")
GPU_ARGUMENT_CHOICES = ("auto", "cpu", *GPU_CHOICES)
MINIMUM_COLAB_VERSION = (0, 6, 0)
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
COMMON_MANAGED_OVERRIDE_KEYS = frozenset(
    {
        "run.output_dir",
        "run.gpus",
        "run.device",
        "hydra.run.dir",
        "hydra.sweep.dir",
        "hydra.sweep.subdir",
        "hydra.output_subdir",
        "hydra.job.chdir",
    }
)
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_CONTRACT = 3
EXIT_NOT_FOUND = 4


class WorkflowInterrupted(WorkflowError):
    """Raised when a local termination signal requests lifecycle cleanup."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz").lower()
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise WorkflowError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _validate_github_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise WorkflowError(f"{label} has an invalid port") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or port
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise WorkflowError(f"{label} must be a credential-free https://github.com URL")
    components = [part for part in parsed.path.split("/") if part]
    if len(components) != 2 or any(
        part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in components
    ):
        raise WorkflowError(f"{label} must identify one GitHub owner/repository")
    return value


def _repository_url(repo_root: Path) -> str:
    value = _git(repo_root, "remote", "get-url", "origin")
    if value.startswith("git@github.com:"):
        value = f"https://github.com/{value.removeprefix('git@github.com:')}"
    elif value.startswith("ssh://git@github.com/"):
        value = f"https://github.com/{value.removeprefix('ssh://git@github.com/')}"
    return _validate_github_url(value, "origin")


def _resolve_repo_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        raise WorkflowError(f"not a tennis-lab git worktree: {root}")
    return root


def _state_root(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _jobs_dir(repo_root: Path, value: str | None) -> Path:
    if value is None:
        return repo_root / "scripts/colab/workflows/jobs"
    return Path(value).expanduser().resolve()


def _run_dir(state_root: Path, run_id: str) -> Path:
    return state_root / validate_run_id(run_id)


def _remote_paths(run_id: str) -> dict[str, str]:
    workspace = f"/content/tennis-lab-runs/{run_id}"
    return {
        "workspace": workspace,
        "request": f"{workspace}/request.json",
        "snapshot": f"{workspace}/source-snapshot.tar.gz",
        "status": f"{workspace}/status.json",
        "artifacts": f"{workspace}/bundle/artifacts.tar.gz",
        "rclone_config": f"{workspace}/.secrets/rclone.conf",
    }


def _source_description(
    repo_root: Path,
    mode: str,
    requested_ref: str,
    snapshot: Snapshot | None,
) -> dict[str, Any]:
    repo_sha = _git(repo_root, "rev-parse", f"{requested_ref}^{{commit}}")
    if not len(repo_sha) == 40:
        raise WorkflowError(
            f"ref did not resolve to a full commit SHA: {requested_ref}"
        )
    repo_tree = _git(repo_root, "rev-parse", f"{repo_sha}^{{tree}}")
    local_tracked_dirty = bool(
        _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    )
    local_working_tree_dirty = bool(
        _git(repo_root, "status", "--porcelain", "--untracked-files=normal")
    )
    if mode == "git" and local_working_tree_dirty:
        raise WorkflowError(
            "git source cannot include local working-tree changes; commit or stash "
            "all tracked and untracked non-ignored changes, or use --source snapshot"
        )
    description: dict[str, Any] = {
        "mode": mode,
        "repository_url": _repository_url(repo_root),
        "requested_ref": requested_ref,
        "repo_sha": repo_sha,
        "repo_tree": repo_tree,
        "tracked_clean": not local_tracked_dirty,
        "working_tree_dirty": local_working_tree_dirty,
    }
    if mode == "snapshot":
        if requested_ref != "HEAD":
            raise WorkflowError("snapshot source mode only supports --ref HEAD")
        assert snapshot is not None
        for item in snapshot.submodules:
            _validate_github_url(item.url, f"submodule {item.path} URL")
        description.update(
            {
                "archive_sha256": snapshot.archive_sha256,
                "content_digest": snapshot.content_digest,
                "included_paths": list(snapshot.included_paths),
                "deleted_paths": list(snapshot.deleted_paths),
                "excluded_secret_count": len(snapshot.excluded_secret_paths),
                "submodules": [
                    {"path": item.path, "url": item.url, "commit": item.commit}
                    for item in snapshot.submodules
                ],
            }
        )
    return description


def _resolve_gpu(accelerator: str, requested: str) -> str | None:
    if accelerator == "gpu" and requested == "cpu":
        raise WorkflowError("this job requires a GPU and cannot use --gpu cpu")
    if requested == "auto":
        return "T4" if accelerator == "gpu" else None
    if requested == "cpu":
        return None
    if requested not in GPU_CHOICES:
        raise WorkflowError(f"unsupported GPU selection: {requested}")
    return requested


def _override_key(argument: str) -> str | None:
    candidate = argument
    while candidate.startswith("+"):
        candidate = candidate[1:]
    if candidate.startswith("~"):
        candidate = candidate[1:]
    if not candidate or candidate.startswith("-"):
        return None
    key = candidate.split("=", 1)[0].split("@", 1)[0]
    return key or None


def _normalized_key_parts(key: str) -> tuple[str, ...]:
    return tuple(re.split(r"[./]", key))


def _keys_overlap(first: str, second: str) -> bool:
    first_parts = _normalized_key_parts(first)
    second_parts = _normalized_key_parts(second)
    shared_length = min(len(first_parts), len(second_parts))
    return first_parts[:shared_length] == second_parts[:shared_length]


def _is_common_managed_key(key: str) -> bool:
    parts = _normalized_key_parts(key)
    normalized = ".".join(parts)
    leaf = parts[-1]
    return (
        normalized in COMMON_MANAGED_OVERRIDE_KEYS
        or normalized in {"hydra", "paths", "run", "runtime", "trainer"}
        or normalized.startswith("hydra.")
        or normalized.startswith("runtime.")
        or (normalized.startswith("paths.") and leaf.endswith("_root"))
        or leaf
        in {
            "accelerator",
            "device",
            "devices",
            "gpus",
            "output_dir",
            "output_directory",
            "output_name",
            "output_root",
            "precision",
            "strategy",
        }
    )


def _validate_overrides(job: Job, overrides: list[str]) -> None:
    protected = set(job.protected_override_keys)
    forbidden_options = {
        "--config-path",
        "--config-name",
        "--run",
        "--multirun",
    }
    for argument in overrides:
        if "\x00" in argument:
            raise WorkflowError("Hydra overrides must not contain NUL bytes")
        option = argument.split("=", 1)[0]
        if option in forbidden_options:
            raise WorkflowError(f"managed Hydra option cannot be overridden: {option}")
        key = _override_key(argument)
        if key is None:
            continue
        if OVERRIDE_KEY_RE.fullmatch(key) is None:
            raise WorkflowError(f"Hydra override key has unsafe syntax: {key!r}")
        overlaps_manifest_key = any(
            _keys_overlap(key, managed) for managed in protected
        )
        if _is_common_managed_key(key) or overlaps_manifest_key:
            raise WorkflowError(f"managed Hydra key cannot be overridden: {key}")


def _request(
    *,
    run_id: str,
    session: str,
    job: Job,
    timeout_seconds: int,
    extra_args: list[str],
    source: dict[str, Any],
    drive_mode: str,
    drive_root: str,
    rclone_remote: str | None,
    gpu: str | None,
    high_mem: bool,
) -> dict[str, Any]:
    resolved_argv = job.resolved_argv(extra_args)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "session": session,
        "created_at": _utc_now(),
        "runtime": {
            "accelerator": "gpu" if gpu is not None else "cpu",
            "gpu": gpu,
            "high_mem": high_mem,
        },
        "source": source,
        "drive": {
            "mode": drive_mode,
            "root": drive_root,
            "remote": rclone_remote,
        },
        "job": {
            "name": job.name,
            "definition_digest": job.definition_digest,
            "accelerator": job.accelerator,
            "setup": list(job.setup),
            "protected_override_keys": list(job.protected_override_keys),
            "argv": resolved_argv,
            "timeout_seconds": timeout_seconds,
            "inputs": [
                {
                    "source": item.source,
                    "destination": item.destination,
                    "writable": item.writable,
                }
                for item in job.inputs
            ],
            "outputs": list(job.outputs),
            "output_storage": job.output_storage,
        },
    }
    value["request_digest"] = digest_json(value)
    return value


def _initial_status(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "session": request["session"],
        "request_digest": request["request_digest"],
        "state": "created",
        "step": "local_preparation",
        "attempt": 0,
        "updated_at": _utc_now(),
        "error": None,
        "provenance": {
            "job_definition_digest": request["job"]["definition_digest"],
            "source": {
                "mode": request["source"]["mode"],
                "repo_sha": request["source"]["repo_sha"],
                "repo_tree": request["source"]["repo_tree"],
                "tracked_clean": request["source"]["tracked_clean"],
                **(
                    {"snapshot_digest": request["source"]["content_digest"]}
                    if request["source"]["mode"] == "snapshot"
                    else {}
                ),
            },
            "resolved_argv": request["job"]["argv"],
            "input_manifest": [],
            "output_manifest": [],
            "runtime": {"collected": False},
        },
        "artifacts": {},
    }


def _local_metadata(
    request: dict[str, Any], config_path: Path, remote: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "session": request["session"],
        "colab_config": str(config_path),
        "request_digest": request["request_digest"],
        "remote": remote,
        "session_state": "not-created",
        "updated_at": _utc_now(),
    }


def _colab_prefix(config_path: Path) -> list[str]:
    return ["colab", "--config", str(config_path.resolve())]


def _invoke(
    config_path: Path,
    args: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [*_colab_prefix(config_path), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except OSError as error:
        raise WorkflowError(f"cannot execute colab: {error}") from error
    if check and result.returncode != 0:
        detail = ""
        if capture:
            detail = (result.stderr or result.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise WorkflowError(
            f"colab {args[0]} failed with exit code {result.returncode}{suffix}"
        )
    return result


def _semantic_versions(value: str) -> set[tuple[int, int, int]]:
    return {
        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        for match in re.finditer(
            r"(?<![0-9])v?(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?(?![0-9])",
            value,
        )
    }


def _help_supports(value: str, *names: str) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?=[\s=,\]\)]|$)", value)
        is not None
        for name in names
    )


def _colab_capabilities(config_path: Path, *, high_mem: bool = False) -> None:
    version_result = _invoke(config_path, ["version"], capture=True)
    version_output = "\n".join(
        part for part in (version_result.stdout, version_result.stderr) if part
    )
    versions = _semantic_versions(version_output)
    if len(versions) != 1:
        raise WorkflowError(
            "unsupported colab CLI: `colab version` did not report exactly one "
            "semantic version"
        )
    version = next(iter(versions))
    if version < MINIMUM_COLAB_VERSION:
        required = ".".join(str(part) for part in MINIMUM_COLAB_VERSION)
        raise WorkflowError(
            f"unsupported colab CLI {'.'.join(str(part) for part in version)}; "
            f"version {required}+ is required"
        )

    exec_help = _invoke(config_path, ["exec", "--help"], capture=True)
    exec_output = "\n".join(
        part for part in (exec_help.stdout, exec_help.stderr) if part
    )
    if not all(
        (
            _help_supports(exec_output, "-s", "--session"),
            _help_supports(exec_output, "-f", "--file"),
            _help_supports(exec_output, "--timeout"),
        )
    ):
        raise WorkflowError(
            "unsupported colab CLI: exec must support -s, -f, and --timeout"
        )
    new_help = _invoke(config_path, ["new", "--help"], capture=True)
    new_output = "\n".join(part for part in (new_help.stdout, new_help.stderr) if part)
    if not _help_supports(new_output, "-s", "--session") or not _help_supports(
        new_output, "--gpu"
    ):
        raise WorkflowError("unsupported colab CLI: new must support -s and --gpu")
    if high_mem and not _help_supports(new_output, "--high-mem"):
        raise WorkflowError(
            "the installed colab CLI does not support --high-mem; omit that option "
            "or install a release whose `colab new --help` lists it"
        )


def _bootstrap_source(
    runner_path: Path,
    run_id: str,
    action: str,
    rclone_config_remote: str | None,
) -> str:
    try:
        runner_source = runner_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise WorkflowError(
            f"cannot read remote runner {runner_path}: {error}"
        ) from error
    assignments = [
        f"TENNIS_COLAB_RUN_ID = {run_id!r}",
        f"TENNIS_COLAB_ACTION = {action!r}",
        f"TENNIS_COLAB_RCLONE_CONFIG = {rclone_config_remote!r}",
        "",
    ]
    return "\n".join(assignments) + runner_source


def _exec_remote(
    repo_root: Path,
    config_path: Path,
    session: str,
    run_id: str,
    action: str,
    *,
    timeout: int,
    rclone_config_remote: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    runner = repo_root / "scripts/colab/workflow/remote_runner.py"
    with tempfile.TemporaryDirectory(prefix="tennis-colab-bootstrap-") as name:
        bootstrap = Path(name) / f"{action}.py"
        bootstrap.write_text(
            _bootstrap_source(runner, run_id, action, rclone_config_remote),
            encoding="utf-8",
        )
        bootstrap.chmod(0o600)
        return _invoke(
            config_path,
            [
                "exec",
                "-s",
                session,
                "-f",
                str(bootstrap),
                "--timeout",
                str(timeout),
            ],
            check=check,
        )


def _upload(
    config_path: Path, session: str, local_path: Path, remote_path: str
) -> None:
    _invoke(
        config_path,
        ["upload", "-s", session, str(local_path.resolve()), remote_path],
    )


def _download_status(
    run_dir: Path,
    config_path: Path,
    session: str,
    remote_path: str,
    request_digest: str,
    run_id: str,
) -> dict[str, Any]:
    validate_run_id(run_id)
    temporary = run_dir / ".status.download"
    temporary.unlink(missing_ok=True)
    try:
        _invoke(
            config_path,
            ["download", "-s", session, remote_path, str(temporary.resolve())],
        )
        status = read_json(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    _validate_local_status(
        status,
        request_digest=request_digest,
        run_id=run_id,
        session=session,
    )
    atomic_write_json(run_dir / "status.json", status)
    return status


def _validate_local_status(
    status: dict[str, Any],
    *,
    request_digest: str,
    run_id: str | None = None,
    session: str | None = None,
) -> None:
    required = {
        "schema_version",
        "run_id",
        "session",
        "request_digest",
        "state",
        "step",
        "attempt",
        "updated_at",
        "error",
        "provenance",
        "artifacts",
    }
    missing = required - set(status)
    unknown = set(status) - (required | {"drive"})
    if missing:
        raise WorkflowError(f"status is missing fields: {sorted(missing)}")
    if unknown:
        raise WorkflowError(f"status has unknown fields: {sorted(unknown)}")
    if status["schema_version"] != SCHEMA_VERSION:
        raise WorkflowError("status has an unsupported schema_version")
    if status["request_digest"] != request_digest:
        raise WorkflowError("status request digest is stale or corrupt")
    if not isinstance(status["run_id"], str):
        raise WorkflowError("status run id is invalid")
    validate_run_id(status["run_id"])
    if not isinstance(status["session"], str) or not SESSION_RE.fullmatch(
        status["session"]
    ):
        raise WorkflowError("status session is invalid")
    if run_id is not None and status["run_id"] != run_id:
        raise WorkflowError("status run id is stale or corrupt")
    if session is not None and status["session"] != session:
        raise WorkflowError("status session is stale or corrupt")
    if status["state"] not in {"created", "running", "failed", "completed"}:
        raise WorkflowError("status state is invalid")
    if not isinstance(status["step"], str) or not status["step"]:
        raise WorkflowError("status step is invalid")
    if not isinstance(status["updated_at"], str) or not status["updated_at"]:
        raise WorkflowError("status updated_at is invalid")
    if status["state"] == "failed":
        if not isinstance(status["error"], dict):
            raise WorkflowError("failed status error is invalid")
    elif status["error"] is not None:
        raise WorkflowError("non-failed status error must be null")
    if (
        not isinstance(status["attempt"], int)
        or isinstance(status["attempt"], bool)
        or status["attempt"] < 0
    ):
        raise WorkflowError("status attempt is invalid")
    if not isinstance(status["provenance"], dict) or not isinstance(
        status["artifacts"], dict
    ):
        raise WorkflowError("status provenance/artifacts are invalid")
    if status["state"] == "completed":
        artifacts = status["artifacts"]
        digest = artifacts.get("archive_sha256")
        size = artifacts.get("archive_size")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or status["step"] != "completed"
        ):
            raise WorkflowError("completed status artifact metadata is invalid")


def _validate_status_manifest_entries(
    value: Any, label: str, declared_root: str
) -> None:
    if not isinstance(value, list):
        raise WorkflowError(f"{label} must be an array")
    paths: set[str] = set()
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "size",
            "sha256",
        }:
            raise WorkflowError(f"{label}[{index}] is invalid")
        path = strict_relative_path(raw_entry["path"], f"{label}[{index}].path")
        size = raw_entry["size"]
        digest = raw_entry["sha256"]
        if (
            path in paths
            or not (path == declared_root or path.startswith(f"{declared_root}/"))
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise WorkflowError(f"{label}[{index}] is invalid")
        paths.add(path)


def _validate_completed_status(status: dict[str, Any], request: dict[str, Any]) -> None:
    required_status_fields = {
        "schema_version",
        "run_id",
        "session",
        "request_digest",
        "state",
        "step",
        "attempt",
        "updated_at",
        "error",
        "provenance",
        "artifacts",
        "drive",
    }
    if set(status) != required_status_fields or status["state"] != "completed":
        raise WorkflowError("published completed status fields are invalid")
    if status["attempt"] < 1:
        raise WorkflowError("published completed status attempt is invalid")

    provenance = status["provenance"]
    provenance_fields = {
        "job_definition_digest",
        "source",
        "resolved_argv",
        "input_manifest",
        "output_manifest",
        "runtime",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise WorkflowError("published status provenance fields are invalid")
    if (
        provenance["job_definition_digest"] != request["job"]["definition_digest"]
        or provenance["resolved_argv"] != request["job"]["argv"]
    ):
        raise WorkflowError("published status job provenance is stale or corrupt")

    source = request["source"]
    expected_source_fields = {"mode", "repo_sha", "repo_tree", "tracked_clean"}
    if source["mode"] == "snapshot":
        expected_source_fields.update(
            {
                "snapshot_digest",
                "snapshot_archive_sha256",
                "snapshot_git_commit",
                "submodules",
            }
        )
    source_provenance = provenance["source"]
    if (
        not isinstance(source_provenance, dict)
        or set(source_provenance) != expected_source_fields
        or source_provenance["mode"] != source["mode"]
        or source_provenance["repo_sha"] != source["repo_sha"]
        or source_provenance["repo_tree"] != source["repo_tree"]
        or source_provenance["tracked_clean"] != source["tracked_clean"]
    ):
        raise WorkflowError("published status source provenance is stale or corrupt")
    if source["mode"] == "snapshot" and (
        source_provenance["snapshot_digest"] != source["content_digest"]
        or source_provenance["snapshot_archive_sha256"] != source["archive_sha256"]
        or source_provenance["submodules"] != source["submodules"]
        or not isinstance(source_provenance["snapshot_git_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", source_provenance["snapshot_git_commit"])
        is None
    ):
        raise WorkflowError("published snapshot provenance is stale or corrupt")

    input_manifest = provenance["input_manifest"]
    expected_inputs = request["job"]["inputs"]
    if not isinstance(input_manifest, list) or len(input_manifest) != len(
        expected_inputs
    ):
        raise WorkflowError("published status input manifest is incomplete")
    input_fields = {"source", "destination", "writable", "entries"}
    for index, (mapping, expected) in enumerate(
        zip(input_manifest, expected_inputs, strict=True)
    ):
        if (
            not isinstance(mapping, dict)
            or set(mapping) != input_fields
            or any(
                mapping[field] != expected[field]
                for field in input_fields - {"entries"}
            )
        ):
            raise WorkflowError("published status input manifest is stale or reordered")
        _validate_status_manifest_entries(
            mapping["entries"],
            f"published status input_manifest[{index}].entries",
            expected["destination"],
        )

    output_manifest = provenance["output_manifest"]
    expected_outputs = request["job"]["outputs"]
    if not isinstance(output_manifest, list) or len(output_manifest) != len(
        expected_outputs
    ):
        raise WorkflowError("published status output manifest is incomplete")
    for index, (mapping, expected) in enumerate(
        zip(output_manifest, expected_outputs, strict=True)
    ):
        if (
            not isinstance(mapping, dict)
            or set(mapping) != {"declared_path", "entries"}
            or mapping["declared_path"] != expected
        ):
            raise WorkflowError(
                "published status output manifest is stale or reordered"
            )
        _validate_status_manifest_entries(
            mapping["entries"],
            f"published status output_manifest[{index}].entries",
            expected,
        )

    runtime = provenance["runtime"]
    runtime_fields = {"collected", "python", "platform", "gpu", "cuda", "torch", "uv"}
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_fields
        or runtime["collected"] is not True
        or any(
            not isinstance(runtime[field], dict)
            for field in runtime_fields - {"collected"}
        )
    ):
        raise WorkflowError("published status runtime provenance is invalid")
    gpu = runtime["gpu"]
    cuda = runtime["cuda"]
    if (
        gpu.get("required") != (request["job"]["accelerator"] == "gpu")
        or gpu.get("requested") != request["runtime"]["gpu"]
        or not isinstance(gpu.get("available"), bool)
        or not isinstance(gpu.get("nvidia_smi"), dict)
        or not isinstance(cuda.get("available"), bool)
        or any(
            not isinstance(runtime[field].get("available"), bool)
            for field in ("python", "platform", "torch", "uv")
        )
    ):
        raise WorkflowError("published status runtime provenance is stale or invalid")
    if request["runtime"]["accelerator"] == "gpu" and not cuda["available"]:
        raise WorkflowError("published GPU runtime does not confirm CUDA availability")

    artifacts = status["artifacts"]
    expected_archive = (
        f"{_remote_paths(request['run_id'])['workspace']}/bundle/artifacts.tar.gz"
    )
    if (
        set(artifacts) != {"archive_path", "archive_sha256", "archive_size"}
        or artifacts["archive_path"] != expected_archive
    ):
        raise WorkflowError("published status artifact provenance is invalid")
    drive = status["drive"]
    expected_final = f"{request['drive']['root']}/colab-runs/{request['run_id']}"
    if (
        not isinstance(drive, dict)
        or set(drive) != {"mode", "final"}
        or drive["mode"] != request["drive"]["mode"]
        or drive["final"] != expected_final
    ):
        raise WorkflowError("published status Drive provenance is stale or corrupt")


def _artifact_destination(destination_dir: Path, run_id: str) -> Path:
    run_id = validate_run_id(run_id)
    destination_dir = destination_dir.expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{run_id}-artifacts.tar.gz"
    return ensure_within(destination_dir, destination, "artifact destination")


def _verified_existing_artifact(
    destination_dir: Path, status: dict[str, Any]
) -> Path | None:
    expected_digest = status.get("artifacts", {}).get("archive_sha256")
    expected_size = status.get("artifacts", {}).get("archive_size")
    if not isinstance(expected_digest, str) or not isinstance(expected_size, int):
        raise WorkflowError(
            "completed status does not contain artifact verification data"
        )
    destination = _artifact_destination(destination_dir, status["run_id"])
    if not destination.exists():
        return None
    if (
        destination.stat().st_size == expected_size
        and sha256_file(destination) == expected_digest
    ):
        return destination
    raise WorkflowError(
        f"artifact destination already exists with different content: {destination}"
    )


def _download_artifact(
    *,
    run_dir: Path,
    config_path: Path,
    session: str,
    remote_path: str,
    destination_dir: Path,
    status: dict[str, Any],
) -> Path:
    expected_digest = status.get("artifacts", {}).get("archive_sha256")
    expected_size = status.get("artifacts", {}).get("archive_size")
    if not isinstance(expected_digest, str) or not isinstance(expected_size, int):
        raise WorkflowError(
            "completed status does not contain artifact verification data"
        )
    existing = _verified_existing_artifact(destination_dir, status)
    if existing is not None:
        print(f"artifact already verified: {existing}")
        return existing
    destination = _artifact_destination(destination_dir, status["run_id"])
    temporary = run_dir / ".artifacts.download"
    temporary.unlink(missing_ok=True)
    try:
        _invoke(
            config_path,
            ["download", "-s", session, remote_path, str(temporary.resolve())],
        )
        if temporary.stat().st_size != expected_size:
            raise WorkflowError("downloaded artifact size does not match remote status")
        if sha256_file(temporary) != expected_digest:
            raise WorkflowError(
                "downloaded artifact SHA-256 does not match remote status"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified artifact: {destination}")
    return destination


def _rclone_copyto(config_path: Path, remote_path: str, local_path: Path) -> None:
    if shutil.which("rclone") is None:
        raise WorkflowError(
            "standalone rclone-mode download requires the local `rclone` executable"
        )
    try:
        result = subprocess.run(
            [
                "rclone",
                "--config",
                str(config_path),
                "copyto",
                remote_path,
                str(local_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkflowError(f"rclone download could not run: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise WorkflowError(
            f"rclone download failed with exit code {result.returncode}: {detail}"
        )


def _published_rclone_root(request: dict[str, Any]) -> str:
    drive = request.get("drive")
    if not isinstance(drive, dict):
        raise WorkflowError("local request Drive metadata is invalid")
    remote = drive.get("remote")
    root = drive.get("root")
    if not isinstance(remote, str) or not RCLONE_REMOTE_RE.fullmatch(remote):
        raise WorkflowError("local request rclone remote is invalid")
    if not isinstance(root, str):
        raise WorkflowError("local request Drive root is invalid")
    root = strict_relative_path(root, "request.drive.root")
    run_id = validate_run_id(request["run_id"])
    return f"{remote}:{root}/colab-runs/{run_id}"


def _validate_published_manifest(
    manifest: Any,
    status_path: Path,
    request_path: Path,
    status: dict[str, Any],
) -> None:
    if not isinstance(manifest, list):
        raise WorkflowError("published bundle manifest must be an array")
    entries: dict[str, dict[str, Any]] = {}
    for raw in manifest:
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            raise WorkflowError("published bundle manifest entry is invalid")
        path = strict_relative_path(raw["path"], "published manifest path")
        size = raw["size"]
        digest = raw["sha256"]
        if (
            path in entries
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise WorkflowError("published bundle manifest entry is invalid")
        entries[path] = raw
    if set(entries) != {"artifacts.tar.gz", "request.json", "status.json"}:
        raise WorkflowError("published bundle manifest has unexpected files")
    expected_files = {"status.json": status_path, "request.json": request_path}
    for name, local_path in expected_files.items():
        if entries[name]["size"] != local_path.stat().st_size or entries[name][
            "sha256"
        ] != sha256_file(local_path):
            raise WorkflowError(f"published {name} does not match bundle manifest")
    artifact_entry = entries["artifacts.tar.gz"]
    if (
        artifact_entry["size"] != status["artifacts"]["archive_size"]
        or artifact_entry["sha256"] != status["artifacts"]["archive_sha256"]
    ):
        raise WorkflowError(
            "published artifact metadata does not match bundle manifest"
        )


def _download_published_status_from_drive(
    *, run_dir: Path, request: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    remote_root = _published_rclone_root(request)
    temporary_status = run_dir / ".published-status.download"
    temporary_request = run_dir / ".published-request.download"
    temporary_manifest = run_dir / ".published-manifest.download"
    temporary_status.unlink(missing_ok=True)
    temporary_request.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)
    try:
        _rclone_copyto(config_path, f"{remote_root}/status.json", temporary_status)
        _rclone_copyto(config_path, f"{remote_root}/request.json", temporary_request)
        _rclone_copyto(config_path, f"{remote_root}/manifest.json", temporary_manifest)
        published_status = read_json(temporary_status)
        _validate_local_status(
            published_status,
            request_digest=request["request_digest"],
            run_id=request["run_id"],
            session=request["session"],
        )
        if published_status["state"] != "completed":
            raise WorkflowError("published Drive run is not completed")
        _validate_completed_status(published_status, request)
        published_request = read_json(temporary_request)
        if published_request != request:
            raise WorkflowError(
                "published Drive request does not match the local request"
            )
        try:
            manifest = json.loads(temporary_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkflowError(
                f"cannot read published bundle manifest: {error}"
            ) from error
        _validate_published_manifest(
            manifest,
            temporary_status,
            temporary_request,
            published_status,
        )
        atomic_write_json(run_dir / "status.json", published_status)
        return published_status
    finally:
        temporary_status.unlink(missing_ok=True)
        temporary_request.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _download_artifact_from_drive(
    *,
    run_dir: Path,
    request: dict[str, Any],
    config_path: Path,
    destination_dir: Path,
) -> Path:
    published_status = _download_published_status_from_drive(
        run_dir=run_dir, request=request, config_path=config_path
    )
    existing = _verified_existing_artifact(destination_dir, published_status)
    if existing is not None:
        print(f"artifact already verified: {existing}")
        return existing
    remote_root = _published_rclone_root(request)
    expected_digest = published_status["artifacts"]["archive_sha256"]
    expected_size = published_status["artifacts"]["archive_size"]
    destination = _artifact_destination(destination_dir, request["run_id"])
    temporary_artifact = run_dir / ".artifacts.download"
    temporary_artifact.unlink(missing_ok=True)
    try:
        _rclone_copyto(
            config_path, f"{remote_root}/artifacts.tar.gz", temporary_artifact
        )
        if temporary_artifact.stat().st_size != expected_size:
            raise WorkflowError("downloaded Drive artifact size does not match status")
        if sha256_file(temporary_artifact) != expected_digest:
            raise WorkflowError(
                "downloaded Drive artifact SHA-256 does not match status"
            )
        os.replace(temporary_artifact, destination)
        print(f"verified artifact: {destination}")
        return destination
    finally:
        temporary_artifact.unlink(missing_ok=True)


def _load_run(
    state_root: Path, run_id: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = _run_dir(state_root, run_id)
    if not run_dir.is_dir():
        raise WorkflowError(f"unknown run id: {run_id}")
    request = read_json(run_dir / "request.json")
    metadata = read_json(run_dir / "local.json")
    if request.get("request_digest") != metadata.get("request_digest"):
        raise WorkflowError("local run metadata has a request digest mismatch")
    request_copy = dict(request)
    digest = request_copy.pop("request_digest", None)
    if digest_json(request_copy) != digest:
        raise WorkflowError("local request content does not match its digest")
    if (
        request.get("schema_version") != SCHEMA_VERSION
        or request.get("run_id") != run_id
    ):
        raise WorkflowError("local request schema/run id is stale or corrupt")
    if (
        not isinstance(request.get("session"), str)
        or not SESSION_RE.fullmatch(request["session"])
        or metadata.get("session") != request["session"]
    ):
        raise WorkflowError("local request/session metadata is stale or corrupt")
    if not isinstance(request.get("drive"), dict) or not isinstance(
        request.get("job"), dict
    ):
        raise WorkflowError("local request Drive/job metadata is invalid")
    return run_dir, request, metadata


def _resolve_rclone_config(value: str | None, *, required: bool) -> Path | None:
    candidate = value or os.environ.get("RCLONE_CONFIG")
    if candidate is None:
        candidate = str(Path.home() / ".config/rclone/rclone.conf")
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        if required:
            raise WorkflowError(
                f"rclone config not found: {path}; pass --rclone-config"
            )
        return None
    mode = path.stat().st_mode & 0o777
    if mode & 0o077 or not mode & 0o400:
        raise WorkflowError(
            f"rclone config must be owner-readable with no group/world access "
            f"(0600-equivalent; found {mode:04o}): {path}; "
            f"run `chmod 600 {path}`"
        )
    return path


def _update_metadata(path: Path, metadata: dict[str, Any], state: str) -> None:
    metadata["session_state"] = state
    metadata["updated_at"] = _utc_now()
    atomic_write_json(path, metadata)


def _stop_session(
    config_path: Path, session: str, metadata_path: Path, metadata: dict[str, Any]
) -> bool:
    try:
        result = _invoke(config_path, ["stop", "-s", session], check=False)
        stopped = result.returncode == 0
    except WorkflowError as error:
        print(f"warning: could not invoke Colab session stop: {error}", file=sys.stderr)
        stopped = False
    state = "stopped" if stopped else "stop-failed"
    _update_metadata(metadata_path, metadata, state)
    return stopped


def _session_is_available(config_path: Path, session: str) -> bool:
    result = _invoke(config_path, ["status", "-s", session], check=False, capture=True)
    if result.returncode != 0:
        return False
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).lower()
    unavailable_markers = (
        "not found",
        "no session",
        "does not exist",
        "unknown session",
    )
    return not any(marker in output for marker in unavailable_markers)


def _cleanup_rclone_secret(
    repo_root: Path,
    config_path: Path,
    session: str,
    run_id: str,
    remote_config: str,
) -> bool:
    try:
        result = _exec_remote(
            repo_root,
            config_path,
            session,
            run_id,
            "cleanup-secret",
            timeout=120,
            rclone_config_remote=remote_config,
            check=False,
        )
    except WorkflowError as error:
        print(
            f"warning: could not invoke remote secret cleanup: {error}", file=sys.stderr
        )
        return False
    return result.returncode == 0


def _signal_handler(signum: int, _frame: Any) -> NoReturn:
    raise WorkflowInterrupted(f"received signal {signal.Signals(signum).name}")


def command_jobs(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    registry = load_registry(_jobs_dir(repo_root, args.jobs_dir))
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": job.name,
                        "description": job.description,
                        "accelerator": job.accelerator,
                        "timeout_seconds": job.timeout_seconds,
                        "setup": list(job.setup),
                    }
                    for job in registry.values()
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        width = max(len(name) for name in registry)
        for job in registry.values():
            print(f"{job.name:<{width}}  {job.description}")
    return 0


def _dry_run_plan(
    request: dict[str, Any],
    config_path: Path,
    remote: dict[str, str],
    repo_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    session = request["session"]
    new_command = [*_colab_prefix(config_path), "new", "-s", session]
    if request["runtime"]["gpu"] is not None:
        new_command.extend(["--gpu", request["runtime"]["gpu"]])
    if request["runtime"]["high_mem"]:
        new_command.append("--high-mem")
    commands: list[list[str]] = [new_command]
    if request["drive"]["mode"] == "mount":
        commands.append(
            [*_colab_prefix(config_path), "drivemount", "-s", session, "/content/drive"]
        )

    def exec_command(action: str, timeout: int) -> list[str]:
        return [
            *_colab_prefix(config_path),
            "exec",
            "-s",
            session,
            "-f",
            f"<generated-bootstrap-{action}.py>",
            "--timeout",
            str(timeout),
        ]

    commands.extend(
        [
            exec_command("prepare", 120),
            [
                *_colab_prefix(config_path),
                "upload",
                "-s",
                session,
                str((run_dir / "request.json").resolve()),
                remote["request"],
            ],
        ]
    )
    if request["source"]["mode"] == "snapshot":
        commands.append(
            [
                *_colab_prefix(config_path),
                "upload",
                "-s",
                session,
                str((run_dir / "source-snapshot.tar.gz").resolve()),
                remote["snapshot"],
            ]
        )
    if request["drive"]["mode"] == "rclone":
        commands.append(
            [
                *_colab_prefix(config_path),
                "upload",
                "-s",
                session,
                "<rclone-config>",
                remote["rclone_config"],
            ]
        )
    commands.extend(
        [
            exec_command("run", request["job"]["timeout_seconds"] + 7200),
            [
                *_colab_prefix(config_path),
                "download",
                "-s",
                session,
                remote["status"],
                "status.json",
            ],
            [*_colab_prefix(config_path), "stop", "-s", session],
        ]
    )
    return {
        "request": request,
        "capability_check": [*_colab_prefix(config_path), "version"],
        "commands": commands,
    }


def command_run(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    jobs = load_registry(_jobs_dir(repo_root, args.jobs_dir))
    if args.job not in jobs:
        raise WorkflowError(
            f"unknown job {args.job!r}; available: {', '.join(sorted(jobs))}"
        )
    job = jobs[args.job]
    if job.output_storage == "drive" and args.drive_mode != "mount":
        raise WorkflowError(
            "this training job writes directly to Drive; use --drive-mode mount"
        )
    _validate_overrides(job, args.overrides)
    if any("/content/drive" in item for item in args.overrides):
        raise WorkflowError(
            "Hydra overrides must not place training data or outputs on the Drive FUSE mount"
        )
    run_id = validate_run_id(args.run_id or _new_run_id())
    session = args.session or f"tennis-{run_id[-13:]}"
    if not SESSION_RE.fullmatch(session):
        raise WorkflowError(
            "session name must be 1-64 letters, digits, underscores, or hyphens"
        )
    drive_root = strict_relative_path(args.drive_root, "--drive-root")
    rclone_remote: str | None = None
    if args.drive_mode == "rclone":
        if not RCLONE_REMOTE_RE.fullmatch(args.rclone_remote):
            raise WorkflowError("--rclone-remote contains unsupported characters")
        rclone_remote = args.rclone_remote
    timeout_seconds = args.timeout or job.timeout_seconds
    if not 60 <= timeout_seconds <= 7 * 24 * 60 * 60:
        raise WorkflowError("--timeout must be between 60 and 604800 seconds")
    state_root = _state_root(args.state_dir)
    run_dir = _run_dir(state_root, run_id)
    if run_dir.exists() and not args.dry_run:
        raise WorkflowError(f"run id already exists: {run_id}")
    resolved_gpu = _resolve_gpu(job.accelerator, args.gpu)

    rclone_config: Path | None = None
    if not args.dry_run:
        if shutil.which("colab") is None:
            raise WorkflowError(
                "colab executable not found; follow the pinned setup in "
                "scripts/colab/README.md"
            )
        if args.drive_mode == "rclone":
            if shutil.which("rclone") is None:
                raise WorkflowError(
                    "rclone executable not found; install it before using --drive-mode rclone"
                )
            rclone_config = _resolve_rclone_config(args.rclone_config, required=True)

    snapshot_archive: Path | None = None
    with tempfile.TemporaryDirectory(prefix="tennis-colab-plan-") as temporary_name:
        snapshot: Snapshot | None = None
        if args.source == "snapshot":
            snapshot_target = Path(temporary_name) / "source-snapshot.tar.gz"
            snapshot = create_snapshot(repo_root, snapshot_target)
        source = _source_description(repo_root, args.source, args.ref, snapshot)
        request = _request(
            run_id=run_id,
            session=session,
            job=job,
            timeout_seconds=timeout_seconds,
            extra_args=args.overrides,
            source=source,
            drive_mode=args.drive_mode,
            drive_root=drive_root,
            rclone_remote=rclone_remote,
            gpu=resolved_gpu,
            high_mem=args.high_mem,
        )
        remote = _remote_paths(run_id)
        default_config = run_dir / "colab-sessions.json"
        config_path = (
            Path(args.colab_config).expanduser().resolve()
            if args.colab_config
            else default_config.resolve()
        )
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_plan(request, config_path, remote, repo_root, run_dir),
                    indent=2,
                )
            )
            return 0

        capability_config = Path(temporary_name) / "colab-sessions.json"
        _colab_capabilities(capability_config, high_mem=args.high_mem)

        run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        if snapshot is not None:
            snapshot_archive = run_dir / "source-snapshot.tar.gz"
            shutil.copy2(snapshot.archive_path, snapshot_archive)

    atomic_write_json(run_dir / "request.json", request)
    atomic_write_json(run_dir / "status.json", _initial_status(request))
    metadata = _local_metadata(request, config_path, remote)
    metadata_path = run_dir / "local.json"
    atomic_write_json(metadata_path, metadata)

    provisioning_attempted = False
    session_created = False
    remote_secret_uploaded = False
    succeeded = False
    forced_stop = False
    status_unknown = False
    try:
        new_args = ["new", "-s", session]
        if resolved_gpu is not None:
            new_args.extend(["--gpu", resolved_gpu])
        if args.high_mem:
            new_args.append("--high-mem")
        provisioning_attempted = True
        _invoke(config_path, new_args)
        session_created = True
        _update_metadata(metadata_path, metadata, "active")
        if args.drive_mode == "mount":
            _invoke(
                config_path,
                ["drivemount", "-s", session, "/content/drive"],
            )
        _exec_remote(repo_root, config_path, session, run_id, "prepare", timeout=120)
        _upload(config_path, session, run_dir / "request.json", remote["request"])
        if snapshot_archive is not None:
            _upload(config_path, session, snapshot_archive, remote["snapshot"])
        if rclone_config is not None:
            remote_secret_uploaded = True
            _upload(
                config_path,
                session,
                rclone_config,
                remote["rclone_config"],
            )
        execution = _exec_remote(
            repo_root,
            config_path,
            session,
            run_id,
            "run",
            timeout=timeout_seconds + 7200,
            rclone_config_remote=(
                remote["rclone_config"] if args.drive_mode == "rclone" else None
            ),
            check=False,
        )
        try:
            status = _download_status(
                run_dir,
                config_path,
                session,
                remote["status"],
                request["request_digest"],
                request["run_id"],
            )
        except WorkflowError as error:
            if request["drive"]["mode"] == "rclone" and rclone_config is not None:
                try:
                    status = _download_published_status_from_drive(
                        run_dir=run_dir,
                        request=request,
                        config_path=rclone_config,
                    )
                except WorkflowError as recovery_error:
                    if execution.returncode == 0:
                        status_unknown = True
                    raise WorkflowError(
                        "no valid VM status was available and published Drive status "
                        f"recovery failed: {recovery_error}"
                    ) from error
            else:
                if execution.returncode == 0:
                    status_unknown = True
                if execution.returncode != 0:
                    raise WorkflowError(
                        f"colab exec failed with exit code {execution.returncode} and no valid remote status"
                    ) from error
                raise WorkflowError(
                    "the remote command returned successfully but its published status "
                    "could not be confirmed; the session was retained for resume"
                ) from error
        succeeded = status.get("state") == "completed"
        if succeeded and args.download_to:
            if request["drive"]["mode"] == "rclone":
                assert rclone_config is not None
                _download_artifact_from_drive(
                    run_dir=run_dir,
                    request=request,
                    config_path=rclone_config,
                    destination_dir=Path(args.download_to),
                )
            else:
                _download_artifact(
                    run_dir=run_dir,
                    config_path=config_path,
                    session=session,
                    remote_path=remote["artifacts"],
                    destination_dir=Path(args.download_to),
                    status=status,
                )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if succeeded else EXIT_RUNTIME
    finally:
        if remote_secret_uploaded and provisioning_attempted:
            cleaned = _cleanup_rclone_secret(
                repo_root,
                config_path,
                session,
                run_id,
                remote["rclone_config"],
            )
            if not cleaned:
                forced_stop = True
                print(
                    "warning: remote rclone secret cleanup failed; forcing session stop",
                    file=sys.stderr,
                )
        should_stop = provisioning_attempted and (
            forced_stop
            or (
                not status_unknown
                and (succeeded or not args.keep_on_failure or not session_created)
            )
        )
        if should_stop:
            if not _stop_session(config_path, session, metadata_path, metadata):
                print(
                    f"warning: failed to stop Colab session {session!r}",
                    file=sys.stderr,
                )
        elif provisioning_attempted:
            state = (
                "retained-status-unknown"
                if status_unknown
                else "retained-after-failure"
            )
            _update_metadata(metadata_path, metadata, state)


def command_status(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_dir)
    run_dir, request, metadata = _load_run(state_root, args.run_id)
    config_path = Path(metadata["colab_config"]).resolve()
    session = metadata["session"]
    locally_live = metadata.get("session_state") in {
        "active",
        "retained-after-failure",
        "retained-status-unknown",
    }
    session_error: WorkflowError | None = None
    session_available = False
    if locally_live:
        try:
            _colab_capabilities(config_path)
            session_available = _session_is_available(config_path, session)
        except WorkflowError as error:
            session_error = error
    elif request["drive"]["mode"] == "rclone":
        cached_status = read_json(run_dir / "status.json")
        _validate_local_status(
            cached_status,
            request_digest=request["request_digest"],
            run_id=request["run_id"],
            session=request["session"],
        )
        if cached_status["state"] != "completed":
            session_error = WorkflowError(
                f"no live Colab session is recorded for {session!r}"
            )
    if session_available:
        try:
            status = _download_status(
                run_dir,
                config_path,
                session,
                metadata["remote"]["status"],
                request["request_digest"],
                request["run_id"],
            )
            _update_metadata(run_dir / "local.json", metadata, "active")
        except WorkflowError as error:
            session_error = error
    elif locally_live and session_error is None:
        session_error = WorkflowError(f"Colab session {session!r} is unavailable")

    if session_available and session_error is None:
        pass
    elif session_error is not None:
        if request["drive"]["mode"] == "rclone":
            rclone_config = _resolve_rclone_config(args.rclone_config, required=True)
            assert rclone_config is not None
            try:
                status = _download_published_status_from_drive(
                    run_dir=run_dir,
                    request=request,
                    config_path=rclone_config,
                )
            except WorkflowError as recovery_error:
                raise WorkflowError(
                    "could not refresh VM status and strict published Drive status "
                    f"recovery failed: {recovery_error}"
                ) from session_error
            state = "active" if session_available else "unavailable"
            _update_metadata(run_dir / "local.json", metadata, state)
        else:
            print(
                "warning: could not refresh VM status; using validated local cached "
                f"status: {session_error}",
                file=sys.stderr,
            )
            if metadata.get("session_state") == "active":
                _update_metadata(run_dir / "local.json", metadata, "unavailable")
            status = read_json(run_dir / "status.json")
    else:
        if metadata.get("session_state") == "active":
            _update_metadata(run_dir / "local.json", metadata, "unavailable")
        if request["drive"]["mode"] == "mount":
            print(
                "warning: no live VM status is available; using validated local "
                "cached status",
                file=sys.stderr,
            )
        status = read_json(run_dir / "status.json")
    _validate_local_status(
        status,
        request_digest=request["request_digest"],
        run_id=request["run_id"],
        session=request["session"],
    )
    value = {
        "status": status,
        "session_state": metadata["session_state"],
        "session_available": session_available,
    }
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


def command_download(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_dir)
    run_dir, request, metadata = _load_run(state_root, args.run_id)
    if request["drive"]["mode"] == "rclone":
        rclone_config = _resolve_rclone_config(args.rclone_config, required=True)
        assert rclone_config is not None
        _download_artifact_from_drive(
            run_dir=run_dir,
            request=request,
            config_path=rclone_config,
            destination_dir=Path(args.to),
        )
        return 0
    status = read_json(run_dir / "status.json")
    _validate_local_status(
        status,
        request_digest=request["request_digest"],
        run_id=request["run_id"],
        session=request["session"],
    )
    if status.get("state") != "completed":
        raise WorkflowError("artifacts can only be downloaded from a completed run")
    existing = _verified_existing_artifact(Path(args.to), status)
    if existing is not None:
        print(f"artifact already verified: {existing}")
        return 0
    if metadata.get("session_state") not in {
        "active",
        "retained-after-failure",
        "retained-status-unknown",
    }:
        raise WorkflowError(
            "mount-mode download requires a live run session; use `run --download-to` "
            "before the default session cleanup"
        )
    config_path = Path(metadata["colab_config"]).resolve()
    _colab_capabilities(config_path)
    if not _session_is_available(config_path, metadata["session"]):
        raise WorkflowError(
            "mount-mode download requires a live run session; use `run --download-to` "
            "before the default session cleanup"
        )
    _download_artifact(
        run_dir=run_dir,
        config_path=config_path,
        session=metadata["session"],
        remote_path=metadata["remote"]["artifacts"],
        destination_dir=Path(args.to),
        status=status,
    )
    return 0


def command_stop(args: argparse.Namespace) -> int:
    state_root = _state_root(args.state_dir)
    run_dir, _request_value, metadata = _load_run(state_root, args.run_id)
    config_path = Path(metadata["colab_config"]).resolve()
    _colab_capabilities(config_path)
    ok = _stop_session(
        config_path,
        metadata["session"],
        run_dir / "local.json",
        metadata,
    )
    return 0 if ok else EXIT_RUNTIME


def command_progress(args: argparse.Namespace) -> int:
    """Read bounded progress through the contents API, without executing kernel code."""
    if args.interval < 2:
        raise WorkflowError("--interval must be at least 2 seconds")
    run_dir, request, metadata = _load_run(_state_root(args.state_dir), args.run_id)
    while True:
        temporary = run_dir / ".progress.download"
        try:
            _invoke(
                Path(metadata["colab_config"]),
                [
                    "download",
                    "-s",
                    metadata["session"],
                    f"/content/tennis-lab-runs/{request['run_id']}/progress.json",
                    str(temporary),
                ],
                capture=True,
            )
            progress = read_json(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        if (
            progress.get("run_id") != request["run_id"]
            or progress.get("request_digest") != request["request_digest"]
        ):
            raise WorkflowError("progress belongs to a different run/request")
        try:
            age = (
                datetime.now(UTC) - datetime.fromisoformat(progress["updated_at"])
            ).total_seconds()
        except (KeyError, TypeError, ValueError) as error:
            raise WorkflowError("invalid progress timestamp") from error
        progress["heartbeat_age_seconds"] = round(age, 1)
        progress["stale"] = age > 30
        training = progress.get("training")
        if training and training.get("updated_at"):
            progress["training_update_age_seconds"] = round(
                (
                    datetime.now(UTC) - datetime.fromisoformat(training["updated_at"])
                ).total_seconds(),
                1,
            )
        atomic_write_json(run_dir / "progress.json", progress)
        if args.command == "logs":
            print("\n".join(progress.get("log_tail", [])[-args.tail :]))
        else:
            print(json.dumps(progress, indent=2, ensure_ascii=False))
        if not args.watch or progress.get("state") != "running":
            return 0
        time.sleep(args.interval)


def command_resume(args: argparse.Namespace) -> int:
    repo_root = _resolve_repo_root(args.repo_root)
    state_root = _state_root(args.state_dir)
    run_dir, request, metadata = _load_run(state_root, args.run_id)
    status = read_json(run_dir / "status.json")
    _validate_local_status(
        status,
        request_digest=request["request_digest"],
        run_id=request["run_id"],
        session=request["session"],
    )
    if status.get("state") == "completed":
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0
    if metadata.get("session_state") not in {
        "active",
        "retained-after-failure",
        "retained-status-unknown",
    }:
        raise WorkflowError(
            "this run has no retained session; rerun with a new run id or use --keep-on-failure"
        )
    config_path = Path(metadata["colab_config"]).resolve()
    session = metadata["session"]
    remote = metadata["remote"]
    _colab_capabilities(config_path)
    rclone_config = None
    if request["drive"]["mode"] == "rclone":
        rclone_config = _resolve_rclone_config(args.rclone_config, required=True)
    remote_secret_uploaded = False
    succeeded = False
    forced_stop = False
    status_unknown = False
    try:
        if request["drive"]["mode"] == "mount":
            _invoke(
                config_path,
                ["drivemount", "-s", session, "/content/drive"],
            )
        if rclone_config is not None:
            remote_secret_uploaded = True
            _upload(config_path, session, rclone_config, remote["rclone_config"])
        execution = _exec_remote(
            repo_root,
            config_path,
            session,
            request["run_id"],
            "run",
            timeout=request["job"]["timeout_seconds"] + 7200,
            rclone_config_remote=(
                remote["rclone_config"]
                if request["drive"]["mode"] == "rclone"
                else None
            ),
            check=False,
        )
        try:
            status = _download_status(
                run_dir,
                config_path,
                session,
                remote["status"],
                request["request_digest"],
                request["run_id"],
            )
        except WorkflowError as error:
            if request["drive"]["mode"] == "rclone" and rclone_config is not None:
                try:
                    status = _download_published_status_from_drive(
                        run_dir=run_dir,
                        request=request,
                        config_path=rclone_config,
                    )
                except WorkflowError as recovery_error:
                    if execution.returncode == 0:
                        status_unknown = True
                    raise WorkflowError(
                        "resume could not confirm VM or published Drive status: "
                        f"{recovery_error}"
                    ) from error
            else:
                if execution.returncode == 0:
                    status_unknown = True
                raise WorkflowError(
                    "resume could not confirm remote status; the session was retained"
                ) from error
        succeeded = status.get("state") == "completed"
        if succeeded and args.download_to:
            if request["drive"]["mode"] == "rclone":
                assert rclone_config is not None
                _download_artifact_from_drive(
                    run_dir=run_dir,
                    request=request,
                    config_path=rclone_config,
                    destination_dir=Path(args.download_to),
                )
            else:
                _download_artifact(
                    run_dir=run_dir,
                    config_path=config_path,
                    session=session,
                    remote_path=remote["artifacts"],
                    destination_dir=Path(args.download_to),
                    status=status,
                )
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return 0 if succeeded else EXIT_RUNTIME
    finally:
        if remote_secret_uploaded:
            cleaned = _cleanup_rclone_secret(
                repo_root,
                config_path,
                session,
                request["run_id"],
                remote["rclone_config"],
            )
            if not cleaned:
                forced_stop = True
        if forced_stop or (
            not status_unknown and (succeeded or not args.keep_on_failure)
        ):
            _stop_session(config_path, session, run_dir / "local.json", metadata)
        else:
            state = (
                "retained-status-unknown"
                if status_unknown
                else "retained-after-failure"
            )
            _update_metadata(run_dir / "local.json", metadata, state)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser and its stable subcommand contract."""

    parser = argparse.ArgumentParser(
        prog="scripts/colab/run.sh",
        description="Run strict, reproducible tennis-lab jobs on Google Colab.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_STATE_ROOT),
        help="local run metadata root (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help="override the strict TOML job registry directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    jobs_parser = subparsers.add_parser("jobs", help="list validated job definitions")
    jobs_parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    jobs_parser.set_defaults(handler=command_jobs)

    run_parser = subparsers.add_parser(
        "run", help="provision a session and run one job"
    )
    run_parser.add_argument("job", help="job name from the TOML registry")
    run_parser.add_argument("--run-id", help="unique run id (generated by default)")
    run_parser.add_argument(
        "--session", help="named Colab session (generated by default)"
    )
    run_parser.add_argument(
        "--gpu",
        choices=GPU_ARGUMENT_CHOICES,
        default="auto",
        help="runtime selection; auto uses CPU for CPU jobs and T4 for GPU jobs",
    )
    run_parser.add_argument(
        "--high-mem",
        action="store_true",
        help="request high-memory provisioning when supported by the Colab CLI",
    )
    run_parser.add_argument("--source", choices=("git", "snapshot"), default="git")
    run_parser.add_argument(
        "--ref", default="HEAD", help="git ref resolved to an exact SHA"
    )
    run_parser.add_argument(
        "--drive-mode", choices=("rclone", "mount"), default="mount"
    )
    run_parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    run_parser.add_argument("--rclone-config")
    run_parser.add_argument("--rclone-remote", default="gdrive")
    run_parser.add_argument(
        "--colab-config", help="absolute or relative Colab session config"
    )
    run_parser.add_argument(
        "--timeout", type=int, help="override job timeout in seconds"
    )
    run_parser.add_argument(
        "--download-to", help="download and verify the artifact archive"
    )
    run_parser.add_argument(
        "--keep-on-failure",
        action="store_true",
        help="retain the session after a failed remote run (secrets are still removed)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved request and commands without calling Colab or Drive",
    )
    run_parser.set_defaults(handler=command_run, overrides=[])

    status_parser = subparsers.add_parser(
        "status", help="show local and live remote status"
    )
    status_parser.add_argument("run_id")
    status_parser.add_argument(
        "--rclone-config",
        help="local rclone config used for strict published-status recovery",
    )
    status_parser.set_defaults(handler=command_status)

    for name in ("progress", "logs"):
        monitor_parser = subparsers.add_parser(
            name, help="read live progress or the last 80 log lines"
        )
        monitor_parser.add_argument("run_id")
        monitor_parser.add_argument("--watch", action="store_true")
        monitor_parser.add_argument("--interval", type=float, default=10)
        monitor_parser.add_argument(
            "--tail", type=int, choices=range(1, 81), default=40, metavar="1..80"
        )
        monitor_parser.set_defaults(handler=command_progress)

    resume_parser = subparsers.add_parser("resume", help="resume a retained failed run")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--rclone-config")
    resume_parser.add_argument("--download-to")
    resume_parser.add_argument("--keep-on-failure", action="store_true")
    resume_parser.set_defaults(handler=command_resume)

    download_parser = subparsers.add_parser(
        "download", help="download and verify published or live-session artifacts"
    )
    download_parser.add_argument("run_id")
    download_parser.add_argument(
        "--to", required=True, help="local destination directory"
    )
    download_parser.add_argument(
        "--rclone-config",
        help="local rclone config for downloading a published rclone-mode run",
    )
    download_parser.set_defaults(handler=command_download)

    stop_parser = subparsers.add_parser(
        "stop", help="stop the session recorded for a run"
    )
    stop_parser.add_argument("run_id")
    stop_parser.set_defaults(handler=command_stop)
    return parser


def _split_overrides(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    separator = argv.index("--")
    return argv[:separator], argv[separator + 1 :]


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, enforce lifecycle cleanup, and return a documented exit code."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parse_argv, overrides = _split_overrides(raw_argv)
    parser = build_parser()
    args = parser.parse_args(parse_argv)
    if overrides and args.command != "run":
        parser.error("arguments after -- are only valid for the run subcommand")
    if args.command == "run":
        args.overrides = overrides
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    for sig in previous_handlers:
        signal.signal(sig, _signal_handler)
    try:
        return int(args.handler(args))
    except WorkflowInterrupted as error:
        print(f"interrupted: {error}", file=sys.stderr)
        return 130
    except WorkflowError as error:
        print(f"error: {error}", file=sys.stderr)
        if "unknown run id" in str(error):
            return EXIT_NOT_FOUND
        return EXIT_CONTRACT
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
