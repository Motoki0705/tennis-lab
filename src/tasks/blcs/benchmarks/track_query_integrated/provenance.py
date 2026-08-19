"""Tracked serial-ownership provenance for Issue #753 CUDA evidence jobs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

from src.tasks.blcs.benchmarks.contracts import (
    ISSUE_NUMBER,
    BenchmarkContractError,
    benchmark_path_resolver,
    repository_root,
)
from src.utils.configuration import PathContractError, PathRole

PROVENANCE_SCHEMA_VERSION = 2
PACKAGE_ORDER = ("6A", "6B", "6C", "6D")
EXPECTED_ORCHESTRATOR_SESSION_UUID = "01a01500-a7a5-7310-837f-3dc88acbefc4"
OWNER_SESSION_SNAPSHOT_DIRECTORY = (
    "src/tasks/blcs/benchmarks/results/issue_753/owner_sessions"
)
SESSION_META_BYTE_SCOPE = "whole-snapshot-exact-first-record-bytes"
SESSION_META_RECORD_SCHEMA = "codex-rollout-session-meta-jsonl-v1"
EXPECTED_COMPONENTS = {
    "6A": "mhc",
    "6B": "compressor",
    "6C": "cswa",
    "6D": "integrated",
}
EXPECTED_OWNERS = {
    "6A": "/root/issue753_attempt6_6a_evidence",
    "6B": "/root/issue753_attempt6_6b_evidence",
    "6C": "/root/issue753_attempt6_6c_evidence",
    "6D": "/root/issue753_attempt6_6d_evidence",
}
EXPECTED_EVIDENCE_PATHS = {
    package_id: f"src/tasks/blcs/benchmarks/results/issue_753/{component}.json"
    for package_id, component in EXPECTED_COMPONENTS.items()
}


@dataclass(frozen=True, slots=True)
class ProvenancePackageSpec:
    """Local source records used to promote one package's historical facts."""

    package_id: str
    owner_task: str
    session_uuid: str
    session_meta_path: Path
    job_id: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _record_bundle_sha256(packages: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(packages),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise BenchmarkContractError(f"{path} must be an object with string keys")
    return cast(Mapping[str, Any], value)


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        raise BenchmarkContractError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"{path} must be a non-empty string")
    return value


def _require_sha256(value: object, path: str) -> str:
    digest = _require_nonempty_string(value, path)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise BenchmarkContractError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _require_uuid(value: object, path: str) -> str:
    raw = _require_nonempty_string(value, path)
    try:
        parsed = UUID(raw)
    except ValueError as error:
        raise BenchmarkContractError(f"{path} must be a canonical UUID") from error
    if str(parsed) != raw:
        raise BenchmarkContractError(f"{path} must be a canonical UUID")
    return raw


def _timestamp(value: object, path: str) -> datetime:
    raw = _require_nonempty_string(value, path)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise BenchmarkContractError(f"{path} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise BenchmarkContractError(f"{path} must include a UTC offset")
    return parsed


def _parse_job_headers(path: Path) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# "):
            continue
        key, separator, value = line[2:].partition(": ")
        if separator:
            headers[key] = value
    required = {"name", "added", "provider", "session", "issue"}
    if not required <= set(headers):
        raise BenchmarkContractError(
            f"queue job headers missing {sorted(required - set(headers))}: {path}"
        )
    return headers


def _parse_session_meta_record(
    record_bytes: bytes,
    *,
    session_uuid: str,
    owner_task: str,
    path: str,
) -> str:
    if not record_bytes:
        raise BenchmarkContractError(f"{path} must not be empty")
    try:
        record = json.loads(record_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkContractError(f"invalid Codex session_meta record: {path}") from error
    record_mapping = _require_mapping(record, path)
    _require_exact_keys(record_mapping, {"timestamp", "type", "payload"}, path)
    _timestamp(record_mapping["timestamp"], f"{path}.timestamp")
    if record_mapping["type"] != "session_meta":
        raise BenchmarkContractError("rollout first record must be session_meta")
    payload = _require_mapping(record_mapping["payload"], f"{path}.payload")
    required_payload_keys = {
        "session_id",
        "id",
        "parent_thread_id",
        "source",
        "thread_source",
        "agent_path",
    }
    if not required_payload_keys <= set(payload):
        raise BenchmarkContractError(
            f"{path}.payload keys missing "
            f"{sorted(required_payload_keys - set(payload))}"
        )
    if payload["id"] != session_uuid:
        raise BenchmarkContractError("session_meta.payload.id mismatch")
    if payload["agent_path"] != owner_task:
        raise BenchmarkContractError("session_meta.payload.agent_path mismatch")
    parent = _require_nonempty_string(
        payload["parent_thread_id"], "session_meta.payload.parent_thread_id"
    )
    _require_uuid(parent, "session_meta.payload.parent_thread_id")
    if payload["session_id"] != parent:
        raise BenchmarkContractError("session_meta.payload.session_id mismatch")
    if payload["thread_source"] != "subagent":
        raise BenchmarkContractError("session_meta.payload.thread_source mismatch")
    source = _require_mapping(payload["source"], "session_meta.payload.source")
    _require_exact_keys(source, {"subagent"}, "session_meta.payload.source")
    subagent = _require_mapping(
        source["subagent"], "session_meta.payload.source.subagent"
    )
    _require_exact_keys(
        subagent, {"thread_spawn"}, "session_meta.payload.source.subagent"
    )
    thread_spawn = _require_mapping(
        subagent["thread_spawn"],
        "session_meta.payload.source.subagent.thread_spawn",
    )
    for key in ("parent_thread_id", "agent_path"):
        _require_nonempty_string(
            thread_spawn.get(key),
            f"session_meta.payload.source.subagent.thread_spawn.{key}",
        )
    if thread_spawn["parent_thread_id"] != parent:
        raise BenchmarkContractError(
            "session_meta.payload.source.subagent.thread_spawn.parent_thread_id "
            "mismatch"
        )
    if thread_spawn["agent_path"] != owner_task:
        raise BenchmarkContractError(
            "session_meta.payload.source.subagent.thread_spawn.agent_path mismatch"
        )
    return parent


def _load_session_meta(
    path: Path, *, session_uuid: str, owner_task: str
) -> tuple[str, bytes]:
    with path.open("rb") as stream:
        first_line = stream.readline()
    parent = _parse_session_meta_record(
        first_line,
        session_uuid=session_uuid,
        owner_task=owner_task,
        path="session_meta",
    )
    return parent, first_line


def _record_origin(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise BenchmarkContractError(f"record path is outside expected root: {path}") from error


def _secure_regular_file_below_root(
    path: str | Path, root: Path, *, label: str
) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise BenchmarkContractError(f"{label} root is missing: {root}") from error
    if not resolved_root.is_dir():
        raise BenchmarkContractError(f"{label} root is not a directory: {root}")
    if isinstance(path, Path):
        if path.is_absolute():
            try:
                relative_path = path.relative_to(resolved_root)
            except ValueError as error:
                raise BenchmarkContractError(
                    f"{label} path escapes its root: {path}"
                ) from error
            relative_parts = relative_path.parts
            resolver_fragment: str | Path = relative_path
        else:
            relative_parts = path.parts
            resolver_fragment = path
    else:
        relative = PurePosixPath(path)
        if relative.is_absolute():
            raise BenchmarkContractError(f"{label} path escapes its root: {path}")
        relative_parts = relative.parts
        resolver_fragment = path
    if not relative_parts:
        raise BenchmarkContractError(f"{label} path is not a regular file: {path}")

    resolver = benchmark_path_resolver(resolved_root)
    try:
        candidate = cast(Path, resolver.resolve(PathRole.PROJECT, resolver_fragment))
    except PathContractError as error:
        raise BenchmarkContractError(f"{label} path escapes its root: {path}") from error

    descriptor = os.open(resolved_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(relative_parts):
            try:
                mode = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                ).st_mode
            except FileNotFoundError as error:
                raise BenchmarkContractError(
                    f"{label} file is missing: {path}"
                ) from error
            if stat.S_ISLNK(mode):
                raise BenchmarkContractError(
                    f"{label} path must not contain symlinks: {path}"
                )
            is_last = index == len(relative_parts) - 1
            if not is_last and not stat.S_ISDIR(mode):
                raise BenchmarkContractError(
                    f"{label} parent is not a directory: {part}"
                )
            if is_last and not stat.S_ISREG(mode):
                raise BenchmarkContractError(
                    f"{label} path is not a regular file: {path}"
                )
            if not is_last:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
    finally:
        with suppress(OSError):
            os.close(descriptor)
    return candidate


def _snapshot_relative_path(package_id: str, session_uuid: str) -> str:
    return (
        f"{OWNER_SESSION_SNAPSHOT_DIRECTORY}/{package_id}-{session_uuid}.jsonl"
    )


def _require_repository_relative_path(value: object, path: str) -> str:
    raw = _require_nonempty_string(value, path)
    parsed = PurePosixPath(raw)
    if (
        parsed.is_absolute()
        or raw != parsed.as_posix()
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise BenchmarkContractError(f"{path} must be a canonical repository-relative path")
    return raw


def _validate_source_origin(origin: object, *, session_uuid: str, path: str) -> str:
    raw = _require_repository_relative_path(origin, path)
    parsed = PurePosixPath(raw)
    if parsed.parts[0] not in {"sessions", "archived_sessions"}:
        raise BenchmarkContractError(f"{path} must identify a Codex rollout origin")
    if parsed.suffix != ".jsonl" or session_uuid not in parsed.name:
        raise BenchmarkContractError(f"{path} does not identify the owner session")
    return raw


def _write_session_snapshot(
    *,
    project_root: Path,
    relative_path: str,
    first_record: bytes,
) -> Path:
    resolved_root = project_root.resolve(strict=True)
    resolver = benchmark_path_resolver(resolved_root)
    snapshot_parts = PurePosixPath(OWNER_SESSION_SNAPSHOT_DIRECTORY).parts
    for index in range(len(snapshot_parts)):
        snapshot_directory = resolver.resolve_symlink_entry(
            PathRole.PROJECT, "/".join(snapshot_parts[: index + 1])
        )
        if snapshot_directory.exists() or snapshot_directory.is_symlink():
            mode = snapshot_directory.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise BenchmarkContractError(
                    "owner session snapshot directory must not contain symlinks"
                )
            if not stat.S_ISDIR(mode):
                raise BenchmarkContractError(
                    "owner session snapshot parent is not a directory"
                )
        else:
            snapshot_directory.mkdir()
    snapshot_path = resolver.resolve_symlink_entry(
        PathRole.PROJECT, relative_path
    )
    if snapshot_path.exists() or snapshot_path.is_symlink():
        existing = _secure_regular_file_below_root(
            snapshot_path, resolved_root, label="owner session snapshot"
        )
        if existing.read_bytes() != first_record:
            raise BenchmarkContractError(
                f"immutable owner session snapshot already differs: {relative_path}"
            )
        return existing
    try:
        with snapshot_path.open("xb") as stream:
            stream.write(first_record)
    except FileExistsError as error:
        raise BenchmarkContractError(
            f"owner session snapshot appeared concurrently: {relative_path}"
        ) from error
    return _secure_regular_file_below_root(
        snapshot_path, resolved_root, label="owner session snapshot"
    )


def _authenticate_session_snapshot(
    session_meta: Mapping[str, Any],
    *,
    package_id: str,
    session_uuid: str,
    owner_task: str,
    project_root: Path,
    path: str,
) -> str:
    _require_exact_keys(
        session_meta,
        {"snapshot_path", "sha256", "byte_scope", "record_schema", "source_origin"},
        path,
    )
    snapshot_relative_path = _require_repository_relative_path(
        session_meta["snapshot_path"], f"{path}.snapshot_path"
    )
    expected_snapshot_path = _snapshot_relative_path(package_id, session_uuid)
    if snapshot_relative_path != expected_snapshot_path:
        raise BenchmarkContractError(f"{path}.snapshot_path mismatch")
    snapshot_sha256 = _require_sha256(session_meta["sha256"], f"{path}.sha256")
    if session_meta["byte_scope"] != SESSION_META_BYTE_SCOPE:
        raise BenchmarkContractError(f"{path}.byte_scope mismatch")
    if session_meta["record_schema"] != SESSION_META_RECORD_SCHEMA:
        raise BenchmarkContractError(f"{path}.record_schema mismatch")
    _validate_source_origin(
        session_meta["source_origin"],
        session_uuid=session_uuid,
        path=f"{path}.source_origin",
    )

    resolver = benchmark_path_resolver(project_root)
    snapshot_path = _secure_regular_file_below_root(
        snapshot_relative_path,
        project_root,
        label="owner session snapshot",
    )
    evidence_root = resolver.resolve(
        PathRole.PROJECT, OWNER_SESSION_SNAPSHOT_DIRECTORY
    )
    try:
        snapshot_path.relative_to(evidence_root)
    except ValueError as error:
        raise BenchmarkContractError(
            f"{path}.snapshot_path is outside the owner-session evidence directory"
        ) from error
    snapshot_bytes = snapshot_path.read_bytes()
    if _sha256_bytes(snapshot_bytes) != snapshot_sha256:
        raise BenchmarkContractError(f"{path}.sha256 does not match snapshot bytes")
    if len(snapshot_bytes.splitlines(keepends=True)) != 1:
        raise BenchmarkContractError(f"{path} snapshot must contain exactly one record")
    return _parse_session_meta_record(
        snapshot_bytes,
        session_uuid=session_uuid,
        owner_task=owner_task,
        path=path,
    )


def build_serial_provenance(
    specs: Sequence[ProvenancePackageSpec],
    *,
    queue_root: Path,
    codex_home: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Promote exact queue/session records into a candidate-tracked document."""
    project_root = repository_root() if root is None else root.resolve(strict=False)
    if tuple(spec.package_id for spec in specs) != PACKAGE_ORDER:
        raise BenchmarkContractError("provenance package specs must be ordered 6A-6D")

    packages: list[dict[str, Any]] = []
    orchestrator_sessions: set[str] = set()
    resolver = benchmark_path_resolver(project_root)
    try:
        validated_queue_root = resolver.validate(PathRole.PROJECT, queue_root)
    except PathContractError as error:
        raise BenchmarkContractError(
            f"queue root is outside the project root: {queue_root}"
        ) from error
    for spec in specs:
        expected_owner = EXPECTED_OWNERS[spec.package_id]
        if spec.owner_task != expected_owner:
            raise BenchmarkContractError(
                f"{spec.package_id} owner mismatch: {spec.owner_task}"
            )
        session_uuid = _require_uuid(
            spec.session_uuid, f"{spec.package_id}.session_uuid"
        )
        source_path = _secure_regular_file_below_root(
            spec.session_meta_path,
            codex_home,
            label=f"{spec.package_id} Codex rollout",
        )
        source_origin = _record_origin(source_path, codex_home.resolve(strict=True))
        _validate_source_origin(
            source_origin,
            session_uuid=session_uuid,
            path=f"{spec.package_id}.source_origin",
        )
        orchestrator_session, first_record = _load_session_meta(
            source_path,
            session_uuid=spec.session_uuid,
            owner_task=spec.owner_task,
        )
        if orchestrator_session != EXPECTED_ORCHESTRATOR_SESSION_UUID:
            raise BenchmarkContractError(
                f"{spec.package_id} owner session parent is not the canonical "
                "orchestrator"
            )
        orchestrator_sessions.add(orchestrator_session)
        snapshot_relative_path = _snapshot_relative_path(
            spec.package_id, session_uuid
        )
        snapshot_path = _write_session_snapshot(
            project_root=project_root,
            relative_path=snapshot_relative_path,
            first_record=first_record,
        )

        done_job_path = resolver.resolve_beneath(
            PathRole.PROJECT,
            validated_queue_root,
            "done",
            f"{spec.job_id}.job",
        )
        run_path = resolver.resolve_beneath(
            PathRole.PROJECT,
            validated_queue_root,
            "repro",
            spec.job_id,
            "run.json",
        )
        log_path = resolver.resolve_beneath(
            PathRole.PROJECT,
            validated_queue_root,
            "logs",
            f"{spec.job_id}.log",
        )
        if not done_job_path.is_file() or not run_path.is_file() or not log_path.is_file():
            raise BenchmarkContractError(
                f"{spec.package_id} queue record is not terminal and complete"
            )
        headers = _parse_job_headers(done_job_path)
        run = _require_mapping(
            json.loads(run_path.read_text(encoding="utf-8")),
            f"{spec.package_id}.run_json",
        )
        expected_run_values = {
            "run_id": spec.job_id,
            "name": headers["name"],
            "provider": headers["provider"],
            "session": spec.session_uuid,
            "issue": str(ISSUE_NUMBER),
        }
        for key, expected in expected_run_values.items():
            if run.get(key) != expected:
                raise BenchmarkContractError(
                    f"{spec.package_id}.run_json.{key} mismatch"
                )
        if headers["session"] != spec.session_uuid:
            raise BenchmarkContractError(f"{spec.package_id} queue session mismatch")
        if headers["issue"] != str(ISSUE_NUMBER):
            raise BenchmarkContractError(f"{spec.package_id} queue issue mismatch")

        added = _timestamp(headers["added"], f"{spec.package_id}.added")
        started = _timestamp(run.get("captured_at"), f"{spec.package_id}.captured_at")
        done = datetime.fromtimestamp(done_job_path.stat().st_ctime, tz=UTC).astimezone(
            added.tzinfo
        )
        if not added <= started <= done:
            raise BenchmarkContractError(
                f"{spec.package_id} queue timestamps are not monotonic"
            )

        evidence_path = EXPECTED_EVIDENCE_PATHS[spec.package_id]
        evidence_file = resolver.resolve(PathRole.PROJECT, evidence_path)
        packages.append(
            {
                "package_id": spec.package_id,
                "component": EXPECTED_COMPONENTS[spec.package_id],
                "owner": {
                    "canonical_agent_task": spec.owner_task,
                    "session_uuid": session_uuid,
                    "session_meta": {
                        "snapshot_path": snapshot_relative_path,
                        "sha256": _sha256_file(snapshot_path),
                        "byte_scope": SESSION_META_BYTE_SCOPE,
                        "record_schema": SESSION_META_RECORD_SCHEMA,
                        "source_origin": source_origin,
                    },
                },
                "queue": {
                    "job_id": spec.job_id,
                    "name": run["name"],
                    "provider": run["provider"],
                    "issue": run["issue"],
                    "command": run["command"],
                    "cwd": run["cwd"],
                    "added_at": headers["added"],
                    "started_at": run["captured_at"],
                    "done_at": done.isoformat(timespec="microseconds"),
                    "terminal_outcome": "done",
                    "exit_code": 0,
                    "records": {
                        "run_json": {
                            "origin": _record_origin(run_path, project_root),
                            "sha256": _sha256_file(run_path),
                        },
                        "done_job": {
                            "origin": _record_origin(done_job_path, project_root),
                            "sha256": _sha256_file(done_job_path),
                        },
                        "log": {
                            "origin": _record_origin(log_path, project_root),
                            "sha256": _sha256_file(log_path),
                        },
                    },
                },
                "stable_evidence": {
                    "path": evidence_path,
                    "sha256": _sha256_file(evidence_file),
                },
            }
        )

    if len(orchestrator_sessions) != 1:
        raise BenchmarkContractError("all package sessions must share one orchestrator")
    orchestrator_session_uuid = next(iter(orchestrator_sessions))
    if orchestrator_session_uuid != EXPECTED_ORCHESTRATOR_SESSION_UUID:
        raise BenchmarkContractError("package orchestrator session is not canonical")
    document = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "issue": ISSUE_NUMBER,
        "orchestrator_session_uuid": orchestrator_session_uuid,
        "packages": packages,
        "record_bundle_sha256": _record_bundle_sha256(packages),
        "serial_validation": {
            "package_order": list(PACKAGE_ORDER),
            "unique_owner_tasks": True,
            "unique_child_sessions": True,
            "unique_queue_jobs": True,
            "job_ids_strictly_increasing": True,
            "no_time_overlap": True,
            "status": "PASS",
        },
    }
    validate_serial_provenance(document, root=project_root)
    return document


def validate_serial_provenance(
    document: Mapping[str, Any], *, root: Path | None = None
) -> None:
    """Validate candidate-tracked ownership, record hashes, and serial order."""
    project_root = repository_root() if root is None else root.resolve(strict=False)
    resolver = benchmark_path_resolver(project_root)
    _require_exact_keys(
        document,
        {
            "schema_version",
            "issue",
            "orchestrator_session_uuid",
            "packages",
            "record_bundle_sha256",
            "serial_validation",
        },
        "provenance",
    )
    if document["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise BenchmarkContractError("provenance.schema_version mismatch")
    if document["issue"] != ISSUE_NUMBER:
        raise BenchmarkContractError("provenance.issue mismatch")
    orchestrator_session_uuid = _require_uuid(
        document["orchestrator_session_uuid"], "provenance.orchestrator_session_uuid"
    )
    raw_packages = document["packages"]
    if not isinstance(raw_packages, list):
        raise BenchmarkContractError("provenance.packages must be an array")
    packages = [
        _require_mapping(package, f"packages[{index}]")
        for index, package in enumerate(raw_packages)
    ]
    if tuple(package.get("package_id") for package in packages) != PACKAGE_ORDER:
        raise BenchmarkContractError("provenance package order must be 6A,6B,6C,6D")

    owners: list[str] = []
    sessions: list[str] = []
    authenticated_parent_sessions: list[str] = []
    job_ids: list[str] = []
    intervals: list[tuple[datetime, datetime]] = []
    for index, package in enumerate(packages):
        path = f"packages[{index}]"
        _require_exact_keys(
            package,
            {"package_id", "component", "owner", "queue", "stable_evidence"},
            path,
        )
        package_id = cast(str, package["package_id"])
        if package["component"] != EXPECTED_COMPONENTS[package_id]:
            raise BenchmarkContractError(f"{path}.component mismatch")
        owner = _require_mapping(package["owner"], f"{path}.owner")
        _require_exact_keys(
            owner,
            {"canonical_agent_task", "session_uuid", "session_meta"},
            f"{path}.owner",
        )
        owner_task = _require_nonempty_string(
            owner["canonical_agent_task"], f"{path}.owner.canonical_agent_task"
        )
        if owner_task != EXPECTED_OWNERS[package_id]:
            raise BenchmarkContractError(f"{path}.owner is not canonical")
        session_uuid = _require_uuid(
            owner["session_uuid"], f"{path}.owner.session_uuid"
        )
        session_meta = _require_mapping(
            owner["session_meta"], f"{path}.owner.session_meta"
        )
        authenticated_parent_sessions.append(
            _authenticate_session_snapshot(
                session_meta,
                package_id=package_id,
                session_uuid=session_uuid,
                owner_task=owner_task,
                project_root=project_root,
                path=f"{path}.owner.session_meta",
            )
        )
        owners.append(owner_task)
        sessions.append(session_uuid)

        queue = _require_mapping(package["queue"], f"{path}.queue")
        _require_exact_keys(
            queue,
            {
                "job_id",
                "name",
                "provider",
                "issue",
                "command",
                "cwd",
                "added_at",
                "started_at",
                "done_at",
                "terminal_outcome",
                "exit_code",
                "records",
            },
            f"{path}.queue",
        )
        job_id = _require_nonempty_string(queue["job_id"], f"{path}.queue.job_id")
        if not job_id.split("_", 1)[0].isdigit():
            raise BenchmarkContractError(f"{path}.queue.job_id is not sortable")
        for key in ("name", "command", "cwd"):
            _require_nonempty_string(queue[key], f"{path}.queue.{key}")
        if queue["provider"] != "codex" or queue["issue"] != str(ISSUE_NUMBER):
            raise BenchmarkContractError(f"{path}.queue attribution mismatch")
        if queue["terminal_outcome"] != "done" or queue["exit_code"] != 0:
            raise BenchmarkContractError(f"{path}.queue terminal outcome is not success")
        added = _timestamp(queue["added_at"], f"{path}.queue.added_at")
        started = _timestamp(queue["started_at"], f"{path}.queue.started_at")
        done = _timestamp(queue["done_at"], f"{path}.queue.done_at")
        if not added <= started <= done:
            raise BenchmarkContractError(f"{path}.queue timestamps are not monotonic")
        intervals.append((started, done))
        job_ids.append(job_id)

        records = _require_mapping(queue["records"], f"{path}.queue.records")
        _require_exact_keys(
            records, {"run_json", "done_job", "log"}, f"{path}.queue.records"
        )
        expected_origins = {
            "run_json": f".training_queue/repro/{job_id}/run.json",
            "done_job": f".training_queue/done/{job_id}.job",
            "log": f".training_queue/logs/{job_id}.log",
        }
        for record_name, expected_origin in expected_origins.items():
            record = _require_mapping(
                records[record_name], f"{path}.queue.records.{record_name}"
            )
            _require_exact_keys(
                record, {"origin", "sha256"}, f"{path}.queue.records.{record_name}"
            )
            if record["origin"] != expected_origin:
                raise BenchmarkContractError(
                    f"{path}.queue.records.{record_name}.origin mismatch"
                )
            _require_sha256(
                record["sha256"], f"{path}.queue.records.{record_name}.sha256"
            )

        stable = _require_mapping(package["stable_evidence"], f"{path}.stable_evidence")
        _require_exact_keys(
            stable, {"path", "sha256"}, f"{path}.stable_evidence"
        )
        expected_evidence_path = EXPECTED_EVIDENCE_PATHS[package_id]
        if stable["path"] != expected_evidence_path:
            raise BenchmarkContractError(f"{path}.stable_evidence.path mismatch")
        evidence_file = resolver.resolve(PathRole.PROJECT, expected_evidence_path)
        expected_evidence_sha = _sha256_file(evidence_file)
        if stable["sha256"] != expected_evidence_sha:
            raise BenchmarkContractError(f"{path}.stable_evidence.sha256 mismatch")

    authenticated_parents = set(authenticated_parent_sessions)
    if len(authenticated_parents) != 1:
        raise BenchmarkContractError(
            "owner session snapshots do not share one parent orchestrator"
        )
    authenticated_parent = next(iter(authenticated_parents))
    if authenticated_parent != EXPECTED_ORCHESTRATOR_SESSION_UUID:
        raise BenchmarkContractError(
            "authenticated owner session parent is not the expected orchestrator"
        )
    if orchestrator_session_uuid != authenticated_parent:
        raise BenchmarkContractError(
            "provenance.orchestrator_session_uuid does not match authenticated snapshots"
        )

    computed_serial_validation = {
        "package_order": list(PACKAGE_ORDER),
        "unique_owner_tasks": len(set(owners)) == len(PACKAGE_ORDER),
        "unique_child_sessions": (
            len(set(sessions)) == len(PACKAGE_ORDER)
            and orchestrator_session_uuid not in sessions
        ),
        "unique_queue_jobs": len(set(job_ids)) == len(PACKAGE_ORDER),
        "job_ids_strictly_increasing": all(
            first < second
            for first, second in zip(job_ids, job_ids[1:], strict=False)
        ),
        "no_time_overlap": all(
            first_done <= second_started
            for (_, first_done), (second_started, _) in zip(
                intervals, intervals[1:], strict=False
            )
        ),
    }
    computed_serial_validation["status"] = (
        "PASS"
        if all(
            value is True
            for key, value in computed_serial_validation.items()
            if key != "package_order"
        )
        else "FAIL"
    )
    serial_validation = _require_mapping(
        document["serial_validation"], "provenance.serial_validation"
    )
    if dict(serial_validation) != computed_serial_validation:
        raise BenchmarkContractError(
            "provenance serial_validation does not match ownership/order records"
        )
    if computed_serial_validation["status"] != "PASS":
        raise BenchmarkContractError("provenance serial execution validation failed")
    expected_record_bundle_sha256 = _record_bundle_sha256(packages)
    if document["record_bundle_sha256"] != expected_record_bundle_sha256:
        raise BenchmarkContractError(
            "provenance.record_bundle_sha256 does not match package records"
        )
