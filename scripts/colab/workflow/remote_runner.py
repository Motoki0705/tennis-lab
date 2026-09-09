"""Execute one validated tennis-lab request inside a Colab runtime.

This file is sent as source through ``colab exec``. It intentionally uses only
the Python standard library until the repository's locked environment exists.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlparse

SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RCLONE_REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
GPU_CHOICES = frozenset({"T4", "L4", "G4", "H100", "A100"})
ACCELERATORS = frozenset({"cpu", "gpu"})
_OVERRIDE_SEGMENT = r"[A-Za-z_][A-Za-z0-9_-]*"
OVERRIDE_KEY_RE = re.compile(
    rf"^(?:{_OVERRIDE_SEGMENT}(?:\.{_OVERRIDE_SEGMENT})*|"
    rf"{_OVERRIDE_SEGMENT}(?:/{_OVERRIDE_SEGMENT})+)$"
)
KNOWN_HOOKS = frozenset({"base", "submodules", "cuda_ops", "nht"})
REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "session",
        "created_at",
        "runtime",
        "source",
        "drive",
        "job",
        "request_digest",
    }
)
STATUS_FIELDS = frozenset(
    {
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
)
MAX_SNAPSHOT_MEMBERS = 200_000
MAX_SNAPSHOT_UNCOMPRESSED_BYTES = 20 * 1024**3
ARTIFACT_DISK_RESERVE_BYTES = 1024**3
SECRET_SNAPSHOT_NAMES = frozenset(
    {
        ".env",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "authorized_user.json",
        "client_secret.json",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "rclone.conf",
        "service-account.json",
    }
)
WORKSPACE_PARENT = Path("/content/tennis-lab-runs")


class RemoteWorkflowError(RuntimeError):
    """A reproducible remote workflow failure."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_id() -> str:
    value = globals().get("TENNIS_COLAB_RUN_ID", "")
    if not isinstance(value, str) or not RUN_ID_RE.fullmatch(value):
        raise RemoteWorkflowError("bootstrap run id is missing or invalid")
    return value


def _workspace() -> Path:
    return WORKSPACE_PARENT / _run_id()


def _strict_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RemoteWorkflowError(f"{label} must be a non-empty POSIX relative path")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise RemoteWorkflowError(f"{label} is not a strict relative path: {value!r}")
    return path.as_posix()


def _expect_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RemoteWorkflowError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _expect_fields(
    value: dict[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise RemoteWorkflowError(f"{label} has unknown fields: {sorted(unknown)}")
    if missing:
        raise RemoteWorkflowError(f"{label} is missing fields: {sorted(missing)}")


def _expect_string(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or "\x00" in value or (nonempty and not value):
        raise RemoteWorkflowError(f"{label} must be a valid string")
    return value


def _expect_string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RemoteWorkflowError(f"{label} must be an array of strings")
    result = cast(list[str], value)
    if nonempty and not result:
        raise RemoteWorkflowError(f"{label} must not be empty")
    if any(not item or "\x00" in item for item in result):
        raise RemoteWorkflowError(f"{label} contains an empty string or NUL byte")
    return result


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteWorkflowError(f"cannot read {label} {path}: {error}") from error
    return _expect_object(value, label)


def _validate_repository_url(value: Any) -> str:
    repository_url = _expect_string(value, "source.repository_url")
    parsed = urlparse(repository_url)
    try:
        port = parsed.port
    except ValueError as error:
        raise RemoteWorkflowError(
            "source.repository_url has an invalid port"
        ) from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RemoteWorkflowError(
            "source.repository_url must be a credential-free https://github.com URL"
        )
    components = [part for part in parsed.path.split("/") if part]
    if len(components) != 2 or any(
        part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in components
    ):
        raise RemoteWorkflowError(
            "source.repository_url must identify one GitHub owner/repository"
        )
    return repository_url


def _validate_mapping_paths(inputs: list[dict[str, Any]], outputs: list[str]) -> None:
    destinations = [mapping["destination"] for mapping in inputs]
    for index, first in enumerate(destinations):
        for second in destinations[index + 1 :]:
            if (
                first == second
                or first.startswith(f"{second}/")
                or second.startswith(f"{first}/")
            ):
                raise RemoteWorkflowError(
                    f"job input destinations overlap: {first!r}, {second!r}"
                )
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            if (
                first == second
                or first.startswith(f"{second}/")
                or second.startswith(f"{first}/")
            ):
                raise RemoteWorkflowError(f"job outputs overlap: {first!r}, {second!r}")
    for mapping in inputs:
        destination = mapping["destination"]
        overlaps = [
            output
            for output in outputs
            if (
                destination == output
                or destination.startswith(f"{output}/")
                or output.startswith(f"{destination}/")
            )
        ]
        if mapping["writable"] and destination not in outputs:
            raise RemoteWorkflowError(
                f"writable input must be an exact declared output: {destination!r}"
            )
        if not mapping["writable"] and overlaps:
            raise RemoteWorkflowError(
                f"read-only input/output paths overlap: {destination!r}, {overlaps[0]!r}"
            )


def _is_secret_snapshot_path(value: str) -> bool:
    parts = tuple(part.lower() for part in PurePosixPath(value).parts)
    lower_name = parts[-1]
    return (
        lower_name in SECRET_SNAPSHOT_NAMES
        or lower_name.startswith(".env.")
        or lower_name.endswith((".key", ".pem", ".p12", ".pfx"))
        or lower_name.startswith("client_secret")
        or ("credential" in lower_name and lower_name.endswith(".json"))
        or ("secret" in lower_name and lower_name.endswith(".json"))
        or ("token" in lower_name and lower_name.endswith(".json"))
        or any(part in {".git", ".secrets", "secrets"} for part in parts)
    )


def _validate_source(value: Any) -> dict[str, Any]:
    source = _expect_object(value, "source")
    mode = source.get("mode")
    common = frozenset(
        {
            "mode",
            "repository_url",
            "requested_ref",
            "repo_sha",
            "repo_tree",
            "tracked_clean",
            "working_tree_dirty",
        }
    )
    snapshot_fields = frozenset(
        {
            "archive_sha256",
            "content_digest",
            "included_paths",
            "deleted_paths",
            "excluded_secret_count",
            "submodules",
        }
    )
    if mode == "git":
        _expect_fields(source, common, common, "source")
    elif mode == "snapshot":
        _expect_fields(
            source, common | snapshot_fields, common | snapshot_fields, "source"
        )
    else:
        raise RemoteWorkflowError("source.mode must be 'git' or 'snapshot'")
    _validate_repository_url(source["repository_url"])
    _expect_string(source["requested_ref"], "source.requested_ref")
    if not isinstance(source["repo_sha"], str) or not GIT_SHA_RE.fullmatch(
        source["repo_sha"]
    ):
        raise RemoteWorkflowError("source.repo_sha must be a full lowercase Git SHA")
    if not isinstance(source["repo_tree"], str) or not GIT_SHA_RE.fullmatch(
        source["repo_tree"]
    ):
        raise RemoteWorkflowError("source.repo_tree must be a full lowercase Git SHA")
    if not isinstance(source["tracked_clean"], bool):
        raise RemoteWorkflowError("source.tracked_clean must be a boolean")
    if mode == "git" and not source["tracked_clean"]:
        raise RemoteWorkflowError("git source requires a clean tracked working tree")
    if not isinstance(source["working_tree_dirty"], bool):
        raise RemoteWorkflowError("source.working_tree_dirty must be a boolean")
    if mode == "snapshot":
        if source["requested_ref"] != "HEAD":
            raise RemoteWorkflowError("snapshot source.requested_ref must be HEAD")
        for field in ("archive_sha256", "content_digest"):
            if not isinstance(source[field], str) or not SHA256_RE.fullmatch(
                source[field]
            ):
                raise RemoteWorkflowError(f"source.{field} must be a lowercase SHA-256")
        included = _expect_string_list(
            source["included_paths"], "source.included_paths"
        )
        deleted = _expect_string_list(source["deleted_paths"], "source.deleted_paths")
        if len(included) > MAX_SNAPSHOT_MEMBERS or len(deleted) > MAX_SNAPSHOT_MEMBERS:
            raise RemoteWorkflowError("snapshot path lists exceed the safety limit")
        normalized: list[str] = []
        for label, paths in (("included_paths", included), ("deleted_paths", deleted)):
            if paths != sorted(paths):
                raise RemoteWorkflowError(f"source.{label} must be sorted")
            for index, path in enumerate(paths):
                relative = _strict_relative(path, f"source.{label}[{index}]")
                if _is_secret_snapshot_path(relative):
                    raise RemoteWorkflowError(
                        f"source.{label} contains a forbidden secret path"
                    )
                normalized.append(relative)
        if len(set(normalized)) != len(normalized):
            raise RemoteWorkflowError("snapshot included/deleted paths must not repeat")
        ordered_paths = sorted(normalized)
        if any(
            second.startswith(f"{first}/")
            for first, second in zip(ordered_paths, ordered_paths[1:], strict=False)
        ):
            raise RemoteWorkflowError(
                "snapshot file/deletion paths must not contain one another"
            )
        excluded_count = source["excluded_secret_count"]
        if (
            not isinstance(excluded_count, int)
            or isinstance(excluded_count, bool)
            or excluded_count < 0
        ):
            raise RemoteWorkflowError(
                "source.excluded_secret_count must be non-negative"
            )
        raw_submodules = source["submodules"]
        if not isinstance(raw_submodules, list):
            raise RemoteWorkflowError("source.submodules must be an array")
        submodule_paths: list[str] = []
        fields = frozenset({"path", "url", "commit"})
        for index, raw_submodule in enumerate(raw_submodules):
            submodule = _expect_object(raw_submodule, f"source.submodules[{index}]")
            _expect_fields(submodule, fields, fields, f"source.submodules[{index}]")
            path = _strict_relative(
                submodule["path"], f"source.submodules[{index}].path"
            )
            _validate_repository_url(submodule["url"])
            if not isinstance(submodule["commit"], str) or not GIT_SHA_RE.fullmatch(
                submodule["commit"]
            ):
                raise RemoteWorkflowError(
                    f"source.submodules[{index}].commit is invalid"
                )
            submodule_paths.append(path)
        if len(set(submodule_paths)) != len(submodule_paths):
            raise RemoteWorkflowError("source.submodule paths must not repeat")
        if submodule_paths != sorted(submodule_paths):
            raise RemoteWorkflowError("source.submodules must be sorted by path")
        for path in normalized:
            if any(
                path == submodule or path.startswith(f"{submodule}/")
                for submodule in submodule_paths
            ):
                raise RemoteWorkflowError("snapshot contains submodule content")
    return source


def _validate_drive(value: Any) -> dict[str, Any]:
    drive = _expect_object(value, "drive")
    fields = frozenset({"mode", "root", "remote"})
    _expect_fields(drive, fields, fields, "drive")
    mode = drive["mode"]
    if mode not in {"mount", "rclone"}:
        raise RemoteWorkflowError("drive.mode must be 'mount' or 'rclone'")
    _strict_relative(drive["root"], "drive.root")
    if mode == "mount":
        if drive["remote"] is not None:
            raise RemoteWorkflowError("drive.remote must be null in mount mode")
    elif not isinstance(drive["remote"], str) or not RCLONE_REMOTE_RE.fullmatch(
        drive["remote"]
    ):
        raise RemoteWorkflowError("drive.remote is invalid for rclone mode")
    return drive


def _validate_job(value: Any) -> dict[str, Any]:
    job = _expect_object(value, "job")
    fields = frozenset(
        {
            "name",
            "definition_digest",
            "accelerator",
            "setup",
            "protected_override_keys",
            "argv",
            "timeout_seconds",
            "inputs",
            "outputs",
        }
    )
    _expect_fields(job, fields, fields, "job")
    if not isinstance(job["name"], str) or not NAME_RE.fullmatch(job["name"]):
        raise RemoteWorkflowError("job.name is invalid")
    if not isinstance(job["definition_digest"], str) or not SHA256_RE.fullmatch(
        job["definition_digest"]
    ):
        raise RemoteWorkflowError("job.definition_digest must be a lowercase SHA-256")
    if job["accelerator"] not in ACCELERATORS:
        raise RemoteWorkflowError("job.accelerator is invalid")
    protected = _expect_string_list(
        job["protected_override_keys"], "job.protected_override_keys"
    )
    if len(set(protected)) != len(protected) or any(
        not OVERRIDE_KEY_RE.fullmatch(key) for key in protected
    ):
        raise RemoteWorkflowError("job.protected_override_keys is invalid")
    setup = _expect_string_list(job["setup"], "job.setup", nonempty=True)
    if setup[0] != "base" or len(set(setup)) != len(setup):
        raise RemoteWorkflowError("job.setup must begin with unique base hook")
    if set(setup) - KNOWN_HOOKS:
        raise RemoteWorkflowError("job.setup contains an unknown hook")
    for dependent in ("cuda_ops", "nht"):
        if dependent in setup and (
            "submodules" not in setup
            or setup.index("submodules") > setup.index(dependent)
        ):
            raise RemoteWorkflowError(
                f"job.setup must run submodules before {dependent}"
            )
    argv = _expect_string_list(job["argv"], "job.argv", nonempty=True)
    if any("/content/drive" in item for item in argv):
        raise RemoteWorkflowError("job.argv must not use the Drive FUSE mount")
    if any(
        PurePosixPath(item).name in {"bash", "dash", "sh", "zsh"}
        and index + 1 < len(argv)
        and argv[index + 1] == "-c"
        for index, item in enumerate(argv)
    ):
        raise RemoteWorkflowError("job.argv must not evaluate a shell command string")
    timeout = job["timeout_seconds"]
    if (
        not isinstance(timeout, int)
        or isinstance(timeout, bool)
        or not 60 <= timeout <= 604800
    ):
        raise RemoteWorkflowError("job.timeout_seconds must be between 60 and 604800")
    raw_inputs = job["inputs"]
    if not isinstance(raw_inputs, list):
        raise RemoteWorkflowError("job.inputs must be an array")
    inputs: list[dict[str, Any]] = []
    mapping_fields = frozenset({"source", "destination", "writable"})
    for index, raw_mapping in enumerate(raw_inputs):
        mapping = _expect_object(raw_mapping, f"job.inputs[{index}]")
        _expect_fields(mapping, mapping_fields, mapping_fields, f"job.inputs[{index}]")
        writable = mapping["writable"]
        if not isinstance(writable, bool):
            raise RemoteWorkflowError(f"job.inputs[{index}].writable must be a boolean")
        inputs.append(
            {
                "source": _strict_relative(
                    mapping["source"], f"job.inputs[{index}].source"
                ),
                "destination": _strict_relative(
                    mapping["destination"], f"job.inputs[{index}].destination"
                ),
                "writable": writable,
            }
        )
    raw_outputs = _expect_string_list(job["outputs"], "job.outputs", nonempty=True)
    outputs = [
        _strict_relative(output, f"job.outputs[{index}]")
        for index, output in enumerate(raw_outputs)
    ]
    _validate_mapping_paths(inputs, outputs)
    return job


def _child(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    candidate = root / _strict_relative(relative, label)
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise RemoteWorkflowError(f"{label} escaped {root}: {relative}")
    if candidate.exists() or candidate.is_symlink():
        resolved_candidate = candidate.resolve()
        if resolved_candidate == root or root not in resolved_candidate.parents:
            raise RemoteWorkflowError(f"{label} escaped {root}: {relative}")
    return candidate


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"[tennis-colab] exec: {argv[0]} ({len(argv) - 1} args)", flush=True)
    result = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        timeout=timeout,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        detail = ""
        if capture:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
        suffix = f": {detail}" if detail else ""
        raise RemoteWorkflowError(
            f"command failed with exit code {result.returncode}: {argv[0]}{suffix}"
        )
    return result


def _prepare_action() -> None:
    workspace = _workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".secrets").mkdir(mode=0o700, exist_ok=True)
    print(f"[tennis-colab] prepared {workspace}", flush=True)


def _cleanup_secret() -> None:
    configured = globals().get("TENNIS_COLAB_RCLONE_CONFIG")
    if not configured:
        return
    if not isinstance(configured, str):
        raise RemoteWorkflowError("temporary rclone configuration path is invalid")
    path = Path(configured)
    workspace = _workspace().resolve()
    resolved = path.resolve()
    if workspace not in resolved.parents or resolved.name != "rclone.conf":
        raise RemoteWorkflowError("refusing to remove an unexpected secret path")
    path.unlink(missing_ok=True)
    print("[tennis-colab] removed temporary rclone configuration", flush=True)


def _safe_extract(archive_path: Path, destination: Path) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members: list[tarfile.TarInfo] = []
        member_types: dict[str, str] = {}
        total_size = 0
        for member in archive:
            if len(members) >= MAX_SNAPSHOT_MEMBERS:
                raise RemoteWorkflowError(
                    f"snapshot exceeds {MAX_SNAPSHOT_MEMBERS} archive members"
                )
            relative = _strict_relative(member.name, "snapshot archive member")
            if relative in member_types:
                raise RemoteWorkflowError(f"snapshot has duplicate member: {relative}")
            if member.isdir():
                member_type = "directory"
            elif member.isfile():
                member_type = "file"
                if member.size < 0:
                    raise RemoteWorkflowError(
                        f"snapshot member has invalid size: {relative}"
                    )
                total_size += member.size
                if total_size > MAX_SNAPSHOT_UNCOMPRESSED_BYTES:
                    raise RemoteWorkflowError(
                        "snapshot uncompressed size exceeds the 20 GiB safety limit"
                    )
            elif member.issym():
                member_type = "symlink"
            else:
                raise RemoteWorkflowError(
                    f"snapshot contains unsupported member type: {member.name}"
                )
            member_types[relative] = member_type
            members.append(member)
        for relative in member_types:
            parts = PurePosixPath(relative).parts
            for offset in range(1, len(parts)):
                parent = PurePosixPath(*parts[:offset]).as_posix()
                if parent in member_types and member_types[parent] != "directory":
                    raise RemoteWorkflowError(
                        f"snapshot member type collision: {parent!r} contains {relative!r}"
                    )
        for member in members:
            relative = _strict_relative(member.name, "snapshot archive member")
            target = _child(root, relative, "snapshot archive member")
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise RemoteWorkflowError(
                        f"snapshot directory collides with existing file: {relative}"
                    )
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute():
                    raise RemoteWorkflowError(
                        f"snapshot contains absolute symlink: {member.name}"
                    )
                resolved_link = (target.parent / member.linkname).resolve()
                if resolved_link == root or root not in resolved_link.parents:
                    raise RemoteWorkflowError(
                        f"snapshot symlink escapes repository: {member.name}"
                    )
                target.symlink_to(member.linkname)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RemoteWorkflowError(f"cannot read snapshot member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
    return tuple(
        sorted(
            name
            for name, member_type in member_types.items()
            if member_type != "directory"
        )
    )


def _clone_exact(repository_url: str, revision: str, repo: Path) -> str:
    _run(["git", "init", str(repo)], timeout=120)
    _run(["git", "-C", str(repo), "remote", "add", "origin", repository_url])
    _run(
        ["git", "-C", str(repo), "fetch", "--depth=1", "origin", revision],
        timeout=900,
    )
    _run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], timeout=120)
    head = _run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    if head != revision:
        raise RemoteWorkflowError(
            f"checked out HEAD {head} does not equal requested revision {revision}"
        )
    return head


def _submodule_paths(repo: Path) -> set[str]:
    result = _run(["git", "-C", str(repo), "ls-files", "--stage"], capture=True).stdout
    paths: set[str] = set()
    for line in result.splitlines():
        try:
            metadata, relative = line.split("\t", 1)
        except ValueError as error:
            raise RemoteWorkflowError(
                "git returned malformed index metadata"
            ) from error
        if metadata.split(" ", 1)[0] == "160000":
            paths.add(_strict_relative(relative, "submodule path"))
    return paths


def _snapshot_content_digest(repo: Path, source: dict[str, Any]) -> str:
    included = set(cast(list[str], source["included_paths"]))
    deleted = set(cast(list[str], source["deleted_paths"]))
    metadata: list[dict[str, object]] = []
    for relative in sorted(included | deleted):
        if relative in deleted:
            deleted_path = repo / relative
            if deleted_path.exists() or deleted_path.is_symlink():
                raise RemoteWorkflowError(
                    f"snapshot deleted path unexpectedly exists: {relative}"
                )
            metadata.append({"path": relative, "kind": "deleted"})
            continue
        absolute = repo / relative
        try:
            file_stat = absolute.lstat()
        except FileNotFoundError as error:
            raise RemoteWorkflowError(
                f"snapshot member is absent after extraction: {relative}"
            ) from error
        if stat.S_ISLNK(file_stat.st_mode):
            metadata.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "mode": stat.S_IMODE(file_stat.st_mode),
                    "target": os.readlink(absolute),
                }
            )
        elif stat.S_ISREG(file_stat.st_mode):
            metadata.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(file_stat.st_mode),
                    "size": file_stat.st_size,
                    "sha256": _sha256(absolute),
                }
            )
        else:
            raise RemoteWorkflowError(
                f"snapshot member has an unsupported extracted type: {relative}"
            )
    return hashlib.sha256(_canonical_json({"entries": metadata})).hexdigest()


def _validate_source_provenance(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    provenance = _expect_object(value, "source marker provenance")
    source = request["source"]
    fields = {"mode", "repo_sha", "repo_tree", "tracked_clean"}
    if source["mode"] == "snapshot":
        fields.update(
            {
                "snapshot_digest",
                "snapshot_archive_sha256",
                "snapshot_git_commit",
                "submodules",
            }
        )
    expected_fields = frozenset(fields)
    _expect_fields(
        provenance, expected_fields, expected_fields, "source marker provenance"
    )
    if (
        provenance["mode"] != source["mode"]
        or provenance["repo_sha"] != source["repo_sha"]
        or provenance["repo_tree"] != source["repo_tree"]
        or provenance["tracked_clean"] != source["tracked_clean"]
    ):
        raise RemoteWorkflowError("source marker provenance does not match the request")
    if source["mode"] == "snapshot" and (
        provenance["snapshot_digest"] != source["content_digest"]
        or provenance["snapshot_archive_sha256"] != source["archive_sha256"]
        or provenance["submodules"] != source["submodules"]
    ):
        raise RemoteWorkflowError(
            "snapshot marker provenance does not match the request"
        )
    if source["mode"] == "snapshot" and (
        not isinstance(provenance["snapshot_git_commit"], str)
        or not GIT_SHA_RE.fullmatch(provenance["snapshot_git_commit"])
    ):
        raise RemoteWorkflowError("snapshot marker git commit is invalid")
    return provenance


def _verify_git_source(repo: Path, source: dict[str, Any]) -> None:
    head = _run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True
    ).stdout.strip()
    tree = _run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], capture=True
    ).stdout.strip()
    tracked_status = _run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        capture=True,
    ).stdout.strip()
    if head != source["repo_sha"] or tree != source["repo_tree"]:
        raise RemoteWorkflowError("existing git source commit/tree is stale or corrupt")
    if tracked_status or not source["tracked_clean"]:
        raise RemoteWorkflowError("existing git source violates tracked-clean contract")


def _initialize_snapshot_repository(repo: Path) -> None:
    _run(["git", "init", str(repo)], timeout=120, capture=True)
    _run(["git", "-C", str(repo), "add", "-A"], timeout=300, capture=True)
    _run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=tennis-lab-colab",
            "-c",
            "user.email=colab@localhost",
            "commit",
            "-m",
            "Colab source snapshot",
        ],
        timeout=300,
        capture=True,
    )


def _prepare_source(request: dict[str, Any], workspace: Path) -> dict[str, Any]:
    source = request["source"]
    marker_path = workspace / "source.json"
    repo = workspace / "repo"
    if marker_path.is_file():
        marker = _read_json_object(marker_path, "source marker")
        marker_fields = frozenset({"request_digest", "provenance"})
        _expect_fields(marker, marker_fields, marker_fields, "source marker")
        if marker["request_digest"] != request["request_digest"]:
            raise RemoteWorkflowError("existing source belongs to a different request")
        if not repo.is_dir():
            raise RemoteWorkflowError("source marker exists but repository is missing")
        existing_provenance = _validate_source_provenance(marker["provenance"], request)
        if source["mode"] == "git":
            _verify_git_source(repo, source)
        else:
            actual_digest = _snapshot_content_digest(repo, source)
            if actual_digest != source["content_digest"]:
                raise RemoteWorkflowError(
                    "existing snapshot content changed since the first attempt"
                )
            snapshot_head = _run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True
            ).stdout.strip()
            if snapshot_head != existing_provenance["snapshot_git_commit"]:
                raise RemoteWorkflowError(
                    "existing snapshot git metadata is stale or corrupt"
                )
        return existing_provenance
    if repo.exists():
        shutil.rmtree(repo)
    repository_url = source["repository_url"]
    revision = source["repo_sha"]
    if not isinstance(repository_url, str) or not repository_url:
        raise RemoteWorkflowError("source.repository_url is invalid")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RemoteWorkflowError("source.repo_sha is invalid")
    provenance: dict[str, Any] = {
        "mode": source["mode"],
        "repo_sha": source["repo_sha"],
        "repo_tree": source["repo_tree"],
        "tracked_clean": source["tracked_clean"],
    }
    if source["mode"] == "snapshot":
        archive = workspace / "source-snapshot.tar.gz"
        expected = source["archive_sha256"]
        if not archive.is_file() or _sha256(archive) != expected:
            raise RemoteWorkflowError("uploaded snapshot archive digest mismatch")
        extracted = _safe_extract(archive, repo)
        if set(extracted) != set(source["included_paths"]):
            raise RemoteWorkflowError(
                "snapshot archive members do not match source.included_paths"
            )
        actual_content_digest = _snapshot_content_digest(repo, source)
        if actual_content_digest != source["content_digest"]:
            raise RemoteWorkflowError(
                "snapshot content digest mismatch after extraction"
            )
        provenance["snapshot_digest"] = source["content_digest"]
        provenance["snapshot_archive_sha256"] = expected
        provenance["submodules"] = source["submodules"]
        _initialize_snapshot_repository(repo)
        provenance["snapshot_git_commit"] = _run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True
        ).stdout.strip()
    elif source["mode"] == "git":
        _clone_exact(repository_url, revision, repo)
        _verify_git_source(repo, source)
    else:
        raise RemoteWorkflowError(f"unsupported source mode: {source['mode']!r}")
    marker = {"request_digest": request["request_digest"], "provenance": provenance}
    _atomic_json(marker_path, marker)
    return provenance


def _hook_base(repo: Path) -> None:
    packages = ["build-essential", "cmake", "ffmpeg", "git", "ninja-build"]
    _run(["apt-get", "update"], timeout=900)
    environment = os.environ.copy()
    environment["DEBIAN_FRONTEND"] = "noninteractive"
    _run(["apt-get", "install", "-y", *packages], timeout=1800, env=environment)
    if shutil.which("uv") is None:
        _run([sys.executable, "-m", "pip", "install", "uv"], timeout=900)
    _run(["uv", "sync", "--locked"], cwd=repo, timeout=3600)


def _hook_submodules(request: dict[str, Any], repo: Path) -> None:
    if request["source"]["mode"] == "snapshot":
        for index, submodule in enumerate(request["source"]["submodules"]):
            path = _child(repo, submodule["path"], f"source.submodules[{index}].path")
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _clone_exact(submodule["url"], submodule["commit"], path)
                except (RemoteWorkflowError, subprocess.TimeoutExpired) as error:
                    if path.exists():
                        shutil.rmtree(path)
                    raise RemoteWorkflowError(
                        f"cannot materialize snapshot submodule {submodule['path']} at "
                        f"recorded gitlink {submodule['commit']}; snapshot archives do "
                        f"not contain submodule working trees, so the commit must be "
                        f"pushed and fetchable from {submodule['url']}: {error}"
                    ) from error
            if not path.is_dir() or not (path / ".git").exists():
                raise RemoteWorkflowError(
                    f"snapshot submodule is not a git checkout: {submodule['path']}"
                )
            origin = _run(
                ["git", "-C", str(path), "remote", "get-url", "origin"], capture=True
            ).stdout.strip()
            head = _run(
                ["git", "-C", str(path), "rev-parse", "HEAD"], capture=True
            ).stdout.strip()
            tree = _run(
                ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"], capture=True
            ).stdout.strip()
            tracked = _run(
                [
                    "git",
                    "-C",
                    str(path),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ],
                capture=True,
            ).stdout.strip()
            if origin != submodule["url"] or head != submodule["commit"]:
                raise RemoteWorkflowError(
                    f"snapshot submodule provenance mismatch: {submodule['path']}"
                )
            if not GIT_SHA_RE.fullmatch(tree) or tracked:
                raise RemoteWorkflowError(
                    f"snapshot submodule content is invalid: {submodule['path']}"
                )
            _run(
                [
                    "git",
                    "-C",
                    str(path),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--checkout",
                ],
                timeout=1800,
            )
            recursive_status = _run(
                ["git", "-C", str(path), "submodule", "status", "--recursive"],
                capture=True,
            ).stdout.splitlines()
            if any(line and line[0] in {"-", "+", "U"} for line in recursive_status):
                raise RemoteWorkflowError(
                    f"snapshot nested submodule content is invalid: {submodule['path']}"
                )
        return
    _run(
        [
            "git",
            "-C",
            str(repo),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--checkout",
        ],
        timeout=1800,
    )


def _hook_cuda_ops(repo: Path) -> None:
    setup = repo / "scripts/colab/setup/install_cuda_ops.sh"
    if not setup.is_file():
        raise RemoteWorkflowError(f"CUDA setup module is missing: {setup}")
    script = 'source "$1"; install_colab_cuda_ops "$2"'
    _run(
        ["bash", "-c", script, "tennis-colab-cuda-hook", str(setup), str(repo)],
        timeout=3600,
    )


def _hook_nht(repo: Path) -> None:
    _run(
        [
            str(repo / ".venv/bin/python"),
            "-m",
            "spin",
            "setup-nht",
            "--with-sfm-learned",
        ],
        cwd=repo,
        timeout=3600,
    )


HOOKS = {
    "base": _hook_base,
    "cuda_ops": _hook_cuda_ops,
    "nht": _hook_nht,
}


def _run_hook(request: dict[str, Any], repo: Path, hook_name: str) -> None:
    print(f"[tennis-colab] setup hook: {hook_name}", flush=True)
    if hook_name == "submodules":
        _hook_submodules(request, repo)
        return
    hook = HOOKS.get(hook_name)
    if hook is None:
        raise RemoteWorkflowError(f"unknown setup hook: {hook_name!r}")
    hook(repo)


def _ensure_rclone(config_path: Path) -> None:
    config_path.chmod(0o600)
    if shutil.which("rclone") is None:
        _run(["apt-get", "update"], timeout=900)
        environment = os.environ.copy()
        environment["DEBIAN_FRONTEND"] = "noninteractive"
        _run(["apt-get", "install", "-y", "rclone"], timeout=1200, env=environment)


def _observe_command(argv: list[str], *, timeout: int = 120) -> dict[str, Any]:
    executable = argv[0]
    available = (
        Path(executable).is_file()
        if "/" in executable
        else shutil.which(executable) is not None
    )
    if not available:
        return {
            "available": False,
            "returncode": None,
            "stdout": None,
            "stderr": f"executable not found: {executable}",
        }
    try:
        result = subprocess.run(
            argv,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": True,
            "returncode": None,
            "stdout": None,
            "stderr": f"{type(error).__name__}: {error}",
        }
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _runtime_provenance(request: dict[str, Any], repo: Path) -> dict[str, Any]:
    training_python = _observe_command([str(repo / ".venv/bin/python"), "--version"])
    nvidia = _observe_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    uv = _observe_command(["uv", "--version"])
    torch_script = (
        "import json, torch; "
        "print(json.dumps({'version': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda_version': torch.version.cuda, "
        "'device_count': torch.cuda.device_count()}))"
    )
    torch_observation = _observe_command(
        [str(repo / ".venv/bin/python"), "-c", torch_script]
    )
    torch_details: dict[str, Any] | None = None
    torch_error: str | None = None
    if torch_observation["returncode"] == 0 and isinstance(
        torch_observation["stdout"], str
    ):
        try:
            raw_torch = json.loads(torch_observation["stdout"])
        except json.JSONDecodeError as error:
            torch_error = f"invalid torch probe JSON: {error}"
        else:
            if (
                isinstance(raw_torch, dict)
                and isinstance(raw_torch.get("version"), str)
                and isinstance(raw_torch.get("cuda_available"), bool)
                and (
                    raw_torch.get("cuda_version") is None
                    or isinstance(raw_torch.get("cuda_version"), str)
                )
                and isinstance(raw_torch.get("device_count"), int)
                and not isinstance(raw_torch.get("device_count"), bool)
            ):
                torch_details = raw_torch
            else:
                torch_error = "torch probe returned invalid fields"
    else:
        torch_error = str(torch_observation["stderr"] or "torch probe failed")
    nvidia_available = nvidia["returncode"] == 0 and bool(nvidia["stdout"])
    torch_available = torch_details is not None
    cuda_available = bool(
        nvidia_available
        and torch_details is not None
        and torch_details["cuda_available"]
        and torch_details["device_count"] > 0
        and torch_details["cuda_version"] is not None
    )
    runtime = {
        "collected": True,
        "python": {
            "available": training_python["returncode"] == 0,
            "version": (
                training_python["stdout"] or training_python["stderr"]
                if training_python["returncode"] == 0
                else None
            ),
            "implementation": platform.python_implementation(),
            "executable": str(repo / ".venv/bin/python"),
            "bootstrap_version": sys.version,
            "probe": training_python,
        },
        "platform": {
            "available": True,
            "os_name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "gpu": {
            "required": request["job"]["accelerator"] == "gpu",
            "requested": request["runtime"]["gpu"],
            "available": nvidia_available,
            "nvidia_smi": nvidia,
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_available": (
                torch_details["cuda_available"] if torch_details is not None else None
            ),
            "torch_cuda_version": (
                torch_details["cuda_version"] if torch_details is not None else None
            ),
            "device_count": (
                torch_details["device_count"] if torch_details is not None else None
            ),
        },
        "torch": {
            "available": torch_available,
            "version": (
                torch_details["version"] if torch_details is not None else None
            ),
            "error": torch_error,
            "probe": torch_observation,
        },
        "uv": {
            "available": uv["returncode"] == 0,
            "version": uv["stdout"] if uv["returncode"] == 0 else None,
            "error": None if uv["returncode"] == 0 else uv["stderr"],
            "probe": uv,
        },
    }
    return runtime


def _manifest_entries(path: Path, relative_root: Path) -> list[dict[str, Any]]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*"))
    entries: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise RemoteWorkflowError(
                f"symlinks are forbidden in staged data: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RemoteWorkflowError(f"unsupported staged data entry: {candidate}")
        entries.append(
            {
                "path": candidate.relative_to(relative_root).as_posix(),
                "size": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    return entries


def _copy_mount_input(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise RemoteWorkflowError(f"Drive input must not be a symlink: {source}")
    if source.is_dir():
        for candidate in source.rglob("*"):
            if candidate.is_symlink():
                raise RemoteWorkflowError(
                    f"Drive input tree must not contain symlinks: {candidate}"
                )
        shutil.copytree(source, destination, symlinks=False)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise RemoteWorkflowError(f"Drive input does not exist: {source}")


def _rclone_path(request: dict[str, Any], relative: str) -> str:
    drive = request["drive"]
    root = _strict_relative(drive["root"], "drive.root")
    child = _strict_relative(relative, "Drive relative path")
    path = PurePosixPath(root, child).as_posix()
    return f"{drive['remote']}:{path}"


def _mount_drive_root(request: dict[str, Any]) -> Path:
    return _child(
        Path("/content/drive/MyDrive"), request["drive"]["root"], "drive.root"
    )


def _copy_rclone_input(
    request: dict[str, Any], source_relative: str, destination: Path, config: Path
) -> None:
    remote_source = _rclone_path(request, source_relative)
    result = _run(
        ["rclone", "--config", str(config), "lsjson", "--stat", remote_source],
        timeout=300,
        capture=True,
    )
    try:
        description = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RemoteWorkflowError(
            f"rclone returned invalid metadata: {error}"
        ) from error
    if (
        not isinstance(description, dict)
        or "IsDir" not in description
        or not isinstance(description["IsDir"], bool)
    ):
        raise RemoteWorkflowError(
            f"rclone source metadata has no boolean IsDir: {remote_source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if description["IsDir"]:
        destination.mkdir()
        _run(
            [
                "rclone",
                "--config",
                str(config),
                "copy",
                remote_source,
                str(destination),
            ],
            timeout=3600,
        )
    else:
        _run(
            [
                "rclone",
                "--config",
                str(config),
                "copyto",
                remote_source,
                str(destination),
            ],
            timeout=3600,
        )


def _stage_inputs(
    request: dict[str, Any],
    repo: Path,
    rclone_config: Path | None,
    *,
    reset_existing: bool,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    drive = request["drive"]
    for index, mapping in enumerate(request["job"]["inputs"]):
        source_relative = _strict_relative(mapping["source"], f"inputs[{index}].source")
        destination_relative = _strict_relative(
            mapping["destination"], f"inputs[{index}].destination"
        )
        destination = _child(repo, destination_relative, f"inputs[{index}].destination")
        if destination.exists() or destination.is_symlink():
            if not reset_existing:
                raise RemoteWorkflowError(
                    f"input destination already exists: {destination}"
                )
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if drive["mode"] == "mount":
            drive_root = _mount_drive_root(request)
            source = _child(drive_root, source_relative, f"inputs[{index}].source")
            _copy_mount_input(source, destination)
        elif drive["mode"] == "rclone" and rclone_config is not None:
            _copy_rclone_input(request, source_relative, destination, rclone_config)
        else:
            raise RemoteWorkflowError(f"unsupported Drive mode: {drive['mode']!r}")
        entries = _manifest_entries(destination, repo)
        manifest.append(
            {
                "source": source_relative,
                "destination": destination_relative,
                "writable": mapping["writable"],
                "entries": entries,
            }
        )
    return manifest


def _collect_outputs(
    request: dict[str, Any], repo: Path, bundle: Path
) -> tuple[list[dict[str, Any]], Path]:
    output_paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for index, relative_value in enumerate(request["job"]["outputs"]):
        relative = _strict_relative(relative_value, f"outputs[{index}]")
        path = _child(repo, relative, f"outputs[{index}]")
        if not path.exists() or path.is_symlink():
            raise RemoteWorkflowError(
                f"declared output is missing or a symlink: {relative}"
            )
        output_paths.append(path)
        manifest.append(
            {"declared_path": relative, "entries": _manifest_entries(path, repo)}
        )
    output_size = sum(
        entry["size"] for declared in manifest for entry in declared["entries"]
    )
    available = shutil.disk_usage(bundle.parent).free
    required = output_size + ARTIFACT_DISK_RESERVE_BYTES
    if available < required:
        raise RemoteWorkflowError(
            "insufficient VM disk space for artifact archive: "
            f"need at least {required} bytes free, found {available}"
        )
    bundle.mkdir(parents=True, exist_ok=True)
    archive_path = bundle / "artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in output_paths:
            archive.add(path, arcname=path.relative_to(repo).as_posix(), recursive=True)
    return manifest, archive_path


def _bundle_manifest(bundle: Path) -> list[dict[str, Any]]:
    return _manifest_entries(bundle, bundle)


def _verify_tree(source: Path, destination: Path) -> None:
    if _manifest_entries(source, source) != _manifest_entries(destination, destination):
        raise RemoteWorkflowError("Drive staging verification failed")


def _publish_mount(request: dict[str, Any], bundle: Path, attempt: int) -> str:
    parent = _mount_drive_root(request) / "colab-runs"
    final = parent / request["run_id"]
    staging = parent / f".{request['run_id']}.uploading-{attempt}-{os.getpid()}"
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise RemoteWorkflowError(
            f"Drive run already exists; refusing overwrite: {final}"
        )
    if staging.exists():
        raise RemoteWorkflowError(f"Drive staging path already exists: {staging}")
    shutil.copytree(bundle, staging)
    _verify_tree(bundle, staging)
    if final.exists():
        raise RemoteWorkflowError(f"Drive run appeared during upload: {final}")
    os.rename(staging, final)
    return str(final)


def _rclone_exists(remote_path: str, config: Path) -> bool:
    result = subprocess.run(
        ["rclone", "--config", str(config), "lsjson", "--stat", remote_path],
        check=False,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 3:
        return False
    raise RemoteWorkflowError(
        f"rclone could not determine whether destination exists (exit {result.returncode})"
    )


def _publish_rclone(
    request: dict[str, Any], bundle: Path, attempt: int, config: Path
) -> str:
    drive = request["drive"]
    parent = PurePosixPath(_strict_relative(drive["root"], "drive.root"), "colab-runs")
    final = f"{drive['remote']}:{parent / request['run_id']}"
    staging = f"{drive['remote']}:{parent / ('.' + request['run_id'] + '.uploading-' + str(attempt) + '-' + str(os.getpid()))}"
    if _rclone_exists(final, config):
        raise RemoteWorkflowError(
            f"Drive run already exists; refusing overwrite: {drive['root']}/colab-runs/{request['run_id']}"
        )
    if _rclone_exists(staging, config):
        raise RemoteWorkflowError("rclone staging path unexpectedly exists")
    _run(
        ["rclone", "--config", str(config), "copy", str(bundle), staging],
        timeout=3600,
    )
    _run(
        ["rclone", "--config", str(config), "check", "--one-way", str(bundle), staging],
        timeout=3600,
    )
    if _rclone_exists(final, config):
        raise RemoteWorkflowError(
            "Drive run appeared during upload; refusing overwrite"
        )
    _run(
        ["rclone", "--config", str(config), "moveto", staging, final],
        timeout=900,
    )
    return f"{drive['root']}/colab-runs/{request['run_id']}"


def _publish(
    request: dict[str, Any], bundle: Path, attempt: int, config: Path | None
) -> str:
    if request["drive"]["mode"] == "mount":
        return _publish_mount(request, bundle, attempt)
    if config is None:
        raise RemoteWorkflowError("rclone configuration is unavailable for publish")
    return _publish_rclone(request, bundle, attempt, config)


def _initial_status(request: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": request["run_id"],
        "session": request["session"],
        "request_digest": request["request_digest"],
        "state": "running",
        "step": "starting",
        "attempt": attempt,
        "updated_at": _utc_now(),
        "error": None,
        "provenance": {
            "job_definition_digest": request["job"]["definition_digest"],
            "source": {},
            "resolved_argv": request["job"]["argv"],
            "input_manifest": [],
            "output_manifest": [],
            "runtime": {
                "collected": False,
                "reason": "runtime inspection has not completed",
            },
        },
        "artifacts": {},
        "drive": {
            "mode": request["drive"]["mode"],
            "final": f"{request['drive']['root']}/colab-runs/{request['run_id']}",
        },
    }


def _write_status(path: Path, status: dict[str, Any], step: str) -> None:
    status["step"] = step
    status["updated_at"] = _utc_now()
    _atomic_json(path, status)


def _validate_manifest_entries(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RemoteWorkflowError(f"{label} must be an array")
    entries: list[dict[str, Any]] = []
    fields = frozenset({"path", "size", "sha256"})
    for index, raw_entry in enumerate(value):
        entry = _expect_object(raw_entry, f"{label}[{index}]")
        _expect_fields(entry, fields, fields, f"{label}[{index}]")
        _strict_relative(entry["path"], f"{label}[{index}].path")
        if (
            not isinstance(entry["size"], int)
            or isinstance(entry["size"], bool)
            or entry["size"] < 0
        ):
            raise RemoteWorkflowError(f"{label}[{index}].size is invalid")
        if not isinstance(entry["sha256"], str) or not SHA256_RE.fullmatch(
            entry["sha256"]
        ):
            raise RemoteWorkflowError(f"{label}[{index}].sha256 is invalid")
        entries.append(entry)
    paths = [entry["path"] for entry in entries]
    if len(set(paths)) != len(paths):
        raise RemoteWorkflowError(f"{label} contains duplicate paths")
    return entries


def _validate_input_manifest(value: Any, request: dict[str, Any]) -> None:
    if not isinstance(value, list):
        raise RemoteWorkflowError("status input_manifest must be an array")
    fields = frozenset({"source", "destination", "writable", "entries"})
    if len(value) > len(request["job"]["inputs"]):
        raise RemoteWorkflowError("status input_manifest has too many mappings")
    for index, raw_mapping in enumerate(value):
        mapping = _expect_object(raw_mapping, f"status input_manifest[{index}]")
        _expect_fields(mapping, fields, fields, f"status input_manifest[{index}]")
        expected = request["job"]["inputs"][index]
        if (
            mapping["source"] != expected["source"]
            or mapping["destination"] != expected["destination"]
            or mapping["writable"] != expected["writable"]
        ):
            raise RemoteWorkflowError("status input_manifest is stale or reordered")
        _validate_manifest_entries(
            mapping["entries"], f"status input_manifest[{index}].entries"
        )


def _validate_runtime_provenance(value: Any, request: dict[str, Any]) -> None:
    runtime = _expect_object(value, "status.provenance.runtime")
    if runtime.get("collected") is False:
        fields = frozenset({"collected", "reason"})
        _expect_fields(runtime, fields, fields, "status.provenance.runtime")
        _expect_string(runtime["reason"], "status.provenance.runtime.reason")
        return
    fields = frozenset(
        {"collected", "python", "platform", "gpu", "cuda", "torch", "uv"}
    )
    _expect_fields(runtime, fields, fields, "status.provenance.runtime")
    if runtime["collected"] is not True:
        raise RemoteWorkflowError("status runtime collected flag is invalid")
    for field in ("python", "platform", "gpu", "cuda", "torch", "uv"):
        _expect_object(runtime[field], f"status.provenance.runtime.{field}")
    gpu = runtime["gpu"]
    if (
        gpu.get("required") != (request["job"]["accelerator"] == "gpu")
        or gpu.get("requested") != request["runtime"]["gpu"]
        or not isinstance(gpu.get("available"), bool)
        or not isinstance(gpu.get("nvidia_smi"), dict)
    ):
        raise RemoteWorkflowError("status GPU provenance is stale or invalid")
    cuda = runtime["cuda"]
    if not isinstance(cuda.get("available"), bool):
        raise RemoteWorkflowError("status CUDA provenance is invalid")
    for field in ("python", "platform", "torch", "uv"):
        if not isinstance(runtime[field].get("available"), bool):
            raise RemoteWorkflowError(
                f"status {field} availability provenance is invalid"
            )


def _validate_output_manifest(value: Any, request: dict[str, Any]) -> None:
    if not isinstance(value, list):
        raise RemoteWorkflowError("status output_manifest must be an array")
    fields = frozenset({"declared_path", "entries"})
    if len(value) > len(request["job"]["outputs"]):
        raise RemoteWorkflowError("status output_manifest has too many mappings")
    for index, raw_mapping in enumerate(value):
        mapping = _expect_object(raw_mapping, f"status output_manifest[{index}]")
        _expect_fields(mapping, fields, fields, f"status output_manifest[{index}]")
        if mapping["declared_path"] != request["job"]["outputs"][index]:
            raise RemoteWorkflowError("status output_manifest is stale or reordered")
        _validate_manifest_entries(
            mapping["entries"], f"status output_manifest[{index}].entries"
        )


def _validate_status(
    value: Any, request: dict[str, Any], *, require_completed: bool = False
) -> dict[str, Any]:
    status = _expect_object(value, "status")
    _expect_fields(status, STATUS_FIELDS, STATUS_FIELDS, "status")
    if status["schema_version"] != SCHEMA_VERSION:
        raise RemoteWorkflowError("status has an unsupported schema_version")
    if status["run_id"] != request["run_id"]:
        raise RemoteWorkflowError("status run_id is stale or corrupt")
    if status["session"] != request["session"]:
        raise RemoteWorkflowError("status session is stale or corrupt")
    if status["request_digest"] != request["request_digest"]:
        raise RemoteWorkflowError("status request_digest is stale or corrupt")
    if status["state"] not in {"running", "failed", "completed"}:
        raise RemoteWorkflowError("status state is invalid")
    if require_completed and status["state"] != "completed":
        raise RemoteWorkflowError("published status is not completed")
    _expect_string(status["step"], "status.step")
    if (
        not isinstance(status["attempt"], int)
        or isinstance(status["attempt"], bool)
        or status["attempt"] < 1
    ):
        raise RemoteWorkflowError("status attempt must be a positive integer")
    _expect_string(status["updated_at"], "status.updated_at")
    error = status["error"]
    if status["state"] == "failed":
        error_object = _expect_object(error, "status.error")
        error_fields = frozenset({"type", "message", "traceback"})
        _expect_fields(error_object, error_fields, error_fields, "status.error")
        for field in error_fields:
            _expect_string(error_object[field], f"status.error.{field}", nonempty=False)
    elif error is not None:
        raise RemoteWorkflowError("non-failed status must have a null error")
    provenance = _expect_object(status["provenance"], "status.provenance")
    provenance_fields = frozenset(
        {
            "job_definition_digest",
            "source",
            "resolved_argv",
            "input_manifest",
            "output_manifest",
            "runtime",
        }
    )
    _expect_fields(
        provenance, provenance_fields, provenance_fields, "status.provenance"
    )
    if provenance["job_definition_digest"] != request["job"]["definition_digest"]:
        raise RemoteWorkflowError("status job definition digest is stale or corrupt")
    if provenance["resolved_argv"] != request["job"]["argv"]:
        raise RemoteWorkflowError("status resolved argv is stale or corrupt")
    _validate_input_manifest(provenance["input_manifest"], request)
    _validate_output_manifest(provenance["output_manifest"], request)
    _validate_runtime_provenance(provenance["runtime"], request)
    source_provenance = provenance["source"]
    if source_provenance:
        _validate_source_provenance(source_provenance, request)
    elif status["state"] == "completed":
        raise RemoteWorkflowError("completed status is missing source provenance")
    artifacts = _expect_object(status["artifacts"], "status.artifacts")
    if artifacts:
        artifact_fields = frozenset({"archive_path", "archive_sha256", "archive_size"})
        _expect_fields(artifacts, artifact_fields, artifact_fields, "status.artifacts")
        expected_path = str(_workspace() / "bundle/artifacts.tar.gz")
        if artifacts["archive_path"] != expected_path:
            raise RemoteWorkflowError("status artifact path is invalid")
        if not isinstance(artifacts["archive_sha256"], str) or not SHA256_RE.fullmatch(
            artifacts["archive_sha256"]
        ):
            raise RemoteWorkflowError("status artifact digest is invalid")
        if (
            not isinstance(artifacts["archive_size"], int)
            or isinstance(artifacts["archive_size"], bool)
            or artifacts["archive_size"] < 0
        ):
            raise RemoteWorkflowError("status artifact size is invalid")
    if status["state"] == "completed":
        if not artifacts:
            raise RemoteWorkflowError("completed status is missing artifact metadata")
        if status["step"] != "completed":
            raise RemoteWorkflowError("completed status must use completed step")
        if len(provenance["input_manifest"]) != len(request["job"]["inputs"]):
            raise RemoteWorkflowError("completed status input manifest is incomplete")
        if len(provenance["output_manifest"]) != len(request["job"]["outputs"]):
            raise RemoteWorkflowError("completed status output manifest is incomplete")
        if not provenance["runtime"].get("collected"):
            raise RemoteWorkflowError(
                "completed status runtime provenance is incomplete"
            )
        if request["runtime"]["accelerator"] == "gpu" and not provenance["runtime"][
            "cuda"
        ].get("available"):
            raise RemoteWorkflowError("completed GPU runtime does not confirm CUDA")
    drive_status = _expect_object(status["drive"], "status.drive")
    drive_fields = frozenset({"mode", "final"})
    _expect_fields(drive_status, drive_fields, drive_fields, "status.drive")
    expected_final = f"{request['drive']['root']}/colab-runs/{request['run_id']}"
    if (
        drive_status["mode"] != request["drive"]["mode"]
        or drive_status["final"] != expected_final
    ):
        raise RemoteWorkflowError("status Drive provenance is stale or corrupt")
    return status


def _validate_request(request: dict[str, Any]) -> None:
    _expect_fields(request, REQUEST_FIELDS, REQUEST_FIELDS, "request")
    if request["schema_version"] != SCHEMA_VERSION:
        raise RemoteWorkflowError("unsupported request schema_version")
    if not isinstance(request["run_id"], str) or not RUN_ID_RE.fullmatch(
        request["run_id"]
    ):
        raise RemoteWorkflowError("request.run_id is invalid")
    if request["run_id"] != _run_id():
        raise RemoteWorkflowError("request run_id does not match the environment")
    if not isinstance(request["session"], str) or not SESSION_RE.fullmatch(
        request["session"]
    ):
        raise RemoteWorkflowError("request.session is invalid")
    _expect_string(request["created_at"], "request.created_at")
    runtime = _expect_object(request["runtime"], "runtime")
    runtime_fields = frozenset({"accelerator", "gpu", "high_mem"})
    _expect_fields(runtime, runtime_fields, runtime_fields, "runtime")
    if runtime["accelerator"] not in ACCELERATORS:
        raise RemoteWorkflowError("runtime.accelerator is unsupported")
    if runtime["gpu"] is not None and runtime["gpu"] not in GPU_CHOICES:
        raise RemoteWorkflowError("runtime.gpu is unsupported")
    if runtime["accelerator"] == "gpu" and runtime["gpu"] is None:
        raise RemoteWorkflowError("GPU runtime has no GPU selection")
    if runtime["accelerator"] == "cpu" and runtime["gpu"] is not None:
        raise RemoteWorkflowError("CPU runtime must not have a GPU selection")
    if not isinstance(runtime["high_mem"], bool):
        raise RemoteWorkflowError("runtime.high_mem must be a boolean")
    _validate_source(request["source"])
    _validate_drive(request["drive"])
    job = _validate_job(request["job"])
    if job["accelerator"] == "gpu" and runtime["accelerator"] != "gpu":
        raise RemoteWorkflowError("GPU-required job was assigned a CPU runtime")
    request_copy = dict(request)
    digest = request_copy.pop("request_digest", None)
    if (
        not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
        or _digest_json(request_copy) != digest
    ):
        raise RemoteWorkflowError("request digest mismatch")


def _published_json(
    request: dict[str, Any], relative: str, config: Path | None
) -> dict[str, Any] | None:
    if request["drive"]["mode"] == "mount":
        final = _mount_drive_root(request) / "colab-runs" / request["run_id"]
        if not final.exists():
            return None
        if final.is_symlink() or not final.is_dir():
            raise RemoteWorkflowError("published Drive run is not a regular directory")
        return _read_json_object(final / relative, f"published {relative}")
    if config is None:
        raise RemoteWorkflowError("rclone config is required to inspect publication")
    final = _rclone_path(request, f"colab-runs/{request['run_id']}")
    if not _rclone_exists(final, config):
        return None
    result = _run(
        ["rclone", "--config", str(config), "cat", f"{final}/{relative}"],
        timeout=300,
        capture=True,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RemoteWorkflowError(f"published {relative} is invalid JSON") from error
    return _expect_object(value, f"published {relative}")


def _recover_published_status(
    request: dict[str, Any], workspace: Path, config: Path | None
) -> dict[str, Any] | None:
    published_status = _published_json(request, "status.json", config)
    if published_status is None:
        return None
    status = _validate_status(published_status, request, require_completed=True)
    published_request = _published_json(request, "request.json", config)
    if published_request is None:
        raise RemoteWorkflowError("published run is missing request.json")
    _validate_request(published_request)
    if published_request != request:
        raise RemoteWorkflowError(
            "published request does not match the current request"
        )
    bundle = workspace / "bundle"
    archive = bundle / "artifacts.tar.gz"
    if not bundle.is_dir() or not archive.is_file():
        raise RemoteWorkflowError(
            "published run exists but the VM verification bundle is unavailable"
        )
    if (
        archive.stat().st_size != status["artifacts"]["archive_size"]
        or _sha256(archive) != status["artifacts"]["archive_sha256"]
    ):
        raise RemoteWorkflowError("VM artifact bundle does not match published status")
    if request["drive"]["mode"] == "mount":
        final = _mount_drive_root(request) / "colab-runs" / request["run_id"]
        _verify_tree(bundle, final)
    else:
        assert config is not None
        final = _rclone_path(request, f"colab-runs/{request['run_id']}")
        _run(
            [
                "rclone",
                "--config",
                str(config),
                "check",
                "--one-way",
                str(bundle),
                final,
            ],
            timeout=3600,
        )
    return status


def _preserve_input_manifest(
    workspace: Path,
    request: dict[str, Any],
    staged_manifest: list[dict[str, Any]],
    previous: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    marker_path = workspace / "input-manifest.json"
    if marker_path.is_file():
        marker = _read_json_object(marker_path, "input manifest marker")
        fields = frozenset({"request_digest", "manifest"})
        _expect_fields(marker, fields, fields, "input manifest marker")
        if marker["request_digest"] != request["request_digest"]:
            raise RemoteWorkflowError("input manifest belongs to a different request")
        _validate_input_manifest(marker["manifest"], request)
        if marker["manifest"] != staged_manifest:
            raise RemoteWorkflowError(
                "restaged inputs differ from the immutable first-attempt manifest"
            )
        manifest = cast(list[dict[str, Any]], marker["manifest"])
    else:
        marker = {
            "request_digest": request["request_digest"],
            "manifest": staged_manifest,
        }
        _atomic_json(marker_path, marker)
        manifest = staged_manifest
    if previous is not None:
        previous_manifest = previous["provenance"]["input_manifest"]
        if previous_manifest and previous_manifest != manifest:
            raise RemoteWorkflowError(
                "previous status input manifest differs from the immutable marker"
            )
    return manifest


def _execute_action() -> int:
    workspace = _workspace()
    request_path = workspace / "request.json"
    status_path = workspace / "status.json"
    request = _read_json_object(request_path, "request")
    _validate_request(request)
    previous: dict[str, Any] | None = None
    if status_path.is_file():
        previous = _validate_status(_read_json_object(status_path, "status"), request)
        if previous["state"] == "completed":
            print("[tennis-colab] run already completed; nothing to resume", flush=True)
            return 0
    attempt = int(previous["attempt"] if previous else 0) + 1
    status = _initial_status(request, attempt)
    _write_status(status_path, status, "preparing_source")
    config_path: Path | None = None
    try:
        if request["drive"]["mode"] == "rclone":
            configured = globals().get("TENNIS_COLAB_RCLONE_CONFIG", "")
            if not isinstance(configured, str):
                raise RemoteWorkflowError(
                    "temporary rclone configuration path is invalid"
                )
            config_path = Path(configured)
            if not config_path.is_file():
                raise RemoteWorkflowError("temporary rclone configuration is missing")
            _ensure_rclone(config_path)
        recovered = _recover_published_status(request, workspace, config_path)
        if recovered is not None:
            _atomic_json(status_path, recovered)
            print(
                f"[tennis-colab] recovered completed publication {request['run_id']}",
                flush=True,
            )
            return 0
        repo = workspace / "repo"
        status["provenance"]["source"] = _prepare_source(request, workspace)
        _write_status(status_path, status, "setup")
        for hook_name in request["job"]["setup"]:
            _run_hook(request, repo, hook_name)
            if hook_name == "base":
                status["provenance"]["runtime"] = _runtime_provenance(request, repo)
                _write_status(status_path, status, "checking_runtime")
                if (
                    request["runtime"]["accelerator"] == "gpu"
                    and not status["provenance"]["runtime"]["cuda"]["available"]
                ):
                    raise RemoteWorkflowError(
                        "GPU runtime cannot start: actual NVIDIA GPU and torch "
                        "CUDA availability were not both confirmed"
                    )
        _write_status(status_path, status, "staging_inputs")
        staged_manifest = _stage_inputs(
            request, repo, config_path, reset_existing=previous is not None
        )
        status["provenance"]["input_manifest"] = _preserve_input_manifest(
            workspace, request, staged_manifest, previous
        )
        _write_status(status_path, status, "inputs_verified")
        _write_status(status_path, status, "running_job")
        environment = os.environ.copy()
        environment["TENNIS_LAB_COLAB_RUN_ID"] = request["run_id"]
        environment["PATH"] = f"{repo / '.venv/bin'}:{environment.get('PATH', '')}"
        try:
            _run(
                list(request["job"]["argv"]),
                cwd=repo,
                timeout=request["job"]["timeout_seconds"],
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise RemoteWorkflowError(
                f"job exceeded timeout of {request['job']['timeout_seconds']} seconds"
            ) from error
        _write_status(status_path, status, "collecting_outputs")
        bundle = workspace / "bundle"
        if bundle.exists():
            shutil.rmtree(bundle)
        output_manifest, archive = _collect_outputs(request, repo, bundle)
        status["provenance"]["output_manifest"] = output_manifest
        status["artifacts"] = {
            "archive_path": str(archive),
            "archive_sha256": _sha256(archive),
            "archive_size": archive.stat().st_size,
        }
        status["state"] = "running"
        _write_status(status_path, status, "publishing")
        completed_status = json.loads(json.dumps(status))
        completed_status["state"] = "completed"
        completed_status["step"] = "completed"
        completed_status["updated_at"] = _utc_now()
        shutil.copy2(request_path, bundle / "request.json")
        _atomic_json(bundle / "status.json", completed_status)
        _atomic_json(bundle / "manifest.json", _bundle_manifest(bundle))
        _publish(request, bundle, attempt, config_path)
        _atomic_json(status_path, completed_status)
        print(f"[tennis-colab] completed run {request['run_id']}", flush=True)
        return 0
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )[-12000:],
        }
        _write_status(status_path, status, status.get("step", "failed"))
        print(f"[tennis-colab] failed: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


def main() -> int:
    """Dispatch the small protocol selected by generated bootstrap literals."""

    action = globals().get("TENNIS_COLAB_ACTION", "")
    if action == "prepare":
        _prepare_action()
        return 0
    elif action == "cleanup-secret":
        _cleanup_secret()
        return 0
    elif action == "run":
        return _execute_action()
    raise RemoteWorkflowError(f"unknown TENNIS_COLAB_ACTION: {action!r}")


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
