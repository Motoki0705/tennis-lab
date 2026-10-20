"""Run storage policy, deletion records and references (CPU and stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

from kg_lib import Node, nodes_dir, repo_root

FILE_LIMIT = 256 * 1024
FIGURE_LIMIT = 512 * 1024
RUN_LIMIT = 2 * 1024 * 1024
MANIFEST = "artifact-manifest.json"
EXCEPTIONS = "run-storage-exceptions.json"
FIGURES = {".png", ".svg", ".pdf"}
RAW = {
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".onnx",
    ".gz",
    ".zip",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".mp4",
    ".gif",
    ".jpg",
    ".jpeg",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".prof",
}
TEXT = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".jsonl",
    ".txt",
    ".log",
    ".md",
    ".patch",
    ".cjs",
    ".state",
    ".job",
    ".ps1",
    ".toml",
    ".js",
    ".mjs",
    ".html",
    ".css",
    ".ts",
}
REASONS = {
    "raw-data-events-archives",
    "figure-over-512KiB",
    "table-or-log-over-256KiB",
    "file-over-256KiB",
    "unsupported-file-type",
    "run-over-2MiB",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
LINK = re.compile(r"!?\[([^\]\n]*)\]\(([^\s)]+)\)")
BARE = re.compile(r"(?<![\w/])knowledge/runs/[^\s`<>\[\]()\"',;|]+")
RELATIVE = re.compile(r"(?<![\w/])(?:\.\./)+runs/[^\s`<>\[\]()\"',;|]+")
DEFINITION = re.compile(r"^\s*\[[^\]\n]+\]:\s*(\S+)", re.MULTILINE)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_sha256(directory: Path) -> str:
    """Hash sorted '<content SHA-256>  <run-relative path>\\n' rows, UTF-8."""
    rows = (
        f"{sha256(p)}  {p.relative_to(directory).as_posix()}\n"
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    )
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def removal_reason(path: Path, size: int) -> str | None:
    ext = path.suffix.lower()
    if ext in RAW or path.name.startswith("events.out.tfevents."):
        return "raw-data-events-archives"
    if ext not in TEXT | FIGURES:
        return "unsupported-file-type"
    if ext in FIGURES:
        return "figure-over-512KiB" if size > FIGURE_LIMIT else None
    if size > FILE_LIMIT:
        return (
            "table-or-log-over-256KiB"
            if ext in {".json", ".csv", ".jsonl", ".txt", ".log"}
            else "file-over-256KiB"
        )
    return None


def _relative(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not value.startswith("/")
        and PurePosixPath(value).as_posix() == value
        and all(part not in (".", "..") for part in value.split("/"))
    )


def _json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(), object_pairs_hook=unique)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError(f"{path}: unsupported schema_version (expected 1)")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    data = _json(path)
    if (
        set(data) != {"schema_version", "run_id", "files"}
        or data["run_id"] != path.parent.name
    ):
        raise ValueError(f"{path}: invalid manifest fields or run_id")
    if not isinstance(data["files"], list) or not data["files"]:
        raise ValueError(f"{path}: files must be a nonempty list")
    seen = set()
    for record in data["files"]:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "reason",
            "removed_from_commit",
        }:
            raise ValueError(f"{path}: invalid deletion record fields")
        relative = record["path"]
        if not _relative(relative) or relative == MANIFEST or relative in seen:
            raise ValueError(f"{path}: invalid or duplicate removed path {relative!r}")
        seen.add(relative)
        if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
            raise ValueError(f"{path}: invalid size_bytes for {relative}")
        if not isinstance(record["sha256"], str) or not SHA256.fullmatch(
            record["sha256"]
        ):
            raise ValueError(f"{path}: invalid SHA-256 for {relative}")
        if not isinstance(record["removed_from_commit"], str) or not COMMIT.fullmatch(
            record["removed_from_commit"]
        ):
            raise ValueError(
                f"{path}: removed_from_commit must be the full pre-deletion HEAD"
            )
        if not isinstance(record["reason"], str) or record["reason"] not in REASONS:
            raise ValueError(f"{path}: unknown deletion reason for {relative}")
        expected = removal_reason(Path(relative), record["size_bytes"])
        if record["reason"] != expected and record["reason"] != "run-over-2MiB":
            raise ValueError(
                f"{path}: deletion reason disagrees with type/size for {relative}"
            )
        target = path.parent / relative
        if target.exists() or target.is_symlink():
            raise ValueError(f"{path}: deleted file still exists: {relative}")
    return data


def _exceptions(base: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = base / EXCEPTIONS
    result: dict[str, dict[str, dict[str, Any]]] = {
        "files": {},
        "runs": {},
        "legacy_manifests": {},
    }
    if not path.exists():
        return result
    data = _json(path)
    if set(data) != {"schema_version", *result}:
        raise ValueError(f"{path}: invalid exception fields")
    for kind in result:
        if not isinstance(data[kind], list):
            raise ValueError(f"{path}: {kind} must be a list")
        for row in data[kind]:
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "reason"}:
                raise ValueError(f"{path}: invalid exception record")
            relative = row["path"]
            if (
                not _relative(relative)
                or not relative.startswith("runs/")
                or relative in result[kind]
            ):
                raise ValueError(
                    f"{path}: invalid or duplicate exception path {relative!r}"
                )
            if not isinstance(row["sha256"], str) or not SHA256.fullmatch(
                row["sha256"]
            ):
                raise ValueError(f"{path}: invalid exception SHA-256")
            if not isinstance(row["reason"], str) or not row["reason"].strip():
                raise ValueError(f"{path}: exception reason is required")
            result[kind][relative] = row
    return result


def validate_runs(
    base: Path, nodes: list[Node], base_ref: str | None = None
) -> list[str]:
    errors: list[str] = []
    manifests: dict[Path, dict[str, Any]] = {}
    try:
        exceptions = _exceptions(base)
    except ValueError as exc:
        errors.append(str(exc))
        exceptions = {"files": {}, "runs": {}, "legacy_manifests": {}}
    used: dict[str, set[str]] = {key: set() for key in exceptions}
    runs = base / "runs"
    for directory in sorted(runs.iterdir()) if runs.exists() else []:
        if directory.is_symlink() or not directory.is_dir():
            errors.append(f"{directory}: run storage requires regular directories")
            continue
        total = 0
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                errors.append(f"{path}: symlinks are not allowed in run storage")
                continue
            if path.is_dir():
                continue
            size = path.stat().st_size
            total += size
            relative = path.relative_to(base).as_posix()
            issue = removal_reason(path, size)
            exception = exceptions["files"].get(relative)
            if exception:
                used["files"].add(relative)
                if sha256(path) != exception["sha256"]:
                    errors.append(f"{path}: exception SHA-256 mismatch")
                if not issue:
                    errors.append(f"{path}: unnecessary file exception")
            elif issue:
                errors.append(f"{path}: run storage policy: {issue} ({size} bytes)")
            if path.name == MANIFEST:
                legacy = exceptions["legacy_manifests"].get(relative)
                if legacy:
                    used["legacy_manifests"].add(relative)
                    try:
                        _json(path)
                    except ValueError as exc:
                        errors.append(str(exc))
                    if sha256(path) != legacy["sha256"]:
                        errors.append(f"{path}: legacy manifest SHA-256 mismatch")
                elif path.parent != directory:
                    errors.append(f"{path}: manifest must be at the run root")
                else:
                    try:
                        manifests[path.resolve()] = load_manifest(path)
                    except ValueError as exc:
                        errors.append(str(exc))
        relative = directory.relative_to(base).as_posix()
        exception = exceptions["runs"].get(relative)
        if exception:
            used["runs"].add(relative)
            if tree_sha256(directory) != exception["sha256"]:
                errors.append(
                    f"{directory}: run exception SHA-256 mismatch (file set and contents are fixed)"
                )
            if total <= RUN_LIMIT:
                errors.append(f"{directory}: unnecessary run exception")
        elif total > RUN_LIMIT:
            errors.append(f"{directory}: run exceeds 2 MiB ({total} bytes)")
    for kind in used:
        for missing in exceptions[kind].keys() - used[kind]:
            errors.append(f"{base / EXCEPTIONS}: stale {kind} exception: {missing}")
    errors.extend(validate_references(base, nodes, manifests))
    if base_ref:
        errors.extend(validate_removal_history(base, base_ref, manifests))
    return errors


def _run_target(base: Path, source: Path, value: str) -> tuple[Path, str] | None:
    parts = urlsplit(value.strip("<>"))
    if parts.scheme or parts.netloc:
        return None
    raw = unquote(parts.path)
    if raw.startswith("knowledge/runs/"):
        target = base / raw.removeprefix("knowledge/")
    else:
        target = Path(raw) if Path(raw).is_absolute() else source.parent / raw
    target = target.resolve()
    if target.is_relative_to((base / "runs").resolve()):
        return target, parts.fragment
    return None


def validate_references(
    base: Path, nodes: list[Node], manifests: dict[Path, dict[str, Any]]
) -> list[str]:
    errors = []

    def reference(
        source: Path, value: str, deleted: bool = False, path_prefix: str | None = None
    ) -> None:
        try:
            resolved = _run_target(base, source, value)
        except ValueError:
            errors.append(f"{source}: invalid artifact reference {value!r}")
            return
        if resolved is None:
            if deleted:
                errors.append(
                    f"{source}: deleted reference must point to a run manifest: {value}"
                )
            return
        target, fragment = resolved
        if deleted:
            match = re.fullmatch(r"/files/(0|[1-9][0-9]*)", fragment)
            manifest = manifests.get(target)
            if path_prefix is not None:
                if (
                    not manifest
                    or fragment
                    or not path_prefix.endswith("/")
                    or not _relative(path_prefix[:-1])
                    or not any(
                        row["path"].startswith(path_prefix) for row in manifest["files"]
                    )
                    or (target.parent / path_prefix).exists()
                ):
                    errors.append(
                        f"{source}: missing deleted directory record: {value}, prefix={path_prefix}"
                    )
            elif not manifest or not match or int(match[1]) >= len(manifest["files"]):
                errors.append(f"{source}: missing deletion record: {value}")
        elif fragment.startswith("/files/") or target.name == MANIFEST and fragment:
            errors.append(
                f"{source}: deletion record reference must be labelled 削除済み or status: deleted: {value}"
            )
        elif "*" in str(target):
            if not list(
                (base / "runs").glob(str(target.relative_to((base / "runs").resolve())))
            ):
                errors.append(f"{source}: broken run glob: {value}")
        elif not target.exists():
            errors.append(
                f"{source}: broken run reference (use a deletion record for deleted files): {value}"
            )

    def artifact(source: Path, value: Any) -> None:
        if isinstance(value, dict):
            if value.get("status") == "deleted":
                prefix = value.get("path_prefix")
                if (
                    set(value)
                    not in ({"status", "record"}, {"status", "record", "path_prefix"})
                    or not isinstance(value.get("record"), str)
                    or "path_prefix" in value
                    and not isinstance(prefix, str)
                ):
                    errors.append(
                        f"{source}: deleted artifact requires only status and record"
                    )
                else:
                    reference(source, value["record"], True, prefix)
            else:
                for child in value.values():
                    artifact(source, child)
        elif isinstance(value, list):
            for child in value:
                artifact(source, child)
        elif isinstance(value, str):
            reference(source, value)

    documents = [(n.path, n.body) for n in nodes]
    documents += [
        (p, p.read_text())
        for p in [base / "summary.md", *sorted((base / "reports").rglob("*.md"))]
        if p.is_file()
    ]
    for node in nodes:
        artifact(node.path, node.meta.get("artifacts", {}))
    for source, body in documents:
        for match in LINK.finditer(body):
            reference(source, match[2], "削除済み" in match[1])
        for match in DEFINITION.finditer(body):
            reference(source, match[1])
        # Link labels can preserve the original path as provenance, not a live reference.
        remaining = DEFINITION.sub("", LINK.sub("", body))
        for pattern in (BARE, RELATIVE):
            for match in pattern.finditer(remaining):
                reference(source, match[0].rstrip("."))
    return errors


def validate_removal_history(
    base: Path, base_ref: str, manifests: dict[Path, dict[str, Any]]
) -> list[str]:
    """Check new deletions against base blobs; keep existing records append-only.

    No lookup of old commits is required after a history rewrite: unchanged
    manifests are compared with the PR base, which already contains the records.
    """
    root = repo_root()

    def git(*args: str) -> bytes:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True)
        if result.returncode:
            raise ValueError(
                f"cannot read artifact history base {base_ref}: {result.stderr.decode().strip()}"
            )
        return result.stdout

    commit = (
        git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}")
        .decode()
        .strip()
    )
    prefix = (base / "runs").relative_to(root).as_posix()
    old: dict[str, tuple[str, int]] = {}
    for row in git("ls-tree", "-rlz", commit, "--", prefix).split(b"\0"):
        if row:
            info, name = row.decode().split("\t")
            _, kind, oid, raw_size = info.split()
            if kind == "blob":
                old[name] = oid, int(raw_size)
    errors = []
    new_records: dict[str, dict[str, Any]] = {}
    for path, manifest in manifests.items():
        name = path.relative_to(root).as_posix()
        before: list[dict[str, Any]] = []
        if name in old:
            previous = json.loads(git("show", f"{commit}:{name}"))
            before = previous["files"]
            if manifest["files"][: len(before)] != before:
                errors.append(
                    f"{path}: existing deletion records are immutable; append new records"
                )
        for record in manifest["files"][len(before) :]:
            name = (path.parent / record["path"]).relative_to(root).as_posix()
            new_records[name] = record
            if name not in old:
                errors.append(
                    f"{path}: new deletion record has no file in base: {record['path']}"
                )
    deleted = {name for name in old if not (root / name).exists()}
    for name in sorted(deleted):
        if name not in new_records:
            errors.append(f"{name}: removed file is missing its deletion record")
    # Stream blobs so large removed arrays never accumulate in process memory.
    with subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as batch:
        assert batch.stdin is not None and batch.stdout is not None
        for name, record in sorted(new_records.items()):
            if name not in old:
                continue
            oid, size = old[name]
            source = git("ls-tree", record["removed_from_commit"], "--", name).decode()
            if not source or source.split()[2] != oid:
                errors.append(
                    f"{name}: removed_from_commit does not contain the base blob"
                )
            batch.stdin.write(f"{oid}\n".encode())
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            remaining = int(header[2])
            digest = hashlib.sha256()
            while remaining:
                chunk = batch.stdout.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"{name}: unexpected end of git blob")
                digest.update(chunk)
                remaining -= len(chunk)
            batch.stdout.read(1)
            if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                errors.append(
                    f"{name}: deletion record size/SHA-256 mismatch with base blob"
                )
        batch.stdin.close()
        if batch.wait():
            raise ValueError("git cat-file failed during deletion verification")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--removed-paths",
        action="store_true",
        required=True,
        help="print sorted repo-relative paths from validated deletion records",
    )
    parser.parse_args()
    base = nodes_dir().parent
    legacy = _exceptions(base)["legacy_manifests"]
    paths = set()
    for manifest_path in sorted((base / "runs").glob(f"*/{MANIFEST}")):
        exception = legacy.get(manifest_path.relative_to(base).as_posix())
        if exception:
            _json(manifest_path)
            if sha256(manifest_path) != exception["sha256"]:
                raise ValueError(f"{manifest_path}: legacy manifest SHA-256 mismatch")
            continue
        for record in load_manifest(manifest_path)["files"]:
            paths.add(
                (manifest_path.parent / record["path"])
                .relative_to(repo_root())
                .as_posix()
            )
    print("\n".join(sorted(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
