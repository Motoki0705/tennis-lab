"""Validate append logs and explicitly preserve interrupted control writes."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

from .filesystem import (
    ATOMIC_TEMPORARY,
    StorageError,
    json_records,
    names,
    read_bytes,
    read_json,
    write_bytes_atomic,
    write_json,
)


def journal_paths(state: Path) -> list[Path]:
    audit = state / "audit.jsonl"
    result = []
    if audit.exists() or audit.is_symlink():
        if not stat.S_ISREG(audit.lstat().st_mode):
            raise StorageError(f"unexpected journal: {audit}")
        result.append(audit)
    folder = state / "removals"
    for name in names(folder) if folder.exists() else []:
        path = folder / name
        if ATOMIC_TEMPORARY.fullmatch(name) and stat.S_ISREG(path.lstat().st_mode):
            continue
        if not name.endswith(".jsonl") or not stat.S_ISREG(path.lstat().st_mode):
            raise StorageError(f"unexpected journal: {path}")
        result.append(path)
    return result


def valid_prefix(data: bytes, path: Path) -> int:
    """Only newline-terminated JSON objects are complete append records."""
    offset = 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            if not line.endswith(b"\n") or not isinstance(json.loads(line), dict):
                raise ValueError("incomplete object line")
        except (ValueError, UnicodeError) as exc:
            if index != len(lines) - 1:
                raise StorageError(
                    f"journal has corrupt interior record: {path}"
                ) from exc
            return offset
        offset += len(line)
    return offset


def repair_log(state: Path, receipt: Path, value: dict[str, Any]) -> dict[str, Any]:
    relative = value["journal"]
    if not isinstance(relative, str) or type(value.get("prefix_bytes")) is not int:
        raise StorageError("invalid journal repair receipt")
    if relative != "audit.jsonl" and not (
        Path(relative).parent == Path("removals")
        and Path(relative).suffix == ".jsonl"
        and not Path(relative).is_absolute()
    ):
        raise StorageError("invalid journal repair path")
    source = state / relative
    archive = state / "journal-history" / (receipt.stem + ".jsonl")
    for path in (source, archive):
        if (path.exists() or path.is_symlink()) and not stat.S_ISREG(
            path.lstat().st_mode
        ):
            raise StorageError(f"invalid journal repair evidence: {path}")
    data = read_bytes(archive if archive.exists() else source)
    if hashlib.sha256(data).hexdigest() != value["sha256"]:
        raise StorageError("journal repair evidence changed")
    if not 0 <= value["prefix_bytes"] < len(data):
        raise StorageError("invalid journal repair prefix")
    prefix = data[: value["prefix_bytes"]]
    if valid_prefix(data, source) != len(prefix) or len(prefix) == len(data):
        raise StorageError("invalid journal repair prefix")
    current = read_bytes(source)
    if current not in (data, prefix):
        raise StorageError("journal changed during repair; preserve both versions")
    if not archive.exists():
        write_bytes_atomic(archive, data)
    if current != prefix:
        write_bytes_atomic(source, prefix)
    write_json(receipt, {**value, "phase": "complete"})
    return {
        "action": "journal_repaired",
        "journal": str(source),
        "archive": str(archive),
        "discarded_tail_bytes": len(data) - len(prefix),
        "sha256": value["sha256"],
        "receipt": str(receipt),
    }


def inspect_logs(state: Path, *, repair: bool) -> list[dict[str, Any]]:
    """Normal operations reject damage; recovery preserves every original byte."""
    folder = state / "journal-repairs"
    results = []
    for name in json_records(folder):
        path = folder / name
        value = read_json(path)
        if value.get("schema_version") != 1 or value.get("phase") not in {
            "prepared",
            "complete",
        }:
            raise StorageError(f"invalid journal repair receipt: {path}")
        if value["phase"] == "prepared":
            if not repair:
                raise StorageError("unfinished journal repair; run recover")
            results.append(repair_log(state, path, value))
    for path in journal_paths(state):
        data = read_bytes(path)
        prefix = valid_prefix(data, path)
        if prefix == len(data):
            continue
        if not repair:
            raise StorageError(f"journal has an incomplete tail; run recover: {path}")
        relative = str(path.relative_to(state))
        digest = hashlib.sha256(data).hexdigest()
        key = hashlib.sha256((relative + digest).encode()).hexdigest()
        receipt = folder / (key + ".json")
        value = {
            "schema_version": 1,
            "phase": "prepared",
            "journal": relative,
            "sha256": digest,
            "prefix_bytes": prefix,
        }
        write_json(receipt, value)
        results.append(repair_log(state, receipt, value))
    return results


def inspect_json(repo: Path, mirror: Path) -> list[dict[str, Any]]:
    """Validate committed JSON and report preserved, uncommitted temporaries."""
    result = []
    metadata = (
        "manifests",
        "transactions",
        "publications",
        "checkpoint-file-operations",
        "recovery-required",
        "publication-evidence",
        "recovered-transactions",
        "legacy-manifests",
        "journal-repairs",
        "journal-history",
        "removals",
    )
    for base in (repo, mirror):
        state = base / ".storage-tier"
        for root in [state, *(state / folder for folder in metadata)]:
            if not root.exists():
                continue
            pending = [root]
            while pending:
                parent = pending.pop()
                for name in names(parent):
                    path = parent / name
                    info = path.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        if root != state:
                            pending.append(path)
                    elif not name.startswith(".") and name.endswith(".json"):
                        if not stat.S_ISREG(info.st_mode):
                            raise StorageError(
                                f"invalid committed control record: {path}"
                            )
                        read_json(path)
                    elif name.startswith(".") and name.endswith(".tmp"):
                        if not stat.S_ISREG(info.st_mode):
                            raise StorageError(f"invalid control temporary: {path}")
                        result.append(
                            {
                                "path": str(path),
                                "bytes": info.st_size,
                                "committed": False,
                            }
                        )
    return result
