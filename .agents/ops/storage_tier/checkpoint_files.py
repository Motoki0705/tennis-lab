"""Checkpoint root-file projection; retain every superseded inode, never swap root."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .filesystem import (
    Snapshot,
    StorageError,
    copy_tree,
    json_records,
    read_json,
    rename,
    snapshot,
    sync_dir,
    write_json,
)
from .layout import Unit

if TYPE_CHECKING:
    from .engine import Engine


def files(path: Path, *, hashes: bool = False) -> Snapshot:
    return snapshot(path, hashes=hashes, root_files_only=True)


def content(tree: Snapshot) -> dict[str, tuple[Any, ...]]:
    # Container permissions belong to each tier, not to the file projection.
    # same_metadata still checks each root's identity/mode within that tier.
    return {name: value for name, value in tree.content().items() if name != "."}


def stable(engine: Engine, unit: Unit, path: Path) -> Snapshot:
    before = files(path)
    engine.quiet(unit, path, before)
    result = files(path, hashes=True)
    if not before.same_metadata(result):
        raise StorageError("checkpoint root files changed during hashing")
    engine.quiet(unit, path, result)
    return result


def recorded_snapshot(engine: Engine, unit: Unit, *, legacy: bool) -> Snapshot | None:
    """Keep existing digest evidence, including direct files of the old schema."""
    path = engine.manifest_path(unit)
    if legacy and path.exists() and read_json(path).get("kind") == "checkpoint":
        saved = engine.manifest(Unit("ckpt", "checkpoint"))
        if saved is None:
            raise StorageError("checkpoint manifest disappeared while reading")
        entries = {
            name: dict(entry)
            for name, entry in saved["mirror"].items()
            if name == "." or ("/" not in name and entry["kind"] == "file")
        }
        entries["."]["mtime_ns"] = entries["."]["ctime_ns"] = 0
        return Snapshot(entries)
    saved = engine.manifest(unit)
    return Snapshot(saved["mirror"]) if saved is not None else None


def check_recorded(previous: Snapshot | None, observed: Snapshot) -> None:
    if previous is None:
        return
    if not previous.same_metadata(observed):
        raise StorageError("E: checkpoint root files changed since verification")
    if content(previous) != content(observed):
        raise StorageError(
            "E: checkpoint root-file SHA-256 does not match its recorded manifest"
        )


def operate(
    engine: Engine, unit: Unit, command: str, *, dry_run: bool
) -> dict[str, Any]:
    if (unit.path, unit.kind) != ("ckpt", "checkpoint-files") or command not in {
        "index",
        "verify",
        "mirror",
        "hydrate",
    }:
        raise StorageError("invalid checkpoint root-file operation")
    engine.require_complete_local(unit)
    operations = engine.config.state / "checkpoint-file-operations"
    if operations.exists():
        for name in json_records(operations):
            record = read_json(operations / name)
            if record.get("phase") != "complete":
                raise StorageError(
                    f"unfinished checkpoint-file publication: {operations / name}; "
                    "pause writers and run recover; preserve the journal and retained history"
                )
    mirror_path = engine.config.mirror / "ckpt"
    previous = recorded_snapshot(engine, unit, legacy=command == "index")
    if command == "index":
        path = engine.path(unit, mirror=True)
        before = files(path)
        tree = stable(engine, unit, path)
        check_recorded(previous, tree)
        # Explicit index is the schema-boundary migration. Preserve the former
        # whole-ckpt manifest before replacing the same path-key with root files.
        old_path = engine.manifest_path(unit)
        if not dry_run and old_path.exists():
            legacy_manifest = read_json(old_path)
            if legacy_manifest.get("kind") == "checkpoint":
                archive = (
                    engine.config.mirror
                    / ".storage-tier/legacy-manifests"
                    / (old_path.stem + ".whole-ckpt.json")
                )
                if archive.exists():
                    if read_json(archive) != legacy_manifest:
                        raise StorageError(
                            "legacy checkpoint manifest archive conflicts with the current evidence"
                        )
                else:
                    write_json(archive, legacy_manifest)
        manifest = engine.record(unit, None, tree, dry_run=dry_run)
        return {
            "action": "would_index" if dry_run else "indexed",
            "bytes": tree.bytes,
            "files": len(tree.entries) - 1,
            "manifest": manifest,
            "atime_preserved": all(
                entry["atime_ns"] == tree.entries[name]["atime_ns"]
                for name, entry in before.entries.items()
            ),
            "sha256_tree": hashlib.sha256(
                json.dumps(content(tree), sort_keys=True).encode()
            ).hexdigest(),
        }
    mirror: Snapshot | None = None
    if mirror_path.exists():
        mirror = stable(engine, unit, engine.path(unit, mirror=True))
        check_recorded(previous, mirror)
    if command in {"verify", "hydrate"} and mirror is None:
        raise StorageError("E: checkpoint root files are missing; index/mirror first")
    local_path = engine.config.repo / "ckpt"
    local = stable(engine, unit, engine.path(unit)) if local_path.exists() else None
    if command == "verify":
        if local is None or mirror is None or content(local) != content(mirror):
            raise StorageError("checkpoint root-file SHA-256/tree mismatch")
        manifest = engine.record(unit, local, mirror, dry_run=dry_run)
        return {
            "action": "would_verify" if dry_run else "verified",
            "bytes": local.bytes,
            "manifest": manifest,
        }
    hydrate = command == "hydrate"
    source, expected = (mirror_path, mirror) if hydrate else (local_path, local)
    destination, old = (local_path, local) if hydrate else (mirror_path, mirror)
    if expected is None:
        raise StorageError("D: checkpoint root files are missing")
    if hydrate and old is not None and len(old.entries) > 1:
        raise StorageError("D: checkpoint root files already exist; refusing to merge")
    if not hydrate and old is not None and content(old) == content(expected):
        manifest = engine.record(unit, expected, old, dry_run=dry_run)
        return {
            "action": "would_verify" if dry_run else "verified",
            "bytes": expected.bytes,
            "manifest": manifest,
        }
    if not hydrate and old is not None and previous is None and len(old.entries) > 1:
        raise StorageError(
            "unrecorded E: checkpoint root files differ; refusing overwrite"
        )
    if dry_run:
        return {
            "action": "would_hydrate" if hydrate else "would_mirror",
            "bytes": expected.bytes,
        }
    engine.check_mount()
    base = engine.config.repo if hydrate else engine.config.mirror
    engine.ensure_parent(destination.parent, base)
    if not destination.exists():
        destination.mkdir(mode=expected.entries["."]["mode"])
        sync_dir(destination.parent)
    engine.path(unit, mirror=not hydrate)
    transaction = uuid.uuid4().hex
    history = base / ".storage-tier/checkpoint-file-history" / transaction
    engine.ensure_parent(history.parent, base)
    journal = operations / f"{transaction}.json"
    record = {
        "schema_version": 2,
        "unit": unit.path,
        "kind": unit.kind,
        "operation": command,
        "phase": "prepared",
        "history": str(history),
        "source": str(source),
        "destination": str(destination),
        "expected": expected.entries,
        "previous": old.entries if old else None,
        "previous_manifest": engine.manifest(unit),
        "published": [],
        "archived": [],
    }
    write_json(journal, record)
    try:
        history.mkdir(mode=0o700)
        staged = history / "payload"
        copy_tree(source, staged, expected, now_ns=engine.now_ns if hydrate else None)
        copied = files(staged, hashes=True)
        if content(copied) != content(expected) or not expected.same_metadata(
            files(source)
        ):
            raise StorageError("checkpoint root-file copy/source changed")
        if old is not None and not old.same_metadata(files(destination)):
            raise StorageError("checkpoint root files changed before publication")
        engine.quiet(unit, source, expected)
        record["copied"] = copied.entries
        record["phase"] = "publishing"
        write_json(journal, record)
        for name in expected.entries:
            if name == ".":
                continue
            engine.check_mount()
            exists = old is not None and name in old.entries
            rename(staged / name, destination / name, exchange=exists)
            record["published"].append(name)
            write_json(journal, record)
        if old is not None:
            retired = history / "removed-from-source"
            for name in old.entries.keys() - expected.entries.keys():
                retired.mkdir(exist_ok=True)
                rename(destination / name, retired / name)
                record["archived"].append(name)
                write_json(journal, record)
        final = stable(engine, unit, destination)
        if content(final) != content(expected) or not expected.same_metadata(
            files(source)
        ):
            raise StorageError("checkpoint root files changed during publication")
        manifest = engine.record(
            unit,
            final if hydrate else expected,
            expected if hydrate else final,
            dry_run=False,
        )
        record["phase"] = "complete"
        write_json(journal, record)
        engine.audit(
            "checkpoint_files_published",
            unit,
            operation=command,
            history=str(history),
            journal=str(journal),
            bytes=final.bytes,
        )
        return {
            "action": "hydrated" if hydrate else "mirrored",
            "bytes": final.bytes,
            "manifest": manifest,
            "retained_history": str(history),
        }
    except BaseException as exc:
        engine.audit(
            "checkpoint_files_interrupted",
            unit,
            operation=command,
            history=str(history),
            journal=str(journal),
            error=str(exc),
        )
        raise


def recover(engine: Engine) -> list[dict[str, Any]]:
    """Complete only a journaled, hash-verified root-file publication."""
    from .filesystem import real_directory
    from .publication import complete, evidence

    folder = engine.config.state / "checkpoint-file-operations"
    results = []
    for name in json_records(folder):
        journal = folder / name
        value = read_json(journal)
        if value.get("phase") == "complete":
            continue
        if (
            value.get("schema_version") != 2
            or value.get("kind") != "checkpoint-files"
            or value.get("unit") != "ckpt"
            or value.get("operation") not in {"mirror", "hydrate"}
            or value.get("phase") not in {"prepared", "publishing"}
        ):
            raise StorageError(
                "unsupported checkpoint publication journal; retain it for manual inspection"
            )
        unit = Unit("ckpt", "checkpoint-files")
        hydrate = value["operation"] == "hydrate"
        source = engine.path(unit, mirror=hydrate)
        destination = engine.path(unit, mirror=not hydrate)
        base = engine.config.repo if hydrate else engine.config.mirror
        anchor = base if hydrate else engine.config.mount
        history = base / ".storage-tier/checkpoint-file-history" / journal.stem
        if (
            value["source"] != str(source)
            or value["destination"] != str(destination)
            or value["history"] != str(history)
        ):
            raise StorageError("invalid checkpoint publication recovery paths")
        expected = Snapshot(value["expected"])
        previous = (
            Snapshot(value["previous"]) if value["previous"] is not None else None
        )
        target = stable(engine, unit, destination)
        if "copied" not in value:
            if (previous is None and content(target)) or (
                previous is not None and content(target) != content(previous)
            ):
                raise StorageError(
                    "unverified checkpoint publication cannot be recovered"
                )
            archive = evidence(engine, "checkpoint-" + name, value)
            complete(journal, value)
            engine.audit(
                "publication_aborted",
                unit,
                evidence_archive=str(archive),
                history=str(history),
            )
            results.append(
                {
                    "unit": unit.path,
                    "action": "publication_aborted",
                    "evidence_archive": str(archive),
                }
            )
            continue
        original = stable(engine, unit, source)
        if content(original) != content(expected):
            raise StorageError(
                "checkpoint publication source SHA-256 differs from verified intent"
            )
        real_directory(history, anchor)
        staged = history / "payload"
        real_directory(staged, anchor)
        staged_tree = files(staged, hashes=True)
        copied = Snapshot(value["copied"])
        old = previous.entries if previous is not None else {}
        extra = set(target.entries) - set(expected.entries)
        if any(label not in old for label in extra):
            raise StorageError("checkpoint publication has unplanned destination files")

        def identity(entry: dict[str, Any]) -> tuple[int, int]:
            return entry["dev"], entry["ino"]

        def equal_file(first: dict[str, Any], second: dict[str, Any]) -> bool:
            return (
                Snapshot({"file": first}).content()
                == Snapshot({"file": second}).content()
            )

        actions = []
        for label, entry in expected.entries.items():
            if label == ".":
                continue
            current = target.entries.get(label)
            if current is not None and equal_file(current, entry):
                continue
            staged_entry = staged_tree.entries.get(label)
            if (
                staged_entry is None
                or not equal_file(staged_entry, entry)
                or identity(staged_entry) != identity(copied.entries[label])
            ):
                raise StorageError(
                    "checkpoint publication staged SHA-256 differs from verified intent"
                )
            if current is not None and (
                label not in old
                or identity(current) != identity(old[label])
                or not equal_file(current, old[label])
            ):
                raise StorageError("checkpoint publication destination is unverified")
            actions.append((label, current is not None))
        for label in extra:
            if (
                label == "."
                or identity(target.entries[label]) != identity(old[label])
                or not equal_file(target.entries[label], old[label])
            ):
                raise StorageError("checkpoint publication retired file is unverified")
        # All inputs were checked before any remaining publication is resumed.
        archive = evidence(engine, "checkpoint-" + name, value)
        for label, exchange in actions:
            engine.check_mount()
            rename(staged / label, destination / label, exchange=exchange)
            value["published"].append(label)
            write_json(journal, value)
        retired = history / "removed-from-source"
        for label in sorted(extra):
            retired.mkdir(exist_ok=True)
            real_directory(retired, anchor)
            rename(destination / label, retired / label)
            value["archived"].append(label)
            write_json(journal, value)
        # Every superseded inode must remain named in the destination or history.
        for label, entry in old.items():
            if label == ".":
                continue
            candidates = (destination / label, staged / label, retired / label)
            if not any(
                p.exists() and (p.lstat().st_dev, p.lstat().st_ino) == identity(entry)
                for p in candidates
            ):
                raise StorageError(
                    "checkpoint publication lost a previous inode; retaining recovery blocker"
                )
        local = stable(engine, unit, engine.path(unit))
        mirror = stable(engine, unit, engine.path(unit, mirror=True))
        if content(local) != content(expected) or content(mirror) != content(expected):
            raise StorageError(
                "checkpoint publication SHA-256 changed before recovery commit"
            )
        engine.record(unit, local, mirror, dry_run=False)
        value["evidence_archive"] = str(archive)
        complete(journal, value)
        engine.audit(
            "publication_recovered",
            unit,
            operation=value["operation"],
            evidence_archive=str(archive),
            history=str(history),
        )
        results.append(
            {
                "unit": unit.path,
                "action": "publication_recovered",
                "evidence_archive": str(archive),
                "history": str(history),
            }
        )
    return results
