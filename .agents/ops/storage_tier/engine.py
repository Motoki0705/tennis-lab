"""Mirror, hydrate and evict whole units with durable verification evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .filesystem import (
    Snapshot,
    StorageError,
    append_json,
    copy_tree,
    names,
    read_json,
    real_directory,
    rename,
    snapshot,
    sync_dir,
    write_json,
)
from .layout import Config, Unit, discover
from .usage import Inspector

IDLE_NS = 14 * 24 * 3600 * 10**9


class BusyUnit(StorageError):
    """Expected temporary use, distinct from unverifiable safety conditions."""


def mounted(config: Config) -> None:
    check = subprocess.run(["mountpoint", "-q", str(config.mount)], check=False)
    if check.returncode != 0:
        raise StorageError(
            f"E: is not mounted at {config.mount}; no mirror or deletion is permitted"
        )
    if config.mount.stat().st_dev == config.repo.stat().st_dev:
        raise StorageError("D: and E: must be different filesystems")
    if config.repo.resolve() != config.repo or config.mount.resolve() != config.mount:
        raise StorageError("storage roots must not traverse symlinks")
    for path, required in ((config.repo, "relatime"), (config.mount, "noatime")):
        info = (
            subprocess.check_output(
                ["findmnt", "-T", str(path), "-n", "-o", "FSTYPE,OPTIONS"], text=True
            )
            .strip()
            .split()
        )
        if len(info) != 2 or info[0] != "ext4" or required not in info[1].split(","):
            raise StorageError(f"{path} must use ext4 with {required}")
    if config.mirror.exists():
        real_directory(config.mirror, config.mount)


class Engine:
    def __init__(
        self,
        config: Config,
        *,
        inspector: Inspector | None = None,
        mount_check: Callable[[Config], None] = mounted,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.inspector = inspector or Inspector(config.repo / ".training_queue")
        self.mount_check = mount_check
        self.clock = clock

    def timestamp(self) -> str:
        return datetime.fromtimestamp(self.clock(), UTC).isoformat()

    @property
    def now_ns(self) -> int:
        return int(self.clock() * 10**9)

    def check_mount(self) -> None:
        self.mount_check(self.config)

    def manifest_path(self, unit: Unit) -> Path:
        key = hashlib.sha256(unit.path.encode()).hexdigest()
        return self.config.mirror / ".storage-tier/manifests" / f"{key}.json"

    def manifest(self, unit: Unit) -> dict[str, Any] | None:
        path = self.manifest_path(unit)
        if not path.exists():
            return None
        value = read_json(path)
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("algorithm") != "sha256"
            or value.get("unit") != unit.path
            or value.get("kind") != unit.kind
            or not isinstance(value.get("mirror"), dict)
            or "." not in value["mirror"]
        ):
            raise StorageError(f"invalid manifest: {path}")
        for name, entry in value["mirror"].items():
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not isinstance(entry, dict)
            ):
                raise StorageError(f"invalid manifest entry: {path}")
            if entry.get("kind") not in {"file", "dir"} or not all(
                type(entry.get(key)) is int
                for key in (
                    "dev",
                    "ino",
                    "size",
                    "mtime_ns",
                    "ctime_ns",
                    "atime_ns",
                    "mode",
                    "nlink",
                    "blocks",
                )
            ):
                raise StorageError(f"invalid manifest metadata: {path}")
            if entry["kind"] == "file" and (
                not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
            ):
                raise StorageError(f"invalid manifest digest: {path}")
        return value

    def record(
        self, unit: Unit, local: Snapshot | None, mirror: Snapshot, *, dry_run: bool
    ) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "unit": unit.path,
            "kind": unit.kind,
            "verified_at": self.timestamp(),
            "algorithm": "sha256",
            "local": local.entries if local is not None else None,
            "mirror": mirror.entries,
        }
        if not dry_run:
            self.check_mount()
            write_json(self.manifest_path(unit), value)
            self.audit(
                "verified",
                unit,
                bytes=mirror.bytes,
                algorithm="sha256",
                both_sides=local is not None,
            )
        return value

    def audit(self, event: str, unit: Unit, **details: Any) -> None:
        append_json(
            self.config.state / "audit.jsonl",
            {"event": event, "unit": unit.path, "time": self.timestamp(), **details},
        )

    @contextmanager
    def locked(self, *, dry_run: bool) -> Iterator[None]:
        self.check_mount()  # Never create even a lock when E: is missing.
        if dry_run:
            yield
            return
        self.config.state.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(
            self.config.state / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "w") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise StorageError("another storage operation is running") from exc
            yield

    def path(self, unit: Unit, *, mirror: bool = False) -> Path:
        base = self.config.mirror if mirror else self.config.repo
        path = base / unit.path
        try:
            real_directory(path, base)
        except FileNotFoundError:
            if not mirror and (self.config.mirror / unit.path).exists():
                raise StorageError(
                    f"unit is archived on E: and missing on D: {unit.path}; use storage_tier hydrate {unit.path}"
                ) from None
            raise
        return path

    def quiet(
        self, unit: Unit, path: Path, tree: Snapshot, *, moved: bool = False
    ) -> None:
        usage = self.inspector.inspect((self.config.repo / unit.path, path), tree)
        if usage.unknown:
            raise StorageError("usage cannot be verified: " + "; ".join(usage.unknown))
        if usage.busy:
            raise BusyUnit("; ".join(usage.busy))
        newest_write = max(
            max(
                e["mtime_ns"],
                e["ctime_ns"] if not moved or name != "." else e["mtime_ns"],
            )
            for name, e in tree.entries.items()
        )
        if newest_write > self.now_ns - self.config.settle_seconds * 10**9:
            raise BusyUnit("unit has recent writes; waiting for the settling interval")

    def assert_idle(self, tree: Snapshot) -> None:
        last = tree.last_access_ns
        if last is None:
            raise BusyUnit("empty unit has no regular-file atime; keeping it")
        if last > self.now_ns - IDLE_NS:
            raise BusyUnit("regular-file atime is less than 14 days old")

    def check_mirror(self, unit: Unit, manifest: dict[str, Any]) -> Snapshot:
        self.check_mount()
        observed = snapshot(self.path(unit, mirror=True))
        if not Snapshot(manifest["mirror"]).same_metadata(observed):
            raise StorageError(
                f"E: size/mtime/ctime or tree changed since verification: {unit.path}"
            )
        return observed

    def ensure_parent(self, path: Path, base: Path) -> None:
        if base == self.config.mirror and not base.exists():
            self.ensure_parent(base, self.config.mount)
        current = base
        for part in path.relative_to(base).parts:
            if not current.exists():
                current.mkdir()
            real_directory(current, base)
            current /= part
        if not current.exists():
            current.mkdir()
        real_directory(current, base)

    def index(self, unit: Unit, *, dry_run: bool) -> dict[str, Any]:
        path = self.path(unit, mirror=True)
        before = snapshot(path)
        if not dry_run:
            self.quiet(unit, path, before)
        tree = snapshot(path, hashes=True)
        if not before.same_metadata(tree):
            raise StorageError("E: changed while indexing")
        if not dry_run:
            self.quiet(unit, path, tree)
        manifest = self.record(unit, None, tree, dry_run=dry_run)
        digest = hashlib.sha256(
            json.dumps(tree.content(), sort_keys=True).encode()
        ).hexdigest()
        preserved = all(
            entry["atime_ns"] == tree.entries[name]["atime_ns"]
            for name, entry in before.entries.items()
        )
        return {
            "action": "would_index" if dry_run else "indexed",
            "bytes": tree.bytes,
            "files": sum(e["kind"] == "file" for e in tree.entries.values()),
            "sha256_tree": digest,
            "atime_preserved": preserved,
            "manifest": manifest,
        }

    def verify(self, unit: Unit, *, dry_run: bool) -> dict[str, Any]:
        local_path, mirror_path = self.path(unit), self.path(unit, mirror=True)
        local_before = snapshot(local_path)
        self.quiet(unit, local_path, local_before)
        mirror_before = snapshot(mirror_path)
        self.quiet(unit, mirror_path, mirror_before)
        local, mirror = (
            snapshot(local_path, hashes=True),
            snapshot(mirror_path, hashes=True),
        )
        if (
            not local_before.same_metadata(local)
            or not mirror_before.same_metadata(mirror)
            or local.content() != mirror.content()
        ):
            raise StorageError(f"SHA-256/tree mismatch: {unit.path}")
        if not local.same_metadata(snapshot(local_path)) or not mirror.same_metadata(
            snapshot(mirror_path)
        ):
            raise StorageError("unit changed during verification")
        self.quiet(unit, local_path, local)
        self.quiet(unit, mirror_path, mirror)
        manifest = self.record(unit, local, mirror, dry_run=dry_run)
        return {
            "action": "would_verify" if dry_run else "verified",
            "bytes": local.bytes,
            "manifest": manifest,
        }

    def mirror(self, unit: Unit, *, dry_run: bool) -> dict[str, Any]:
        source = self.path(unit)
        before = snapshot(source)
        self.quiet(unit, source, before)
        destination = self.config.mirror / unit.path
        old_manifest = self.manifest(unit)
        old: Snapshot | None = None
        if os.path.lexists(destination):
            old = snapshot(self.path(unit, mirror=True))
            self.quiet(unit, destination, old)
            if old_manifest is not None:
                self.check_mirror(unit, old_manifest)
                if old_manifest.get("local") is not None and Snapshot(
                    old_manifest["local"]
                ).same_metadata(before):
                    return {
                        "action": "unchanged",
                        "bytes": before.bytes,
                        "manifest": old_manifest,
                    }
        local = snapshot(source, hashes=True)
        if not before.same_metadata(local):
            raise StorageError("D: changed while mirroring")
        if old is not None:
            mirror = snapshot(destination, hashes=True)
            if not old.same_metadata(mirror):
                raise StorageError("E: changed while mirroring")
            if local.content() == mirror.content():
                self.quiet(unit, source, snapshot(source))
                self.quiet(unit, destination, mirror)
                if not local.same_metadata(snapshot(source)):
                    raise StorageError("D: changed while verifying E:")
                manifest = self.record(unit, local, mirror, dry_run=dry_run)
                return {
                    "action": "would_verify" if dry_run else "verified",
                    "bytes": local.bytes,
                    "manifest": manifest,
                }
            if old_manifest is None:
                raise StorageError(
                    "unrecorded E: unit differs from D:; refusing to overwrite the authoritative copy"
                )
        if dry_run:
            return {
                "action": "would_mirror",
                "bytes": local.bytes,
                "eviction": "requires a published and verified mirror",
            }
        self.check_mount()
        self.ensure_parent(destination.parent, self.config.mirror)
        stage = destination.with_name(f".storage-tier-stage-{uuid.uuid4().hex}")
        published = False
        try:
            copy_tree(source, stage, local)
            copied = snapshot(stage, hashes=True)
            if copied.content() != local.content() or not local.same_metadata(
                snapshot(source)
            ):
                raise StorageError(
                    "copy/source failed SHA-256 or stability verification"
                )
            self.quiet(unit, source, snapshot(source))
            self.check_mount()
            if old is not None:
                if not old.same_metadata(snapshot(destination)):
                    raise StorageError("E: changed before publication")
                self.quiet(unit, destination, old)
            rename(stage, destination, exchange=old is not None)
            published = True
            final = snapshot(destination)
            for name, entry in final.entries.items():
                if entry["kind"] == "file":
                    entry["sha256"] = copied.entries[name]["sha256"]
            manifest = self.record(unit, local, final, dry_run=False)
            if old is not None:
                shutil.rmtree(stage)
                sync_dir(stage.parent)
            return {"action": "mirrored", "bytes": local.bytes, "manifest": manifest}
        finally:
            # After an exchange, preserve the old mirror if manifest publication
            # failed; its named staging path is reported by discovery on retry.
            if not published and stage.exists():
                shutil.rmtree(stage)

    def hydrate(self, unit: Unit, *, dry_run: bool) -> dict[str, Any]:
        destination = self.config.repo / unit.path
        with suppress(FileNotFoundError):
            real_directory(destination.parent, self.config.repo)
        if os.path.lexists(destination):
            raise StorageError(
                f"D: destination already exists; refusing to merge or overwrite: {destination}"
            )
        source = self.path(unit, mirror=True)
        before = snapshot(source)
        self.quiet(unit, source, before)
        previous = self.manifest(unit)
        if previous is not None:
            self.check_mirror(unit, previous)
        mirror = snapshot(source, hashes=True)
        if not before.same_metadata(mirror):
            raise StorageError("E: changed during hydrate")
        if (
            previous is not None
            and Snapshot(previous["mirror"]).content() != mirror.content()
        ):
            raise StorageError("E: SHA-256 does not match its recorded manifest")
        if dry_run:
            return {"action": "would_hydrate", "bytes": mirror.bytes}
        self.check_mount()
        self.ensure_parent(destination.parent, self.config.repo)
        stage = destination.with_name(f".storage-tier-hydrate-{uuid.uuid4().hex}")
        try:
            copy_tree(source, stage, mirror, now_ns=self.now_ns)
            local = snapshot(stage, hashes=True)
            if local.content() != mirror.content() or not mirror.same_metadata(
                snapshot(source)
            ):
                raise StorageError("hydrate copy/source verification failed")
            self.quiet(unit, source, snapshot(source))
            self.check_mount()
            rename(stage, destination)
            final = snapshot(destination)
            for name, entry in final.entries.items():
                if entry["kind"] == "file":
                    entry["sha256"] = local.entries[name]["sha256"]
            self.record(unit, final, mirror, dry_run=False)
            self.audit(
                "hydrated", unit, bytes=mirror.bytes, verification="sha256-both-sides"
            )
            return {"action": "hydrated", "bytes": mirror.bytes}
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def evict(
        self, unit: Unit, *, dry_run: bool, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if evidence is not None and not dry_run:
            raise StorageError("unpublished manifests may only be used in a dry-run")
        if not unit.sweepable:
            return {"action": "kept", "reason": "ckpt is mirror-only"}
        source = self.path(unit)
        before = snapshot(source)
        self.assert_idle(before)
        self.quiet(unit, source, before)
        manifest = evidence if evidence is not None else self.manifest(unit)
        if manifest is None:
            raise StorageError(
                f"no verified mirror manifest: {unit.path}; run mirror first"
            )
        self.check_mirror(unit, manifest)
        local = snapshot(source, hashes=True)
        if (
            not before.same_metadata(local)
            or local.content() != Snapshot(manifest["mirror"]).content()
        ):
            raise StorageError(
                f"D: SHA-256/tree mismatch against manifest: {unit.path}"
            )
        final = snapshot(source)
        if not local.same_metadata(final):
            raise StorageError("D: changed before eviction")
        self.assert_idle(final)
        self.quiet(unit, source, final)
        self.check_mirror(unit, manifest)
        detail = {
            **final.allocation(),
            "verification": "D-sha256+E-size-mtime-ctime",
            "manifest_verified_at": manifest["verified_at"],
            "last_access_ns": final.last_access_ns,
        }
        if dry_run:
            return {"action": "would_evict", **detail}
        transaction_id = uuid.uuid4().hex
        quarantine = source.with_name(f".storage-tier-trash-{transaction_id}")
        transaction = self.config.state / "transactions" / f"{transaction_id}.json"
        record = {
            "schema_version": 1,
            "unit": unit.path,
            "kind": unit.kind,
            "quarantine": str(quarantine.relative_to(self.config.repo)),
            "dev": final.entries["."]["dev"],
            "ino": final.entries["."]["ino"],
            "phase": "prepared",
            **detail,
        }
        write_json(transaction, record)
        free_before = os.statvfs(source).f_bavail * os.statvfs(source).f_frsize
        try:
            rename(source, quarantine)
            moved = snapshot(quarantine)
            if not final.same_metadata(moved, moved=True):
                raise StorageError("D: unit was replaced or changed during quarantine")
            self.assert_idle(moved)
            self.quiet(unit, quarantine, moved, moved=True)
            self.check_mirror(unit, manifest)
            record["phase"] = "verified"
            write_json(transaction, record)
            self.audit("deletion_verified", unit, transaction=transaction_id, **detail)
            shutil.rmtree(quarantine)
            sync_dir(quarantine.parent)
            free_after = (
                os.statvfs(self.config.repo).f_bavail
                * os.statvfs(self.config.repo).f_frsize
            )
            result = {
                "action": "evicted",
                "observed_free_delta_bytes": free_after - free_before,
                **detail,
            }
            self.audit("deleted", unit, transaction=transaction_id, **result)
            record["phase"] = "complete"
            write_json(transaction, record)
            transaction.unlink()
            sync_dir(transaction.parent)
            return result
        except BaseException:
            # Restore only our pinned unit. A successor at the original name is
            # preserved, and an occupied origin leaves a recoverable quarantine.
            if quarantine.exists():
                with suppress(OSError):
                    rename(quarantine, source)
            raise

    def recover(self) -> list[dict[str, Any]]:
        directory = self.config.state / "transactions"
        results: list[dict[str, Any]] = []
        if not directory.exists():
            return results
        for name in names(directory):
            value = read_json(directory / name)
            if value.get("schema_version") != 1 or value.get("phase") not in {
                "prepared",
                "verified",
                "complete",
            }:
                raise StorageError(
                    "invalid deletion transaction; manual inspection required"
                )
            unit = Unit(value["unit"], value["kind"])
            if unit not in discover(self.config, self.config.mirror).units:
                raise StorageError(
                    "transaction unit is not an independently discovered E: unit"
                )
            source = self.config.repo / unit.path
            quarantine = self.config.repo / value["quarantine"]
            real_directory(source.parent, self.config.repo)
            if quarantine.parent != source.parent or not quarantine.name.startswith(
                ".storage-tier-trash-"
            ):
                raise StorageError("invalid transaction quarantine")
            if quarantine.exists():
                info = quarantine.lstat()
                if (info.st_dev, info.st_ino) != (value["dev"], value["ino"]):
                    rename(quarantine, source)
                    self.audit("successor_restored", unit, transaction=name)
                    (directory / name).unlink()
                    sync_dir(directory)
                    raise StorageError(
                        "a successor was quarantined during a race; restored without deletion"
                    )
                rename(quarantine, source)
                action = "restored_after_interruption"
            elif source.exists():
                action = "kept_after_interruption"
            elif value["phase"] in {"verified", "complete"}:
                action = "removed_before_interruption"
            else:
                raise StorageError(
                    "prepared transaction has neither source nor quarantine"
                )
            self.audit(
                "deletion_recovery",
                unit,
                action=action,
                transaction=name,
                prior_phase=value["phase"],
            )
            (directory / name).unlink()
            sync_dir(directory)
            results.append(
                {
                    "unit": unit.path,
                    "action": action,
                    "partial_deletion_possible": value["phase"] == "verified",
                }
            )
        return results

    def run(
        self, command: str, requested: list[str], *, dry_run: bool
    ) -> tuple[int, dict[str, Any]]:
        if command == "cycle" and not dry_run and not requested:
            marker_path = self.config.state / "active.json"
            if not marker_path.exists():
                raise StorageError(
                    "automatic cycle requires activate after migration and host validation"
                )
            marker = read_json(marker_path)
            if marker != {
                "schema_version": 1,
                "repository": str(self.config.repo),
                "mirror_mount": str(self.config.mount),
            }:
                raise StorageError(
                    "automatic cycle activation does not match this repository and mount"
                )
        with self.locked(dry_run=dry_run or command == "scan"):
            local = discover(self.config, self.config.repo)
            mirror = discover(self.config, self.config.mirror)
            inventory = {u.path: u for u in [*local.units, *mirror.units]}
            report: dict[str, Any] = {
                "command": command,
                "dry_run": dry_run or command == "scan",
                "time": self.timestamp(),
                "repo": str(self.config.repo),
                "mirror": str(self.config.mirror),
                "layout_errors": local.errors + mirror.errors,
                "unmanaged": {"D": local.unmanaged, "E": mirror.unmanaged},
                "results": [],
            }
            if command == "scan":
                usage = self.inspector.inspect((), Snapshot({}))
                report["usage"] = {"busy": usage.busy, "unknown": usage.unknown}
                report["units"] = [
                    {
                        "unit": u.path,
                        "kind": u.kind,
                        "local": u in local.units,
                        "mirror": u in mirror.units,
                    }
                    for u in sorted(inventory.values())
                ]
                return int(bool(report["layout_errors"] or usage.unknown)), report
            if command == "recover":
                if dry_run:
                    raise StorageError(
                        "recover changes transaction state; inspect scan/audit first"
                    )
                report["results"] = self.recover()
                return 0, report
            transactions = self.config.state / "transactions"
            if transactions.exists() and names(transactions):
                raise StorageError(
                    "unfinished deletion transactions; run recover before continuing"
                )
            if not requested and report["layout_errors"]:
                raise StorageError(
                    "layout has errors; inspect scan before selecting units"
                )
            selected: list[Unit] = []
            for path in requested:
                candidate = Path(path)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise StorageError(f"unit must be repo-relative: {path}")
                matches = [
                    u for u in inventory.values() if candidate.is_relative_to(u.path)
                ]
                if len(matches) != 1:
                    raise StorageError(
                        f"unknown/ambiguous unit: {path}; inspect scan and dataset boundaries"
                    )
                if matches[0] not in selected:
                    selected.append(matches[0])
            if not requested:
                selected = (
                    mirror.units if command in {"index", "hydrate"} else local.units
                )
            failed = False
            for unit in selected:
                try:
                    if command == "cycle":
                        if unit.sweepable and self.manifest(unit) is not None:
                            try:
                                self.assert_idle(snapshot(self.path(unit)))
                            except BusyUnit:
                                pass
                            else:
                                result = self.evict(unit, dry_run=dry_run)
                                report["results"].append({"unit": unit.path, **result})
                                continue
                        result = self.mirror(unit, dry_run=dry_run)
                        proposed = result.pop("manifest", None)
                        if result["action"] == "would_mirror":
                            result["next"] = (
                                "verify the published mirror before eviction"
                            )
                        else:
                            try:
                                result["eviction"] = self.evict(
                                    unit,
                                    dry_run=dry_run,
                                    evidence=proposed if dry_run else None,
                                )
                            except BusyUnit as exc:
                                result["eviction"] = {
                                    "action": "kept",
                                    "reason": str(exc),
                                }
                    else:
                        method = {
                            "index": self.index,
                            "mirror": self.mirror,
                            "verify": self.verify,
                            "hydrate": self.hydrate,
                            "evict": self.evict,
                        }[command]
                        result = method(unit, dry_run=dry_run)
                    result.pop("manifest", None)
                    report["results"].append({"unit": unit.path, **result})
                except BusyUnit as exc:
                    report["results"].append(
                        {"unit": unit.path, "action": "kept", "reason": str(exc)}
                    )
                except (OSError, ValueError, StorageError) as exc:
                    failed = True
                    report["results"].append(
                        {"unit": unit.path, "action": "failed", "error": str(exc)}
                    )
                    if not dry_run:
                        self.audit("failed", unit, operation=command, error=str(exc))
            return int(failed), report
