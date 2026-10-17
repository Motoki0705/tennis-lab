"""Explicit, journalled collection of expired, catalog-managed annotation files.

Retirement commits before unlink. Every reference writer uses the same registry
transaction and refuses retired paths, including during interruption recovery.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..artifacts import artifact_lock, read_document, write_document
from ..contracts import ArtifactRef, Document
from ..paths import AnnotationLayout
from .catalog import StorageCatalog
from .contracts import GCPlan, Inventory, InventoryEntry
from .inventory import _scope_directory
from .planner import plan_gc

_SCHEMA = """
CREATE TABLE IF NOT EXISTS storage_gc_jobs (
    request_digest TEXT PRIMARY KEY,
    request TEXT NOT NULL,
    receipt TEXT
);
"""


class GCRequest(Document):
    """A separate, explicit decision about one saved dry-run, never a TTL default."""

    inventory: ArtifactRef
    plan: ArtifactRef
    reason: str = Field(min_length=1)


class GCReceipt(Document):
    schema_version: Literal["annotation.storage.gc-receipt.v1"] = (
        "annotation.storage.gc-receipt.v1"
    )
    request: ArtifactRef
    removed: tuple[ArtifactRef, ...]
    unlinked_logical_bytes: int = Field(ge=0)
    # An inode estimate from the reviewed dry-run, not measured disk free space.
    planned_reclaimable_allocated_bytes: int = Field(ge=0)


class GCStore:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.catalog = StorageCatalog(layout)
        self.registry = self.catalog.registry
        self.registry.initialize("annotation-storage-gc-v1", _SCHEMA)

    def execute(self, request: GCRequest, *, max_entries: int) -> ArtifactRef:
        """Collect one explicitly reviewed plan; replay its journal after failure.

        This API does not grant resource admission or discover legacy consumers.
        Consumers of managed files must pin their references before reading them.
        Unknown files and files outside the annotation root are never removed.
        """
        if not request.reason.strip():
            raise ValueError("GC requires an explicit deletion decision reason")
        observed = read_document(self.layout, request.inventory, Inventory)
        plan = read_document(self.layout, request.plan, GCPlan)
        if plan.inventory_digest != observed.sha256:
            raise ValueError("GC plan and inventory do not match")
        if (
            request.inventory.role != "annotation"
            or request.plan.role != "annotation"
            or self.layout.artifact_path(request.inventory)
            != self.layout.path("storage_inventories", observed.sha256 + ".json")
            or self.layout.artifact_path(request.plan)
            != self.layout.path("storage_plans", plan.sha256 + ".json")
        ):
            raise ValueError("GC requires the evidence saved by save_dry_run")
        if max_entries < 1 or len(observed.entries) > max_entries:
            raise ValueError("GC inventory exceeds the explicit entry budget")
        # GC jobs also exclude each other across plans. Reference writers only
        # need the registry transaction; no scheduler recovery or TTL release.
        with artifact_lock(self.layout, "storage-gc"):
            request_ref = write_document(
                self.layout,
                self.layout.path("storage_gc", request.sha256, "request.json"),
                request,
            )
            self._retire(request, observed, plan)
            with self.registry.transaction() as database:
                rows = database.execute(
                    "SELECT path FROM storage_retired WHERE collection_id=? ORDER BY path",
                    (request.sha256,),
                ).fetchall()
            for row in rows:
                self._remove(request.sha256, row[0])
            receipt = self._finish(request, request_ref, plan)
            return write_document(
                self.layout,
                self.layout.path("storage_gc", request.sha256, "receipt.json"),
                receipt,
            )

    def _retire(self, request: GCRequest, observed: Inventory, plan: GCPlan) -> None:
        with self.registry.transaction() as database:
            previous = database.execute(
                "SELECT request FROM storage_gc_jobs WHERE request_digest=?",
                (request.sha256,),
            ).fetchone()
            if previous is not None:
                if previous[0] != request.canonical_bytes().decode():
                    raise ValueError("GC request identity changed")
                return
            # The durable decision remains valid if wall time moves backwards
            # during recovery; only a new retirement uses the current clock.
            if plan.created_at_ns > time.time_ns():
                raise ValueError("GC cannot execute a future-dated dry-run")
            snapshot = self.catalog.snapshot(database=database)
            if snapshot.digest != plan.catalog_digest:
                raise ValueError(
                    "storage references changed; a new dry-run is required"
                )
            # Recompute while holding the same write transaction as protect/register.
            # This also rejects fabricated eligible rows and altered file metadata.
            checked = plan_gc(
                self.catalog, observed, now_ns=plan.created_at_ns, database=database
            )
            if (
                tuple(row for row in checked.decisions if row.eligible)
                != tuple(row for row in plan.decisions if row.eligible)
                or checked.eligible_logical_bytes != plan.eligible_logical_bytes
                or checked.eligible_allocated_bytes != plan.eligible_allocated_bytes
            ):
                raise ValueError("storage files or decisions changed; repeat dry-run")
            entries = {entry.path: entry for entry in observed.entries}
            selected = [row.path for row in plan.decisions if row.eligible]
            if not selected:
                raise ValueError("GC plan contains no eligible files")
            for path in selected:
                artifact = snapshot.artifacts[path]
                self._target(artifact.reference)
                entry = entries[path]
                if entry.kind != "file":
                    raise ValueError("GC only accepts regular files")
                database.execute(
                    "INSERT INTO storage_retired VALUES (?, ?, ?, ?, 'pending')",
                    (
                        path,
                        request.sha256,
                        artifact.reference.canonical_bytes().decode(),
                        entry.canonical_bytes().decode(),
                    ),
                )
            database.execute(
                "INSERT INTO storage_gc_jobs VALUES (?, ?, NULL)",
                (request.sha256, request.canonical_bytes().decode()),
            )

    def _target(self, reference: ArtifactRef) -> Path:
        if reference.role != "annotation":
            raise ValueError("GC only operates inside the fixed annotation root")
        path: Path = self.layout.artifact_path(reference)
        relative = path.relative_to(self.layout.root)
        if (
            len(relative.parts) < 2
            or relative.parts[0] == "storage"
            or relative.parts[:2] == ("cache", "locks")
        ):
            raise ValueError("GC cannot retire registry, locks or storage evidence")
        return path

    def _remaining_links(
        self, database: sqlite3.Connection, collection_id: str, entry: InventoryEntry
    ) -> int:
        missing = 0
        rows = database.execute(
            "SELECT reference, entry, state FROM storage_retired WHERE collection_id=?",
            (collection_id,),
        ).fetchall()
        for raw_reference, raw_entry, state in rows:
            other = InventoryEntry.model_validate_json(raw_entry)
            if (other.device, other.inode) != (entry.device, entry.inode):
                continue
            path = self._target(ArtifactRef.model_validate_json(raw_reference))
            with _scope_directory(path.parent) as parent:
                try:
                    os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    if state == "pending":
                        raise ValueError(
                            "GC candidate disappeared before removal intent"
                        ) from None
                    missing += 1
                else:
                    if state == "deleted":
                        raise ValueError("a retired GC path was recreated")
        remaining: int = entry.links - missing
        return remaining

    def _remove(self, collection_id: str, path: str) -> None:
        # The durable intent distinguishes our unlink/commit gap from an
        # unexpectedly absent pending file. A killed process never releases pins.
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT reference, entry, state FROM storage_retired WHERE path=? AND collection_id=?",
                (path, collection_id),
            ).fetchone()
            if row is None:
                raise ValueError("GC retirement is missing")
            reference = ArtifactRef.model_validate_json(row[0])
            entry = InventoryEntry.model_validate_json(row[1])
            target = self._target(reference)
            if row[2] == "deleted":
                with _scope_directory(target.parent) as parent:
                    try:
                        os.stat(target.name, dir_fd=parent, follow_symlinks=False)
                    except FileNotFoundError:
                        return
                    raise ValueError("a retired GC path was recreated")
            if row[2] == "pending":
                with _scope_directory(target.parent) as parent:
                    self._verify(
                        parent,
                        target.name,
                        reference,
                        entry,
                        self._remaining_links(database, collection_id, entry),
                    )
                database.execute(
                    "UPDATE storage_retired SET state='removing' WHERE path=?", (path,)
                )
        with (
            self.registry.transaction() as database,
            _scope_directory(target.parent) as parent,
        ):
            try:
                os.stat(target.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                # Intent was committed above; unlink may have succeeded while
                # its completion transaction was rolled back by process death.
                pass
            else:
                self._verify(
                    parent,
                    target.name,
                    reference,
                    entry,
                    self._remaining_links(database, collection_id, entry),
                )
                # A pinned fd must still name the authorized parent, not a
                # directory moved elsewhere while content verification ran.
                with _scope_directory(target.parent) as current_parent:
                    pinned = os.fstat(parent)
                    current = os.fstat(current_parent)
                    if (pinned.st_dev, pinned.st_ino) != (
                        current.st_dev,
                        current.st_ino,
                    ):
                        raise ValueError("GC parent directory changed")
                os.unlink(target.name, dir_fd=parent)
            os.fsync(parent)
            database.execute(
                "UPDATE storage_retired SET state='deleted' WHERE path=?", (path,)
            )

    @staticmethod
    def _verify(
        parent: int,
        name: str,
        reference: ArtifactRef,
        entry: InventoryEntry,
        remaining_links: int,
    ) -> None:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or (
                before.st_dev,
                before.st_ino,
                before.st_nlink,
                before.st_size,
                before.st_blocks * 512,
                before.st_mtime_ns,
            ) != (
                entry.device,
                entry.inode,
                remaining_links,
                entry.logical_bytes,
                entry.allocated_bytes,
                entry.modified_ns,
            ):
                raise ValueError("GC candidate identity, contents or links changed")
            if hashlib.file_digest(source, "sha256").hexdigest() != reference.digest:
                raise ValueError("GC candidate hash changed")
            after = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_nlink,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            ):
                raise ValueError("GC candidate changed during verification")

    def _finish(
        self, request: GCRequest, request_ref: ArtifactRef, plan: GCPlan
    ) -> GCReceipt:
        with self.registry.transaction() as database:
            previous = database.execute(
                "SELECT receipt FROM storage_gc_jobs WHERE request_digest=?",
                (request.sha256,),
            ).fetchone()
            if previous[0] is not None:
                saved: GCReceipt = GCReceipt.model_validate_json(previous[0])
                return saved
            rows = database.execute(
                "SELECT reference, state FROM storage_retired WHERE collection_id=? ORDER BY path",
                (request.sha256,),
            ).fetchall()
            if any(row[1] != "deleted" for row in rows):
                raise ValueError("GC has incomplete removals")
            receipt = GCReceipt(
                request=request_ref,
                removed=tuple(ArtifactRef.model_validate_json(row[0]) for row in rows),
                unlinked_logical_bytes=plan.eligible_logical_bytes,
                planned_reclaimable_allocated_bytes=plan.eligible_allocated_bytes,
            )
            database.execute(
                "UPDATE storage_gc_jobs SET receipt=? WHERE request_digest=?",
                (receipt.canonical_bytes().decode(), request.sha256),
            )
            return receipt
