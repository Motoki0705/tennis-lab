"""Journal before copy, verify both readers, then atomically publish a root pointer."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from ...artifacts import (
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from ...contracts import ArtifactRef, content_digest
from ...paths import AnnotationLayout
from ..catalog import StorageCatalog
from ..contracts import InventoryScope, ManagedArtifact, Protection, Retention
from ..inventory import inventory
from .contracts import Bundle, MigrationPlan, MigrationReader, MigrationReceipt
from .files import checked_root, copy_file, verify_file

_SCHEMA = """
CREATE TABLE IF NOT EXISTS storage_migrations (
    plan_digest TEXT PRIMARY KEY,
    plan TEXT NOT NULL,
    receipt TEXT,
    state TEXT NOT NULL CHECK (state IN ('planned', 'published'))
);
CREATE TABLE IF NOT EXISTS storage_imported_roots (
    root_id TEXT PRIMARY KEY,
    plan_digest TEXT NOT NULL REFERENCES storage_migrations(plan_digest)
);
"""


def _source(layout: AnnotationLayout, plan: MigrationPlan) -> Path:
    root = checked_root(Path(plan.source_root))
    if plan.mode == "in_place":
        if root != layout.root:
            raise ValueError("in-place import requires the fixed annotation root")
    elif root.is_relative_to(layout.root) or layout.root.is_relative_to(root):
        raise ValueError("copy import requires disjoint source and annotation roots")
    return root


def _destination(layout: AnnotationLayout, plan: MigrationPlan) -> Path:
    result: Path
    if plan.mode == "in_place":
        result = layout.root
    else:
        result = layout.path("migration_archive", plan.sha256, "root")
    return result


def _copy_tree(
    layout: AnnotationLayout, root: Path, bundle: Bundle, *, sync: bool
) -> None:
    files = {root / item.path for item in bundle.files}
    directories = {root}
    for path in files:
        directories.update(
            parent for parent in path.parents if parent.is_relative_to(root)
        )
    observed = inventory(
        layout,
        (
            InventoryScope(
                role="annotation", path=root.relative_to(layout.root).as_posix()
            ),
        ),
        max_entries=len(files) + len(directories),
    )
    if (
        {Path(item.path) for item in observed.entries if item.kind == "file"} != files
        or {Path(item.path) for item in observed.entries if item.kind == "directory"}
        != directories
        or any(item.kind not in {"file", "directory"} for item in observed.entries)
    ):
        raise ValueError("migration destination contains unplanned entries")
    if sync:
        for directory in sorted(
            directories, key=lambda path: len(path.parts), reverse=True
        ):
            sync_directory(directory)


def _inspect(root: Path, reader: MigrationReader, expected: Bundle) -> None:
    if reader.inspect(root) != expected:
        raise ValueError("migration reader result differs from dry-run")
    # The common layer independently verifies the task reader's file declarations.
    for item in expected.files:
        verify_file(root, item)


def dry_run(
    layout: AnnotationLayout,
    reader: MigrationReader,
    *,
    root_id: str,
    source_root: Path,
    mode: Literal["in_place", "copy"],
    max_files: int,
    expected_previous: str | None = None,
) -> MigrationPlan:
    """Read only: no registry, staging, plan file or old-root changes."""
    checked_root(source_root)
    bundle = reader.inspect(source_root)
    if max_files < 1 or len(bundle.files) > max_files:
        raise ValueError("migration file count exceeds the explicit limit")
    plan = MigrationPlan(
        root_id=root_id,
        source_root=source_root.as_posix(),
        mode=mode,
        expected_previous=expected_previous,
        bundle=bundle,
        copy_payload_bytes=sum(item.size_bytes for item in bundle.files)
        if mode == "copy"
        else 0,
    )
    _source(layout, plan)
    for item in bundle.files:
        verify_file(source_root, item)
    return plan


def _references(
    layout: AnnotationLayout, plan: MigrationPlan
) -> tuple[ArtifactRef, ...]:
    root = _destination(layout, plan)
    references = tuple(
        ArtifactRef(
            role="annotation",
            path=(root / item.path).relative_to(layout.root).as_posix(),
            digest=item.digest,
            size_bytes=item.size_bytes,
        )
        for item in plan.bundle.files
    )
    for reference in references:
        layout.artifact_path(reference)
    return references


def _check_pointer(database: sqlite3.Connection, plan: MigrationPlan) -> None:
    row = database.execute(
        "SELECT plan_digest FROM storage_imported_roots WHERE root_id=?",
        (plan.root_id,),
    ).fetchone()
    previous = row[0] if row else None
    if previous not in (plan.expected_previous, plan.sha256):
        raise ValueError("imported root pointer changed; a new dry-run is required")


class MigrationStore:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.catalog = StorageCatalog(layout)
        self.registry = self.catalog.registry
        self.registry.initialize("annotation-storage-migration-v1", _SCHEMA)

    def apply(
        self,
        plan: MigrationPlan,
        reader: MigrationReader,
        *,
        max_copy_bytes: int,
        max_files: int,
    ) -> ArtifactRef:
        """Explicit execution, suitable for a budget-admitted CPU/I/O stage.

        This API does not grant resource admission, move originals, rewrite legacy
        campaign configurations, or authorize any future cleanup.
        """
        if max_copy_bytes < plan.copy_payload_bytes or max_copy_bytes < 0:
            raise ValueError("migration copy exceeds the explicit payload budget")
        if max_files < 1 or len(plan.bundle.files) > max_files:
            raise ValueError("migration file count exceeds the explicit limit")
        source = _source(self.layout, plan)
        with artifact_lock(self.layout, "migration-" + plan.sha256):
            _inspect(source, reader, plan.bundle)
            plan_ref = write_document(
                self.layout,
                self.layout.path("migration_plans", plan.sha256 + ".json"),
                plan,
            )
            with self.registry.transaction() as database:
                _check_pointer(database, plan)
                database.execute(
                    "INSERT OR IGNORE INTO storage_migrations VALUES (?, ?, NULL, 'planned')",
                    (plan.sha256, plan_ref.canonical_bytes().decode()),
                )
                self.catalog.register(
                    ManagedArtifact(
                        reference=plan_ref,
                        owner=plan.root_id,
                        stage="legacy-import",
                        kind="configuration",
                        regenerable=False,
                        retention=Retention(mode="retain", reason="migration journal"),
                    ),
                    database=database,
                )
            destination = _destination(self.layout, plan)
            if plan.mode == "copy" and not destination.exists():
                staging = self.layout.path("migration_staging", plan.sha256, "root")
                staging.mkdir(parents=True, exist_ok=True)
                for item in plan.bundle.files:
                    copy_file(source, staging, item)
                _copy_tree(self.layout, staging, plan.bundle, sync=True)
                _inspect(staging, reader, plan.bundle)
                # A file change since copying invalidates this plan, never the source.
                _inspect(source, reader, plan.bundle)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(staging, destination)
                sync_directory(destination.parent)
                sync_directory(destination.parent.parent)
                sync_directory(staging.parent)
            # Also recovers a crash after rename and before SQLite publication.
            if plan.mode == "copy":
                _copy_tree(self.layout, destination, plan.bundle, sync=False)
            _inspect(destination, reader, plan.bundle)
            _inspect(source, reader, plan.bundle)
            references = _references(self.layout, plan)
            receipt = MigrationReceipt(
                plan=plan_ref, bundle_digest=plan.bundle.sha256, files=references
            )
            receipt_ref = write_document(
                self.layout,
                self.layout.path("migration_plans", plan.sha256 + ".receipt.json"),
                receipt,
            )
            self._publish(plan, plan_ref, receipt_ref, references)
            return receipt_ref

    def _publish(
        self,
        plan: MigrationPlan,
        plan_ref: ArtifactRef,
        receipt_ref: ArtifactRef,
        references: tuple[ArtifactRef, ...],
    ) -> None:
        mapped = dict(
            zip((item.path for item in plan.bundle.files), references, strict=True)
        )
        retention = Retention(mode="retain", reason="preserved legacy import")
        with self.registry.transaction() as database:
            _check_pointer(database, plan)
            for item in plan.bundle.files:
                self.catalog.register(
                    ManagedArtifact(
                        reference=mapped[item.path],
                        owner="legacy-object-" + content_digest(mapped[item.path].path),
                        stage="legacy-import",
                        kind=item.kind,
                        regenerable=False,
                        retention=retention,
                        dependencies=tuple(mapped[path] for path in item.dependencies),
                    ),
                    database=database,
                )
            self.catalog.register(
                ManagedArtifact(
                    reference=receipt_ref,
                    owner=plan.root_id,
                    stage="legacy-import",
                    kind="adoption_evidence",
                    regenerable=False,
                    retention=retention,
                    dependencies=(plan_ref, *references),
                ),
                database=database,
            )
            self.catalog.protect(
                Protection(
                    protection_id="legacy-import-" + plan.sha256,
                    kind="publication",
                    reason="validated legacy root and its original evidence",
                    references=(receipt_ref,),
                ),
                database=database,
            )
            database.execute(
                "UPDATE storage_migrations SET receipt=?, state='published' WHERE plan_digest=?",
                (receipt_ref.canonical_bytes().decode(), plan.sha256),
            )
            database.execute(
                "INSERT INTO storage_imported_roots VALUES (?, ?) "
                "ON CONFLICT(root_id) DO UPDATE SET plan_digest=excluded.plan_digest",
                (plan.root_id, plan.sha256),
            )


def imported_root(
    layout: AnnotationLayout, root_id: str, reader: MigrationReader
) -> Path:
    """Read a consistent pointer and revalidate its pinned files without writes."""
    with closing(
        sqlite3.connect(layout.registry.as_uri() + "?mode=ro", uri=True)
    ) as database:
        row = database.execute(
            "SELECT m.plan, m.receipt FROM storage_imported_roots r "
            "JOIN storage_migrations m ON r.plan_digest=m.plan_digest "
            "WHERE r.root_id=? AND m.state='published'",
            (root_id,),
        ).fetchone()
    if row is None:
        raise ValueError("legacy root has no completed import")
    plan_ref = ArtifactRef.model_validate_json(row[0])
    receipt_ref = ArtifactRef.model_validate_json(row[1])
    plan = read_document(layout, plan_ref, MigrationPlan)
    receipt = read_document(layout, receipt_ref, MigrationReceipt)
    if (
        plan.root_id != root_id
        or receipt.plan != plan_ref
        or receipt.bundle_digest != plan.bundle.sha256
        or receipt.files != _references(layout, plan)
    ):
        raise ValueError("legacy import receipt does not match its plan")
    for reference in receipt.files:
        verify_reference(layout, reference)
    root = _destination(layout, plan)
    if plan.mode == "copy":
        _copy_tree(layout, root, plan.bundle, sync=False)
    _inspect(root, reader, plan.bundle)
    return root
