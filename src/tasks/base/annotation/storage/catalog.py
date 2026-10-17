"""Reference registration in the same SQLite database as business state.

Callers can pass their transaction to keep a publication/training pointer and its
protection atomic. Construct the catalog before entering that transaction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass

from ..artifacts import verify_reference
from ..contracts import ArtifactRef, content_digest
from ..paths import AnnotationLayout
from ..registry import Registry
from .contracts import ManagedArtifact, Protection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS storage_artifacts (
    path TEXT PRIMARY KEY,
    document TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS storage_protections (
    protection_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    released_reason TEXT
);
"""


@dataclass(frozen=True)
class CatalogSnapshot:
    artifacts: dict[str, ManagedArtifact]
    protections: tuple[Protection, ...]
    digest: str


class StorageCatalog:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.registry = Registry(layout)
        self.registry.initialize("annotation-storage-v1", _SCHEMA)

    @contextmanager
    def _transaction(
        self, database: sqlite3.Connection | None
    ) -> Iterator[sqlite3.Connection]:
        if database is not None:
            if not database.in_transaction:
                raise ValueError("registration requires an active transaction")
            paths = {row[1]: row[2] for row in database.execute("PRAGMA database_list")}
            if paths.get("main") != str(self.layout.registry):
                raise ValueError("registration transaction belongs to another registry")
            yield database
        else:
            with self.registry.transaction() as connection:
                yield connection

    def key(self, reference: ArtifactRef) -> str:
        # Role aliases for the same path must not create separate graph nodes.
        return str(self.layout.artifact_path(reference))

    def register(
        self, artifact: ManagedArtifact, *, database: sqlite3.Connection | None = None
    ) -> None:
        for reference in (artifact.reference, *artifact.dependencies):
            verify_reference(self.layout, reference)
        path = self.key(artifact.reference)
        dependencies = [self.key(item) for item in artifact.dependencies]
        if path in dependencies or len(dependencies) != len(set(dependencies)):
            raise ValueError("artifact dependencies must be distinct other paths")
        document = artifact.canonical_bytes().decode()
        with self._transaction(database) as connection:
            old = connection.execute(
                "SELECT document FROM storage_artifacts WHERE path=?", (path,)
            ).fetchone()
            if old is not None and old[0] != document:
                raise ValueError(
                    "storage ownership/retention registration is immutable"
                )
            connection.execute(
                "INSERT OR IGNORE INTO storage_artifacts VALUES (?, ?)",
                (path, document),
            )

    def protect(
        self, protection: Protection, *, database: sqlite3.Connection | None = None
    ) -> None:
        for reference in protection.references:
            verify_reference(self.layout, reference)
        paths = [self.key(item) for item in protection.references]
        if len(paths) != len(set(paths)):
            raise ValueError("protection references must have distinct physical paths")
        document = protection.canonical_bytes().decode()
        with self._transaction(database) as connection:
            old = connection.execute(
                "SELECT document, released_reason FROM storage_protections WHERE protection_id=?",
                (protection.protection_id,),
            ).fetchone()
            if old is not None and (old[0] != document or old[1] is not None):
                raise ValueError(
                    "protection identity is immutable and cannot be reused"
                )
            connection.execute(
                "INSERT OR IGNORE INTO storage_protections VALUES (?, ?, NULL)",
                (protection.protection_id, document),
            )

    def release(
        self,
        protection_id: str,
        *,
        expected_digest: str,
        reason: str,
        database: sqlite3.Connection | None = None,
    ) -> None:
        if not reason.strip():
            raise ValueError("releasing protection requires an explicit reason")
        with self._transaction(database) as connection:
            old = connection.execute(
                "SELECT document, released_reason FROM storage_protections WHERE protection_id=?",
                (protection_id,),
            ).fetchone()
            if (
                old is None
                or Protection.model_validate_json(old[0]).sha256 != expected_digest
            ):
                raise ValueError("protection identity changed or is unknown")
            if old[1] is not None and old[1] != reason:
                raise ValueError("protection was already released for another reason")
            connection.execute(
                "UPDATE storage_protections SET released_reason=? WHERE protection_id=?",
                (reason, protection_id),
            )

    def protect_once(
        self, protection: Protection, *, database: sqlite3.Connection | None = None
    ) -> None:
        """Verify files on first registration; replay only an identical live pin.

        This is for immutable execution inputs already checked by their readers.
        Replaying a pin does not assert that the current file bytes were re-read.
        """
        with self._transaction(database) as connection:
            old = connection.execute(
                "SELECT document,released_reason FROM storage_protections WHERE protection_id=?",
                (protection.protection_id,),
            ).fetchone()
            if old is not None:
                if old[0] != protection.canonical_bytes().decode() or old[1] is not None:
                    raise ValueError("input protection changed or was released")
                return
            self.protect(protection, database=connection)

    def snapshot(self) -> CatalogSnapshot:
        # No scheduler status/recovery: inspection never changes owners or leases.
        with closing(
            sqlite3.connect(self.layout.registry.as_uri() + "?mode=ro", uri=True)
        ) as database:
            database.execute("BEGIN")
            artifacts = database.execute(
                "SELECT path, document FROM storage_artifacts ORDER BY path"
            ).fetchall()
            protections = database.execute(
                "SELECT protection_id, document, released_reason FROM storage_protections ORDER BY protection_id"
            ).fetchall()
        return CatalogSnapshot(
            artifacts={
                path: ManagedArtifact.model_validate_json(raw)
                for path, raw in artifacts
            },
            protections=tuple(
                Protection.model_validate_json(raw)
                for _, raw, released in protections
                if released is None
            ),
            digest=content_digest([artifacts, protections]),
        )
