"""Immutable revision bundles and an atomic current-revision pointer."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from .document import (
    content_changes,
    evidence_changes,
    inspect_document,
    validate_candidate,
)
from .paths import SafeDirectory, relative_path
from .schema import (
    Candidate,
    CheckResult,
    ContractError,
    Digest,
    Document,
    DocumentTarget,
    EditRequest,
    Identifier,
    Model,
    RestoreRequest,
    Revision,
    RevisionId,
    RevisionInfo,
    SnapshotResponse,
    StagedCandidate,
    ValidationResult,
    candidate_digest,
    canonical_bytes,
    parse,
)


class _Head(Model):
    revision_id: RevisionId
    record_digest: Digest


class _Stage(Model):
    staged: StagedCandidate
    candidate: Candidate


class _Record(Model):
    revision: Revision
    candidate: Candidate
    validation: ValidationResult


P = ParamSpec("P")
R = TypeVar("R")
M = TypeVar("M", bound=Model)


def _storage_boundary(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def call(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except OSError as exc:
            raise ContractError("storage_error") from exc

    return call


def _identifier(value: str) -> str:
    try:
        result: str = TypeAdapter(Identifier).validate_python(value)
        return result
    except ValidationError as exc:
        raise ContractError("invalid_request") from exc


def _revision_id(value: str) -> str:
    try:
        result: str = TypeAdapter(RevisionId).validate_python(value)
        return result
    except ValidationError as exc:
        raise ContractError("invalid_request") from exc


def _json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_files(document: Document) -> dict[str, bytes]:
    """The closed export allowlist. Asset payloads are added from the manifest."""
    return {
        "index.html": document.html.encode("utf-8"),
        "evidence.json": _json(
            [item.model_dump(mode="json") for item in document.evidence]
        ),
        "figures.json": _json(
            [item.model_dump(mode="json") for item in document.figures]
        ),
    }


class RevisionStore:
    @_storage_boundary
    def __init__(self, root: Path, *, fault: Callable[[str], None] | None = None):
        self.root = root.absolute()
        self.fault = fault
        with SafeDirectory(self.root, create=True) as filesystem:
            filesystem.mkdir("private/documents")
            filesystem.mkdir("exports")

    def _checkpoint(self, name: str) -> None:
        if self.fault is not None:
            self.fault(name)

    @staticmethod
    def _document_path(document_id: str) -> str:
        return f"private/documents/{_identifier(document_id)}"

    @contextmanager
    def _lock(self, filesystem: SafeDirectory, document_id: str) -> Iterator[str]:
        path = self._document_path(document_id)
        filesystem.mkdir(path)
        with filesystem.open_file(f"{path}/lock", os.O_RDWR | os.O_CREAT) as descriptor:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield path
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)

    @staticmethod
    def _read_model(filesystem: SafeDirectory, path: str, model: type[M]) -> M:
        try:
            result: M = model.model_validate_json(filesystem.read(path))
            return result
        except (ValidationError, ValueError) as exc:
            raise ContractError("corrupt_store") from exc

    def _record(
        self,
        filesystem: SafeDirectory,
        document_id: str,
        revision_id: str,
        *,
        expected_digest: str | None = None,
    ) -> _Record:
        path = (
            f"{self._document_path(document_id)}/revisions/{_revision_id(revision_id)}"
        )
        try:
            data = filesystem.read(f"{path}/record.json")
        except FileNotFoundError as exc:
            raise ContractError("revision_not_found") from exc
        if expected_digest is not None and _hash(data) != expected_digest:
            raise ContractError("corrupt_store")
        try:
            record: _Record = _Record.model_validate_json(data)
            if (
                record.revision.revision_id != revision_id
                or record.revision.document.document_id != document_id
                or record.revision.document != record.candidate.document
                or record.candidate.request.document_id != document_id
                or record.revision.parent_revision
                != record.candidate.request.base_revision
                or record.revision.request_id != record.candidate.request.request_id
                or record.revision.change_summary != record.candidate.change_summary
                or record.revision.changes != record.candidate.changes
                or record.revision.evidence_changes != record.candidate.evidence_changes
                or record.validation.candidate_digest
                != candidate_digest(record.candidate)
                or not record.validation.passed
            ):
                raise ContractError("corrupt_store")
            self._check_bundle(filesystem, f"{path}/public", record.revision.document)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            raise ContractError("corrupt_store") from exc
        return record

    def _current(self, filesystem: SafeDirectory, document_id: str) -> Revision | None:
        head_path = f"{self._document_path(document_id)}/current.json"
        if not filesystem.exists(head_path):
            return None
        head = self._read_model(filesystem, head_path, _Head)
        try:
            return self._record(
                filesystem,
                document_id,
                head.revision_id,
                expected_digest=head.record_digest,
            ).revision
        except ContractError as exc:
            if exc.error.code == "revision_not_found":
                raise ContractError("corrupt_store") from exc
            raise

    @staticmethod
    def _check_bundle(filesystem: SafeDirectory, path: str, document: Document) -> None:
        inspect_document(document)
        for name, data in public_files(document).items():
            if filesystem.read(f"{path}/{name}") != data:
                raise ContractError("corrupt_store")
        for asset in document.assets:
            if _hash(filesystem.read(f"{path}/{asset.path}")) != asset.sha256:
                raise ContractError("corrupt_store")

    def _write_bundle(
        self,
        filesystem: SafeDirectory,
        destination: str,
        candidate: Candidate,
        assets: Mapping[str, bytes],
        current: Revision | None,
    ) -> None:
        document = candidate.document
        declared = {asset.asset_id for asset in document.assets}
        if not assets.keys() <= declared:
            raise ContractError("invalid_candidate")
        previous = (
            {asset.asset_id: asset for asset in current.document.assets}
            if current
            else {}
        )
        for name, data in public_files(document).items():
            filesystem.write(f"{destination}/{name}", data)
        for asset in document.assets:
            if asset.asset_id in assets:
                payload = assets[asset.asset_id]
            elif (
                current
                and asset.asset_id in previous
                and previous[asset.asset_id].sha256 == asset.sha256
            ):
                # Explicitly content-addressed reuse; missing/changed assets fail.
                old_path = previous[asset.asset_id].path
                payload = filesystem.read(
                    f"{self._document_path(document.document_id)}/revisions/{current.revision_id}/public/{old_path}"
                )
            else:
                raise ContractError("broken_reference")
            if not isinstance(payload, bytes) or _hash(payload) != asset.sha256:
                raise ContractError("broken_reference")
            filesystem.write(f"{destination}/{asset.path}", payload)

    @_storage_boundary
    def stage(
        self, candidate: Candidate, assets: Mapping[str, bytes] | None = None
    ) -> StagedCandidate:
        # Reparse even model instances: frozen Pydantic models contain mutable lists.
        candidate = parse(Candidate, canonical_bytes(candidate))
        with (
            SafeDirectory(self.root) as filesystem,
            self._lock(filesystem, candidate.request.document_id) as path,
        ):
            current = self._current(filesystem, candidate.request.document_id)
            validate_candidate(candidate, current)
            stage_id = uuid4().hex
            staged = StagedCandidate(
                document_id=candidate.request.document_id,
                stage_id=stage_id,
                candidate_digest=candidate_digest(candidate),
                request_id=candidate.request.request_id,
                base_revision=candidate.request.base_revision,
            )
            pending = f"{path}/stages/.pending-{stage_id}"
            self._write_bundle(
                filesystem,
                f"{pending}/public",
                candidate,
                assets if assets is not None else {},
                current,
            )
            filesystem.write(
                f"{pending}/stage.json",
                canonical_bytes(_Stage(staged=staged, candidate=candidate)),
            )
            filesystem.rename(pending, f"{path}/stages/{stage_id}")
            return staged

    def _load_stage(
        self, filesystem: SafeDirectory, document_id: str, stage_id: str
    ) -> _Stage:
        try:
            stage = self._read_model(
                filesystem,
                f"{self._document_path(document_id)}/stages/{_revision_id(stage_id)}/stage.json",
                _Stage,
            )
        except FileNotFoundError as exc:
            raise ContractError("candidate_not_found") from exc
        if (
            stage.staged.document_id != document_id
            or stage.staged.stage_id != stage_id
            or stage.staged.candidate_digest != candidate_digest(stage.candidate)
            or stage.staged.request_id != stage.candidate.request.request_id
            or stage.staged.base_revision != stage.candidate.request.base_revision
        ):
            raise ContractError("corrupt_store")
        return stage

    @_storage_boundary
    def staged_candidate(self, document_id: str, stage_id: str) -> Candidate:
        with SafeDirectory(self.root) as filesystem:
            stage = self._load_stage(filesystem, document_id, stage_id)
            self._check_bundle(
                filesystem,
                f"{self._document_path(document_id)}/stages/{stage_id}/public",
                stage.candidate.document,
            )
            return stage.candidate

    @_storage_boundary
    def staged_asset_bytes(
        self, document_id: str, stage_id: str, asset_id: str
    ) -> bytes:
        candidate = self.staged_candidate(document_id, stage_id)
        asset = next(
            (
                item
                for item in candidate.document.assets
                if item.asset_id == _identifier(asset_id)
            ),
            None,
        )
        if asset is None:
            raise ContractError("broken_reference")
        with SafeDirectory(self.root) as filesystem:
            data = filesystem.read(
                f"{self._document_path(document_id)}/stages/{stage_id}/public/{asset.path}"
            )
            if _hash(data) != asset.sha256:
                raise ContractError("corrupt_store")
            return data

    def _history(self, filesystem: SafeDirectory, document_id: str) -> list[Revision]:
        current = self._current(filesystem, document_id)
        if current is None:
            raise ContractError("document_not_found")
        result = []
        seen = set()
        while current:
            if current.revision_id in seen:
                raise ContractError("corrupt_store")
            seen.add(current.revision_id)
            result.append(current)
            current = (
                self._record(filesystem, document_id, current.parent_revision).revision
                if current.parent_revision
                else None
            )
        return result

    def _commit(
        self,
        filesystem: SafeDirectory,
        stage: _Stage,
        validation: ValidationResult,
        *,
        restored_from: str | None = None,
    ) -> Revision:
        staged, candidate = stage.staged, stage.candidate
        path = self._document_path(staged.document_id)
        current = self._current(filesystem, staged.document_id)
        if validation.candidate_digest != staged.candidate_digest:
            raise ContractError("validation_mismatch")
        if not validation.passed:
            raise ContractError("validation_failed")
        if current:
            for existing in self._history(filesystem, staged.document_id):
                if existing.request_id == staged.request_id:
                    record = self._record(
                        filesystem, staged.document_id, existing.revision_id
                    )
                    if candidate_digest(record.candidate) == staged.candidate_digest:
                        return existing
                    raise ContractError("request_conflict")
        validate_candidate(candidate, current)
        self._check_bundle(
            filesystem, f"{path}/stages/{staged.stage_id}/public", candidate.document
        )
        revision = Revision(
            revision_id=uuid4().hex,
            parent_revision=staged.base_revision,
            restored_from=restored_from,
            request_id=staged.request_id,
            change_summary=candidate.change_summary,
            changes=candidate.changes,
            evidence_changes=candidate.evidence_changes,
            document=candidate.document,
        )
        pending = f"{path}/revisions/.pending-{revision.revision_id}"
        source = f"{path}/stages/{staged.stage_id}/public"
        for name in list(public_files(candidate.document)) + [
            asset.path for asset in candidate.document.assets
        ]:
            filesystem.write(
                f"{pending}/public/{name}", filesystem.read(f"{source}/{name}")
            )
        self._checkpoint("public_files_written")
        record_bytes = canonical_bytes(
            _Record(revision=revision, candidate=candidate, validation=validation)
        )
        filesystem.write(f"{pending}/record.json", record_bytes)
        self._checkpoint("record_written")
        filesystem.rename(pending, f"{path}/revisions/{revision.revision_id}")
        self._checkpoint("revision_installed")
        self._check_bundle(
            filesystem,
            f"{path}/revisions/{revision.revision_id}/public",
            candidate.document,
        )
        head = _Head(
            revision_id=revision.revision_id, record_digest=_hash(record_bytes)
        )
        self._checkpoint("before_current_replace")
        filesystem.atomic_write(f"{path}/current.json", canonical_bytes(head))
        return revision

    @_storage_boundary
    def commit(self, staged: StagedCandidate, validation: ValidationResult) -> Revision:
        staged = parse(StagedCandidate, canonical_bytes(staged))
        validation = parse(ValidationResult, canonical_bytes(validation))
        with (
            SafeDirectory(self.root) as filesystem,
            self._lock(filesystem, staged.document_id),
        ):
            stage = self._load_stage(filesystem, staged.document_id, staged.stage_id)
            if stage.staged != staged:
                raise ContractError("validation_mismatch")
            return self._commit(filesystem, stage, validation)

    def import_initial(
        self,
        candidate: Candidate,
        assets: Mapping[str, bytes],
        validation: ValidationResult,
    ) -> Revision:
        if candidate.request.base_revision is not None:
            raise ContractError("invalid_request")
        return self.commit(self.stage(candidate, assets), validation)

    @_storage_boundary
    def current(self, document_id: str) -> Revision:
        with SafeDirectory(self.root) as filesystem:
            current = self._current(filesystem, document_id)
            if current is None:
                raise ContractError("document_not_found")
            return current

    @_storage_boundary
    def revision(self, document_id: str, revision_id: str) -> Revision:
        with SafeDirectory(self.root) as filesystem:
            # Only revisions reachable from current are published. An orphan left
            # before pointer replacement is private, even if its UUID is known.
            for revision in self._history(filesystem, document_id):
                if revision.revision_id == _revision_id(revision_id):
                    return revision
            raise ContractError("revision_not_found")

    @_storage_boundary
    def history(self, document_id: str) -> list[RevisionInfo]:
        with SafeDirectory(self.root) as filesystem:
            return [
                RevisionInfo.model_validate_json(
                    item.model_dump_json(exclude={"document"})
                )
                for item in self._history(filesystem, document_id)
            ]

    @_storage_boundary
    def restore(self, request: RestoreRequest) -> Revision:
        request = parse(RestoreRequest, canonical_bytes(request))
        with (
            SafeDirectory(self.root) as filesystem,
            self._lock(filesystem, request.document_id),
        ):
            history = self._history(filesystem, request.document_id)
            current = history[0]
            existing = next(
                (item for item in history if item.request_id == request.request_id),
                None,
            )
            if existing is not None:
                if (
                    existing.restored_from == request.revision_id
                    and existing.parent_revision == request.base_revision
                ):
                    return existing
                raise ContractError("request_conflict")
            if current.revision_id != request.base_revision:
                raise ContractError(
                    "stale_revision", current_revision=current.revision_id
                )
            target = next(
                (item for item in history if item.revision_id == request.revision_id),
                None,
            )
            if target is None:
                raise ContractError("revision_not_found")
            candidate = Candidate(
                request=EditRequest(
                    document_id=request.document_id,
                    base_revision=request.base_revision,
                    request_id=request.request_id,
                    target=DocumentTarget(),
                    instruction=f"Restore revision {request.revision_id}.",
                ),
                document=target.document,
                changes=content_changes(current.document, target.document),
                evidence_changes=evidence_changes(
                    current.document,
                    target.document,
                    reason="Restore previously validated evidence.",
                ),
                change_summary=f"Restored revision {request.revision_id}.",
            )
            # Stay under the same document lock throughout; do not recursively
            # call stage(), which would acquire a different flock descriptor.
            validate_candidate(candidate, current)
            stage_id = uuid4().hex
            staged = StagedCandidate(
                document_id=request.document_id,
                stage_id=stage_id,
                candidate_digest=candidate_digest(candidate),
                request_id=request.request_id,
                base_revision=request.base_revision,
            )
            stage = _Stage(staged=staged, candidate=candidate)
            stage_path = f"{self._document_path(request.document_id)}/stages/{stage_id}"
            assets = {
                asset.asset_id: filesystem.read(
                    f"{self._document_path(request.document_id)}/revisions/{target.revision_id}/public/{asset.path}"
                )
                for asset in target.document.assets
            }
            self._write_bundle(
                filesystem, f"{stage_path}/public", candidate, assets, current
            )
            filesystem.write(f"{stage_path}/stage.json", canonical_bytes(stage))
            old_validation = self._record(
                filesystem, request.document_id, target.revision_id
            ).validation
            # No generated content: reuse semantic/rendering checks on exactly
            # the old document, while rechecking structure and every byte.
            validation = ValidationResult(
                candidate_digest=staged.candidate_digest,
                structural=CheckResult(status="passed"),
                semantic=old_validation.semantic,
                rendering=old_validation.rendering,
            )
            return self._commit(
                filesystem, stage, validation, restored_from=target.revision_id
            )

    @_storage_boundary
    def asset_bytes(self, document_id: str, revision_id: str, asset_id: str) -> bytes:
        revision = self.revision(document_id, revision_id)
        asset = next(
            (
                asset
                for asset in revision.document.assets
                if asset.asset_id == _identifier(asset_id)
            ),
            None,
        )
        if asset is None:
            raise ContractError("broken_reference")
        with SafeDirectory(self.root) as filesystem:
            data = filesystem.read(
                f"{self._document_path(document_id)}/revisions/{revision.revision_id}/public/{asset.path}"
            )
            if _hash(data) != asset.sha256:
                raise ContractError("corrupt_store")
            return data

    def content_file(self, document_id: str, revision_id: str, name: str) -> bytes:
        """Serve only the three public document files; assets resolve by asset ID."""
        relative_path(name)
        revision = self.revision(document_id, revision_id)
        files = public_files(revision.document)
        if name not in files:
            raise ContractError("broken_reference")
        return files[name]

    @_storage_boundary
    def snapshot(self, document_id: str, revision_id: str) -> SnapshotResponse:
        revision = self.revision(document_id, revision_id)
        snapshot_id = uuid4().hex
        files = sorted(
            [
                *public_files(revision.document),
                *(asset.path for asset in revision.document.assets),
                "manifest.json",
            ]
        )
        response = SnapshotResponse(
            snapshot_id=snapshot_id,
            document_id=document_id,
            revision_id=revision_id,
            files=files,
        )
        published = f"exports/{snapshot_id}"
        try:
            with SafeDirectory(self.root) as filesystem:
                pending = f"exports/.pending-{snapshot_id}"
                source = (
                    f"{self._document_path(document_id)}/revisions/{revision_id}/public"
                )
                for name in files:
                    data = (
                        canonical_bytes(response)
                        if name == "manifest.json"
                        else filesystem.read(f"{source}/{name}")
                    )
                    filesystem.write(f"{pending}/{name}", data)
                self._check_bundle(filesystem, pending, revision.document)
                # Visibility switches only after the whole bundle exists.
                filesystem.rename(pending, published)
        except OSError as exc:
            # rename() can succeed before directory fsync fails. A complete,
            # exact public snapshot is a successful result, not a lost export.
            with SafeDirectory(self.root) as filesystem:
                if not filesystem.exists(published):
                    raise
                manifest = self._read_model(
                    filesystem, f"{published}/manifest.json", SnapshotResponse
                )
                if manifest != response:
                    raise ContractError("corrupt_store") from exc
                self._check_bundle(filesystem, published, revision.document)
        return response

    @_storage_boundary
    def snapshot_file(self, snapshot_id: str, name: str) -> bytes:
        relative_path(name)
        path = f"exports/{_revision_id(snapshot_id)}"
        with SafeDirectory(self.root) as filesystem:
            try:
                manifest = self._read_model(
                    filesystem, f"{path}/manifest.json", SnapshotResponse
                )
            except FileNotFoundError as exc:
                raise ContractError("snapshot_not_found") from exc
            if manifest.snapshot_id != snapshot_id or name not in manifest.files:
                raise ContractError("snapshot_not_found")
            return filesystem.read(f"{path}/{name}")
