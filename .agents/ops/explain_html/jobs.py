"""Durable, per-document jobs. Only this controller may publish editor results."""

from __future__ import annotations

import fcntl
import logging
import os
from collections import defaultdict, deque
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Condition, Event, Thread
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from .document import validate_request
from .editor import EditorialResult, WritingContext
from .paths import SafeDirectory
from .schema import (
    JOB_TRANSITIONS,
    Candidate,
    CheckResult,
    ContractError,
    EditRequest,
    Identifier,
    JobResponse,
    JobState,
    Model,
    PublicError,
    RestoreRequest,
    Revision,
    RevisionId,
    RevisionInfo,
    StagedCandidate,
    ValidationResult,
    canonical_bytes,
    parse,
)
from .store import RevisionStore

LOG = logging.getLogger(__name__)
ACTIVE = frozenset({"queued", "running", "validating"})


class Editor(Protocol):
    def edit(
        self,
        request: EditRequest,
        current: Revision,
        context: WritingContext,
        *,
        cancel: Event,
    ) -> EditorialResult: ...


class BundleValidator(Protocol):
    def __call__(
        self, candidate: Candidate, assets: Mapping[str, bytes], *, cancel: Event
    ) -> CheckResult:
        """Check sanitization, renderer policy AND actual rendering of these bytes."""
        ...


@dataclass(frozen=True)
class PreparedEdit:
    context: WritingContext
    assets: Mapping[str, bytes] = field(default_factory=dict)


class AssetPreparer(Protocol):
    def __call__(
        self,
        request: EditRequest,
        current: Revision,
        history: list[RevisionInfo],
        *,
        cancel: Event,
    ) -> PreparedEdit:
        """Trusted asset/image planning. Raise image_generation_unavailable if needed.

        This is not browser JSON. It may supply approved assets/figures and an
        authorized evidence correction, but cannot grant a publication verdict.
        """
        ...


def preserve_resources(
    request: EditRequest,
    current: Revision,
    history: list[RevisionInfo],
    *,
    cancel: Event,
) -> PreparedEdit:
    """Use existing resources and evidence; no implicit generation or correction."""
    document = current.document
    return PreparedEdit(
        WritingContext(
            purpose=document.purpose,
            audience=document.audience,
            evidence=document.evidence,
            assets=document.assets,
            figures=document.figures,
            change_summaries=[item.change_summary for item in reversed(history)],
        )
    )


class _Record(Model):
    request: EditRequest | RestoreRequest
    job: JobResponse | None = None
    staged: StagedCandidate | None = None
    restored_revision: RevisionId | None = None
    error: PublicError | None = None


def identifier(value: str) -> str:
    try:
        result: str = TypeAdapter(Identifier).validate_python(value)
        return result
    except ValidationError as exc:
        raise ContractError("invalid_request") from exc


class JobController:
    def __init__(
        self,
        store: RevisionStore,
        editor: Editor,
        validator: BundleValidator,
        *,
        prepare: AssetPreparer = preserve_resources,
    ):
        self.store, self.editor, self.validator, self.prepare = (
            store,
            editor,
            validator,
            prepare,
        )
        self._condition = Condition()
        self._closed = False
        self._stopped = Event()
        self._failure: ContractError | None = None
        self._records: dict[tuple[str, str], _Record] = {}
        self._cancels: dict[tuple[str, str], Event] = {}
        self._queues: dict[str, deque[str]] = defaultdict(deque)
        self._workers: dict[str, Thread] = {}
        self._threads: set[Thread] = set()
        self._resources = ExitStack()
        try:
            filesystem = self._resources.enter_context(SafeDirectory(store.root))
            if os.fstat(filesystem.fd).st_mode & 0o077:
                raise ContractError("unsafe_path")
            filesystem.mkdir("private/requests")
            descriptor = self._resources.enter_context(
                filesystem.open_file("private/service.lock", os.O_RDWR | os.O_CREAT)
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("request_conflict") from exc
            self._recover()
        except BaseException:
            self._resources.close()
            raise

    @staticmethod
    def _key(request: EditRequest | RestoreRequest) -> tuple[str, str]:
        return request.document_id, request.request_id

    def _save(self, record: _Record) -> None:
        request = record.request
        try:
            with SafeDirectory(self.store.root) as filesystem:
                filesystem.atomic_write(
                    f"private/requests/{request.document_id}/{request.request_id}.json",
                    canonical_bytes(record),
                )
        except (OSError, ContractError) as exc:
            error = (
                exc
                if isinstance(exc, ContractError)
                else ContractError("storage_error")
            )
            self._fail_runtime(error)
            raise error from exc
        self._records[self._key(request)] = record
        self._condition.notify_all()

    def _fail_runtime(self, error: ContractError) -> None:
        """No job can proceed after uncertain journal I/O; restart reconciles it."""
        self._failure = error
        for cancel in self._cancels.values():
            cancel.set()
        self._condition.notify_all()

    def _published(self, record: _Record) -> Revision | None:
        request = record.request
        for info in self.store.history(request.document_id):
            if info.request_id != request.request_id:
                continue
            revision = self.store.revision(request.document_id, info.revision_id)
            if isinstance(request, RestoreRequest):
                if (
                    revision.restored_from != request.revision_id
                    or revision.parent_revision != request.base_revision
                ):
                    raise ContractError("corrupt_store")
            else:
                if record.staged is None:
                    raise ContractError("corrupt_store")
                candidate = self.store.staged_candidate(
                    request.document_id, record.staged.stage_id
                )
                if (
                    candidate.request != request
                    or revision.document != candidate.document
                    or revision.parent_revision != request.base_revision
                    or revision.changes != candidate.changes
                    or revision.evidence_changes != candidate.evidence_changes
                    or revision.change_summary != candidate.change_summary
                ):
                    raise ContractError("corrupt_store")
            return revision
        return None

    def _success(self, record: _Record, revision: Revision) -> _Record:
        request = record.request
        if isinstance(request, RestoreRequest):
            restored: _Record = record.model_copy(
                update={"restored_revision": revision.revision_id}
            )
            return restored
        assert revision.parent_revision is not None
        result: _Record = record.model_copy(
            update={
                "job": JobResponse(
                    document_id=request.document_id,
                    request_id=request.request_id,
                    base_revision=revision.parent_revision,
                    state="succeeded",
                    revision_id=revision.revision_id,
                    changes=revision.changes,
                    change_summary=revision.change_summary,
                )
            }
        )

        return result

    def _recover(self) -> None:
        with (
            self._condition,
            SafeDirectory(self.store.root / "private/requests") as documents,
        ):
            for document_id in sorted(os.listdir(documents.fd)):
                identifier(document_id)
                with SafeDirectory(
                    self.store.root / "private/requests" / document_id
                ) as requests:
                    for name in sorted(os.listdir(requests.fd)):
                        # An atomic write interrupted before rename is private,
                        # unreachable and never a request to execute.
                        if name.startswith(".pending-"):
                            continue
                        if not name.endswith(".json"):
                            raise ContractError("corrupt_store")
                        request_id = identifier(name[:-5])
                        try:
                            record = _Record.model_validate_json(requests.read(name))
                        except (ValidationError, ValueError) as exc:
                            raise ContractError("corrupt_store") from exc
                        if self._key(record.request) != (document_id, request_id):
                            raise ContractError("corrupt_store")
                        job = record.job
                        if isinstance(record.request, EditRequest):
                            if job is None or (
                                job.document_id,
                                job.request_id,
                                job.base_revision,
                            ) != (
                                document_id,
                                request_id,
                                record.request.base_revision,
                            ):
                                raise ContractError("corrupt_store")
                        elif job is not None:
                            raise ContractError("corrupt_store")
                        self._records[(document_id, request_id)] = record
                        if (job and job.state in ACTIVE) or (
                            job is None
                            and record.restored_revision is None
                            and record.error is None
                        ):
                            published = self._published(record)
                            if published:
                                self._save(self._success(record, published))
                            elif job:
                                self._transition(record, "interrupted")
                            else:
                                self._save(
                                    record.model_copy(
                                        update={
                                            "error": ContractError("interrupted").error
                                        }
                                    )
                                )

    def _existing(self, request: EditRequest | RestoreRequest) -> _Record | None:
        record = self._records.get(self._key(request))
        if record:
            if canonical_bytes(record.request) != canonical_bytes(request):
                raise ContractError("request_conflict")
            return record
        # Include initial imports and publications from trusted external tooling.
        if any(
            info.request_id == request.request_id
            for info in self.store.history(request.document_id)
        ):
            raise ContractError("request_conflict")
        return None

    def _open(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise ContractError("interrupted")

    def submit(self, request: EditRequest) -> JobResponse:
        request = parse(EditRequest, canonical_bytes(request))
        with self._condition:
            self._open()
            existing = self._existing(request)
            if existing:
                assert existing.job is not None
                return existing.job
            current = self.store.current(request.document_id)
            validate_request(request, current)
            job = JobResponse(
                document_id=request.document_id,
                request_id=request.request_id,
                base_revision=current.revision_id,
                state="queued",
            )
            self._save(_Record(request=request, job=job))
            self._cancels[self._key(request)] = Event()
            self._queues[request.document_id].append(request.request_id)
            if request.document_id not in self._workers:
                thread = Thread(
                    target=self._work, args=(request.document_id,), daemon=False
                )
                self._workers[request.document_id] = thread
                self._threads.add(thread)
                thread.start()
            return job

    def status(self, document_id: str, request_id: str) -> JobResponse:
        with self._condition:
            if self._failure is not None:
                raise self._failure
            key = identifier(document_id), identifier(request_id)
            record = self._records.get(key)
            if record is None or record.job is None:
                raise ContractError("request_not_found")
            return parse(JobResponse, canonical_bytes(record.job))

    def wait(self, document_id: str, request_id: str, *, timeout: float) -> JobResponse:
        """An embedding host may wait for completion without re-running a request."""
        with self._condition:
            done = self._condition.wait_for(
                lambda: self.status(document_id, request_id).state not in ACTIVE,
                timeout=timeout,
            )
            if not done:
                raise TimeoutError("Job is still active.")
            return self.status(document_id, request_id)

    def _transition(
        self, record: _Record, state: JobState, error: PublicError | None = None
    ) -> _Record:
        assert record.job is not None
        if state not in JOB_TRANSITIONS[record.job.state]:
            raise RuntimeError("Invalid job transition.")
        if state in {"cancelled", "interrupted"}:
            error = ContractError(state).error
        job = parse(
            JobResponse,
            record.job.model_copy(
                update={"state": state, "error": error}
            ).model_dump_json(),
        )
        updated: _Record = record.model_copy(update={"job": job})
        self._save(updated)
        return updated

    def cancel(self, document_id: str, request_id: str) -> JobResponse:
        with self._condition:
            self.status(document_id, request_id)
            key = document_id, request_id
            record = self._records[key]
            assert record.job is not None
            if record.job.state in ACTIVE:
                # Even a storage failure must fence the late generator result.
                self._cancels[key].set()
                record = self._transition(record, "cancelled")
            assert record.job is not None
            return record.job

    def restore(self, request: RestoreRequest) -> Revision:
        request = parse(RestoreRequest, canonical_bytes(request))
        with self._condition:
            self._open()
            existing = self._existing(request)
            if existing:
                if existing.error:
                    raise ContractError(
                        existing.error.code,
                        current_revision=existing.error.current_revision,
                    )
                if existing.restored_revision:
                    return self.store.revision(
                        request.document_id, existing.restored_revision
                    )
                raise ContractError("interrupted")
            record = _Record(request=request)
            self._save(record)
            try:
                revision = self.store.restore(request)
            except ContractError as exc:
                self._save(record.model_copy(update={"error": exc.error}))
                raise
            self._save(self._success(record, revision))
            for key, active in list(self._records.items()):
                if (
                    key[0] == request.document_id
                    and active.job
                    and active.job.state in ACTIVE
                ):
                    self._cancels[key].set()
                    self._transition(
                        active,
                        "failed",
                        ContractError(
                            "stale_revision", current_revision=revision.revision_id
                        ).error,
                    )
            return revision

    def _check_active(self, key: tuple[str, str]) -> _Record:
        record = self._records[key]
        self._open()
        if (
            self._cancels[key].is_set()
            or not record.job
            or record.job.state not in ACTIVE
        ):
            raise ContractError("cancelled")
        assert isinstance(record.request, EditRequest)
        validate_request(record.request, self.store.current(key[0]))
        return record

    def _work(self, document_id: str) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._closed
                            or self._failure is not None
                            or bool(self._queues[document_id])
                        )
                    )
                    if self._closed or self._failure is not None:
                        return
                    key = document_id, self._queues[document_id].popleft()
                    record = self._records[key]
                    if not record.job or record.job.state not in ACTIVE:
                        continue
                self._execute(key)
        finally:
            with self._condition:
                self._workers.pop(document_id, None)

    def _execute(self, key: tuple[str, str]) -> None:
        try:
            with self._condition:
                record = self._check_active(key)
                assert isinstance(record.request, EditRequest)
                request = record.request
                current = self.store.current(key[0])
                self._transition(record, "running")
            cancel = self._cancels[key]
            prepared = self.prepare(
                request, current, self.store.history(key[0]), cancel=cancel
            )
            with self._condition:
                self._check_active(key)
            # Do not override the adapter's shared writer/verifier timeout.
            result = self.editor.edit(request, current, prepared.context, cancel=cancel)
            candidate = parse(Candidate, canonical_bytes(result.candidate))
            if candidate.request != request:
                raise ContractError("invalid_candidate")
            validation = parse(ValidationResult, canonical_bytes(result.validation))
            with self._condition:
                record = self._check_active(key)
                self._transition(record, "validating")
                staged = self.store.stage(candidate, prepared.assets)
                record = self._records[key].model_copy(update={"staged": staged})
                self._save(record)
            if validation.candidate_digest != staged.candidate_digest:
                raise ContractError("validation_mismatch")
            if not all(
                check.status == "passed" and not check.issues
                for check in (validation.structural, validation.semantic)
            ):
                raise ContractError("validation_failed")
            # Validators receive the immutable staged candidate and every byte,
            # including reused assets, rather than mutable editor output.
            candidate = self.store.staged_candidate(key[0], staged.stage_id)
            assets = {
                asset.asset_id: self.store.staged_asset_bytes(
                    key[0], staged.stage_id, asset.asset_id
                )
                for asset in candidate.document.assets
            }
            with self._condition:
                self._check_active(key)
            rendering = parse(
                CheckResult,
                canonical_bytes(self.validator(candidate, assets, cancel=cancel)),
            )
            validation = validation.model_copy(update={"rendering": rendering})
            with self._condition:
                record = self._check_active(key)
                revision = self.store.commit(staged, validation)
                self._save(self._success(record, revision))
        except Exception as exc:
            LOG.exception("Private job failure")
            with self._condition:
                if self._failure is not None:
                    return
                record = self._records[key]
                if record.job and record.job.state in ACTIVE:
                    # Pointer replacement can succeed just before journal I/O
                    # fails. Never mislabel a committed revision as failed.
                    try:
                        published = self._published(record)
                        if published:
                            self._save(self._success(record, published))
                        else:
                            error = (
                                exc
                                if isinstance(exc, ContractError)
                                else ContractError("generation_failed")
                            )
                            state: JobState = (
                                error.error.code
                                if error.error.code in {"cancelled", "interrupted"}
                                else "failed"
                            )
                            self._transition(record, state, error.error)
                    except (OSError, ContractError) as failure:
                        self._fail_runtime(
                            failure
                            if isinstance(failure, ContractError)
                            else ContractError("storage_error")
                        )

    def close(self) -> None:
        with self._condition:
            already_closing = self._closed
            if not already_closing:
                self._closed = True
                threads = list(self._threads)
                for cancel in self._cancels.values():
                    cancel.set()
                self._condition.notify_all()
        if already_closing:
            self._stopped.wait()
            return
        try:
            with self._condition:
                # On uncertain journal I/O leave the last durable state intact;
                # a pointer replacement may already have committed this job.
                if self._failure is None:
                    for record in list(self._records.values()):
                        if record.job and record.job.state in ACTIVE:
                            self._transition(record, "interrupted")
        finally:
            for thread in threads:
                thread.join()
            self._resources.close()
            self._stopped.set()

    def __enter__(self) -> JobController:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
