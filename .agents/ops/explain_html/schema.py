"""The single source for explain-html wire models, schemas and wire examples.

No server, generator or browser is imported here. JSON Schema/OpenAPI is generated
from these models; downstream callers must not maintain copies of the schemas.

Python usage (writer_json and validation_json come from separate trusted
writer/validator orchestration; a writer must never author its own verdict)::

    from pathlib import Path
    from explain_html.schema import Candidate, ValidationResult, parse
    from explain_html.store import RevisionStore

    store = RevisionStore(Path("outputs/explain-html"))
    candidate = parse(Candidate, writer_json)
    staged = store.stage(candidate, asset_bytes_by_id)
    validation = parse(ValidationResult, validation_json)
    revision = store.commit(staged, validation)
    snapshot = store.snapshot(candidate.document.document_id, revision.revision_id)

For initial import, use the same Candidate with base_revision=None and a document
target, then store.import_initial(candidate, asset_bytes_by_id, validation).
The examples() function contains all JSON examples, including an initial
Candidate and an incomplete ValidationResult that correctly cannot be committed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from pydantic.json_schema import models_json_schema

IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN)]
RevisionId = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class Provenance(Model):
    kind: Literal[
        "measured", "hypothetical", "generated_concept", "derived", "reference"
    ]
    source: Text


class Evidence(Model):
    evidence_id: Identifier
    statement: Text
    provenance: Provenance
    values: dict[str, JsonValue] = Field(default_factory=dict)
    conditions: list[Text] = Field(default_factory=list)


class Asset(Model):
    asset_id: Identifier
    path: Text
    media_type: Text
    sha256: Digest
    kind: Literal["image", "svg", "data", "stylesheet", "renderer"]
    caption: Text
    alt: Text
    provenance: Provenance


class Renderer(Model):
    asset_id: Identifier
    version: Text


class Figure(Model):
    figure_id: Identifier
    content_id: Identifier
    kind: Literal["image", "svg", "graph"]
    asset_ids: list[Identifier]
    caption: Text
    alt: Text
    provenance: Provenance
    renderer: Renderer | None = None
    definition: dict[str, JsonValue] = Field(default_factory=dict)


class Document(Model):
    schema_version: Literal[1] = 1
    document_id: Identifier
    purpose: Text
    audience: Text
    evidence: list[Evidence]
    html: Text
    assets: list[Asset] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)


class DocumentTarget(Model):
    kind: Literal["document"] = "document"


class ContentTarget(Model):
    kind: Literal["content"] = "content"
    content_id: Identifier


class SelectionTarget(Model):
    """Offsets are UTF-16 code units in the block's DOM textContent.

    Count after HTML input CRLF/CR-to-LF preprocessing, character-reference
    expansion and the initial-LF rule for pre/textarea/listing. Other whitespace
    is preserved. exact, prefix and suffix are decoded Unicode strings. Positions
    and quotes must both match; a quote is never relocated. Non-BMP characters
    occupy two code units, matching browser JavaScript string/Range offsets.
    """

    kind: Literal["selection"] = "selection"
    content_id: Identifier
    exact: Text
    prefix: str = ""
    suffix: str = ""
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]


Target = Annotated[
    DocumentTarget | ContentTarget | SelectionTarget, Field(discriminator="kind")
]


class EditRequest(Model):
    document_id: Identifier
    base_revision: RevisionId | None
    request_id: Identifier
    target: Target
    instruction: Text


class ContentChange(Model):
    content_id: Identifier
    kind: Literal["added", "updated", "removed"]


class EvidenceChange(Model):
    evidence_id: Identifier
    kind: Literal["added", "updated", "removed"]
    reason: Text


class Candidate(Model):
    request: EditRequest
    document: Document
    changes: list[ContentChange]
    evidence_changes: list[EvidenceChange]
    change_summary: Text


class ValidationIssue(Model):
    rule: Text
    message: Text
    content_id: Identifier | None = None
    evidence_id: Identifier | None = None


class CheckResult(Model):
    status: Literal["passed", "failed", "not_run"]
    issues: list[ValidationIssue] = Field(default_factory=list)


class ValidationResult(Model):
    candidate_digest: Digest
    structural: CheckResult
    semantic: CheckResult
    rendering: CheckResult

    @property
    def passed(self) -> bool:
        return all(
            check.status == "passed" and not check.issues
            for check in (self.structural, self.semantic, self.rendering)
        )


class StagedCandidate(Model):
    document_id: Identifier
    stage_id: RevisionId
    candidate_digest: Digest
    request_id: Identifier
    base_revision: RevisionId | None


class RevisionInfo(Model):
    revision_id: RevisionId
    parent_revision: RevisionId | None
    restored_from: RevisionId | None
    request_id: Identifier
    change_summary: Text
    changes: list[ContentChange]
    evidence_changes: list[EvidenceChange]


class Revision(RevisionInfo):
    document: Document


ErrorCode = Literal[
    "invalid_request",
    "document_not_found",
    "revision_not_found",
    "request_not_found",
    "candidate_not_found",
    "snapshot_not_found",
    "document_exists",
    "stale_revision",
    "request_conflict",
    "invalid_target",
    "duplicate_id",
    "invalid_document",
    "broken_reference",
    "unsafe_path",
    "invalid_candidate",
    "validation_failed",
    "validation_mismatch",
    "generation_failed",
    "timeout",
    "cancelled",
    "interrupted",
    "model_mismatch",
    "invalid_model_output",
    "image_generation_unavailable",
    "asset_generation_failed",
    "unauthorized",
    "invalid_origin",
    "invalid_host",
    "storage_error",
    "corrupt_store",
    "snapshot_exists",
]

# status, public message, retryable. Never include paths, prompts or raw exceptions.
ERRORS: dict[ErrorCode, tuple[int, str, bool]] = {
    "invalid_request": (400, "The request does not match the contract.", False),
    "document_not_found": (404, "The document does not exist.", False),
    "revision_not_found": (404, "The revision does not exist.", False),
    "request_not_found": (404, "The request does not exist.", False),
    "candidate_not_found": (404, "The staged candidate does not exist.", False),
    "snapshot_not_found": (404, "The snapshot does not exist.", False),
    "document_exists": (409, "The document already has an initial revision.", False),
    "stale_revision": (409, "The displayed revision is no longer current.", False),
    "request_conflict": (409, "The request ID was used for different content.", False),
    "invalid_target": (422, "The target does not match the displayed content.", False),
    "duplicate_id": (422, "Content or artifact identifiers are not unique.", False),
    "invalid_document": (422, "The document structure is invalid.", False),
    "broken_reference": (422, "A document reference cannot be resolved.", False),
    "unsafe_path": (422, "An artifact path is outside the permitted namespace.", False),
    "invalid_candidate": (422, "The candidate does not describe its changes.", False),
    "validation_failed": (
        422,
        "The candidate did not pass all validation checks.",
        False,
    ),
    "validation_mismatch": (409, "Validation belongs to a different candidate.", False),
    "generation_failed": (502, "Document generation failed.", True),
    "timeout": (504, "Document generation timed out.", True),
    "cancelled": (409, "The request was cancelled.", False),
    "interrupted": (409, "The request was interrupted; retry explicitly.", True),
    "model_mismatch": (502, "The effective model or effort did not match.", False),
    "invalid_model_output": (502, "The model returned an invalid candidate.", True),
    "image_generation_unavailable": (422, "Image generation is unavailable.", False),
    "asset_generation_failed": (502, "Asset generation failed.", True),
    "unauthorized": (401, "A valid session token is required.", False),
    "invalid_origin": (403, "The request origin is not permitted.", False),
    "invalid_host": (403, "The request host is not permitted.", False),
    "storage_error": (500, "The artifact could not be persisted.", True),
    "corrupt_store": (500, "Stored revision integrity checks failed.", False),
    "snapshot_exists": (409, "The export destination already exists.", False),
}


class PublicError(Model):
    code: ErrorCode
    message: Text
    retryable: bool
    current_revision: RevisionId | None = None


class ErrorResponse(Model):
    error: PublicError


class ContractError(Exception):
    def __init__(self, code: ErrorCode, *, current_revision: str | None = None):
        status, message, retryable = ERRORS[code]
        self.status_code = status
        self.error = PublicError(
            code=code,
            message=message,
            retryable=retryable,
            current_revision=current_revision,
        )
        super().__init__(code)


JobState = Literal[
    "queued", "running", "validating", "succeeded", "failed", "cancelled", "interrupted"
]
JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    "queued": frozenset({"running", "failed", "cancelled", "interrupted"}),
    "running": frozenset({"validating", "failed", "cancelled", "interrupted"}),
    "validating": frozenset({"succeeded", "failed", "cancelled", "interrupted"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}


class JobResponse(Model):
    document_id: Identifier
    request_id: Identifier
    base_revision: RevisionId
    state: JobState
    revision_id: RevisionId | None = None
    change_summary: str | None = None
    changes: list[ContentChange] = Field(default_factory=list)
    error: PublicError | None = None

    def model_post_init(self, context: Any) -> None:
        if self.state == "succeeded":
            if (
                self.revision_id is None
                or self.revision_id == self.base_revision
                or not self.change_summary
                or self.error is not None
            ):
                raise ValueError("Success must identify a newly committed revision.")
        else:
            if (
                self.revision_id is not None
                or self.change_summary is not None
                or self.changes
            ):
                raise ValueError("Only committed jobs may contain a result.")
            if self.state in {"failed", "cancelled", "interrupted"}:
                if self.error is None:
                    raise ValueError("A terminal failure must contain a public error.")
                if (
                    self.state in {"cancelled", "interrupted"}
                    and self.error.code != self.state
                ):
                    raise ValueError("The error must agree with the terminal state.")
            elif self.error is not None:
                raise ValueError("An active job cannot contain a terminal error.")


class DocumentResponse(Model):
    revision: Revision


class HistoryResponse(Model):
    document_id: Identifier
    revisions: list[RevisionInfo]


class CancelRequest(Model):
    document_id: Identifier
    request_id: Identifier


class RestoreRequest(Model):
    document_id: Identifier
    request_id: Identifier
    base_revision: RevisionId
    revision_id: RevisionId


class SnapshotRequest(Model):
    document_id: Identifier
    revision_id: RevisionId


class SnapshotResponse(Model):
    snapshot_id: RevisionId
    document_id: Identifier
    revision_id: RevisionId
    entrypoint: Literal["index.html"] = "index.html"
    files: list[Text]


T = TypeVar("T", bound=Model)


def parse(model: type[T], payload: str | bytes) -> T:
    """Strict wire parsing with a path/prompt-free public error."""
    try:
        result: T = model.model_validate_json(payload)
        return result
    except (ValidationError, ValueError) as exc:
        raise ContractError("invalid_request") from exc


def canonical_bytes(model: Model) -> bytes:
    try:
        return json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ContractError("invalid_request") from exc


def candidate_digest(candidate: Candidate) -> str:
    """Binds request, HTML, metadata, declared changes and every asset digest."""
    return hashlib.sha256(canonical_bytes(candidate)).hexdigest()


def examples() -> dict[str, Any]:
    """Wire examples live here only; README and adapters refer to this function."""
    document = Document(
        document_id="triangulation",
        purpose="Explain how two views locate a point.",
        audience="Readers familiar with image coordinates.",
        evidence=[],
        html='<html><body><p data-content-id="geometry">Intersect two rays.</p></body></html>',
    )
    candidate = Candidate(
        request=EditRequest(
            document_id="triangulation",
            base_revision=None,
            request_id="initial",
            target=DocumentTarget(),
            instruction="Explain the geometry and state missing assumptions.",
        ),
        document=document,
        changes=[ContentChange(content_id="geometry", kind="added")],
        evidence_changes=[],
        change_summary="Initial explanation draft.",
    )
    validation = ValidationResult(
        candidate_digest=candidate_digest(candidate),
        structural=CheckResult(status="passed"),
        semantic=CheckResult(status="not_run"),
        rendering=CheckResult(status="not_run"),
    )
    return {
        "Document": document.model_dump(mode="json"),
        "Candidate": candidate.model_dump(mode="json"),
        "ValidationResult": validation.model_dump(mode="json"),
        "EditRequest": {
            "document_id": "triangulation",
            "base_revision": "1" * 32,
            "request_id": "add-example",
            "instruction": "Add a numerical example.",
            "target": {
                "kind": "selection",
                "content_id": "geometry",
                "exact": "two rays",
                "prefix": "Intersect ",
                "suffix": ".",
                "start": 10,
                "end": 18,
            },
        },
        "RestoreRequest": {
            "document_id": "triangulation",
            "base_revision": "2" * 32,
            "request_id": "undo-example",
            "revision_id": "1" * 32,
        },
        "ErrorResponse": {
            "error": {
                "code": "stale_revision",
                "message": ERRORS["stale_revision"][1],
                "retryable": False,
                "current_revision": "2" * 32,
            },
        },
    }


def contract_schema() -> dict[str, Any]:
    """OpenAPI 3.1, including editor/store models not exposed as HTTP endpoints."""
    models: list[type[Model]] = [
        Document,
        Candidate,
        ValidationResult,
        StagedCandidate,
        EditRequest,
        RestoreRequest,
        CancelRequest,
        SnapshotRequest,
        DocumentResponse,
        HistoryResponse,
        JobResponse,
        SnapshotResponse,
        ErrorResponse,
    ]
    _, schema = models_json_schema(
        [(model, "validation") for model in models],
        ref_template="#/components/schemas/{model}",
    )
    endpoints: list[tuple[str, str, str | None, str | None, int]] = [
        ("get", "/api/documents/{document_id}", None, "DocumentResponse", 200),
        ("get", "/api/documents/{document_id}/revisions", None, "HistoryResponse", 200),
        (
            "get",
            "/api/documents/{document_id}/revisions/{revision_id}",
            None,
            "DocumentResponse",
            200,
        ),
        (
            "post",
            "/api/documents/{document_id}/requests",
            "EditRequest",
            "JobResponse",
            202,
        ),
        (
            "get",
            "/api/documents/{document_id}/requests/{request_id}",
            None,
            "JobResponse",
            200,
        ),
        (
            "post",
            "/api/documents/{document_id}/requests/{request_id}/cancel",
            "CancelRequest",
            "JobResponse",
            200,
        ),
        (
            "post",
            "/api/documents/{document_id}/undo",
            "RestoreRequest",
            "DocumentResponse",
            200,
        ),
        (
            "post",
            "/api/documents/{document_id}/exports",
            "SnapshotRequest",
            "SnapshotResponse",
            201,
        ),
        (
            "get",
            "/api/documents/{document_id}/revisions/{revision_id}/content/{file_name}",
            None,
            None,
            200,
        ),
        (
            "get",
            "/api/documents/{document_id}/revisions/{revision_id}/content/assets/{asset_name}",
            None,
            None,
            200,
        ),
        ("get", "/api/exports/{snapshot_id}/{file_path}", None, None, 200),
    ]
    paths: dict[str, Any] = {}
    for method, path, request, response, success in endpoints:

        def content(name: str) -> dict[str, Any]:
            return {
                "application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}
            }

        operation: dict[str, Any] = {
            "responses": {
                str(success): {
                    "description": "Success",
                    **({"content": content(response)} if response else {}),
                },
                "default": {
                    "description": "Explicit failure",
                    "content": content("ErrorResponse"),
                },
            },
            "parameters": [
                {
                    "name": item[1:-1],
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
                for item in path.split("/")
                if item.startswith("{")
            ],
        }
        for parameter in operation["parameters"]:
            if parameter["name"] == "file_name":
                parameter["schema"]["enum"] = [
                    "index.html",
                    "reader.html",
                    "evidence.json",
                    "figures.json",
                ]
        if request:
            operation["requestBody"] = {"required": True, "content": content(request)}
        if response is not None:
            operation["security"] = [{"SessionToken": []}]
        paths.setdefault(path, {})[method] = operation
    return {
        "openapi": "3.1.0",
        "info": {"title": "Explain HTML contract", "version": "1"},
        "paths": paths,
        "components": {
            "schemas": schema["$defs"],
            "securitySchemes": {
                "SessionToken": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-Explain-Token",
                }
            },
        },
        "x-examples": examples(),
        "x-job-transitions": {
            key: sorted(value) for key, value in JOB_TRANSITIONS.items()
        },
        "x-errors": {
            key: {"status": value[0], "message": value[1], "retryable": value[2]}
            for key, value in ERRORS.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(contract_schema(), ensure_ascii=False, indent=2))
