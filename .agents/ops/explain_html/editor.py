"""Initial writing and editing return candidates with independent semantic checks.

Only trusted orchestration supplies WritingContext, including evidence changes.
No store or public filesystem is opened by this module. Rendering remains a
separate check; a returned ValidationResult cannot yet authorize store.commit().
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Event
from typing import Any

from pydantic import Field

from .document import (
    HTMLIndex,
    content_changes,
    evidence_changes,
    validate_candidate,
    validate_request,
)
from .gemini import Gemini, TurnEvidence, check_cancel
from .schema import (
    Asset,
    Candidate,
    CheckResult,
    ContractError,
    Digest,
    Document,
    EditRequest,
    Evidence,
    EvidenceChange,
    Figure,
    Identifier,
    Model,
    Revision,
    Text,
    ValidationIssue,
    ValidationResult,
    candidate_digest,
    parse,
)

DocumentCompiler = Callable[[Document], Document]
WRITER_TIMEOUT = 900
VERIFIER_TIMEOUT = 240
FIGURE_REPAIR_TIMEOUT = 600


class WritingContext(Model):
    """Trusted inputs, never taken from the writer's proposed changes.

    evidence is the exact desired evidence set. For edits, any difference from
    current evidence also needs an explicit authorized_evidence_changes entry.
    The verifier still checks that those changes follow the user's instruction.
    """

    purpose: Text
    audience: Text
    evidence: list[Evidence]
    authorized_evidence_changes: list[EvidenceChange] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    change_summaries: list[Text] = Field(default_factory=list)
    preferences: str = ""


class SemanticFinding(Model):
    """A quoted claim and its problem, scoped to the inspected candidate."""

    quote: Text
    reason: Text
    content_id: Identifier | None = None
    evidence_id: Identifier | None = None


class SemanticVerdict(Model):
    candidate_digest: Digest
    check: CheckResult
    reviewed_evidence_ids: list[Identifier]
    # Required even when empty: an old evidence-only review is not sufficient.
    unsupported_claims: list[SemanticFinding]
    technical_errors: list[SemanticFinding]

    def semantic_check(self) -> CheckResult:
        """Every finding vetoes approval, even if the model reports passed."""
        issues = list(self.check.issues)
        for rule, findings in (
            ("unsupported_claim", self.unsupported_claims),
            ("technical_error", self.technical_errors),
        ):
            issues.extend(
                ValidationIssue(
                    rule=rule,
                    message=f"{json.dumps(finding.quote, ensure_ascii=False)}: "
                    + finding.reason,
                    content_id=finding.content_id,
                    evidence_id=finding.evidence_id,
                )
                for finding in findings
            )
        return CheckResult(
            status="passed"
            if self.check.status == "passed" and not issues
            else "failed",
            issues=issues,
        )


class SemanticValidationError(ContractError):
    """Reject publication while retaining the digest-bound private diagnostics."""

    def __init__(self, validation: ValidationResult):
        self.validation = validation
        super().__init__("validation_failed")


@dataclass(frozen=True)
class EditorialResult:
    candidate: Candidate
    validation: ValidationResult
    writer: TurnEvidence
    verifier: TurnEvidence
    repairs: tuple[TurnEvidence, ...] = ()


def _figure_issues(document: Document) -> list[dict[str, str]]:
    """Diagnose only the writer's figure/slot correspondence, never repair HTML."""
    index = HTMLIndex(document.html)
    figures = {figure.figure_id: figure for figure in document.figures}
    issues = []
    for figure in document.figures:
        problem = (
            "missing_slot"
            if figure.figure_id not in index.figures
            else "wrong_content_block"
            if figure.content_id not in index.figure_blocks[figure.figure_id]
            else None
        )
        if problem:
            issues.append(
                {
                    "figure_id": figure.figure_id,
                    "content_id": figure.content_id,
                    "problem": problem,
                }
            )
    for figure_id in sorted(index.figures - figures.keys()):
        issues.append({"figure_id": figure_id, "problem": "missing_metadata"})
    return issues


class _EvidenceIndex(HTMLIndex):
    """Collect text scoped by evidence references using the contract's HTML parser."""

    def __init__(self, html: str):
        self.evidence_text: dict[str, list[str]] = {}
        self.evidence_stack: list[str | None] = []
        super().__init__(html)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        depth = len(self.stack)
        super().handle_starttag(tag, attrs)
        evidence_id = dict(attrs).get("data-evidence-id")
        if evidence_id:
            self.evidence_text.setdefault(evidence_id, [])
        if len(self.stack) > depth:
            self.evidence_stack.append(evidence_id)

    def _end(self, tag: str, end: int) -> None:
        super()._end(tag, end)
        self.evidence_stack.pop()

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        for evidence_id in set(self.evidence_stack) - {None}:
            if evidence_id is not None:
                self.evidence_text[evidence_id].append(data)


def _numbers(text: str) -> set[Decimal]:
    return {
        Decimal(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9_.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text
        )
    }


def _values(value: Any) -> set[Decimal]:
    if type(value) in (int, float):
        return {Decimal(str(value))}
    if isinstance(value, dict):
        return set().union(*(_values(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_values(item) for item in value))
    return set()


def _evidence_map(evidence: list[Evidence]) -> dict[str, Evidence]:
    result = {item.evidence_id: item for item in evidence}
    if len(result) != len(evidence):
        raise ContractError("invalid_request")
    return result


def _check_context(context: WritingContext, current: Revision | None) -> None:
    _evidence_map(context.evidence)
    # Compare declarations through the shared contract helper, not a second
    # implementation of evidence change accounting.
    desired = Document(
        document_id=current.document.document_id if current else "initial",
        purpose=context.purpose,
        audience=context.audience,
        evidence=context.evidence,
        html="<p data-content-id='context'>Context</p>",
    )
    expected = evidence_changes(
        current.document if current else None, desired, reason="context"
    )
    authorized = context.authorized_evidence_changes
    if current is None:
        if authorized:
            raise ContractError("invalid_request")
    elif {(item.evidence_id, item.kind) for item in expected} != {
        (item.evidence_id, item.kind) for item in authorized
    } or len(expected) != len(authorized):
        raise ContractError("invalid_request")


def _check_preservation(
    candidate: Candidate, context: WritingContext, current: Revision | None
) -> None:
    document = candidate.document
    if (
        document.purpose != context.purpose
        or document.audience != context.audience
        or _evidence_map(document.evidence) != _evidence_map(context.evidence)
    ):
        raise ContractError("validation_failed")
    new = _EvidenceIndex(document.html)
    old = _EvidenceIndex(current.document.html) if current else None
    previous = _evidence_map(current.document.evidence) if current else {}
    for fact in document.evidence:
        was = previous.get(fact.evidence_id)
        if (
            old
            and fact.evidence_id in old.evidence
            and fact.evidence_id not in new.evidence
        ):
            raise ContractError("validation_failed")
        # Keep an existing rendered critical numeric value in its evidence scope.
        # Extra numbers, paraphrases and new examples still need semantic review.
        if old and was == fact:
            before = _numbers("".join(old.evidence_text.get(fact.evidence_id, [])))
            after = _numbers("".join(new.evidence_text.get(fact.evidence_id, [])))
            if not (_values(fact.values) & before) <= after:
                raise ContractError("validation_failed")
        source = fact.provenance.source
        if source.startswith(("https://", "http://")):
            links = {url for url, citation in new.references if citation}
            if source not in links:
                raise ContractError("validation_failed")


class GeminiEditor:
    def __init__(self, gemini: Gemini, *, compile_document: DocumentCompiler):
        """Require an explicit trusted compiler, even for figure-free documents.

        The compiler receives an isolated Document and returns its final form
        (including generated asset metadata), or raises ContractError. The caller
        retains generated bytes privately for staging and rendering validation.
        """
        if not callable(compile_document):
            raise ContractError("invalid_request")
        self.gemini = gemini
        self.compile_document = compile_document

    def write_initial(
        self,
        request: EditRequest,
        context: WritingContext,
        *,
        writer_timeout: float = WRITER_TIMEOUT,
        verifier_timeout: float = VERIFIER_TIMEOUT,
        cancel: Event | None = None,
    ) -> EditorialResult:
        """Generate an initial Candidate using the null-base document request."""
        return self._generate(
            request,
            context,
            None,
            writer_timeout=writer_timeout,
            verifier_timeout=verifier_timeout,
            cancel=cancel,
        )

    def edit(
        self,
        request: EditRequest,
        current: Revision,
        context: WritingContext,
        *,
        writer_timeout: float = WRITER_TIMEOUT,
        verifier_timeout: float = VERIFIER_TIMEOUT,
        cancel: Event | None = None,
    ) -> EditorialResult:
        """Generate a complete candidate against the supplied immutable revision."""
        return self._generate(
            request,
            context,
            current,
            writer_timeout=writer_timeout,
            verifier_timeout=verifier_timeout,
            cancel=cancel,
        )

    def _generate(
        self,
        request: EditRequest,
        context: WritingContext,
        current: Revision | None,
        *,
        writer_timeout: float,
        verifier_timeout: float,
        cancel: Event | None,
    ) -> EditorialResult:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (writer_timeout, verifier_timeout)
        ):
            raise ContractError("invalid_request")
        check_cancel(cancel)
        validate_request(request, current)
        _check_context(context, current)
        # There is no previous session or implicit memory. Every turn receives
        # the complete context, current document, request and change history.
        inputs: dict[str, Any] = {
            "mode": "initial" if current is None else "edit",
            "context": context.model_dump(mode="json"),
            "current_revision": current.model_dump(mode="json") if current else None,
            "request": request.model_dump(mode="json"),
        }
        # A concrete wire-shape example prevents ID-only evidence lists and
        # invented request fields observed in native finish attempts. It carries
        # no prose/layout; the writer must fill HTML and change accounting.
        writer_inputs = {
            **inputs,
            "candidate_template": {
                "request": inputs["request"],
                "document": {
                    "document_id": request.document_id,
                    **{
                        key: inputs["context"][key]
                        for key in (
                            "purpose",
                            "audience",
                            "evidence",
                            "assets",
                            "figures",
                        )
                    },
                    "html": "",
                },
                "changes": [],
                "evidence_changes": [
                    {
                        "evidence_id": item.evidence_id,
                        "kind": "added",
                        "reason": "Supplied initial evidence.",
                    }
                    for item in context.evidence
                ]
                if current is None
                else [
                    item.model_dump(mode="json")
                    for item in context.authorized_evidence_changes
                ],
                "change_summary": "",
            },
        }
        prompts = Path(__file__).with_name("prompts")
        writer = self.gemini.turn(
            (prompts / "writer.md").read_text()
            + "\nINPUT_JSON\n"
            + json.dumps(writer_inputs, ensure_ascii=False),
            Candidate.model_json_schema(),
            timeout=writer_timeout,
            cancel=cancel,
        )
        repairs: tuple[TurnEvidence, ...] = ()
        try:
            candidate = parse(Candidate, json.dumps(writer.output, ensure_ascii=False))
            if candidate.request != request:
                raise ContractError("invalid_candidate")
            issues = _figure_issues(candidate.document)
        except ContractError as exc:
            raise ContractError("invalid_model_output") from exc
        if issues:
            check_cancel(cancel)
            repaired = self.gemini.turn(
                (prompts / "writer.md").read_text()
                + "\nINPUT_JSON\n"
                + json.dumps(
                    {
                        **writer_inputs,
                        "figure_repair": {
                            "attempt": 1,
                            "max_attempts": 1,
                            "timeout_seconds": FIGURE_REPAIR_TIMEOUT,
                            "issues": issues,
                            "rejected_candidate": candidate.model_dump(mode="json"),
                        },
                    },
                    ensure_ascii=False,
                ),
                Candidate.model_json_schema(),
                timeout=FIGURE_REPAIR_TIMEOUT,
                cancel=cancel,
            )
            repairs = (repaired.evidence,)
            try:
                candidate = parse(
                    Candidate, json.dumps(repaired.output, ensure_ascii=False)
                )
            except ContractError as exc:
                raise ContractError("invalid_model_output") from exc
        try:
            if candidate.request != request or _figure_issues(candidate.document):
                raise ContractError("invalid_candidate")
            validate_candidate(candidate, current)
        except ContractError as exc:
            raise ContractError("invalid_model_output") from exc
        # Only the trusted compiler may add generated assets. The writer still
        # has to preserve the exact asset manifest supplied by orchestration.
        if candidate.document.assets != context.assets:
            raise ContractError("validation_failed")
        _check_preservation(candidate, context, current)
        check_cancel(cancel)
        compiled = self.compile_document(candidate.document.model_copy(deep=True))
        if not isinstance(compiled, Document):
            raise ContractError("invalid_document")
        # Reparse to validate and detach mutable lists even from model_copy().
        compiled = parse(Document, compiled.model_dump_json())
        candidate = candidate.model_copy(
            update={
                "document": compiled,
                "changes": content_changes(
                    current.document if current else None, compiled
                ),
            }
        )
        validate_candidate(candidate, current)
        _check_preservation(candidate, context, current)
        digest = candidate_digest(candidate)
        inputs.update(
            candidate=candidate.model_dump(mode="json"), candidate_digest=digest
        )
        verifier = self.gemini.turn(
            (prompts / "verifier.md").read_text()
            + "\nINPUT_JSON\n"
            + json.dumps(inputs, ensure_ascii=False),
            SemanticVerdict.model_json_schema(),
            timeout=verifier_timeout,
            cancel=cancel,
        )
        try:
            verdict = parse(
                SemanticVerdict, json.dumps(verifier.output, ensure_ascii=False)
            )
        except ContractError as exc:
            raise ContractError("invalid_model_output") from exc
        if verdict.candidate_digest != digest:
            raise ContractError("validation_mismatch")
        reviewed = verdict.reviewed_evidence_ids
        expected_ids = {item.evidence_id for item in context.evidence}
        if current:
            expected_ids.update(item.evidence_id for item in current.document.evidence)
        if set(reviewed) != expected_ids or len(reviewed) != len(expected_ids):
            raise ContractError("invalid_model_output")
        semantic = verdict.semantic_check()
        validation = ValidationResult(
            candidate_digest=digest,
            structural=CheckResult(status="passed"),
            semantic=semantic,
            rendering=CheckResult(status="not_run"),
        )
        if semantic.status != "passed":
            raise SemanticValidationError(validation)
        check_cancel(cancel)
        return EditorialResult(
            candidate,
            validation,
            writer.evidence,
            verifier.evidence,
            repairs,
        )
