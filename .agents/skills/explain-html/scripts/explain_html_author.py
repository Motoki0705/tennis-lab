"""Initial authoring and launch glue for the explain-html skill.

Wire models, Gemini pins, compilation, publication and service lifecycle belong
to explain_html; this helper supplies the brief and real browser boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Literal
from uuid import uuid4

from pydantic import Field

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / ".agents/ops"))

from explain_html.editor import GeminiEditor, WritingContext  # noqa: E402
from explain_html.gemini import DEFAULT_AGY, Gemini  # noqa: E402
from explain_html.managed import ManagedResources  # noqa: E402
from explain_html.paths import SafeDirectory  # noqa: E402
from explain_html.schema import (  # noqa: E402
    Candidate,
    CheckResult,
    ContractError,
    Digest,
    DocumentTarget,
    EditRequest,
    ErrorResponse,
    Identifier,
    Model,
    StagedCandidate,
    Text,
    ValidationResult,
    canonical_bytes,
    parse,
)
from explain_html.store import RevisionStore  # noqa: E402
from explain_html.visuals import managed_renderer, validate_visuals  # noqa: E402
from lib.opsproc import CommandError, ProcessCancelled, run_process_group  # noqa: E402


class Brief(Model):
    document_id: Identifier
    instruction: Text
    context: WritingContext
    asset_files: dict[Identifier, Text] = Field(default_factory=dict)


class EvidenceReview(Model):
    evidence_id: Identifier
    assessment: Text


class ReasoningReview(Model):
    claim: Text
    evidence_ids: list[Identifier]
    assessment: Text


class CodexReview(Model):
    """Trusted local review record, not a model verdict or proof of thought."""

    reviewer: Literal["codex"]
    candidate_digest: Digest
    check: CheckResult
    evidence_checks: list[EvidenceReview]
    asynchronous_observations: ReasoningReview
    sensitivity_fixed_quantities: ReasoningReview
    conditional_generalizations: ReasoningReview


class _PendingInitial(Model):
    staged: StagedCandidate
    validation: ValidationResult


def validate_bundle(
    candidate: Candidate, assets: Mapping[str, bytes], *, cancel: Event
) -> CheckResult:
    """Inspect the exact staged bytes in Chromium, also for reading-time edits."""
    validate_visuals(candidate.document, dict(assets))
    if cancel.is_set():
        raise ContractError("cancelled")
    with tempfile.TemporaryDirectory(prefix="explain-html-render-") as temporary:
        root = Path(temporary)
        (root / "index.html").write_text(candidate.document.html, encoding="utf-8")
        for asset in candidate.document.assets:
            destination = root / asset.path  # validated canonical managed path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(assets[asset.asset_id])
        try:
            output = run_process_group(
                ["node", str(SCRIPTS / "render_check.cjs"), str(root / "index.html")],
                cwd=REPO,
                timeout=45,
                cancel=cancel,
            )
            return parse(CheckResult, output)
        except ProcessCancelled as exc:
            raise ContractError("cancelled") from exc
        except CommandError as exc:
            raise ContractError("validation_failed") from exc


def create(brief_path: Path, root: Path, agy: Path) -> dict[str, object]:
    brief = parse(Brief, brief_path.read_text(encoding="utf-8"))
    store = RevisionStore(root)
    # Fail before spending a generation call when the document already exists.
    try:
        store.current(brief.document_id)
    except ContractError as exc:
        if exc.error.code != "document_not_found":
            raise
    else:
        raise ContractError("document_exists")
    payloads = {
        name: (brief_path.parent / path).read_bytes()
        for name, path in brief.asset_files.items()
    }
    context = brief.context
    if any(figure.kind == "graph" for figure in context.figures):
        imported = managed_renderer()
        renderer, content = imported.asset, imported.content
        if renderer.asset_id not in {asset.asset_id for asset in context.assets}:
            context = context.model_copy(update={"assets": [*context.assets, renderer]})
        if renderer.asset_id not in payloads:
            payloads[renderer.asset_id] = content
    if set(payloads) != {asset.asset_id for asset in context.assets}:
        raise ContractError("broken_reference")
    if any(
        hashlib.sha256(payloads[asset.asset_id]).hexdigest() != asset.sha256
        for asset in context.assets
    ):
        raise ContractError("invalid_document")
    request = EditRequest(
        document_id=brief.document_id,
        base_revision=None,
        request_id="initial-" + uuid4().hex,
        target=DocumentTarget(),
        instruction=brief.instruction,
    )
    editor = GeminiEditor(
        Gemini(root / "private/gemini", agy=agy),
        compile_document=ManagedResources().compiler(payloads),
    )
    result = editor.write_initial(request, context)
    staged = store.stage(result.candidate, payloads)
    candidate = store.staged_candidate(brief.document_id, staged.stage_id)
    staged_assets = {
        asset.asset_id: store.staged_asset_bytes(
            brief.document_id, staged.stage_id, asset.asset_id
        )
        for asset in candidate.document.assets
    }
    rendering = validate_bundle(candidate, staged_assets, cancel=Event())
    if rendering.status != "passed" or rendering.issues:
        logging.getLogger(__name__).warning(
            "Rendering check: %s", rendering.model_dump_json()
        )
        raise ContractError("validation_failed")
    validation = result.validation.model_copy(update={"rendering": rendering})
    pending = _PendingInitial(staged=staged, validation=validation)
    review_path = f"private/codex-reviews/{staged.stage_id}"
    with SafeDirectory(root) as filesystem:
        filesystem.write(f"{review_path}/candidate.json", canonical_bytes(candidate))
        filesystem.write(f"{review_path}/context.json", canonical_bytes(context))
        filesystem.write(f"{review_path}/request.json", canonical_bytes(request))
        filesystem.write(
            f"{review_path}/turns.json",
            json.dumps(
                {
                    "writer": asdict(result.writer),
                    "repairs": [asdict(turn) for turn in result.repairs],
                    "verifier": asdict(result.verifier),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8"),
        )
        filesystem.write(
            f"{review_path}/content/index.html", candidate.document.html.encode("utf-8")
        )
        for asset in candidate.document.assets:
            filesystem.write(
                f"{review_path}/content/{asset.path}", staged_assets[asset.asset_id]
            )
        filesystem.write(f"{review_path}/pending.json", canonical_bytes(pending))
    return {
        "state": "awaiting_codex_review",
        "document_id": brief.document_id,
        "stage_id": staged.stage_id,
        "candidate_digest": staged.candidate_digest,
        "review_directory": str(root / review_path),
    }


def publish(root: Path, stage_id: str, review_file: Path) -> dict[str, object]:
    review_path = f"private/codex-reviews/{stage_id}"
    with SafeDirectory(root) as filesystem:
        try:
            pending = parse(
                _PendingInitial, filesystem.read(f"{review_path}/pending.json")
            )
        except FileNotFoundError as exc:
            raise ContractError("candidate_not_found") from exc
    staged = pending.staged
    if staged.stage_id != stage_id or not pending.validation.passed:
        raise ContractError("validation_mismatch")
    store = RevisionStore(root)
    candidate = store.staged_candidate(staged.document_id, staged.stage_id)
    review = parse(CodexReview, review_file.read_bytes())
    if review.candidate_digest != staged.candidate_digest:
        raise ContractError("validation_mismatch")
    evidence_ids = {item.evidence_id for item in candidate.document.evidence}
    reviewed = [item.evidence_id for item in review.evidence_checks]
    if set(reviewed) != evidence_ids or len(reviewed) != len(evidence_ids):
        raise ContractError("invalid_request")
    for item in (
        review.asynchronous_observations,
        review.sensitivity_fixed_quantities,
        review.conditional_generalizations,
    ):
        if not set(item.evidence_ids) <= evidence_ids:
            raise ContractError("invalid_request")
    # A negative review is durable. Correct the writer input and create a new
    # candidate; do not turn the same rejected candidate into an approval.
    with SafeDirectory(root) as filesystem:
        record = f"{review_path}/codex-review.json"
        encoded = canonical_bytes(review)
        if filesystem.exists(record):
            if filesystem.read(record) != encoded:
                raise ContractError("request_conflict")
        else:
            filesystem.write(record, encoded)
    if review.check.status != "passed" or review.check.issues:
        raise ContractError("validation_failed")
    try:
        revision = store.commit(staged, pending.validation)
    except ContractError as error:
        # An acknowledgement can fail after the atomic current replacement.
        # Report success only if the public store verifies this exact request.
        try:
            revision = store.current(staged.document_id)
        except ContractError as lookup:
            if lookup.error.code == "document_not_found":
                raise error from lookup
            raise
        if (
            revision.request_id != candidate.request.request_id
            or revision.parent_revision is not None
            or revision.document != candidate.document
            or revision.changes != candidate.changes
            or revision.evidence_changes != candidate.evidence_changes
        ):
            raise error
    return {"document_id": staged.document_id, "revision_id": revision.revision_id}


def runtime(command: str, root: Path, *options: str) -> dict[str, object]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / ".agents/ops"), str(SCRIPTS), environment.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "explain_html", command, "--root", str(root), *options],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        error = parse(ErrorResponse, completed.stdout)
        raise ContractError(error.error.code)
    result: dict[str, object] = json.loads(completed.stdout)
    return result


def open_document(root: Path, document_id: str, agy: Path) -> dict[str, object]:
    RevisionStore(root).current(document_id)
    status = runtime("status", root)
    if status["running"]:
        # Keep the live session and token; only select another published document.
        token = str(status["url"]).split("token=", 1)[1]
        url = str(status["url"]).split("#", 1)[0]
        return {
            "running": True,
            "url": f"{url}#document_id={document_id}&token={token}",
        }
    return runtime(
        "start",
        root,
        "--document",
        document_id,
        "--agy",
        str(agy),
        "--ui",
        "explain_html.web:files",
        "--validator",
        "explain_html_author:validate_bundle",
    )


def main() -> int:
    logging.getLogger().addHandler(logging.NullHandler())
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "publish", "open", "status", "stop", "export"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "create":
            command.add_argument("--brief", type=Path, required=True)
        if name == "publish":
            command.add_argument("--stage", required=True)
            command.add_argument("--review-file", type=Path, required=True)
        if name in {"create", "open"}:
            command.add_argument("--agy", type=Path, default=DEFAULT_AGY)
        if name in {"open", "export"}:
            command.add_argument("--document", required=True)
        if name == "open":
            command.add_argument("--browser", action="store_true")
        if name == "export":
            command.add_argument("--revision")
    arguments = parser.parse_args()
    root = arguments.root.absolute()
    try:
        with SafeDirectory(root, create=True) as filesystem:
            filesystem.mkdir("private")
            with filesystem.open_file(
                "private/author.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND
            ) as descriptor:
                logging.getLogger().addHandler(
                    logging.StreamHandler(os.fdopen(os.dup(descriptor), "a"))
                )
        if arguments.command == "create":
            result = create(arguments.brief.absolute(), root, arguments.agy)
        elif arguments.command == "publish":
            result = publish(root, arguments.stage, arguments.review_file)
        elif arguments.command == "open":
            result = open_document(root, arguments.document, arguments.agy)
            if arguments.browser:
                result["browser_opened"] = webbrowser.open(str(result["url"]))
        else:
            options = []
            if arguments.command == "export":
                options = ["--document", arguments.document]
                if arguments.revision:
                    options += ["--revision", arguments.revision]
            result = runtime(arguments.command, root, *options)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Private authoring failure")
        error = (
            exc if isinstance(exc, ContractError) else ContractError("invalid_request")
        )
        print(ErrorResponse(error=error.error).model_dump_json())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
