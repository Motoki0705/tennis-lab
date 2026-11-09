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
from pathlib import Path
from threading import Event
from uuid import uuid4

from pydantic import Field

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / ".agents/ops"))

from explain_html.editor import GeminiEditor, WritingContext  # noqa: E402
from explain_html.gemini import DEFAULT_AGY, Gemini  # noqa: E402
from explain_html.jobs import PreparedEdit  # noqa: E402
from explain_html.managed import ManagedResources  # noqa: E402
from explain_html.paths import SafeDirectory  # noqa: E402
from explain_html.schema import (  # noqa: E402
    Candidate,
    CheckResult,
    ContractError,
    DocumentTarget,
    EditRequest,
    ErrorResponse,
    Identifier,
    Model,
    Revision,
    RevisionInfo,
    Text,
    parse,
)
from explain_html.store import RevisionStore  # noqa: E402
from explain_html.visuals import managed_renderer, validate_visuals  # noqa: E402
from lib.opsproc import CommandError, ProcessCancelled, run_process_group  # noqa: E402

WRITING_HINT = (
    "The runtime's conservative CSS validator rejects declaration names ending "
    "in behavior, including scroll-behavior. Omit those declarations entirely; "
    "use normal instant scrolling. Use prefers-reduced-motion without that property."
)


class Brief(Model):
    document_id: Identifier
    instruction: Text
    context: WritingContext
    asset_files: dict[Identifier, Text] = Field(default_factory=dict)


def prepare_edit(
    request: EditRequest,
    current: Revision,
    history: list[RevisionInfo],
    *,
    cancel: Event,
) -> PreparedEdit:
    prepared = ManagedResources().prepare(request, current, history, cancel=cancel)
    return PreparedEdit(
        prepared.context.model_copy(
            update={"preferences": prepared.context.preferences + "\n" + WRITING_HINT}
        ),
        prepared.assets,
    )


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
    context = brief.context.model_copy(
        update={"preferences": brief.context.preferences + "\n" + WRITING_HINT}
    )
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
    try:
        revision = store.commit(
            staged, result.validation.model_copy(update={"rendering": rendering})
        )
    except ContractError as error:
        # An acknowledgement can fail after the atomic current replacement.
        # Report success only if the public store verifies this exact request.
        try:
            revision = store.current(brief.document_id)
        except ContractError as lookup:
            if lookup.error.code == "document_not_found":
                raise error from lookup
            raise
        if (
            revision.request_id != request.request_id
            or revision.parent_revision is not None
            or revision.document != candidate.document
            or revision.changes != candidate.changes
            or revision.evidence_changes != candidate.evidence_changes
        ):
            raise error
    return {"document_id": brief.document_id, "revision_id": revision.revision_id}


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
        "--prepare",
        "explain_html_author:prepare_edit",
        "--validator",
        "explain_html_author:validate_bundle",
    )


def main() -> int:
    logging.getLogger().addHandler(logging.NullHandler())
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "open", "status", "stop", "export"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "create":
            command.add_argument("--brief", type=Path, required=True)
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
