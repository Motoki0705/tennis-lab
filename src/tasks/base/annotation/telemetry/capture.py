"""Stream one completed rollout; whitelist operational records, never reasoning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..artifacts import (
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..contracts import ArtifactRef, canonical_bytes, content_digest
from ..paths import AnnotationLayout
from ..plan import file_digest, load_plan, references
from ..workers.contracts import ExitReceipt, Launch
from .contracts import (
    RenderedImage,
    ResponseEntry,
    ResponsePage,
    ResponseRecord,
    TelemetryReport,
    ToolEvidence,
    ToolOutput,
    Usage,
)
from .images import archive_image
from .viewer import index_page, response_page

MAX_RECORD_BYTES = 64 * 1024**2
MAX_PENDING_OUTPUTS = 1024
Classifier = Callable[[str | None, str], str]
EvidenceResolver = Callable[[str, str], ToolEvidence | None]


def locate_rollout(launch: Launch, execution: ExitReceipt) -> Path:
    thread = execution.thread_id
    if (
        thread is None
        or not thread
        or any(char not in "0123456789abcdef-" for char in thread)
    ):
        raise ValueError("a valid worker thread ID is required to locate its rollout")
    paths = list(
        (Path(launch.codex_home) / "sessions").glob(f"*/*/*/rollout-*-{thread}.jsonl")
    )
    if len(paths) != 1:
        raise ValueError(
            f"expected exactly one rollout for this thread, found {len(paths)}"
        )
    return paths[0]


def _usage(value: object) -> Usage | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("usage must be an object or explicitly missing")
    fields = {key: value[key] for key in Usage.model_fields if key in value}
    if not fields:
        return None
    result: Usage = Usage.model_validate(fields)
    if (
        result.input_tokens is not None
        and result.cached_input_tokens is not None
        and result.cached_input_tokens > result.input_tokens
    ):
        raise ValueError("cached input tokens exceed total input tokens")
    return result


@dataclass
class _Call:
    name: str | None
    digest: str
    chars: int
    category: str
    timestamp: str | None
    views: list[str] = field(default_factory=list)


def _records(path: Path, digest: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        number = 0
        while raw := handle.readline(MAX_RECORD_BYTES + 1):
            number += 1
            if len(raw) > MAX_RECORD_BYTES:
                raise ValueError("rollout record exceeds the bounded parser size")
            digest.update(raw)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("rollout event is not an object")
            yield number, value


def capture(
    layout: AnnotationLayout,
    launch_ref: ArtifactRef,
    execution_ref: ArtifactRef,
    catalog: tuple[RenderedImage, ...],
    *,
    rollout: Path | None = None,
    classifier: Classifier | None = None,
    classifier_id: str = "unclassified.v1",
    evidence_resolver: EvidenceResolver | None = None,
) -> ArtifactRef:
    if not classifier_id.strip() or (
        (classifier is not None or evidence_resolver is not None)
        and classifier_id == "unclassified.v1"
    ):
        raise ValueError("task classification requires an explicit version identity")
    launch = read_document(layout, launch_ref, Launch)
    execution = read_document(layout, execution_ref, ExitReceipt)
    if execution.launch_digest != launch_ref.digest or execution.thread_id is None:
        raise ValueError(
            "telemetry execution must identify this exact launch and thread"
        )
    if rollout is not None and layout.mode != "test":
        raise ValueError(
            "production rollouts must be located from the owning Codex home/thread"
        )
    path = rollout if rollout is not None else locate_rollout(launch, execution)
    plan = load_plan(verify_reference(layout, launch.plan))
    for entry in catalog:
        for ref in references(entry):
            verify_reference(layout, ref)
    catalog_hash: str = content_digest(
        [entry.model_dump(mode="json") for entry in catalog]
    )
    directory = layout.path(
        "campaigns", plan.campaign_id, "reports", "telemetry", launch.attempt
    )
    with artifact_lock(layout, "telemetry-" + launch.attempt):
        report_path = directory / "report.json"
        if report_path.exists():
            from ..artifacts import annotation_reference

            reference = annotation_reference(layout, report_path)
            prior = read_document(layout, reference, TelemetryReport)
            if (
                prior.launch,
                prior.execution,
                prior.catalog_digest,
                prior.rollout_digest,
                prior.classifier_id,
            ) != (
                launch_ref,
                execution_ref,
                catalog_hash,
                file_digest(path),
                classifier_id,
            ):
                raise ValueError("completed telemetry input changed")
            verify_report(layout, reference)
            return reference
        return _capture(
            layout,
            directory,
            path,
            launch_ref,
            execution_ref,
            launch,
            execution,
            catalog,
            catalog_hash,
            classifier,
            classifier_id,
            evidence_resolver,
        )


def _capture(
    layout: AnnotationLayout,
    directory: Path,
    path: Path,
    launch_ref: ArtifactRef,
    execution_ref: ArtifactRef,
    launch: Launch,
    execution: ExitReceipt,
    catalog: tuple[RenderedImage, ...],
    catalog_hash: str,
    classifier: Classifier | None,
    classifier_id: str,
    evidence_resolver: EvidenceResolver | None,
) -> ArtifactRef:
    before = path.stat()
    digest = hashlib.sha256()
    calls: dict[str, _Call] = {}
    seen_ids: dict[str, str] = {}
    pending: list[ToolOutput] = []
    orphans: list[ToolOutput] = []
    entries: list[ResponseEntry] = []
    page_refs: list[ArtifactRef] = []
    duplicate = conflicting = missing_id = missing_usage = notifications = (
        compactions
    ) = 0
    image_count = unbound_count = 0
    context_window: int | None = None
    metadata: dict[str, Any] | None = None
    models: set[str] = set()
    known = {key: 0 for key in Usage.model_fields}
    missing = {key: 0 for key in Usage.model_fields}
    for line, event in _records(path, digest):
        kind = event.get("type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("rollout payload must be an object")
        subtype = payload.get("type")
        stamp = event.get("timestamp")
        if line == 1 and kind != "session_meta":
            raise ValueError("rollout must begin with its session identity")
        if kind == "session_meta":
            if (
                metadata is not None
                or payload.get("id") != execution.thread_id
                or payload.get("cwd") != str(Path(launch.directory) / "work")
            ):
                raise ValueError(
                    "rollout identity/CWD does not match the worker attempt"
                )
            metadata = {
                key: payload.get(key)
                for key in ("id", "cwd", "cli_version", "model_provider")
            }
        elif kind == "turn_context":
            if isinstance(payload.get("model"), str):
                models.add(payload["model"])
        elif kind == "event_msg" and subtype == "token_count":
            notifications += 1
            info = payload.get("info") or {}
            window = info.get("model_context_window")
            if window is not None:
                if type(window) is not int or window <= 0:
                    raise ValueError("invalid logged context window")
                context_window = window
        elif kind in {"compacted", "compaction"} or subtype in {
            "context_compacted",
            "compaction",
        }:
            compactions += 1
            # A replacement context need not retain the preceding tool images.
            # Preserve them as unassigned instead of claiming a later API saw them.
            orphans.extend(pending)
            pending.clear()
            if len(orphans) > MAX_PENDING_OUTPUTS:
                raise ValueError("too many unassigned tool outputs")
        elif kind == "response_item" and subtype in {
            "custom_tool_call",
            "function_call",
        }:
            call_id = payload.get("call_id")
            raw_input = payload.get("input", payload.get("arguments", ""))
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in calls
                or not isinstance(raw_input, str)
            ):
                raise ValueError("invalid or duplicate outstanding tool call")
            name = payload.get("name")
            if name is not None and not isinstance(name, str):
                raise ValueError("tool name must be a string")
            calls[call_id] = _Call(
                name,
                hashlib.sha256(raw_input.encode()).hexdigest(),
                len(raw_input),
                "tool" if classifier is None else classifier(name, raw_input),
                stamp,
            )
            if len(calls) > MAX_PENDING_OUTPUTS:
                raise ValueError("too many outstanding tool calls")
        elif kind == "event_msg" and subtype == "item_completed":
            item = payload.get("item") or {}
            if item.get("type") == "ImageView":
                source = item.get("path")
                if not isinstance(source, str):
                    raise ValueError("ImageView path is missing")
                source = (
                    unquote(urlparse(source).path)
                    if source.startswith("file:")
                    else source
                )
                explicit = payload.get("call_id")
                call = (
                    calls.get(explicit)
                    if isinstance(explicit, str)
                    else next(iter(calls.values()))
                    if len(calls) == 1
                    else None
                )
                if call is not None:
                    call.views.append(source)
        elif kind == "response_item" and subtype in {
            "custom_tool_call_output",
            "function_call_output",
        }:
            call_id = payload.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("tool output call ID is missing")
            call = calls.pop(call_id, None)
            blocks = payload.get("output", [])
            if isinstance(blocks, str):
                blocks = [{"type": "input_text", "text": blocks}]
            if not isinstance(blocks, list) or any(
                not isinstance(block, dict) for block in blocks
            ):
                raise ValueError("unsupported tool-output blocks")
            views = () if call is None else tuple(call.views)
            images = tuple(
                archive_image(
                    layout,
                    block,
                    line=line,
                    block_index=i,
                    catalog=catalog,
                    view_paths=views,
                )
                for i, block in enumerate(blocks)
                if block.get("type") == "input_image"
            )
            text_parts = [
                block.get("text", "")
                for block in blocks
                if block.get("type") in {"input_text", "text"}
            ]
            if any(not isinstance(text, str) for text in text_parts):
                raise ValueError("tool text block must contain text")
            category = "unknown" if call is None else call.category
            evidence = (
                evidence_resolver(category, "\n".join(text_parts))
                if evidence_resolver is not None
                else None
            )
            if evidence is not None:
                for reference in references(evidence):
                    verify_reference(layout, reference)
            pending.append(
                ToolOutput(
                    call_id=call_id,
                    name=None if call is None else call.name,
                    input_digest=None if call is None else call.digest,
                    input_chars=0 if call is None else call.chars,
                    input_timestamp=None if call is None else call.timestamp,
                    output_digest=hashlib.sha256(canonical_bytes(blocks)).hexdigest(),
                    text_chars=sum(len(text) for text in text_parts),
                    output_line=line,
                    timestamp=stamp,
                    images=images,
                    observed_view_paths=views,
                    category=category,
                    evidence=evidence,
                )
            )
            if len(pending) > MAX_PENDING_OUTPUTS:
                raise ValueError("too many tool outputs without a response boundary")
        elif kind == "token_usage_record":
            if any(
                payload.get(key) is not None and payload[key] != execution.thread_id
                for key in ("session_id", "thread_id")
            ):
                raise ValueError("usage record belongs to another thread/session")
            response_id = payload.get("response_id")
            usage = _usage(payload.get("usage"))
            if not isinstance(response_id, str) or not response_id:
                missing_id += 1
                orphans.extend(pending)
                pending.clear()
                if len(orphans) > MAX_PENDING_OUTPUTS:
                    raise ValueError("too many unassigned tool outputs")
                continue
            signature: str = content_digest(
                None if usage is None else usage.model_dump(mode="json")
            )
            if response_id in seen_ids:
                duplicate += 1
                if seen_ids[response_id] != signature:
                    conflicting += 1
                continue
            seen_ids[response_id] = signature
            if usage is None:
                missing_usage += 1
            for key in known:
                value = None if usage is None else getattr(usage, key)
                if value is None:
                    missing[key] += 1
                else:
                    known[key] += value
            proxy = (
                None
                if usage is None
                or usage.input_tokens is None
                or usage.output_tokens is None
                else usage.input_tokens + usage.output_tokens
            )
            record = ResponseRecord(
                index=len(entries) + 1,
                response_id=response_id,
                thread_id=payload.get("thread_id"),
                session_id=payload.get("session_id"),
                turn_id=payload.get("turn_id"),
                root_turn_id=payload.get("root_turn_id"),
                rollout_line=line,
                timestamp=stamp,
                usage=usage,
                logged_thread_usage=_usage(payload.get("thread_token_usage")),
                logged_turn_usage=_usage(payload.get("turn_token_usage")),
                context_window=context_window,
                context_proxy_tokens=proxy,
                tool_outputs=tuple(pending),
            )
            count = sum(len(output.images) for output in pending)
            image_count += count
            unbound_count += sum(
                image.source is None for output in pending for image in output.images
            )
            reference = write_document(
                layout, directory / f"response-{record.index:05d}.json", record
            )
            entries.append(
                ResponseEntry(
                    index=record.index,
                    response_id=response_id,
                    record=reference,
                    page=response_page(layout, directory, record),
                    images=count,
                    text_chars=sum(output.text_chars for output in pending),
                    input_tokens=None if usage is None else usage.input_tokens,
                    output_tokens=None if usage is None else usage.output_tokens,
                )
            )
            pending.clear()
    if metadata is None:
        raise ValueError("empty rollout has no session identity")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("rollout changed while telemetry was captured")
    orphans.extend(pending)
    if len(orphans) > MAX_PENDING_OUTPUTS:
        raise ValueError("too many unassigned tool outputs")
    orphan_ref = write_asset(
        layout,
        directory / "orphan-outputs.json",
        canonical_bytes([output.model_dump(mode="json") for output in orphans]) + b"\n",
    )
    for start in range(0, len(entries), 100):
        page_refs.append(
            write_document(
                layout,
                directory / f"responses-{start // 100:04d}.json",
                ResponsePage(entries=tuple(entries[start : start + 100])),
            )
        )
    complete_totals = {
        key: value
        if entries and not missing[key] and not missing_id and not conflicting
        else None
        for key, value in known.items()
    }
    summary = f"応答ID {len(entries)}件 / 重複通知 {duplicate}件 / ID欠損 {missing_id}件 / usage欠損 {missing_usage}件。画像単独のtoken料金は算出しません。"
    if not entries:
        summary += " 応答別usage未観測のため合計は不明です。"
    return write_document(
        layout,
        directory / "report.json",
        TelemetryReport(
            launch=launch_ref,
            execution=execution_ref,
            thread_id=execution.thread_id or "",
            rollout_path=str(path),
            rollout_digest=digest.hexdigest(),
            rollout_bytes=after.st_size,
            cli_version=metadata.get("cli_version"),
            model_provider=metadata.get("model_provider"),
            model_contexts=tuple(sorted(models)),
            responses=len(entries),
            duplicate_usage_records=duplicate,
            conflicting_usage_records=conflicting,
            missing_response_ids=missing_id,
            missing_usage=missing_usage,
            token_notifications=notifications,
            compactions=compactions,
            orphan_outputs=orphan_ref,
            orphan_output_count=len(orphans),
            pages=tuple(page_refs),
            index_page=index_page(layout, directory, entries, summary),
            totals=complete_totals,
            known_subtotals=known,
            missing_metric_counts=missing,
            image_inputs=image_count,
            unbound_image_inputs=unbound_count,
            catalog_digest=catalog_hash,
            classifier_id=classifier_id,
            prompt_chars=len(
                verify_reference(layout, launch.prepared.prompt).read_text()
            ),
            elapsed_seconds=execution.elapsed_seconds,
        ),
    )


def iter_responses(
    layout: AnnotationLayout, reference: ArtifactRef
) -> Iterator[ResponseRecord]:
    report = read_document(layout, reference, TelemetryReport)
    count = 0
    for page_ref in report.pages:
        response_index = read_document(layout, page_ref, ResponsePage)
        for entry in response_index.entries:
            record = read_document(layout, entry.record, ResponseRecord)
            count += 1
            if (record.index, record.response_id) != (count, entry.response_id):
                raise ValueError("response index and payload disagree")
            yield record
    if count != report.responses:
        raise ValueError("response count and report disagree")


def verify_report(layout: AnnotationLayout, reference: ArtifactRef) -> TelemetryReport:
    report: TelemetryReport = read_document(layout, reference, TelemetryReport)
    for ref in references(report):
        verify_reference(layout, ref)
    for page_ref in report.pages:
        page = read_document(layout, page_ref, ResponsePage)
        for ref in references(page):
            verify_reference(layout, ref)
    for response in iter_responses(layout, reference):
        for ref in references(response):
            verify_reference(layout, ref)
    orphans = json.loads(verify_reference(layout, report.orphan_outputs).read_bytes())
    if not isinstance(orphans, list) or len(orphans) != report.orphan_output_count:
        raise ValueError("orphan output count and report disagree")
    for value in orphans:
        output = ToolOutput.model_validate_json(json.dumps(value))
        for ref in references(output):
            verify_reference(layout, ref)
    return report


class ImagePresentationUnavailable(ValueError):
    """Authentic telemetry cannot establish which model received the images."""


def presented_images(
    layout: AnnotationLayout, reference: ArtifactRef
) -> tuple[ArtifactRef, ...]:
    """Only exact, catalog-bound bytes followed by an identified response qualify."""
    report = verify_report(layout, reference)
    if report.conflicting_usage_records:
        raise ImagePresentationUnavailable(
            "conflicting response IDs cannot prove image presentation"
        )
    launch = read_document(layout, report.launch, Launch)
    plan = load_plan(verify_reference(layout, launch.plan))
    if not report.model_contexts:
        raise ImagePresentationUnavailable(
            "logged model is missing; image presentation cannot be verified"
        )
    if report.model_contexts != (plan.agent_model,):
        raise ImagePresentationUnavailable(
            "logged model does not match the frozen campaign"
        )
    result: set[ArtifactRef] = set()
    for response in iter_responses(layout, reference):
        for output in response.tool_outputs:
            for image in output.images:
                verify_reference(layout, image.image)
                if image.source is not None:
                    for ref in references(image.source):
                        verify_reference(layout, ref)
                    result.add(image.source.image)
    return tuple(sorted(result, key=lambda ref: (ref.role, ref.path, ref.digest)))
