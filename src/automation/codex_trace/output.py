"""JSON, SQLite, and terminal renderers for trace analysis reports."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.automation.codex_trace.model import AnalysisReport


def write_json_report(report: AnalysisReport, path: Path, *, force: bool) -> None:
    """Atomically write the complete report as formatted JSON."""

    with atomic_output(path, force=force) as temporary:
        temporary.write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_sqlite_report(report: AnalysisReport, path: Path, *, force: bool) -> None:
    """Atomically write normalized inference, segment, and tool-call tables."""

    with atomic_output(path, force=force) as temporary:
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            _insert_report(connection, report)
            connection.commit()
        finally:
            connection.close()


def render_summary(report: AnalysisReport) -> str:
    """Render a compact distinction between exact totals and estimates."""

    totals = report.totals_exact
    estimated_input = sum(
        step.input_tokens_estimated for step in report.inference_steps
    )
    residuals = [
        step.input_residual_tokens
        for step in report.inference_steps
        if step.input_residual_tokens is not None
    ]
    lines = [
        f"Trace: {report.source.trace_id} ({report.source.status})",
        f"Rollout: {report.source.rollout_id}",
        (
            "Inference calls: "
            f"{totals.inference_calls} "
            f"({totals.inference_calls_with_usage} with exact usage)"
        ),
        (
            "Exact tokens: "
            f"input={totals.input_tokens}, cached_input={totals.cached_input_tokens}, "
            f"output={totals.output_tokens}, reasoning={totals.reasoning_output_tokens}"
        ),
        (
            f"Estimated attributed input: {estimated_input} "
            f"({sum(residuals)} exact-minus-estimated residual)"
        ),
        f"Tool calls: {len(report.tool_calls)}",
        f"Warnings: {len(report.warnings)}",
    ]
    return "\n".join(lines)


@contextmanager
def atomic_output(path: Path, *, force: bool) -> Iterator[Path]:
    """Yield a temporary sibling and atomically install it on success."""

    resolved = path.expanduser().resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"output already exists (pass --force): {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{resolved.name}.", dir=resolved.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE inference_steps (
            inference_call_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            codex_turn_id TEXT NOT NULL,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            started_at_unix_ms INTEGER NOT NULL,
            ended_at_unix_ms INTEGER,
            status TEXT NOT NULL,
            input_tokens_exact INTEGER,
            cached_input_tokens_exact INTEGER,
            output_tokens_exact INTEGER,
            reasoning_tokens_exact INTEGER,
            input_tokens_estimated INTEGER NOT NULL,
            input_residual_tokens INTEGER,
            input_tokens_by_cluster_json TEXT NOT NULL,
            reasoning_cluster_probabilities_json TEXT NOT NULL,
            reasoning_tokens_by_cluster_json TEXT NOT NULL,
            reasoning_evidence_mode TEXT NOT NULL,
            request_payload_id TEXT NOT NULL,
            response_payload_id TEXT
        );
        CREATE TABLE segments (
            segment_id TEXT PRIMARY KEY,
            inference_call_id TEXT,
            direction TEXT NOT NULL,
            item_id TEXT,
            call_id TEXT,
            structural_type TEXT NOT NULL,
            semantic_cluster TEXT NOT NULL,
            cluster_probability REAL NOT NULL,
            tokens_estimated INTEGER NOT NULL,
            token_method TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            raw_payload_id TEXT,
            evidence_mode TEXT NOT NULL
        );
        CREATE TABLE tool_calls (
            tool_call_id TEXT PRIMARY KEY,
            inference_call_id TEXT,
            thread_id TEXT NOT NULL,
            codex_turn_id TEXT,
            kind TEXT NOT NULL,
            requester TEXT NOT NULL,
            started_at_unix_ms INTEGER NOT NULL,
            ended_at_unix_ms INTEGER,
            status TEXT NOT NULL,
            invocation_tokens_estimated INTEGER,
            raw_output_tokens_estimated INTEGER,
            original_output_tokens INTEGER,
            model_visible_output_tokens_estimated INTEGER NOT NULL,
            model_visible_output_evidence TEXT NOT NULL,
            token_method TEXT NOT NULL,
            invocation_payload_id TEXT,
            result_payload_id TEXT,
            files_touched_json TEXT NOT NULL,
            lines_added INTEGER NOT NULL,
            lines_deleted INTEGER NOT NULL
        );
        """
    )


def _insert_report(connection: sqlite3.Connection, report: AnalysisReport) -> None:
    metadata = {
        "analysis_schema_version": report.schema_version,
        "source": report.as_dict()["source"],
        "totals_exact": report.as_dict()["totals_exact"],
        "warnings": report.as_dict()["warnings"],
    }
    connection.executemany(
        "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
        [
            (key, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            for key, value in metadata.items()
        ],
    )
    for step in report.inference_steps:
        usage = step.usage_exact
        connection.execute(
            """
            INSERT INTO inference_steps VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                step.inference_call_id,
                step.thread_id,
                step.codex_turn_id,
                step.model,
                step.provider,
                step.started_at_unix_ms,
                step.ended_at_unix_ms,
                step.status,
                usage.input_tokens if usage else None,
                usage.cached_input_tokens if usage else None,
                usage.output_tokens if usage else None,
                usage.reasoning_output_tokens if usage else None,
                step.input_tokens_estimated,
                step.input_residual_tokens,
                _json(step.input_tokens_by_cluster_estimated),
                _json(step.reasoning_cluster_probabilities),
                _json(step.reasoning_tokens_by_cluster_estimated),
                step.reasoning_evidence_mode,
                step.request_payload_id,
                step.response_payload_id,
            ),
        )
    connection.executemany(
        "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                segment.segment_id,
                segment.inference_call_id,
                segment.direction,
                segment.item_id,
                segment.call_id,
                segment.structural_type,
                segment.semantic_cluster,
                segment.cluster_probability,
                segment.tokens_estimated,
                segment.token_method,
                segment.byte_count,
                segment.raw_payload_id,
                segment.evidence_mode,
            )
            for segment in report.segments
        ],
    )
    connection.executemany(
        "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                tool.tool_call_id,
                tool.inference_call_id,
                tool.thread_id,
                tool.codex_turn_id,
                tool.kind,
                tool.requester,
                tool.started_at_unix_ms,
                tool.ended_at_unix_ms,
                tool.status,
                tool.invocation_tokens_estimated,
                tool.raw_output_tokens_estimated,
                tool.original_output_tokens,
                tool.model_visible_output_tokens_estimated,
                tool.model_visible_output_evidence,
                tool.token_method,
                tool.invocation_payload_id,
                tool.result_payload_id,
                _json(tool.files_touched),
                tool.lines_added,
                tool.lines_deleted,
            )
            for tool in report.tool_calls
        ],
    )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
