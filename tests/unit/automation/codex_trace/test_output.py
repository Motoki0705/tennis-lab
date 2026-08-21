from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.automation.codex_trace.analyzer import TraceAnalyzer
from src.automation.codex_trace.bundle import TraceBundle
from src.automation.codex_trace.output import write_json_report, write_sqlite_report
from tests.support.codex_trace import write_sample_trace_bundle


def test_output_writes_queryable_json_and_sqlite(tmp_path: Path) -> None:
    report = TraceAnalyzer(
        TraceBundle.load(write_sample_trace_bundle(tmp_path), auto_reduce=False)
    ).analyze()
    json_path = tmp_path / "analysis.json"
    sqlite_path = tmp_path / "analysis.sqlite"

    write_json_report(report, json_path, force=False)
    write_sqlite_report(report, sqlite_path, force=False)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["totals_exact"]["input_tokens"] == 240
    connection = sqlite3.connect(sqlite_path)
    try:
        inference_count = connection.execute(
            "SELECT COUNT(*) FROM inference_steps"
        ).fetchone()
        tool_count = connection.execute("SELECT COUNT(*) FROM tool_calls").fetchone()
        assert inference_count == (2,)
        assert tool_count == (2,)
    finally:
        connection.close()


def test_output_refuses_to_replace_without_force(tmp_path: Path) -> None:
    report = TraceAnalyzer(
        TraceBundle.load(write_sample_trace_bundle(tmp_path), auto_reduce=False)
    ).analyze()
    output = tmp_path / "analysis.json"
    write_json_report(report, output, force=False)

    with pytest.raises(FileExistsError, match="--force"):
        write_json_report(report, output, force=False)
