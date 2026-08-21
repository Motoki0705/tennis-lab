from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.automation.codex_trace.analyzer import TraceAnalyzer
from src.automation.codex_trace.bundle import TraceBundle
from src.automation.codex_trace.model import AnalysisReport
from src.automation.codex_trace.visualization import (
    render_html_report,
    write_html_report,
    write_png_report,
)
from tests.support.codex_trace import write_sample_trace_bundle


def _report(tmp_path: Path) -> AnalysisReport:
    bundle = TraceBundle.load(write_sample_trace_bundle(tmp_path), auto_reduce=False)
    return TraceAnalyzer(bundle).analyze()


def test_html_report_contains_graphs_but_not_sensitive_raw_content(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    rendered = render_html_report(report)

    assert rendered.count("<svg") >= 5
    assert "Exact tokens by inference" in rendered
    assert "Estimated input attribution" in rendered
    assert "Execution timeline" in rendered
    assert "direct_tool_items" in rendered
    assert report.source.bundle_path not in rendered
    assert "Inspect and update src/example.py" not in rendered


def test_html_and_png_reports_are_written_atomically(tmp_path: Path) -> None:
    report = _report(tmp_path)
    html_path = tmp_path / "report.html"
    png_path = tmp_path / "report.png"

    write_html_report(report, html_path, force=False)
    write_png_report(report, png_path, force=False)

    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    with Image.open(png_path) as image:
        assert image.format == "PNG"
        assert image.width >= 2_000
        assert image.height >= 1_000
    with pytest.raises(FileExistsError, match="--force"):
        write_html_report(report, html_path, force=False)
