from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.codex_trace import write_sample_trace_bundle


@pytest.mark.e2e
def test_codex_trace_cli_analyzes_reduced_bundle(tmp_path: Path) -> None:
    bundle_path = write_sample_trace_bundle(tmp_path)
    json_path = tmp_path / "report.json"
    sqlite_path = tmp_path / "report.sqlite"
    html_path = tmp_path / "report.html"
    png_path = tmp_path / "report.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.automation.codex_trace",
            str(bundle_path),
            "--no-reduce",
            "--json",
            str(json_path),
            "--sqlite",
            str(sqlite_path),
            "--html",
            str(html_path),
            "--png",
            str(png_path),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Exact tokens: input=240" in result.stdout
    assert (
        json.loads(json_path.read_text(encoding="utf-8"))["source"]["trace_id"]
        == "trace-test"
    )
    assert sqlite_path.is_file()
    assert html_path.is_file()
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
