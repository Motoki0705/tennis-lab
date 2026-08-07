from __future__ import annotations

from src.automation.chatgpt_mcp.server import _run_probe


def test_run_probe_reports_missing_executable() -> None:
    result = _run_probe(["/definitely/missing/tennis-mcp-command"])

    assert result["ok"] is False
    assert "FileNotFoundError" in result["output"]
