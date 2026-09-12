from __future__ import annotations

from src.automation.chatgpt_mcp.scripts.capabilities import check_compiler


def test_compiler_probe_builds_and_runs_c_program() -> None:
    result = check_compiler()

    assert result["compiler"].endswith("cc")
    assert result["version"]
    assert result["compile_and_run"] is True
