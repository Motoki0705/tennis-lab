from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.types import LATEST_PROTOCOL_VERSION
from starlette.testclient import TestClient

from src.automation.chatgpt_mcp.server import _run_probe, build_gateway
from src.automation.chatgpt_mcp.settings import GatewaySettings

_EXPECTED_TOOLS = {
    "get_host_status",
    "prepare_revision_workspace",
    "get_revision_status",
    "start_command",
    "get_command_job",
    "list_command_jobs",
    "get_command_output",
    "cancel_command_job",
    "enqueue_training",
    "get_training_job",
    "get_training_output",
}

_FORBIDDEN_REPOSITORY_TOOLS = {
    "create_workspace",
    "list_workspace_files",
    "read_workspace_file",
    "search_workspace_code",
    "apply_workspace_patch",
    "get_workspace_diff",
    "commit_workspace",
    "push_workspace_branch",
}


def _settings(tmp_path: Path) -> GatewaySettings:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".venv/bin").mkdir(parents=True)
    uv_root = tmp_path / "uv-python"
    uv_root.mkdir()
    return GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        public_base_url=None,
        host="127.0.0.1",
        port=8767,
        uv_python_root=uv_root,
    )


def test_run_probe_reports_missing_executable() -> None:
    result = _run_probe(["/definitely/missing/tennis-mcp-command"])

    assert result["ok"] is False
    assert "FileNotFoundError" in result["output"]


def test_private_gateway_advertises_only_execution_plane_tools(tmp_path: Path) -> None:
    app = build_gateway(_settings(tmp_path), authenticated=False).streamable_http_app()
    headers = {
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
    }

    with TestClient(app, base_url="http://127.0.0.1:8767") as client:
        initialize = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1"},
                },
            },
            headers=headers,
        )
        assert initialize.status_code == 200, initialize.text

        tools = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers=headers,
        )
        assert tools.status_code == 200, tools.text
        advertised = {tool["name"]: tool for tool in tools.json()["result"]["tools"]}

    assert set(advertised) == _EXPECTED_TOOLS
    assert not (_FORBIDDEN_REPOSITORY_TOOLS & set(advertised))
    assert advertised["start_command"]["annotations"]["destructiveHint"] is True
    assert advertised["start_command"]["annotations"]["openWorldHint"] is False
    assert advertised["enqueue_training"]["annotations"]["destructiveHint"] is True
    assert advertised["enqueue_training"]["annotations"]["openWorldHint"] is False
    assert advertised["prepare_revision_workspace"]["annotations"]["openWorldHint"] is True
