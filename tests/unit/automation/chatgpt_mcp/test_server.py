from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.types import LATEST_PROTOCOL_VERSION
from starlette.testclient import TestClient

from src.automation.chatgpt_mcp.server import _run_probe, build_gateway
from src.automation.chatgpt_mcp.settings import GatewaySettings

_EXPECTED_TOOLS = {
    "get_host_status",
    "get_execution_layout",
    "prepare_revision_workspace",
    "get_revision_status",
    "start_command",
    "get_command_job",
    "list_command_jobs",
    "get_command_output",
    "cancel_command_job",
    "enqueue_training",
    "get_training_job",
    "list_training_jobs",
    "get_training_output",
    "cancel_training_job",
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
    control = tmp_path / "control"
    (control / "venv/bin").mkdir(parents=True)
    (control / "current").mkdir()
    (control / "repository.git").mkdir()
    uv_root = tmp_path / "uv-python"
    uv_root.mkdir()
    return GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        control_dir=control,
        public_base_url=None,
        host="127.0.0.1",
        port=8767,
        uv_python_root=uv_root,
    )


def test_run_probe_reports_missing_executable() -> None:
    result = _run_probe(["/definitely/missing/tennis-mcp-command"])

    assert result["ok"] is False
    assert "FileNotFoundError" in result["output"]


def test_private_gateway_advertises_flexible_execution_plane_tools(
    tmp_path: Path,
) -> None:
    app = build_gateway(_settings(tmp_path), authenticated=False).streamable_http_app()
    headers = {
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
        "Mcp-Method": "tools/list",
    }

    with TestClient(app, base_url="http://127.0.0.1:8767") as client:
        tools = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": LATEST_PROTOCOL_VERSION,
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "pytest",
                            "version": "1",
                        },
                        "io.modelcontextprotocol/clientCapabilities": {},
                    }
                },
            },
            headers=headers,
        )
        assert tools.status_code == 200, tools.text
        advertised = {tool["name"]: tool for tool in tools.json()["result"]["tools"]}

        service = client.get("/")
        assert service.status_code == 200
        assert service.json()["role"] == (
            "arbitrary tennis-lab execution, validation, and GPU training"
        )

    assert set(advertised) == _EXPECTED_TOOLS
    assert not (_FORBIDDEN_REPOSITORY_TOOLS & set(advertised))
    assert advertised["start_command"]["annotations"]["destructiveHint"] is True
    assert advertised["start_command"]["annotations"]["openWorldHint"] is False
    assert advertised["enqueue_training"]["annotations"]["destructiveHint"] is True
    assert advertised["cancel_training_job"]["annotations"]["destructiveHint"] is True
    assert (
        advertised["prepare_revision_workspace"]["annotations"]["openWorldHint"] is True
    )
    start_schema = advertised["start_command"]["inputSchema"]["properties"]
    assert start_schema["execution_root"]["enum"] == ["revision", "project"]
    assert "working_directory" in start_schema
    training_schema = advertised["enqueue_training"]["inputSchema"]["properties"]
    assert training_schema["resource"]["enum"] == ["half", "all"]
    assert training_schema["resource"]["default"] == "all"
    assert "MIG or VRAM hard cap" in advertised["enqueue_training"]["description"]
    assert "observably non-running" in advertised["cancel_training_job"]["description"]
    assert "terminating remains nonterminal" in advertised["get_training_job"]["description"]
