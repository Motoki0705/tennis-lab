from __future__ import annotations

import base64
import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp.types import LATEST_PROTOCOL_VERSION
from starlette.testclient import TestClient

from src.automation.chatgpt_mcp.server import build_gateway
from src.automation.chatgpt_mcp.settings import GatewaySettings

pytestmark = pytest.mark.integration

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
    settings = GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        control_dir=control,
        public_base_url="https://mcp.example.test",
        uv_python_root=uv_root,
    )
    settings.ensure_state()
    return settings


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def test_oauth_discovery_token_and_reduced_tool_surface(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_gateway(settings).streamable_http_app()
    redirect_uri = "https://chatgpt.com/connector/oauth/test-callback"
    verifier = "v" * 64

    with TestClient(app, base_url="https://mcp.example.test") as client:
        health = client.get("/healthz")
        assert health.status_code == 200

        resource_metadata = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resource_metadata.status_code == 200
        assert resource_metadata.json()["resource"] == settings.resource_url
        authorization_servers = resource_metadata.json()["authorization_servers"]
        assert [value.rstrip("/") for value in authorization_servers] == [
            settings.public_base_url
        ]

        authorization_metadata = client.get("/.well-known/oauth-authorization-server")
        assert authorization_metadata.status_code == 200
        assert authorization_metadata.json()["registration_endpoint"].endswith(
            "/register"
        )
        assert (
            "S256" in authorization_metadata.json()["code_challenge_methods_supported"]
        )

        registration = client.post(
            "/register",
            json={
                "redirect_uris": [redirect_uri],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": "wsl:read wsl:write",
                "client_name": "ChatGPT integration test",
            },
        )
        assert registration.status_code == 201, registration.text
        client_id = registration.json()["client_id"]
        client_secret = registration.json()["client_secret"]

        authorization = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": "wsl:read wsl:write",
                "state": "integration-state",
                "code_challenge": _pkce_challenge(verifier),
                "code_challenge_method": "S256",
                "resource": settings.resource_url,
            },
            follow_redirects=False,
        )
        assert authorization.status_code == 302, authorization.text
        transaction = parse_qs(urlsplit(authorization.headers["location"]).query)[
            "transaction"
        ][0]

        approval = client.post(
            "/oauth/approve",
            data={
                "transaction": transaction,
                "owner_secret": settings.read_owner_secret(),
            },
            follow_redirects=False,
        )
        assert approval.status_code == 303
        callback_query = parse_qs(urlsplit(approval.headers["location"]).query)
        assert callback_query["state"] == ["integration-state"]

        token = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "client_secret": client_secret,
                "code": callback_query["code"][0],
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
                "resource": settings.resource_url,
            },
        )
        assert token.status_code == 200, token.text
        access_token = token.json()["access_token"]

        unauthorized = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert unauthorized.status_code == 401

        mcp_headers = {
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {access_token}",
            "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
            "Mcp-Method": "tools/list",
        }
        tools = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
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
            headers=mcp_headers,
        )
        assert tools.status_code == 200, tools.text
        advertised = {tool["name"]: tool for tool in tools.json()["result"]["tools"]}

    assert set(advertised) == _EXPECTED_TOOLS
    assert advertised["start_command"]["annotations"]["destructiveHint"] is True
    assert advertised["enqueue_training"]["annotations"]["destructiveHint"] is True
    assert (
        advertised["start_command"]["_meta"]["securitySchemes"][0]["type"] == "oauth2"
    )


def test_private_tunnel_mode_uses_loopback_without_oauth(tmp_path: Path) -> None:
    settings = replace(
        _settings(tmp_path), public_base_url=None, host="127.0.0.1", port=8767
    )
    app = build_gateway(settings, authenticated=False).streamable_http_app()
    headers = {
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION,
        "Mcp-Method": "tools/list",
    }

    with TestClient(app, base_url="http://127.0.0.1:8767") as client:
        service = client.get("/")
        assert service.status_code == 200
        assert service.json()["authentication"] == "OpenAI Secure MCP Tunnel"
        assert service.json()["role"] == (
            "arbitrary tennis-lab execution, validation, and GPU training"
        )
        assert (
            client.get("/.well-known/oauth-protected-resource/mcp").status_code == 404
        )

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
                            "name": "tunnel-test",
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

    assert set(advertised) == _EXPECTED_TOOLS
    assert advertised["start_command"].get("_meta", {}).get("securitySchemes") is None
