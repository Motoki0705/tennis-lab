from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.secure_tunnel import (
    PRIVATE_SERVICE_NAME,
    TUNNEL_SERVICE_NAME,
    SecureTunnelError,
    SecureTunnelManager,
    validate_tunnel_id,
)
from src.automation.chatgpt_mcp.settings import GatewaySettings


def _manager(tmp_path: Path) -> SecureTunnelManager:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    source = tmp_path / "source directory"
    source.mkdir()
    executable_target = tmp_path / "python-runtime"
    executable_target.touch(mode=0o700)
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(executable_target)
    tunnel_client = tmp_path / "tunnel-client"
    tunnel_client.touch(mode=0o700)
    settings = GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        public_base_url=None,
        tunnel_client_path=tunnel_client,
    )
    return SecureTunnelManager(
        settings,
        source_root=source,
        python_executable=executable,
        service_dir=tmp_path / "systemd",
    )


@pytest.mark.parametrize(
    "value",
    ["", "tun_abc", "tunnel_short", "tunnel_has a space", "other_1234567890123456"],
)
def test_validate_tunnel_id_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SecureTunnelError):
        validate_tunnel_id(value)


def test_configure_writes_private_files_and_uses_file_secret_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        profile_dir = Path(command[command.index("--profile-dir") + 1])
        profile_dir.mkdir(parents=True, exist_ok=True)
        key_reference = command[command.index("--control-plane-api-key-ref") + 1]
        (profile_dir / "tennis-lab.yaml").write_text(
            f"control_plane:\n  api_key: {key_reference}\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "created\n", "")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )
    tunnel_id = "tunnel_0123456789abcdef0123456789abcdef"
    runtime_key = "sk-test-runtime-key-that-remains-private"

    profile = manager.configure(tunnel_id=tunnel_id, runtime_api_key=runtime_key)

    assert commands[0][1:4] == ["init", "--sample", "sample_mcp_remote_no_auth"]
    assert runtime_key not in commands[0]
    assert commands[0][commands[0].index("--mcp-server-url") + 1] == (
        "http://127.0.0.1:8767/mcp"
    )
    assert manager.settings.secure_tunnel_id_path.read_text().strip() == tunnel_id
    assert manager.settings.secure_tunnel_key_path.read_text().strip() == runtime_key
    assert "file:" in profile.read_text(encoding="utf-8")
    assert runtime_key not in profile.read_text(encoding="utf-8")
    assert stat.S_IMODE(profile.stat().st_mode) == 0o600
    assert (
        stat.S_IMODE(manager.settings.secure_tunnel_key_path.stat().st_mode) == 0o600
    )
    assert (
        stat.S_IMODE(manager.settings.secure_tunnel_dir.stat().st_mode) == 0o700
    )


def test_install_services_keeps_secret_out_of_units_and_starts_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.settings.secure_tunnel_profile_dir.mkdir(parents=True)
    manager.settings.secure_tunnel_profile_path.write_text(
        "config_version: 1\n", encoding="utf-8"
    )
    runtime_key = "sk-secret-must-not-appear-in-systemd"
    manager.settings.secure_tunnel_key_path.write_text(runtime_key, encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )
    ready_urls: list[tuple[str, str]] = []

    def fake_wait_for_http(
        url: str, *, service_name: str
    ) -> None:
        ready_urls.append((url, service_name))

    monkeypatch.setattr(manager, "_wait_for_http", fake_wait_for_http)

    paths = manager.install_user_services()
    manager.start()

    private_unit = paths.private_service.read_text(encoding="utf-8")
    tunnel_unit = paths.tunnel_service.read_text(encoding="utf-8")
    assert "serve-private" in private_unit
    assert f"WorkingDirectory={tmp_path}/source\\x20directory" in private_unit
    assert 'WorkingDirectory="' not in private_unit
    assert f'ExecStart="{tmp_path}/.venv/bin/python"' in private_unit
    assert "TENNIS_MCP_HOST=127.0.0.1" in private_unit
    assert "TENNIS_MCP_PORT=8767" in private_unit
    assert "--profile-file" in tunnel_unit
    assert f"Requires={PRIVATE_SERVICE_NAME}" in tunnel_unit
    assert runtime_key not in private_unit + tunnel_unit
    verify_command = commands[0]
    assert verify_command[:3] == ["systemd-analyze", "--user", "verify"]
    private_candidate = Path(verify_command[3])
    tunnel_candidate = Path(verify_command[4])
    assert private_candidate.name == PRIVATE_SERVICE_NAME
    assert tunnel_candidate.name == TUNNEL_SERVICE_NAME
    assert private_candidate.parent == tunnel_candidate.parent
    assert private_candidate.parent != paths.private_service.parent
    assert commands[1] == ["systemctl", "--user", "daemon-reload"]
    assert commands[2] == [
        "systemctl",
        "--user",
        "enable",
        PRIVATE_SERVICE_NAME,
        TUNNEL_SERVICE_NAME,
    ]
    assert commands[3:] == [
        ["systemctl", "--user", "restart", PRIVATE_SERVICE_NAME],
        ["systemctl", "--user", "restart", TUNNEL_SERVICE_NAME],
        ["systemctl", "--user", "is-active", PRIVATE_SERVICE_NAME],
        ["systemctl", "--user", "is-active", TUNNEL_SERVICE_NAME],
    ]
    assert ready_urls == [
        ("http://127.0.0.1:8767/healthz", PRIVATE_SERVICE_NAME),
        ("http://127.0.0.1:8768/readyz", TUNNEL_SERVICE_NAME),
    ]


def test_install_services_rejects_invalid_systemd_units_without_replacing_active_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.settings.secure_tunnel_profile_dir.mkdir(parents=True)
    manager.settings.secure_tunnel_profile_path.write_text(
        "config_version: 1\n", encoding="utf-8"
    )
    manager.service_dir.mkdir(parents=True)
    manager.private_service_path.write_text(
        "old private unit\n", encoding="utf-8"
    )
    manager.tunnel_service_path.write_text(
        "old tunnel unit\n", encoding="utf-8"
    )

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["systemd-analyze", "--user", "verify"]
        assert Path(command[3]).name == PRIVATE_SERVICE_NAME
        assert Path(command[4]).name == TUNNEL_SERVICE_NAME
        assert Path(command[3]).parent != manager.service_dir
        return subprocess.CompletedProcess(command, 1, "", "invalid unit")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )

    with pytest.raises(SecureTunnelError, match="systemd unit verification failed"):
        manager.install_user_services()

    assert manager.private_service_path.read_text(encoding="utf-8") == (
        "old private unit\n"
    )
    assert manager.tunnel_service_path.read_text(encoding="utf-8") == (
        "old tunnel unit\n"
    )

def test_start_rejects_service_that_did_not_become_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[-2:] == ["is-active", PRIVATE_SERVICE_NAME]:
            return subprocess.CompletedProcess(command, 3, "inactive\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )
    monkeypatch.setattr(manager, "_wait_for_http", lambda *args, **kwargs: None)

    with pytest.raises(SecureTunnelError, match="did not become active: inactive"):
        manager.start()

    assert commands[-1] == [
        "systemctl",
        "--user",
        "is-active",
        PRIVATE_SERVICE_NAME,
    ]


def test_doctor_accepts_missing_oauth_metadata_for_no_auth_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.settings.secure_tunnel_profile_dir.mkdir(parents=True)
    manager.settings.secure_tunnel_profile_path.write_text(
        "config_version: 1\n", encoding="utf-8"
    )
    doctor_payload = {
        "result": "fail",
        "failed_checks": ["oauth_metadata"],
        "checks": [
            {"id": "mcp_server_reachable", "status": "PASS"},
            {"id": "oauth_metadata", "status": "FAIL", "summary": "HTTP 404"},
        ],
    }
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, json.dumps(doctor_payload), "")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )

    result = manager.doctor()

    normalized = json.loads(result.stdout)
    assert result.returncode == 0
    assert normalized["result"] == "pass"
    assert normalized["failed_checks"] == []
    assert normalized["checks"][1]["status"] == "SKIP"
    assert commands[0][commands[0].index("--health.listen-addr") + 1] == (
        "127.0.0.1:0"
    )


def test_doctor_preserves_unexpected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manager.settings.secure_tunnel_profile_dir.mkdir(parents=True)
    manager.settings.secure_tunnel_profile_path.write_text(
        "config_version: 1\n", encoding="utf-8"
    )
    doctor_payload = {
        "result": "fail",
        "failed_checks": ["mcp_server_reachable"],
        "checks": [{"id": "mcp_server_reachable", "status": "FAIL"}],
    }

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, json.dumps(doctor_payload), "")

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.secure_tunnel.subprocess.run", fake_run
    )

    result = manager.doctor()

    assert result.returncode == 1
    assert json.loads(result.stdout) == doctor_payload
