from __future__ import annotations

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
    source = tmp_path / "source"
    source.mkdir()
    executable = tmp_path / "python"
    executable.touch(mode=0o700)
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

    paths = manager.install_user_services()
    manager.start()

    private_unit = paths.private_service.read_text(encoding="utf-8")
    tunnel_unit = paths.tunnel_service.read_text(encoding="utf-8")
    assert "serve-private" in private_unit
    assert "TENNIS_MCP_HOST=127.0.0.1" in private_unit
    assert "TENNIS_MCP_PORT=8767" in private_unit
    assert "--profile-file" in tunnel_unit
    assert f"Requires={PRIVATE_SERVICE_NAME}" in tunnel_unit
    assert runtime_key not in private_unit + tunnel_unit
    assert commands[0] == ["systemctl", "--user", "daemon-reload"]
    assert commands[1] == [
        "systemctl",
        "--user",
        "enable",
        "--now",
        PRIVATE_SERVICE_NAME,
        TUNNEL_SERVICE_NAME,
    ]
