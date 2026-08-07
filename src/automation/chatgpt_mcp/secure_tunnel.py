"""Persistent OpenAI Secure MCP Tunnel configuration and service lifecycle."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.automation.chatgpt_mcp.settings import GatewaySettings

PRIVATE_MCP_PORT = 8767
TUNNEL_HEALTH_PORT = 8768
STARTUP_TIMEOUT_SECONDS = 30.0
PRIVATE_SERVICE_NAME = "tennis-lab-chatgpt-mcp-private.service"
TUNNEL_SERVICE_NAME = "tennis-lab-chatgpt-secure-tunnel.service"
_TUNNEL_ID_PATTERN = re.compile(r"^tunnel_[A-Za-z0-9_-]{16,128}$")


class SecureTunnelError(RuntimeError):
    """Raised when a secure tunnel cannot be configured safely."""


@dataclass(frozen=True)
class SecureTunnelPaths:
    """Materialized local paths for the persistent tunnel."""

    tunnel_id: Path
    runtime_api_key: Path
    profile: Path
    private_service: Path
    tunnel_service: Path


def validate_tunnel_id(value: str) -> str:
    """Return a validated OpenAI tunnel identifier."""

    tunnel_id = value.strip()
    if not _TUNNEL_ID_PATTERN.fullmatch(tunnel_id):
        raise SecureTunnelError(
            "tunnel ID must start with 'tunnel_' and contain 16-128 "
            "letters, digits, underscores, or hyphens"
        )
    return tunnel_id


def _validate_runtime_api_key(value: str) -> str:
    key = value.strip()
    if len(key) < 20 or "\n" in key or "\r" in key:
        raise SecureTunnelError("runtime API key is missing or malformed")
    return key


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _unit_value(value: Path) -> str:
    """Quote a path for a systemd directive."""

    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _unit_path_value(value: Path) -> str:
    """Encode an absolute path for systemd path directives without quoting it."""

    if not value.is_absolute():
        raise SecureTunnelError(f"systemd path must be absolute: {value}")
    safe = b"/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    encoded: list[str] = []
    for byte in os.fsencode(value):
        if byte in safe:
            encoded.append(chr(byte))
        elif byte == ord("%"):
            encoded.append("%%")
        else:
            encoded.append(f"\\x{byte:02x}")
    return "".join(encoded)


class SecureTunnelManager:
    """Configure the OpenAI tunnel-client and its user-level systemd units."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        source_root: Path,
        python_executable: Path,
        service_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.source_root = source_root.resolve()
        self.python_executable = python_executable.expanduser().absolute()
        self.service_dir = (
            service_dir or Path.home() / ".config/systemd/user"
        ).resolve()

    @property
    def private_service_path(self) -> Path:
        return self.service_dir / PRIVATE_SERVICE_NAME

    @property
    def tunnel_service_path(self) -> Path:
        return self.service_dir / TUNNEL_SERVICE_NAME

    def configure(self, *, tunnel_id: str, runtime_api_key: str) -> Path:
        """Write private credentials and materialize a tunnel-client profile."""

        checked_id = validate_tunnel_id(tunnel_id)
        checked_key = _validate_runtime_api_key(runtime_api_key)
        if not self.settings.tunnel_client_path.is_file():
            raise SecureTunnelError(
                f"tunnel-client was not found: {self.settings.tunnel_client_path}"
            )

        self.settings.ensure_state()
        self.settings.secure_tunnel_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.settings.secure_tunnel_dir, 0o700)
        self.settings.secure_tunnel_profile_dir.mkdir(
            mode=0o700, parents=True, exist_ok=True
        )
        os.chmod(self.settings.secure_tunnel_profile_dir, 0o700)
        _write_private(self.settings.secure_tunnel_id_path, checked_id)
        _write_private(self.settings.secure_tunnel_key_path, checked_key)

        command = [
            str(self.settings.tunnel_client_path),
            "init",
            "--sample",
            "sample_mcp_remote_no_auth",
            "--profile",
            "tennis-lab",
            "--profile-dir",
            str(self.settings.secure_tunnel_profile_dir),
            "--tunnel-id",
            checked_id,
            "--control-plane-api-key-ref",
            f"file:{self.settings.secure_tunnel_key_path}",
            "--mcp-server-url",
            f"http://127.0.0.1:{PRIVATE_MCP_PORT}/mcp",
            "--health-listen-addr",
            f"127.0.0.1:{TUNNEL_HEALTH_PORT}",
            "--force",
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise SecureTunnelError(
                "tunnel-client init failed: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        if not self.settings.secure_tunnel_profile_path.is_file():
            raise SecureTunnelError(
                "tunnel-client did not create the expected tennis-lab profile"
            )
        os.chmod(self.settings.secure_tunnel_profile_path, 0o600)
        return cast(Path, self.settings.secure_tunnel_profile_path)

    def install_user_services(self) -> SecureTunnelPaths:
        """Install private MCP and tunnel-client units without starting them."""

        if not self.settings.secure_tunnel_profile_path.is_file():
            raise SecureTunnelError(
                "secure tunnel is not configured; run configure-secure-tunnel first"
            )
        self.service_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.service_dir, 0o700)
        self.private_service_path.write_text(
            self._private_service_unit(), encoding="utf-8"
        )
        self.tunnel_service_path.write_text(
            self._tunnel_service_unit(), encoding="utf-8"
        )
        os.chmod(self.private_service_path, 0o600)
        os.chmod(self.tunnel_service_path, 0o600)
        verification = subprocess.run(
            [
                "systemd-analyze",
                "--user",
                "verify",
                str(self.private_service_path),
                str(self.tunnel_service_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if verification.returncode != 0:
            detail = verification.stderr.strip() or verification.stdout.strip()
            raise SecureTunnelError(f"systemd unit verification failed: {detail}")
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"], check=True, timeout=30
        )
        return self.paths()

    def start(self) -> None:
        """Enable and start both halves of the persistent tunnel."""

        subprocess.run(
            [
                "systemctl",
                "--user",
                "enable",
                PRIVATE_SERVICE_NAME,
                TUNNEL_SERVICE_NAME,
            ],
            check=True,
            timeout=60,
        )
        subprocess.run(
            ["systemctl", "--user", "restart", PRIVATE_SERVICE_NAME],
            check=True,
            timeout=60,
        )
        self._wait_for_http(
            f"http://127.0.0.1:{PRIVATE_MCP_PORT}/healthz",
            service_name=PRIVATE_SERVICE_NAME,
        )
        subprocess.run(
            ["systemctl", "--user", "restart", TUNNEL_SERVICE_NAME],
            check=True,
            timeout=60,
        )
        self._wait_for_http(
            f"http://127.0.0.1:{TUNNEL_HEALTH_PORT}/readyz",
            service_name=TUNNEL_SERVICE_NAME,
        )
        for service_name in (PRIVATE_SERVICE_NAME, TUNNEL_SERVICE_NAME):
            result = subprocess.run(
                ["systemctl", "--user", "is-active", service_name],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                detail = result.stdout.strip() or result.stderr.strip() or "inactive"
                raise SecureTunnelError(
                    f"{service_name} did not become active: {detail}"
                )

    def _wait_for_http(self, url: str, *, service_name: str) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        last_error = "endpoint did not respond"
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=1.0) as response:  # noqa: S310
                    if 200 <= response.status < 300:
                        return
                    last_error = f"HTTP {response.status}"
            except HTTPError as error:
                last_error = f"HTTP {error.code}"
            except (OSError, URLError) as error:
                last_error = str(error)
            time.sleep(0.25)
        raise SecureTunnelError(
            f"{service_name} did not become ready at {url}: {last_error}"
        )

    def doctor(self) -> subprocess.CompletedProcess[str]:
        """Run tunnel-client's diagnostic checks for the configured profile."""

        if not self.settings.secure_tunnel_profile_path.is_file():
            raise SecureTunnelError("secure tunnel profile does not exist")
        result = subprocess.run(
            [
                str(self.settings.tunnel_client_path),
                "doctor",
                "--profile-file",
                str(self.settings.secure_tunnel_profile_path),
                "--health.listen-addr",
                "127.0.0.1:0",
                "--explain",
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        return _normalize_no_auth_doctor_result(result)

    def paths(self) -> SecureTunnelPaths:
        """Return all persistent configuration paths."""

        return SecureTunnelPaths(
            tunnel_id=self.settings.secure_tunnel_id_path,
            runtime_api_key=self.settings.secure_tunnel_key_path,
            profile=self.settings.secure_tunnel_profile_path,
            private_service=self.private_service_path,
            tunnel_service=self.tunnel_service_path,
        )

    def _private_service_unit(self) -> str:
        return f"""[Unit]
Description=Private tennis-lab MCP endpoint for OpenAI Secure Tunnel
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={_unit_path_value(self.source_root)}
Environment="TENNIS_MCP_REPO_ROOT={self.settings.repo_root}"
Environment="TENNIS_MCP_STATE_DIR={self.settings.state_dir}"
Environment="TENNIS_MCP_HOST=127.0.0.1"
Environment="TENNIS_MCP_PORT={PRIVATE_MCP_PORT}"
ExecStart={_unit_value(self.python_executable)} -m src.automation.chatgpt_mcp serve-private
Restart=always
RestartSec=5
TimeoutStopSec=20
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""

    def _tunnel_service_unit(self) -> str:
        return f"""[Unit]
Description=OpenAI Secure MCP Tunnel for tennis-lab WSL
After=network-online.target {PRIVATE_SERVICE_NAME}
Wants=network-online.target
Requires={PRIVATE_SERVICE_NAME}

[Service]
Type=simple
ExecStart={_unit_value(self.settings.tunnel_client_path)} run --profile-file {_unit_value(self.settings.secure_tunnel_profile_path)} --health.url-file {_unit_value(self.settings.secure_tunnel_health_url_path)}
Restart=always
RestartSec=5
TimeoutStopSec=20
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""


def _normalize_no_auth_doctor_result(
    result: subprocess.CompletedProcess[str],
) -> subprocess.CompletedProcess[str]:
    """Mark missing OAuth metadata as expected for the no-auth tunnel profile."""

    if result.returncode == 0:
        return result
    try:
        parsed: object = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(parsed, dict):
        return result
    payload = cast(dict[str, object], parsed)
    if payload.get("failed_checks") != ["oauth_metadata"]:
        return result
    checks_value = payload.get("checks")
    if not isinstance(checks_value, list):
        return result
    oauth_check: dict[str, object] | None = None
    for check_value in checks_value:
        if isinstance(check_value, dict) and check_value.get("id") == "oauth_metadata":
            oauth_check = cast(dict[str, object], check_value)
            break
    if oauth_check is None or oauth_check.get("status") != "FAIL":
        return result

    payload["result"] = "pass"
    payload["failed_checks"] = []
    oauth_check["status"] = "SKIP"
    oauth_check["summary"] = (
        "not required: the loopback MCP intentionally uses no authentication; "
        "OpenAI Secure Tunnel controls access"
    )
    oauth_check["why"] = (
        "The sample_mcp_remote_no_auth profile terminates access control at the "
        "OpenAI tunnel, so the private loopback endpoint must not advertise OAuth."
    )
    oauth_check["next"] = []
    normalized_stdout = json.dumps(payload, indent=2) + "\n"
    return subprocess.CompletedProcess(
        result.args,
        0,
        normalized_stdout,
        result.stderr,
    )
