"""Command-line lifecycle for the ChatGPT WSL MCP execution gateway."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

from src.automation.chatgpt_mcp.sandbox_exec import run_from_spec
from src.automation.chatgpt_mcp.secure_tunnel import SecureTunnelManager
from src.automation.chatgpt_mcp.server import run_gateway
from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.tunnel import QuickTunnel
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

PATH_BOUNDARY = NonHydraPathBoundary(
    name="automation.chatgpt_mcp",
    fields=(
        BoundaryPathField(
            "repo_root",
            PathRole.PROJECT,
            PathDirection.INPUT,
            PathKind.DIRECTORY,
            must_exist=True,
            allow_role_root=True,
        ),
        BoundaryPathField(
            "state_dir",
            PathRole.ARTIFACT,
            PathDirection.OUTPUT,
            PathKind.DIRECTORY,
            allow_role_root=True,
        ),
    ),
)


def _state_dir() -> Path:
    value = os.environ.get(
        "TENNIS_MCP_STATE_DIR",
        str(Path.home() / ".local/state/tennis-lab-chatgpt-mcp"),
    )
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("TENNIS_MCP_STATE_DIR must be absolute")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path.resolve()


def serve(public_base_url: str) -> None:
    """Serve behind an already configured HTTPS reverse proxy."""

    settings = GatewaySettings.from_env(public_base_url=public_base_url)
    settings.ensure_state()
    settings.public_url_path.write_text(settings.resource_url + "\n", encoding="utf-8")
    os.chmod(settings.public_url_path, 0o600)
    run_gateway(settings)


def serve_public() -> None:
    """Create a temporary public HTTPS origin and serve until interrupted."""

    state_dir = _state_dir()
    local_port = int(os.environ.get("TENNIS_MCP_PORT", "8765"))
    cloudflared = Path(
        os.environ.get(
            "TENNIS_MCP_CLOUDFLARED",
            "/home/kamimura/.local/bin/cloudflared",
        )
    ).expanduser()
    tunnel = QuickTunnel(
        cloudflared_path=cloudflared.resolve(),
        local_port=local_port,
        log_path=state_dir / "cloudflared.log",
    )
    try:
        public_base_url = tunnel.start()
        settings = GatewaySettings.from_env(public_base_url=public_base_url)
        settings.ensure_state()
        settings.public_url_path.write_text(
            settings.resource_url + "\n", encoding="utf-8"
        )
        os.chmod(settings.public_url_path, 0o600)
        print(f"ChatGPT MCP URL: {settings.resource_url}", flush=True)
        run_gateway(settings)
    finally:
        tunnel.stop()


def serve_private() -> None:
    """Serve only on WSL loopback for the OpenAI Secure MCP Tunnel client."""

    settings = GatewaySettings.from_env(require_public_base_url=False)
    if settings.host != "127.0.0.1":
        raise ValueError("serve-private requires TENNIS_MCP_HOST=127.0.0.1")
    settings.ensure_state()
    run_gateway(settings, authenticated=False)


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError("command must run from a tennis-lab git checkout")
    return Path(result.stdout.strip()).resolve()


def _require_canonical_source(settings: GatewaySettings) -> Path:
    source_root = _git_root()
    if source_root != settings.repo_root:
        raise RuntimeError(
            "persistent MCP services must be installed from the canonical repository root "
            f"{settings.repo_root}, not from worktree {source_root}. Merge the PR, update "
            "main, then rerun this command from the canonical checkout."
        )
    return source_root


def install_user_service(*, start: bool) -> Path:
    """Install the legacy public service from the canonical checkout only."""

    settings = GatewaySettings.from_env(require_public_base_url=False)
    source_root = _require_canonical_source(settings)
    repo_root = settings.repo_root
    state_dir = _state_dir()
    service_dir = Path.home() / ".config/systemd/user"
    service_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    service_path = service_dir / "tennis-lab-chatgpt-mcp.service"
    unit = f"""[Unit]
Description=Authenticated ChatGPT MCP execution gateway for tennis-lab WSL
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={source_root}
Environment=TENNIS_MCP_REPO_ROOT={repo_root}
Environment=TENNIS_MCP_STATE_DIR={state_dir}
ExecStart={sys.executable} -m src.automation.chatgpt_mcp serve-public
Restart=always
RestartSec=5
TimeoutStopSec=20
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""
    service_path.write_text(unit, encoding="utf-8")
    os.chmod(service_path, 0o600)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=30)
    if start:
        subprocess.run(
            [
                "systemctl",
                "--user",
                "enable",
                "--now",
                "tennis-lab-chatgpt-mcp.service",
            ],
            check=True,
            timeout=30,
        )
    return service_path


def _secure_tunnel_manager(*, require_canonical_source: bool = False) -> SecureTunnelManager:
    settings = GatewaySettings.from_env(require_public_base_url=False)
    source_root = (
        _require_canonical_source(settings) if require_canonical_source else _git_root()
    )
    return SecureTunnelManager(
        settings,
        source_root=source_root,
        python_executable=Path(sys.executable),
    )


def _read_runtime_api_key(key_file: Path | None) -> str:
    if key_file is not None:
        resolved = key_file.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"runtime API key file does not exist: {resolved}")
        return resolved.read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        raise ValueError(
            "interactive input is unavailable; pass --runtime-key-file instead"
        )
    return getpass.getpass("OpenAI tunnel runtime API key (input hidden): ").strip()


def configure_secure_tunnel(
    *, tunnel_id: str, runtime_key_file: Path | None, start: bool
) -> str:
    """Persist a tunnel ID/key and install services from canonical main."""

    manager = _secure_tunnel_manager(require_canonical_source=True)
    runtime_api_key = _read_runtime_api_key(runtime_key_file)
    profile_path = manager.configure(
        tunnel_id=tunnel_id, runtime_api_key=runtime_api_key
    )
    manager.install_user_services()
    if start:
        manager.start()
    return "\n".join(
        [
            f"Tunnel ID: {tunnel_id.strip()}",
            f"Profile: {profile_path}",
            "Connection: Tunnel",
            "Authentication: None (access is controlled by the OpenAI tunnel)",
            f"Services started: {'yes' if start else 'no'}",
        ]
    )


def show_secure_connection() -> str:
    """Return the stable identifier and ChatGPT plugin fields."""

    manager = _secure_tunnel_manager()
    tunnel_id_path = manager.settings.secure_tunnel_id_path
    if not tunnel_id_path.is_file():
        raise RuntimeError("secure tunnel has not been configured yet")
    return "\n".join(
        [
            "Name: tennis-lab WSL",
            "Description: Exact-revision validation, CUDA, and training on tennis-lab WSL",
            "Connection: Tunnel",
            f"Tunnel ID: {tunnel_id_path.read_text(encoding='utf-8').strip()}",
            "Authentication: None",
        ]
    )


def show_connection() -> str:
    """Return the exact fields needed by the legacy public plugin form."""

    state_dir = _state_dir()
    url_path = state_dir / "public-url"
    secret_path = state_dir / "owner-secret"
    if not url_path.is_file() or not secret_path.is_file():
        raise RuntimeError("gateway has not started yet")
    return "\n".join(
        [
            "Name: tennis-lab WSL",
            "Description: Exact-revision validation, CUDA, and training on tennis-lab WSL",
            f"Server URL: {url_path.read_text(encoding='utf-8').strip()}",
            "Authentication: OAuth",
            f"Owner secret: {secret_path.read_text(encoding='utf-8').strip()}",
        ]
    )


def main() -> int:
    """Dispatch gateway lifecycle subcommands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--public-base-url", required=True)
    subparsers.add_parser("serve-public")
    subparsers.add_parser("serve-private")

    install_parser = subparsers.add_parser("install-user-service")
    install_parser.add_argument("--start", action="store_true")
    subparsers.add_parser("show-connection")

    secure_parser = subparsers.add_parser("configure-secure-tunnel")
    secure_parser.add_argument("--tunnel-id", required=True)
    secure_parser.add_argument("--runtime-key-file", type=Path)
    secure_parser.add_argument("--start", action="store_true")
    subparsers.add_parser("show-secure-connection")
    subparsers.add_parser("doctor-secure-tunnel")
    sandbox_parser = subparsers.add_parser("sandbox-exec", help=argparse.SUPPRESS)
    sandbox_parser.add_argument("--spec", type=Path, required=True)

    arguments = parser.parse_args()
    boundary_settings = GatewaySettings.from_env(
        public_base_url=(
            arguments.public_base_url if arguments.command == "serve" else None
        ),
        require_public_base_url=arguments.command == "serve",
    )
    roots = RuntimePathRoots(
        project_root=boundary_settings.repo_root,
        data_root=boundary_settings.repo_root,
        checkpoint_root=boundary_settings.repo_root,
        artifact_root=boundary_settings.state_dir,
        output_root=boundary_settings.state_dir,
        cache_root=boundary_settings.state_dir,
        external_asset_root=boundary_settings.repo_root,
    )
    PATH_BOUNDARY.validate(
        {
            "repo_root": boundary_settings.repo_root,
            "state_dir": boundary_settings.state_dir,
        },
        resolver=PathResolver(roots),
    )
    if arguments.command == "serve":
        serve(arguments.public_base_url)
    elif arguments.command == "serve-public":
        serve_public()
    elif arguments.command == "serve-private":
        serve_private()
    elif arguments.command == "install-user-service":
        path = install_user_service(start=arguments.start)
        print(path)
    elif arguments.command == "show-connection":
        print(show_connection())
    elif arguments.command == "configure-secure-tunnel":
        print(
            configure_secure_tunnel(
                tunnel_id=arguments.tunnel_id,
                runtime_key_file=arguments.runtime_key_file,
                start=arguments.start,
            )
        )
    elif arguments.command == "show-secure-connection":
        print(show_secure_connection())
    elif arguments.command == "doctor-secure-tunnel":
        result = _secure_tunnel_manager().doctor()
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        return 0 if result.returncode == 0 else 1
    elif arguments.command == "sandbox-exec":
        exit_code: int = run_from_spec(arguments.spec)
        return exit_code
    return 0
