"""Strict runtime settings for the ChatGPT WSL MCP gateway."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def _required_absolute_directory(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} does not exist or is not a directory: {resolved}")
    return resolved


def normalize_public_base_url(value: str) -> str:
    """Validate and normalize the externally reachable HTTPS origin."""

    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("MCP public base URL must be an absolute HTTPS URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("MCP public base URL must contain only the HTTPS origin")
    return normalized


@dataclass(frozen=True)
class GatewaySettings:
    """Validated settings shared by OAuth, MCP tools, and the tunnel launcher."""

    repo_root: Path
    state_dir: Path
    public_base_url: str | None
    host: str = "127.0.0.1"
    port: int = 8765
    docker_image: str = "nvidia/cuda:13.0.0-base-ubuntu24.04"
    cloudflared_path: Path = Path("/home/kamimura/.local/bin/cloudflared")
    tunnel_client_path: Path = Path("/home/kamimura/.local/bin/tunnel-client")
    uv_python_root: Path = Path("/home/kamimura/.local/share/uv/python")
    access_token_ttl_seconds: int = 3600
    refresh_token_ttl_seconds: int = 30 * 24 * 3600
    authorization_ttl_seconds: int = 300

    @classmethod
    def from_env(
        cls,
        *,
        public_base_url: str | None = None,
        require_public_base_url: bool = True,
    ) -> GatewaySettings:
        """Load settings from explicit `TENNIS_MCP_*` environment variables."""

        repo_root = _required_absolute_directory(
            os.environ.get(
                "TENNIS_MCP_REPO_ROOT", "/home/kamimura/projects/tennis-lab"
            ),
            "TENNIS_MCP_REPO_ROOT",
        )
        if not (repo_root / ".git").exists():
            raise ValueError(f"TENNIS_MCP_REPO_ROOT is not a git checkout: {repo_root}")

        state_value = os.environ.get(
            "TENNIS_MCP_STATE_DIR",
            str(Path.home() / ".local/state/tennis-lab-chatgpt-mcp"),
        )
        state_dir = Path(state_value).expanduser()
        if not state_dir.is_absolute():
            raise ValueError("TENNIS_MCP_STATE_DIR must be an absolute path")

        base_value = public_base_url or os.environ.get("TENNIS_MCP_PUBLIC_BASE_URL")
        if not base_value and require_public_base_url:
            raise ValueError("TENNIS_MCP_PUBLIC_BASE_URL is required")

        port = int(os.environ.get("TENNIS_MCP_PORT", "8765"))
        if not 1024 <= port <= 65535:
            raise ValueError("TENNIS_MCP_PORT must be between 1024 and 65535")

        cloudflared = Path(
            os.environ.get(
                "TENNIS_MCP_CLOUDFLARED",
                "/home/kamimura/.local/bin/cloudflared",
            )
        ).expanduser()
        tunnel_client = Path(
            os.environ.get(
                "TENNIS_MCP_TUNNEL_CLIENT",
                "/home/kamimura/.local/bin/tunnel-client",
            )
        ).expanduser()
        uv_python_root = Path(
            os.environ.get(
                "TENNIS_MCP_UV_PYTHON_ROOT",
                "/home/kamimura/.local/share/uv/python",
            )
        ).expanduser()

        return cls(
            repo_root=repo_root,
            state_dir=state_dir.resolve(),
            public_base_url=(
                normalize_public_base_url(base_value) if base_value else None
            ),
            host=os.environ.get("TENNIS_MCP_HOST", "127.0.0.1"),
            port=port,
            docker_image=os.environ.get(
                "TENNIS_MCP_DOCKER_IMAGE",
                "nvidia/cuda:13.0.0-base-ubuntu24.04",
            ),
            cloudflared_path=cloudflared.resolve(),
            tunnel_client_path=tunnel_client.resolve(),
            uv_python_root=uv_python_root.resolve(),
        )

    @property
    def resource_url(self) -> str:
        """Canonical OAuth protected-resource identifier for the MCP endpoint."""

        if self.public_base_url is None:
            raise ValueError("public_base_url is required for OAuth mode")
        return f"{self.public_base_url}/mcp"

    @property
    def secure_tunnel_dir(self) -> Path:
        return self.state_dir / "secure-tunnel"

    @property
    def secure_tunnel_id_path(self) -> Path:
        return self.secure_tunnel_dir / "tunnel-id"

    @property
    def secure_tunnel_key_path(self) -> Path:
        return self.secure_tunnel_dir / "runtime-api-key"

    @property
    def secure_tunnel_profile_dir(self) -> Path:
        return self.secure_tunnel_dir / "profiles"

    @property
    def secure_tunnel_profile_path(self) -> Path:
        return self.secure_tunnel_profile_dir / "tennis-lab.yaml"

    @property
    def secure_tunnel_health_url_path(self) -> Path:
        return self.secure_tunnel_dir / "health-url"

    @property
    def database_path(self) -> Path:
        return self.state_dir / "gateway.sqlite3"

    @property
    def owner_secret_path(self) -> Path:
        return self.state_dir / "owner-secret"

    @property
    def public_url_path(self) -> Path:
        return self.state_dir / "public-url"

    @property
    def tunnel_log_path(self) -> Path:
        return self.state_dir / "cloudflared.log"

    @property
    def job_specs_dir(self) -> Path:
        return self.state_dir / "training-specs"

    def ensure_state(self) -> None:
        """Create private state directories and a high-entropy owner secret."""

        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        self.job_specs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.job_specs_dir, 0o700)

        try:
            descriptor = os.open(
                self.owner_secret_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            os.chmod(self.owner_secret_path, 0o600)
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
                stream.write("\n")

    def read_owner_secret(self) -> str:
        """Read the local owner secret without exposing it through MCP tools."""

        secret = self.owner_secret_path.read_text(encoding="utf-8").strip()
        if len(secret) < 32:
            raise ValueError("owner secret is missing or too short")
        return secret
