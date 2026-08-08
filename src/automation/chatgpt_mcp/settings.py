"""Strict runtime settings for the ChatGPT WSL MCP gateway."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

_DEFAULT_PROJECT_ROOT = Path("/home/kamimura/projects/tennis-lab")
_DEFAULT_STATE_DIR = Path.home() / ".local/state/tennis-lab-chatgpt-mcp"
_DEFAULT_CONTROL_DIR = Path.home() / ".local/share/tennis-lab-chatgpt-mcp"
_DEFAULT_ORIGIN_URL = "https://github.com/Motoki0705/tennis-lab.git"
_ALLOWED_ORIGIN_URLS = {
    _DEFAULT_ORIGIN_URL,
    "https://github.com/Motoki0705/tennis-lab",
    "git@github.com:Motoki0705/tennis-lab.git",
}


def _required_absolute_directory(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{name} does not exist or is not a directory: {resolved}")
    return resolved


def _absolute_path(value: str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {path}")
    return path.resolve()


def normalize_public_base_url(value: str) -> str:
    """Validate and normalize the externally reachable HTTPS origin."""

    normalized = value.rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("MCP public base URL must be an absolute HTTPS URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("MCP public base URL must contain only the HTTPS origin")
    return normalized


def normalize_origin_url(value: str) -> str:
    """Restrict the trusted mirror to the tennis-lab origin."""

    normalized = value.strip().rstrip("/")
    if normalized == "https://github.com/Motoki0705/tennis-lab":
        normalized = _DEFAULT_ORIGIN_URL
    if normalized not in _ALLOWED_ORIGIN_URLS:
        raise ValueError(
            "TENNIS_MCP_ORIGIN_URL must name Motoki0705/tennis-lab on GitHub"
        )
    return normalized


@dataclass(frozen=True)
class GatewaySettings:
    """Validated settings shared by auth, execution jobs, and tunnel launchers."""

    repo_root: Path
    state_dir: Path
    public_base_url: str | None
    control_dir: Path = _DEFAULT_CONTROL_DIR
    origin_url: str = _DEFAULT_ORIGIN_URL
    host: str = "127.0.0.1"
    port: int = 8765
    docker_image: str = "nvidia/cuda:13.0.0-base-ubuntu24.04"
    cloudflared_path: Path = Path("/home/kamimura/.local/bin/cloudflared")
    tunnel_client_path: Path = Path("/home/kamimura/.local/bin/tunnel-client")
    uv_python_root: Path = Path("/home/kamimura/.local/share/uv/python")
    access_token_ttl_seconds: int = 3600
    refresh_token_ttl_seconds: int = 30 * 24 * 3600
    authorization_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        repo_root = self.repo_root.resolve()
        state_dir = self.state_dir.resolve()
        control_dir = self.control_dir.resolve()
        if state_dir == repo_root or state_dir.is_relative_to(repo_root):
            raise ValueError(
                "MCP state must be outside the destructible tennis-lab tree"
            )
        if control_dir == repo_root or control_dir.is_relative_to(repo_root):
            raise ValueError(
                "MCP control plane must be outside the destructible tennis-lab tree"
            )
        if state_dir == control_dir:
            raise ValueError("MCP state and control directories must be distinct")
        if not 1024 <= self.port <= 65535:
            raise ValueError("MCP port must be between 1024 and 65535")

    @classmethod
    def from_env(
        cls,
        *,
        public_base_url: str | None = None,
        require_public_base_url: bool = True,
    ) -> GatewaySettings:
        """Load settings from explicit ``TENNIS_MCP_*`` environment variables."""

        repo_root = _required_absolute_directory(
            os.environ.get("TENNIS_MCP_REPO_ROOT", str(_DEFAULT_PROJECT_ROOT)),
            "TENNIS_MCP_REPO_ROOT",
        )
        if not (repo_root / ".git").exists():
            raise ValueError(f"TENNIS_MCP_REPO_ROOT is not a git checkout: {repo_root}")

        state_dir = _absolute_path(
            os.environ.get("TENNIS_MCP_STATE_DIR", str(_DEFAULT_STATE_DIR)),
            "TENNIS_MCP_STATE_DIR",
        )
        control_dir = _absolute_path(
            os.environ.get("TENNIS_MCP_CONTROL_DIR", str(_DEFAULT_CONTROL_DIR)),
            "TENNIS_MCP_CONTROL_DIR",
        )
        base_value = public_base_url or os.environ.get("TENNIS_MCP_PUBLIC_BASE_URL")
        if not base_value and require_public_base_url:
            raise ValueError("TENNIS_MCP_PUBLIC_BASE_URL is required")

        port = int(os.environ.get("TENNIS_MCP_PORT", "8765"))
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
            state_dir=state_dir,
            public_base_url=(
                normalize_public_base_url(base_value) if base_value else None
            ),
            control_dir=control_dir,
            origin_url=normalize_origin_url(
                os.environ.get("TENNIS_MCP_ORIGIN_URL", _DEFAULT_ORIGIN_URL)
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
    def project_venv_link(self) -> Path:
        return self.repo_root / ".venv"

    @property
    def runtime_venv_root(self) -> Path:
        return self.control_dir / "venv"

    @property
    def venv_root(self) -> Path:
        """Return the trusted venv after deployment, or the bootstrap venv before it."""

        if self.runtime_venv_root.is_dir():
            return self.runtime_venv_root
        return self.project_venv_link

    @property
    def runtime_releases_dir(self) -> Path:
        return self.control_dir / "releases"

    @property
    def runtime_current_dir(self) -> Path:
        return self.control_dir / "current"

    @property
    def runtime_bin_dir(self) -> Path:
        return self.control_dir / "bin"

    @property
    def runtime_version_path(self) -> Path:
        return self.control_dir / "runtime-version"

    @property
    def trusted_git_dir(self) -> Path:
        return self.control_dir / "repository.git"

    @property
    def trusted_git_home(self) -> Path:
        return self.control_dir / "git-home"

    @property
    def trusted_queue_script(self) -> Path:
        return self.runtime_bin_dir / "training_queue.sh"

    @property
    def trusted_queue_dir(self) -> Path:
        return self.state_dir / "training-queue"

    @property
    def revision_workspace_dir(self) -> Path:
        return self.state_dir / "revisions"

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

    @property
    def sandbox_jobs_dir(self) -> Path:
        return self.state_dir / "sandboxes"

    @property
    def git_file_mask_path(self) -> Path:
        return self.state_dir / "masked-git-file"

    @property
    def git_dir_mask_path(self) -> Path:
        return self.state_dir / "masked-git-directory"

    @property
    def git_mask_path(self) -> Path:
        """Compatibility alias for the exact-revision worktree ``.git`` file mask."""

        return self.git_file_mask_path

    def ensure_state(self) -> None:
        """Create private state directories and high-entropy local secrets."""

        for directory in (
            self.state_dir,
            self.job_specs_dir,
            self.sandbox_jobs_dir,
            self.revision_workspace_dir,
            self.trusted_queue_dir,
        ):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(directory, 0o700)

        try:
            descriptor = os.open(
                self.owner_secret_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            os.chmod(self.owner_secret_path, 0o600)
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
                stream.write("\n")

        if not self.git_file_mask_path.exists():
            descriptor = os.open(
                self.git_file_mask_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(
                    "git metadata is intentionally unavailable in MCP sandboxes\n"
                )
        os.chmod(self.git_file_mask_path, 0o400)

        self.git_dir_mask_path.mkdir(mode=0o500, parents=True, exist_ok=True)
        os.chmod(self.git_dir_mask_path, 0o500)

    def ensure_control_directories(self) -> None:
        """Create owner-only trusted control-plane directories."""

        for directory in (
            self.control_dir,
            self.runtime_releases_dir,
            self.runtime_bin_dir,
            self.trusted_git_home,
        ):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(directory, 0o700)

    def read_owner_secret(self) -> str:
        """Read the local owner secret without exposing it through MCP tools."""

        secret = self.owner_secret_path.read_text(encoding="utf-8").strip()
        if len(secret) < 32:
            raise ValueError("owner secret is missing or too short")
        return secret
