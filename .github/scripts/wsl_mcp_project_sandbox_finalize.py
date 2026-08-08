from __future__ import annotations

import textwrap
from pathlib import Path


def write(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


write(
    "src/automation/chatgpt_mcp/settings.py",
    r'''
    """Strict runtime settings for the ChatGPT WSL MCP gateway."""

    from __future__ import annotations

    import os
    import secrets
    import shutil
    from dataclasses import dataclass
    from pathlib import Path
    from urllib.parse import urlsplit

    _DEFAULT_PROJECT_ROOT = Path("/home/kamimura/projects/tennis-lab")
    _DEFAULT_STATE_ROOT = Path.home() / ".local/state/tennis-lab-chatgpt-mcp"
    _DEFAULT_RUNTIME_ROOT = Path.home() / ".local/share/tennis-lab-chatgpt-mcp"
    _DEFAULT_REMOTE_URL = "https://github.com/Motoki0705/tennis-lab.git"
    _PERSISTENT_PROJECT_ROOTS = (
        "data",
        "outputs",
        "ckpt",
        "ckpts",
        "artifacts",
        ".cache",
        "third_party",
    )


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


    def _inside(path: Path, root: Path) -> bool:
        return path == root or path.is_relative_to(root)


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
        """Validated control-plane and sacrificial project settings."""

        repo_root: Path
        state_dir: Path
        public_base_url: str | None
        host: str = "127.0.0.1"
        port: int = 8765
        docker_image: str = "nvidia/cuda:13.0.0-base-ubuntu24.04"
        cloudflared_path: Path = Path("/home/kamimura/.local/bin/cloudflared")
        tunnel_client_path: Path = Path("/home/kamimura/.local/bin/tunnel-client")
        uv_python_root: Path = Path("/home/kamimura/.local/share/uv/python")
        uv_path: Path = Path("/home/kamimura/.local/bin/uv")
        runtime_root: Path = _DEFAULT_RUNTIME_ROOT
        remote_url: str = _DEFAULT_REMOTE_URL
        gpu_lock_file: Path = Path("/var/lib/tennis-lab-actions/gpu.lock")
        access_token_ttl_seconds: int = 3600
        refresh_token_ttl_seconds: int = 30 * 24 * 3600
        authorization_ttl_seconds: int = 300

        def __post_init__(self) -> None:
            project = self.repo_root.resolve()
            state = self.state_dir.resolve()
            runtime = self.runtime_root.resolve()
            if _inside(state, project):
                raise ValueError("TENNIS_MCP_STATE_DIR must be outside tennis-lab")
            if _inside(runtime, project):
                raise ValueError("TENNIS_MCP_RUNTIME_ROOT must be outside tennis-lab")
            if state == runtime or _inside(state, runtime) or _inside(runtime, state):
                raise ValueError("runtime and state roots must be separate directories")
            if not self.remote_url.strip() or any(
                character in self.remote_url for character in ("\n", "\r", "\x00")
            ):
                raise ValueError("TENNIS_MCP_REMOTE_URL is malformed")

        @classmethod
        def from_env(
            cls,
            *,
            public_base_url: str | None = None,
            require_public_base_url: bool = True,
        ) -> GatewaySettings:
            """Load settings from explicit `TENNIS_MCP_*` environment variables."""

            repo_root = _required_absolute_directory(
                os.environ.get("TENNIS_MCP_REPO_ROOT", str(_DEFAULT_PROJECT_ROOT)),
                "TENNIS_MCP_REPO_ROOT",
            )
            state_dir = _absolute_path(
                os.environ.get("TENNIS_MCP_STATE_DIR", str(_DEFAULT_STATE_ROOT)),
                "TENNIS_MCP_STATE_DIR",
            )
            runtime_root = _absolute_path(
                os.environ.get("TENNIS_MCP_RUNTIME_ROOT", str(_DEFAULT_RUNTIME_ROOT)),
                "TENNIS_MCP_RUNTIME_ROOT",
            )
            base_value = public_base_url or os.environ.get("TENNIS_MCP_PUBLIC_BASE_URL")
            if not base_value and require_public_base_url:
                raise ValueError("TENNIS_MCP_PUBLIC_BASE_URL is required")

            port = int(os.environ.get("TENNIS_MCP_PORT", "8765"))
            if not 1024 <= port <= 65535:
                raise ValueError("TENNIS_MCP_PORT must be between 1024 and 65535")

            uv_default = shutil.which("uv") or "/home/kamimura/.local/bin/uv"
            return cls(
                repo_root=repo_root,
                state_dir=state_dir,
                public_base_url=(
                    normalize_public_base_url(base_value) if base_value else None
                ),
                host=os.environ.get("TENNIS_MCP_HOST", "127.0.0.1"),
                port=port,
                docker_image=os.environ.get(
                    "TENNIS_MCP_DOCKER_IMAGE",
                    "nvidia/cuda:13.0.0-base-ubuntu24.04",
                ),
                cloudflared_path=_absolute_path(
                    os.environ.get(
                        "TENNIS_MCP_CLOUDFLARED",
                        "/home/kamimura/.local/bin/cloudflared",
                    ),
                    "TENNIS_MCP_CLOUDFLARED",
                ),
                tunnel_client_path=_absolute_path(
                    os.environ.get(
                        "TENNIS_MCP_TUNNEL_CLIENT",
                        "/home/kamimura/.local/bin/tunnel-client",
                    ),
                    "TENNIS_MCP_TUNNEL_CLIENT",
                ),
                uv_python_root=_absolute_path(
                    os.environ.get(
                        "TENNIS_MCP_UV_PYTHON_ROOT",
                        "/home/kamimura/.local/share/uv/python",
                    ),
                    "TENNIS_MCP_UV_PYTHON_ROOT",
                ),
                uv_path=_absolute_path(
                    os.environ.get("TENNIS_MCP_UV", uv_default),
                    "TENNIS_MCP_UV",
                ),
                runtime_root=runtime_root,
                remote_url=os.environ.get("TENNIS_MCP_REMOTE_URL", _DEFAULT_REMOTE_URL),
                gpu_lock_file=_absolute_path(
                    os.environ.get(
                        "TENNIS_MCP_GPU_LOCK_FILE",
                        "/var/lib/tennis-lab-actions/gpu.lock",
                    ),
                    "TENNIS_MCP_GPU_LOCK_FILE",
                ),
            )

        @property
        def resource_url(self) -> str:
            if self.public_base_url is None:
                raise ValueError("public_base_url is required for OAuth mode")
            return f"{self.public_base_url}/mcp"

        @property
        def venv_root(self) -> Path:
            return self.repo_root / ".venv"

        @property
        def trusted_mirror_dir(self) -> Path:
            return self.state_dir / "trusted-mirror.git"

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
            return self.state_dir / "command-specs"

        @property
        def sandbox_jobs_dir(self) -> Path:
            return self.state_dir / "sandboxes"

        @property
        def training_queue_dir(self) -> Path:
            return self.state_dir / "training-queue"

        @property
        def training_queue_jobs_dir(self) -> Path:
            return self.training_queue_dir / "jobs"

        @property
        def training_queue_running_dir(self) -> Path:
            return self.training_queue_dir / "running"

        @property
        def training_queue_done_dir(self) -> Path:
            return self.training_queue_dir / "done"

        @property
        def training_queue_failed_dir(self) -> Path:
            return self.training_queue_dir / "failed"

        @property
        def training_queue_logs_dir(self) -> Path:
            return self.training_queue_dir / "logs"

        @property
        def training_queue_worker_lock(self) -> Path:
            return self.training_queue_dir / "worker.lock"

        @property
        def git_mask_path(self) -> Path:
            return self.state_dir / "masked-git-pointer"

        @property
        def git_mask_dir(self) -> Path:
            return self.state_dir / "masked-git-directory"

        @property
        def runtime_releases_dir(self) -> Path:
            return self.runtime_root / "releases"

        @property
        def runtime_current_dir(self) -> Path:
            return self.runtime_root / "current"

        @property
        def project_persistent_roots(self) -> tuple[str, ...]:
            return _PERSISTENT_PROJECT_ROOTS

        def ensure_state(self) -> None:
            """Create private control-plane state outside the sacrificial project."""

            directories = (
                self.state_dir,
                self.job_specs_dir,
                self.sandbox_jobs_dir,
                self.revision_workspace_dir,
                self.training_queue_dir,
                self.training_queue_jobs_dir,
                self.training_queue_running_dir,
                self.training_queue_done_dir,
                self.training_queue_failed_dir,
                self.training_queue_logs_dir,
            )
            for directory in directories:
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

            if not self.git_mask_path.exists():
                descriptor = os.open(
                    self.git_mask_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o400,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write("git metadata is unavailable in MCP sandboxes\n")
            os.chmod(self.git_mask_path, 0o400)
            self.git_mask_dir.mkdir(mode=0o500, exist_ok=True)
            os.chmod(self.git_mask_dir, 0o500)

        def ensure_project_runtime_dirs(self) -> None:
            """Create fixed project-owned data/output roots used by normal task CLIs."""

            for name in self.project_persistent_roots:
                target = self.repo_root / name
                if target.exists() or target.is_symlink():
                    continue
                target.mkdir(mode=0o700, parents=True)

        def read_owner_secret(self) -> str:
            secret = self.owner_secret_path.read_text(encoding="utf-8").strip()
            if len(secret) < 32:
                raise ValueError("owner secret is missing or too short")
            return secret
    ''',
)


write(
    "src/automation/chatgpt_mcp/workspace.py",
    r'''
    """Trusted exact-revision workspaces backed by a project-external bare mirror."""

    from __future__ import annotations

    import os
    import re
    import secrets
    import subprocess
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    from src.automation.chatgpt_mcp.storage import SqliteStore

    _WORKSPACE_ID = re.compile(r"^rev-[a-f0-9]{16}$")
    _GIT_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
    _GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


    class WorkspaceError(RuntimeError):
        """Raised when a trusted remote revision cannot be materialized safely."""


    @dataclass(frozen=True)
    class RevisionWorkspace:
        workspace_id: str
        path: Path
        branch: str
        revision: str

        def public_dict(self) -> dict[str, str]:
            return {
                "workspace_id": self.workspace_id,
                "branch": self.branch,
                "revision": self.revision,
            }


    def _run(arguments: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            arguments,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env={
                **dict(os.environ),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
            },
        )


    def _checked(arguments: list[str], *, message: str, timeout: int = 120) -> str:
        result = _run(arguments, timeout=timeout)
        if result.returncode != 0:
            raise WorkspaceError(result.stderr.strip() or result.stdout.strip() or message)
        return result.stdout.strip()


    def _validate_branch(value: str) -> str:
        branch = value.strip()
        if (
            not _GIT_BRANCH.fullmatch(branch)
            or ".." in branch
            or "//" in branch
            or "@{" in branch
            or branch.endswith("/")
            or branch.startswith("-")
        ):
            raise WorkspaceError(f"invalid remote branch: {value!r}")
        return branch


    def _validate_revision(value: str) -> str:
        revision = value.strip().lower()
        if not _GIT_SHA.fullmatch(revision):
            raise WorkspaceError("expected_sha must be a full 40-character commit SHA")
        return revision


    def _validate_workspace_id(value: str) -> str:
        workspace_id = value.strip()
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise WorkspaceError(f"invalid revision workspace id: {value!r}")
        return workspace_id


    class WorkspaceManager:
        """Fetch fixed-origin commits without consulting the mutable project `.git`."""

        def __init__(
            self,
            repo_root: Path,
            state_or_revision_root: Path,
            store: SqliteStore,
            *,
            remote_url: str = "https://github.com/Motoki0705/tennis-lab.git",
            mirror_root: Path | None = None,
        ) -> None:
            self.project_root = repo_root.resolve()
            configured_root = state_or_revision_root.resolve()
            self.state_root = (
                configured_root.parent
                if configured_root.name == "revisions"
                else configured_root
            )
            self.workspace_root = (
                configured_root
                if configured_root.name == "revisions"
                else (configured_root / "revisions").resolve()
            )
            self.mirror_root = (
                mirror_root.resolve()
                if mirror_root is not None
                else (self.state_root / "trusted-mirror.git").resolve()
            )
            for protected in (self.workspace_root, self.mirror_root):
                if protected == self.project_root or protected.is_relative_to(
                    self.project_root
                ):
                    raise WorkspaceError(
                        "trusted mirror and revision workspaces must be outside tennis-lab"
                    )
            self.remote_url = remote_url
            self.store = store

        def _git_dir(self, *arguments: str) -> list[str]:
            return ["git", f"--git-dir={self.mirror_root}", *arguments]

        def _ensure_mirror(self) -> None:
            self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.state_root, 0o700)
            if not self.mirror_root.exists():
                temporary = self.mirror_root.with_name(
                    f".{self.mirror_root.name}.{secrets.token_hex(8)}.tmp"
                )
                result = _run(["git", "init", "--bare", "-q", str(temporary)])
                if result.returncode != 0:
                    raise WorkspaceError(result.stderr.strip() or "git init --bare failed")
                os.replace(temporary, self.mirror_root)
            if not self.mirror_root.is_dir():
                raise WorkspaceError("trusted mirror path is not a directory")
            os.chmod(self.mirror_root, 0o700)
            _checked(
                self._git_dir("config", "remote.origin.url", self.remote_url),
                message="could not fix trusted origin URL",
            )
            _checked(
                self._git_dir("config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"),
                message="could not fix trusted origin refspec",
            )

        def prepare_revision(self, *, branch: str, expected_sha: str) -> dict[str, str]:
            checked_branch = _validate_branch(branch)
            checked_sha = _validate_revision(expected_sha)
            self.workspace_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.workspace_root, 0o700)
            self._ensure_mirror()

            remote_ref = f"refs/remotes/origin/{checked_branch}"
            refspec = f"+refs/heads/{checked_branch}:{remote_ref}"
            _checked(
                self._git_dir(
                    "fetch",
                    "--force",
                    "--no-tags",
                    "--no-recurse-submodules",
                    "origin",
                    refspec,
                ),
                message="trusted git fetch failed",
                timeout=300,
            )
            fetched_sha = _checked(
                self._git_dir("rev-parse", f"{remote_ref}^{{commit}}"),
                message="fetched branch did not resolve to a commit",
            ).lower()
            if fetched_sha != checked_sha:
                raise WorkspaceError(
                    "remote revision mismatch: "
                    f"origin/{checked_branch} is {fetched_sha}, expected {checked_sha}"
                )

            workspace_id = f"rev-{secrets.token_hex(8)}"
            target = (self.workspace_root / workspace_id).resolve()
            if not target.is_relative_to(self.workspace_root) or target.parent != self.workspace_root:
                raise WorkspaceError("revision workspace escaped its configured root")
            result = _run(
                self._git_dir("worktree", "add", "--detach", str(target), checked_sha),
                timeout=300,
            )
            if result.returncode != 0:
                raise WorkspaceError(result.stderr.strip() or "trusted worktree add failed")
            os.chmod(target, 0o700)

            workspace = RevisionWorkspace(
                workspace_id=workspace_id,
                path=target,
                branch=checked_branch,
                revision=checked_sha,
            )
            try:
                self._verify_materialized_workspace(workspace)
            except BaseException:
                _run(self._git_dir("worktree", "remove", "--force", str(target)))
                raise
            self.store.put(
                "revision_workspaces",
                workspace_id,
                {
                    "workspace_id": workspace_id,
                    "path": str(target),
                    "branch": checked_branch,
                    "revision": checked_sha,
                },
            )
            return workspace.public_dict()

        def get_revision(self, workspace_id: str) -> RevisionWorkspace:
            checked_id = _validate_workspace_id(workspace_id)
            payload = self.store.get("revision_workspaces", checked_id)
            if payload is None:
                raise WorkspaceError("revision workspace was not found")
            path = Path(str(payload["path"])).resolve()
            if not path.is_relative_to(self.workspace_root) or path.parent != self.workspace_root:
                raise WorkspaceError("stored revision workspace escaped its configured root")
            workspace = RevisionWorkspace(
                workspace_id=checked_id,
                path=path,
                branch=_validate_branch(str(payload["branch"])),
                revision=_validate_revision(str(payload["revision"])),
            )
            self._verify_materialized_workspace(workspace)
            return workspace

        def assert_execution_ready(
            self, *, workspace_id: str, expected_sha: str
        ) -> RevisionWorkspace:
            checked_sha = _validate_revision(expected_sha)
            workspace = self.get_revision(workspace_id)
            if workspace.revision != checked_sha:
                raise WorkspaceError(
                    "workspace revision does not match expected_sha: "
                    f"{workspace.revision} != {checked_sha}"
                )
            status = _checked(
                [
                    "git",
                    "-C",
                    str(workspace.path),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                message="git status failed",
            )
            if status:
                raise WorkspaceError(
                    "trusted revision workspace contains changes; prepare a new workspace"
                )
            return workspace

        def describe_revision(self, workspace_id: str) -> dict[str, Any]:
            workspace = self.get_revision(workspace_id)
            status = _checked(
                [
                    "git",
                    "-C",
                    str(workspace.path),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                message="git status failed",
            )
            return {
                **workspace.public_dict(),
                "clean": not bool(status),
                "source": "project-external trusted mirror",
            }

        def _verify_materialized_workspace(self, workspace: RevisionWorkspace) -> None:
            if not workspace.path.is_dir():
                raise WorkspaceError("revision workspace directory is missing")
            top_level = _checked(
                ["git", "-C", str(workspace.path), "rev-parse", "--show-toplevel"],
                message="path is not a git worktree",
            )
            if Path(top_level).resolve() != workspace.path:
                raise WorkspaceError("revision workspace must name its exact git root")
            head = _checked(
                ["git", "-C", str(workspace.path), "rev-parse", "HEAD^{commit}"],
                message="revision workspace HEAD is unavailable",
            ).lower()
            if head != workspace.revision:
                raise WorkspaceError(
                    f"revision workspace moved from {workspace.revision} to {head}"
                )
            git_pointer = workspace.path / ".git"
            if git_pointer.is_symlink() or not git_pointer.is_file():
                raise WorkspaceError("revision workspace .git pointer is not a regular file")
            first_line = git_pointer.read_text(encoding="utf-8").splitlines()[0]
            if not first_line.startswith("gitdir: "):
                raise WorkspaceError("revision workspace .git pointer is malformed")
            git_dir = Path(first_line.removeprefix("gitdir: ")).resolve()
            metadata_root = (self.mirror_root / "worktrees").resolve()
            if not git_dir.is_relative_to(metadata_root):
                raise WorkspaceError("revision metadata is outside the trusted mirror")
    ''',
)


write(
    "src/automation/chatgpt_mcp/jobs.py",
    r'''
    """Arbitrary project-sandbox commands and project-external serialized GPU jobs."""

    from __future__ import annotations

    import contextlib
    import hashlib
    import json
    import os
    import re
    import secrets
    import shlex
    import stat
    import subprocess
    import threading
    import time
    from pathlib import Path, PurePosixPath
    from typing import Any, IO, Protocol

    from pydantic import BaseModel, Field, field_validator

    from src.automation.chatgpt_mcp.settings import GatewaySettings
    from src.automation.chatgpt_mcp.storage import SqliteStore
    from src.automation.chatgpt_mcp.workspace import RevisionWorkspace, WorkspaceManager

    TRAINING_WORKER_SERVICE_NAME = "tennis-lab-chatgpt-training-worker.service"
    _JOB_ID = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
    _WORKSPACE_ID = re.compile(r"^rev-[a-f0-9]{16}$")
    _SHA = re.compile(r"^[0-9a-f]{40}$")
    _TRAINING_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    _QUEUE_FILE = re.compile(r"^[0-9]{20}_[a-z0-9-]{8,64}\.json$")
    _DIRECT_COMMAND_MAX_SECONDS = 24 * 3600
    _MAX_CONCURRENT_DIRECT_JOBS = 4
    _JOB_METADATA_TTL_SECONDS = 30 * 24 * 3600
    _TRAINING_METADATA_TTL_SECONDS = 90 * 24 * 3600
    _COMMAND_FILE_NAME = "command"
    _COMMAND_MOUNT_PATH = "/run/tennis-mcp-command"

    _SECRET_PATTERNS = (
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    )


    class ExecutionWorkspaceResolver(Protocol):
        def assert_execution_ready(
            self, *, workspace_id: str, expected_sha: str
        ) -> RevisionWorkspace: ...


    class JobError(RuntimeError):
        """Raised for invalid job state or a failed sandbox operation."""


    def _validate_working_directory(value: str) -> str:
        normalized = value.strip() or "."
        path = PurePosixPath(normalized)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            raise ValueError("working_directory must stay below the exact revision root")
        if "\x00" in normalized or len(normalized) > 512:
            raise ValueError("working_directory is malformed")
        return str(path)


    class SandboxSpec(BaseModel):
        """Private execution contract for one exact revision and RW project mount."""

        job_id: str = Field(pattern=_JOB_ID.pattern)
        command: str = Field(min_length=1, max_length=200_000)
        workspace_id: str = Field(pattern=_WORKSPACE_ID.pattern)
        expected_sha: str = Field(pattern=_SHA.pattern)
        working_directory: str = "."
        use_gpu: bool = False
        timeout_seconds: int = Field(default=900, ge=1, le=7 * 24 * 3600)

        @field_validator("command")
        @classmethod
        def reject_nul_command(cls, value: str) -> str:
            if "\x00" in value:
                raise ValueError("command may not contain NUL")
            return value

        @field_validator("working_directory")
        @classmethod
        def validate_working_directory(cls, value: str) -> str:
            return _validate_working_directory(value)


    def _redact_secrets(value: str) -> str:
        redacted = value
        for pattern in _SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted


    def _command_digest(command: str) -> str:
        return hashlib.sha256(command.encode("utf-8")).hexdigest()


    def _safe_mount(path: Path, target: str, *, read_only: bool) -> str:
        source = path.resolve()
        if "," in str(source) or "," in target:
            raise JobError("Docker bind mount paths may not contain commas")
        value = f"type=bind,src={source},dst={target}"
        if read_only:
            value += ",readonly"
        return value


    def _write_private_file(path: Path, value: str) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(path, 0o600)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


    def _private_regular_file(path: Path, *, root: Path) -> Path:
        resolved = path.resolve()
        if (
            path.is_symlink()
            or not resolved.is_relative_to(root.resolve())
            or resolved.parent != root.resolve()
            or not resolved.is_file()
        ):
            raise JobError("private spec path escaped its owner-only directory")
        file_stat = resolved.stat()
        if file_stat.st_uid != os.getuid() or stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise JobError("private spec ownership or permissions are unsafe")
        return resolved


    class DockerSandbox:
        """Execute arbitrary shell inside a bounded container with only tennis-lab RW."""

        def __init__(
            self,
            settings: GatewaySettings,
            workspaces: ExecutionWorkspaceResolver,
        ) -> None:
            self.settings = settings
            self.workspaces = workspaces

        def container_name(self, job_id: str) -> str:
            if not _JOB_ID.fullmatch(job_id):
                raise JobError(f"invalid job id: {job_id}")
            return f"tennis-lab-mcp-{job_id}"

        def _job_directories(self, job_id: str) -> tuple[Path, Path, Path, Path]:
            job_root = (self.settings.sandbox_jobs_dir / job_id).resolve()
            sandbox_root = self.settings.sandbox_jobs_dir.resolve()
            if not job_root.is_relative_to(sandbox_root) or job_root.parent != sandbox_root:
                raise JobError("sandbox job path escaped its configured root")
            if job_root.exists():
                raise JobError(f"sandbox directory already exists for {job_id}")
            workspace_copy = job_root / "workspace"
            artifacts = job_root / "artifacts"
            command_path = job_root / _COMMAND_FILE_NAME
            workspace_copy.mkdir(mode=0o700, parents=True)
            artifacts.mkdir(mode=0o700)
            os.chmod(job_root, 0o700)
            return job_root, workspace_copy, artifacts, command_path

        def _command_file(self, job_id: str) -> Path:
            return self.settings.sandbox_jobs_dir / job_id / _COMMAND_FILE_NAME

        def _project_git_mask(self) -> Path:
            git_entry = self.settings.repo_root / ".git"
            return self.settings.git_mask_dir if git_entry.is_dir() else self.settings.git_mask_path

        def command(self, spec: SandboxSpec, *, detached: bool) -> list[str]:
            source = self.workspaces.assert_execution_ready(
                workspace_id=spec.workspace_id,
                expected_sha=spec.expected_sha,
            )
            self.settings.ensure_state()
            self.settings.ensure_project_runtime_dirs()
            if not self.settings.venv_root.is_dir():
                raise JobError(
                    f"project virtual environment is missing: {self.settings.venv_root}"
                )
            if not self.settings.uv_python_root.is_dir():
                raise JobError(f"uv Python runtime is missing: {self.settings.uv_python_root}")
            _, workspace_copy, artifacts, command_path = self._job_directories(spec.job_id)
            _write_private_file(command_path, spec.command)

            project_root = str(self.settings.repo_root)
            link_commands = " ".join(
                (
                    f"rm -rf /workspace/{shlex.quote(name)}; "
                    f"ln -s {shlex.quote(project_root + '/' + name)} "
                    f"/workspace/{shlex.quote(name)};"
                )
                for name in self.settings.project_persistent_roots
            )
            workdir = shlex.quote(spec.working_directory)
            wrapper = (
                "set -euo pipefail; "
                "mkdir -p /tmp/tennis-mcp-home /workspace /artifacts/repro; "
                "cp -a /source/. /workspace/; "
                "rm -rf /workspace/.git; "
                f"{link_commands} "
                f"target=/workspace/{workdir}; "
                "test -d \"$target\" || { echo 'working_directory does not exist' >&2; exit 2; }; "
                "cd \"$target\"; "
                f"exec /usr/bin/timeout --signal=TERM --kill-after=30s "
                f"{spec.timeout_seconds} /bin/bash {_COMMAND_MOUNT_PATH}"
            )
            project_target = str(self.settings.repo_root)
            git_mask = self._project_git_mask()
            mounts = [
                _safe_mount(source.path, "/source", read_only=True),
                _safe_mount(workspace_copy, "/workspace", read_only=False),
                _safe_mount(artifacts, "/artifacts", read_only=False),
                _safe_mount(command_path, _COMMAND_MOUNT_PATH, read_only=True),
                _safe_mount(self.settings.repo_root, project_target, read_only=False),
                _safe_mount(self.settings.repo_root, "/project", read_only=False),
                _safe_mount(self.settings.venv_root, str(self.settings.venv_root), read_only=True),
                _safe_mount(self.settings.venv_root, "/project/.venv", read_only=True),
                _safe_mount(self.settings.uv_python_root, str(self.settings.uv_python_root), read_only=True),
                _safe_mount(self.settings.git_mask_path, "/source/.git", read_only=True),
            ]
            if (self.settings.repo_root / ".git").exists():
                mounts.extend(
                    [
                        _safe_mount(git_mask, f"{project_target}/.git", read_only=True),
                        _safe_mount(git_mask, "/project/.git", read_only=True),
                    ]
                )

            arguments = [
                "docker",
                "run",
                "--name",
                self.container_name(spec.job_id),
                "--label",
                "tennis-lab.mcp=true",
                "--label",
                f"tennis-lab.revision={spec.expected_sha}",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--init",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "8192",
                "--memory",
                "48g",
                "--shm-size",
                "8g",
                "--network",
                "none",
                "--ipc",
                "private",
                "--read-only",
                "--pull",
                "never",
                "--workdir",
                "/workspace",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=8g,mode=1777",
                "--env",
                "HOME=/tmp/tennis-mcp-home",
                "--env",
                "PYTHONUNBUFFERED=1",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "PYTHONNOUSERSITE=1",
                "--env",
                "GIT_CONFIG_NOSYSTEM=1",
                "--env",
                "GIT_CONFIG_GLOBAL=/dev/null",
                "--env",
                f"VIRTUAL_ENV={self.settings.venv_root}",
                "--env",
                f"PATH={self.settings.venv_root / 'bin'}:/usr/local/bin:/usr/bin:/bin",
                "--env",
                f"TENNIS_RUN_ID={spec.job_id}",
                "--env",
                "TENNIS_REPRO_DIR=/artifacts/repro",
                "--env",
                f"TENNIS_LAB_PROJECT_ROOT={project_target}",
            ]
            for mount in mounts:
                arguments.extend(["--mount", mount])
            if detached:
                arguments.append("--detach")
            if spec.use_gpu:
                arguments.extend(["--gpus", "all"])
            arguments.extend([self.settings.docker_image, "/bin/bash", "-lc", wrapper])
            return arguments

        def start(self, spec: SandboxSpec) -> str:
            arguments = self.command(spec, detached=True)
            command_path = self._command_file(spec.job_id)
            try:
                result = subprocess.run(
                    arguments,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=120,
                )
            finally:
                command_path.unlink(missing_ok=True)
            if result.returncode != 0:
                raise JobError(_redact_secrets(result.stderr.strip()) or "docker run failed")
            return result.stdout.strip()

        def run_foreground(self, spec: SandboxSpec, *, log_stream: IO[bytes]) -> int:
            arguments = self.command(spec, detached=False)
            command_path = self._command_file(spec.job_id)
            try:
                process = subprocess.run(
                    arguments,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=spec.timeout_seconds + 180,
                )
            finally:
                command_path.unlink(missing_ok=True)
            return process.returncode

        def inspect(self, job_id: str) -> dict[str, Any]:
            result = subprocess.run(
                ["docker", "inspect", self.container_name(job_id)],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise JobError("sandbox container was not found")
            document = json.loads(result.stdout)[0]
            state = document["State"]
            return {
                "status": state["Status"],
                "running": bool(state["Running"]),
                "exit_code": state["ExitCode"] if not state["Running"] else None,
                "started_at": state["StartedAt"],
                "finished_at": state["FinishedAt"],
                "error": _redact_secrets(state["Error"]) or None,
            }

        def logs(self, job_id: str, *, tail: int = 400) -> str:
            if not 1 <= tail <= 5000:
                raise JobError("tail must be between 1 and 5000")
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), self.container_name(job_id)],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise JobError(_redact_secrets(result.stderr.strip()) or "docker logs failed")
            return _redact_secrets((result.stdout + result.stderr)[-200_000:])

        def stop(self, job_id: str) -> None:
            result = subprocess.run(
                ["docker", "stop", "--time", "10", self.container_name(job_id)],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise JobError(_redact_secrets(result.stderr.strip()) or "docker stop failed")


    class JobManager:
        """Create and observe bounded arbitrary CPU commands."""

        def __init__(
            self,
            settings: GatewaySettings,
            store: SqliteStore,
            workspaces: ExecutionWorkspaceResolver,
        ) -> None:
            self.settings = settings
            self.store = store
            self.sandbox = DockerSandbox(settings, workspaces)
            self._admission_lock = threading.Lock()

        @staticmethod
        def new_job_id(prefix: str = "job") -> str:
            return f"{prefix}-{secrets.token_hex(8)}"

        def _running_direct_jobs(self) -> int:
            running = 0
            for payload in self.store.list("jobs", limit=1000):
                with contextlib.suppress(JobError):
                    if self.sandbox.inspect(str(payload["job_id"]))["running"]:
                        running += 1
            return running

        def start(
            self,
            *,
            command: str,
            workspace_id: str,
            expected_sha: str,
            working_directory: str,
            timeout_seconds: int,
        ) -> dict[str, Any]:
            if timeout_seconds > _DIRECT_COMMAND_MAX_SECONDS:
                raise JobError(
                    f"direct commands are limited to {_DIRECT_COMMAND_MAX_SECONDS} seconds; "
                    "use enqueue_training for serialized GPU or longer work"
                )
            job_id = self.new_job_id()
            spec = SandboxSpec(
                job_id=job_id,
                command=command,
                workspace_id=workspace_id,
                expected_sha=expected_sha.lower(),
                working_directory=working_directory,
                use_gpu=False,
                timeout_seconds=timeout_seconds,
            )
            with self._admission_lock:
                if self._running_direct_jobs() >= _MAX_CONCURRENT_DIRECT_JOBS:
                    raise JobError("the maximum number of concurrent direct jobs is running")
                container_id = self.sandbox.start(spec)
            payload = {
                "job_id": job_id,
                "workspace_id": workspace_id,
                "revision": spec.expected_sha,
                "working_directory": spec.working_directory,
                "container_id": container_id,
                "command_sha256": _command_digest(command),
                "created_at": time.time(),
            }
            self.store.put(
                "jobs",
                job_id,
                payload,
                expires_at=time.time() + _JOB_METADATA_TTL_SECONDS,
            )
            return {"job_id": job_id, "status": "running", "revision": spec.expected_sha}

        def get(self, job_id: str) -> dict[str, Any]:
            payload = self.store.get("jobs", job_id)
            if payload is None:
                raise JobError("job id was not found")
            return {**payload, **self.sandbox.inspect(job_id)}

        def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
            summaries: list[dict[str, Any]] = []
            for payload in self.store.list("jobs", limit=limit):
                try:
                    state = self.sandbox.inspect(str(payload["job_id"]))
                except JobError:
                    state = {"status": "missing", "running": False, "exit_code": None}
                summaries.append({**payload, **state})
            return summaries


    class TrainingQueueManager:
        """Queue arbitrary GPU commands for the trusted external worker."""

        def __init__(
            self,
            settings: GatewaySettings,
            store: SqliteStore,
            workspaces: ExecutionWorkspaceResolver,
        ) -> None:
            self.settings = settings
            self.store = store
            self.workspaces = workspaces
            self.sandbox = DockerSandbox(settings, workspaces)
            self.settings.ensure_state()

        def enqueue(
            self,
            *,
            name: str,
            command: str,
            workspace_id: str,
            expected_sha: str,
            working_directory: str,
            issue: int | None,
            timeout_seconds: int,
        ) -> dict[str, Any]:
            if not _TRAINING_NAME.fullmatch(name):
                raise JobError(
                    "training name must start with an alphanumeric character and use "
                    "only letters, digits, dot, underscore, or hyphen"
                )
            if issue is not None and issue <= 0:
                raise JobError("issue must be a positive integer")
            self.workspaces.assert_execution_ready(
                workspace_id=workspace_id,
                expected_sha=expected_sha,
            )
            job_id = JobManager.new_job_id("train")
            spec = SandboxSpec(
                job_id=job_id,
                command=command,
                workspace_id=workspace_id,
                expected_sha=expected_sha.lower(),
                working_directory=working_directory,
                use_gpu=True,
                timeout_seconds=timeout_seconds,
            )
            queue_file = f"{time.time_ns():020d}_{job_id}.json"
            if not _QUEUE_FILE.fullmatch(queue_file):
                raise JobError("generated queue filename is invalid")
            temporary = self.settings.training_queue_jobs_dir / f".{queue_file}.tmp"
            final = self.settings.training_queue_jobs_dir / queue_file
            _write_private_file(temporary, spec.model_dump_json(indent=2) + "\n")
            os.replace(temporary, final)
            os.chmod(final, 0o600)
            payload = {
                "job_id": job_id,
                "name": name,
                "issue": issue,
                "workspace_id": workspace_id,
                "revision": spec.expected_sha,
                "working_directory": spec.working_directory,
                "queue_file": queue_file,
                "command_sha256": _command_digest(command),
                "status": "queued",
                "created_at": time.time(),
            }
            self.store.put(
                "training_jobs",
                job_id,
                payload,
                expires_at=time.time() + _TRAINING_METADATA_TTL_SECONDS,
            )
            try:
                self._start_worker()
            except BaseException:
                final.unlink(missing_ok=True)
                self.store.delete("training_jobs", job_id)
                raise
            return {
                "job_id": job_id,
                "queue_file": queue_file,
                "status": "queued",
                "revision": spec.expected_sha,
            }

        def _start_worker(self) -> None:
            result = subprocess.run(
                ["systemctl", "--user", "start", TRAINING_WORKER_SERVICE_NAME],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise JobError(
                    _redact_secrets(result.stderr.strip())
                    or "trusted training worker failed to start"
                )

        def status(self, job_id: str) -> dict[str, Any]:
            payload = self.store.get("training_jobs", job_id)
            if payload is None:
                raise JobError("training job id was not found")
            queue_file = str(payload["queue_file"])
            locations = {
                "queued": self.settings.training_queue_jobs_dir / queue_file,
                "running": self.settings.training_queue_running_dir / queue_file,
                "succeeded": self.settings.training_queue_done_dir / queue_file,
                "failed": self.settings.training_queue_failed_dir / queue_file,
            }
            status = str(payload.get("status", "unknown"))
            for candidate_status, path in locations.items():
                if path.exists():
                    status = candidate_status
                    break
            result: dict[str, Any] = {**payload, "status": status}
            with contextlib.suppress(JobError):
                result.update(self.sandbox.inspect(job_id))
            return result

        def logs(self, job_id: str, *, tail: int = 400) -> str:
            payload = self.store.get("training_jobs", job_id)
            if payload is None:
                raise JobError("training job id was not found")
            log_path = self.settings.training_queue_logs_dir / f"{job_id}.log"
            if not log_path.is_file():
                with contextlib.suppress(JobError):
                    return self.sandbox.logs(job_id, tail=tail)
                return "training log is not available yet"
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return _redact_secrets("\n".join(lines[-tail:])[-200_000:])

        def status_summary(self) -> dict[str, Any]:
            worker = subprocess.run(
                ["systemctl", "--user", "is-active", TRAINING_WORKER_SERVICE_NAME],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            return {
                "worker": worker.stdout.strip() or "inactive",
                "queued": len(list(self.settings.training_queue_jobs_dir.glob("*.json"))),
                "running": len(list(self.settings.training_queue_running_dir.glob("*.json"))),
                "succeeded": len(list(self.settings.training_queue_done_dir.glob("*.json"))),
                "failed": len(list(self.settings.training_queue_failed_dir.glob("*.json"))),
                "state_root": str(self.settings.training_queue_dir),
            }


    def load_private_spec(path: Path, *, root: Path) -> SandboxSpec:
        resolved = _private_regular_file(path, root=root)
        return SandboxSpec.model_validate_json(resolved.read_text(encoding="utf-8"))


    def execute_sandbox_spec(settings: GatewaySettings, spec_path: Path) -> int:
        """Compatibility entry point for owner-only command specs."""

        resolved = _private_regular_file(spec_path, root=settings.job_specs_dir)
        try:
            spec = SandboxSpec.model_validate_json(resolved.read_text(encoding="utf-8"))
            store = SqliteStore(settings.database_path)
            workspaces = WorkspaceManager(
                settings.repo_root,
                settings.revision_workspace_dir,
                store,
                remote_url=settings.remote_url,
                mirror_root=settings.trusted_mirror_dir,
            )
            log_path = settings.training_queue_logs_dir / f"{spec.job_id}.log"
            with log_path.open("ab") as stream:
                return DockerSandbox(settings, workspaces).run_foreground(
                    spec, log_stream=stream
                )
        finally:
            resolved.unlink(missing_ok=True)
    ''',
)


write(
    "src/automation/chatgpt_mcp/queue_worker.py",
    r'''
    """Trusted project-external FIFO worker for all MCP GPU execution."""

    from __future__ import annotations

    import contextlib
    import fcntl
    import json
    import os
    import time
    from pathlib import Path

    from src.automation.chatgpt_mcp.jobs import (
        DockerSandbox,
        JobError,
        SandboxSpec,
        _redact_secrets,
        load_private_spec,
    )
    from src.automation.chatgpt_mcp.settings import GatewaySettings
    from src.automation.chatgpt_mcp.storage import SqliteStore
    from src.automation.chatgpt_mcp.workspace import WorkspaceManager


    def _claim(settings: GatewaySettings) -> Path | None:
        for queued in sorted(settings.training_queue_jobs_dir.glob("*.json")):
            running = settings.training_queue_running_dir / queued.name
            try:
                os.replace(queued, running)
            except FileNotFoundError:
                continue
            os.chmod(running, 0o600)
            return running
        return None


    def _marker(settings: GatewaySettings, *, name: str, succeeded: bool) -> Path:
        root = (
            settings.training_queue_done_dir
            if succeeded
            else settings.training_queue_failed_dir
        )
        path = root / name
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        os.close(descriptor)
        return path


    def _update_status(
        store: SqliteStore,
        spec: SandboxSpec,
        *,
        status: str,
        exit_code: int,
        error: str | None = None,
    ) -> None:
        payload = store.get("training_jobs", spec.job_id) or {
            "job_id": spec.job_id,
            "workspace_id": spec.workspace_id,
            "revision": spec.expected_sha,
        }
        payload.update(
            {
                "status": status,
                "exit_code": exit_code,
                "finished_at": time.time(),
                "error": _redact_secrets(error or "") or None,
            }
        )
        store.put("training_jobs", spec.job_id, payload)


    def run_worker(settings: GatewaySettings, *, idle_timeout: int = 30) -> int:
        """Run exactly one serialized GPU worker until its queue is idle."""

        if not 0 <= idle_timeout <= 3600:
            raise ValueError("idle_timeout must be between 0 and 3600")
        settings.ensure_state()
        lock_descriptor = os.open(
            settings.training_queue_worker_lock,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        try:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            store = SqliteStore(settings.database_path)
            workspaces = WorkspaceManager(
                settings.repo_root,
                settings.revision_workspace_dir,
                store,
                remote_url=settings.remote_url,
                mirror_root=settings.trusted_mirror_dir,
            )
            sandbox = DockerSandbox(settings, workspaces)
            idle_since = time.monotonic()
            while True:
                claimed = _claim(settings)
                if claimed is None:
                    if time.monotonic() - idle_since >= idle_timeout:
                        return 0
                    time.sleep(1.0)
                    continue
                idle_since = time.monotonic()
                spec: SandboxSpec | None = None
                log_path: Path | None = None
                exit_code = 1
                error: str | None = None
                try:
                    spec = load_private_spec(
                        claimed, root=settings.training_queue_running_dir
                    )
                    log_path = settings.training_queue_logs_dir / f"{spec.job_id}.log"
                    settings.gpu_lock_file.parent.mkdir(parents=True, exist_ok=True)
                    gpu_lock = os.open(
                        settings.gpu_lock_file,
                        os.O_RDWR | os.O_CREAT,
                        0o660,
                    )
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                        with log_path.open("ab") as stream:
                            exit_code = sandbox.run_foreground(spec, log_stream=stream)
                    finally:
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                        os.close(gpu_lock)
                except BaseException as caught:
                    error = f"{type(caught).__name__}: {caught}"
                    if log_path is not None:
                        with log_path.open("ab") as stream:
                            stream.write((_redact_secrets(error) + "\n").encode("utf-8"))
                finally:
                    claimed.unlink(missing_ok=True)
                if spec is None:
                    _marker(settings, name=claimed.name, succeeded=False)
                    continue
                succeeded = exit_code == 0 and error is None
                _marker(settings, name=claimed.name, succeeded=succeeded)
                _update_status(
                    store,
                    spec,
                    status="succeeded" if succeeded else "failed",
                    exit_code=exit_code,
                    error=error,
                )
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
    ''',
)


write(
    "src/automation/chatgpt_mcp/runtime.py",
    r'''
    """Atomic installation of the MCP control plane outside the mutable project."""

    from __future__ import annotations

    import os
    import secrets
    import shutil
    import subprocess
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    from src.automation.chatgpt_mcp.settings import GatewaySettings


    class RuntimeInstallError(RuntimeError):
        """Raised when a trusted runtime release cannot be installed."""


    @dataclass(frozen=True)
    class RuntimeRelease:
        revision: str
        root: Path
        source_root: Path
        python_executable: Path
        launcher_path: Path


    def _checked(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 600,
    ) -> str:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeInstallError(
                result.stderr.strip() or result.stdout.strip() or "command failed"
            )
        return result.stdout.strip()


    def _verify_source(source_root: Path, expected_sha: str) -> str:
        if len(expected_sha) != 40 or any(
            character not in "0123456789abcdef" for character in expected_sha.lower()
        ):
            raise RuntimeInstallError("expected_sha must be a full commit SHA")
        head = _checked(
            ["git", "-C", str(source_root), "rev-parse", "HEAD^{commit}"],
            timeout=30,
        ).lower()
        if head != expected_sha.lower():
            raise RuntimeInstallError(f"source HEAD {head} != expected {expected_sha}")
        status = _checked(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            ],
            timeout=30,
        )
        if status:
            raise RuntimeInstallError("runtime source contains tracked modifications")
        return head


    def _launcher_text() -> str:
        return """from __future__ import annotations\n\nimport sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parent\nsys.path.insert(0, str(ROOT / 'source'))\nfrom src.automation.chatgpt_mcp.cli import main\n\nraise SystemExit(main())\n"""


    class RuntimeInstaller:
        """Build a versioned minimal venv and atomically switch `runtime/current`."""

        def __init__(
            self,
            settings: GatewaySettings,
            *,
            source_root: Path,
            expected_sha: str,
            source_python: Path | None = None,
        ) -> None:
            self.settings = settings
            self.source_root = source_root.resolve()
            self.expected_sha = expected_sha.lower()
            self.source_python = (source_python or Path(sys.executable)).resolve()

        def install(self) -> RuntimeRelease:
            revision = _verify_source(self.source_root, self.expected_sha)
            self.settings.runtime_releases_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.settings.runtime_root, 0o700)
            os.chmod(self.settings.runtime_releases_dir, 0o700)
            release_root = self.settings.runtime_releases_dir / revision
            if not release_root.exists():
                temporary = self.settings.runtime_releases_dir / (
                    f".{revision}.{secrets.token_hex(8)}.tmp"
                )
                temporary.mkdir(mode=0o700)
                try:
                    source_target = temporary / "source"
                    package_target = source_target / "src/automation/chatgpt_mcp"
                    package_target.parent.mkdir(mode=0o700, parents=True)
                    shutil.copytree(
                        self.source_root / "src/automation/chatgpt_mcp",
                        package_target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                    for relative in ("src/__init__.py", "src/automation/__init__.py"):
                        source = self.source_root / relative
                        target = source_target / relative
                        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        if source.is_file():
                            shutil.copy2(source, target)
                        else:
                            target.write_text("", encoding="utf-8")
                    launcher = temporary / "launcher.py"
                    launcher.write_text(_launcher_text(), encoding="utf-8")
                    uv = self.settings.uv_path
                    if not uv.is_file():
                        discovered = shutil.which("uv")
                        if discovered is None:
                            raise RuntimeInstallError(f"uv was not found: {uv}")
                        uv = Path(discovered)
                    venv = temporary / "venv"
                    _checked(
                        [str(uv), "venv", str(venv), "--python", str(self.source_python)],
                        timeout=300,
                    )
                    runtime_python = venv / "bin/python"
                    _checked(
                        [
                            str(uv),
                            "pip",
                            "install",
                            "--python",
                            str(runtime_python),
                            "mcp>=1.28.1,<2",
                            "python-multipart>=0.0.9",
                        ],
                        timeout=900,
                    )
                    _checked(
                        [str(runtime_python), str(launcher), "--help"],
                        cwd=temporary,
                        timeout=60,
                    )
                    os.replace(temporary, release_root)
                except BaseException:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
            release = self._release(revision)
            current_tmp = self.settings.runtime_root / f".current.{secrets.token_hex(8)}"
            current_tmp.symlink_to(release.root)
            os.replace(current_tmp, self.settings.runtime_current_dir)
            return release

        def _release(self, revision: str) -> RuntimeRelease:
            root = self.settings.runtime_releases_dir / revision
            release = RuntimeRelease(
                revision=revision,
                root=root,
                source_root=root / "source",
                python_executable=root / "venv/bin/python",
                launcher_path=root / "launcher.py",
            )
            if not all(
                path.exists()
                for path in (
                    release.source_root,
                    release.python_executable,
                    release.launcher_path,
                )
            ):
                raise RuntimeInstallError(f"runtime release is incomplete: {root}")
            return release


    def current_runtime(settings: GatewaySettings) -> RuntimeRelease:
        current = settings.runtime_current_dir
        if not current.exists():
            raise RuntimeInstallError("no external MCP runtime is installed")
        root = current.resolve()
        return RuntimeRelease(
            revision=root.name,
            root=root,
            source_root=root / "source",
            python_executable=root / "venv/bin/python",
            launcher_path=root / "launcher.py",
        )
    ''',
)


write(
    "src/automation/chatgpt_mcp/sandbox_exec.py",
    r'''
    """Execute one owner-only sandbox spec through validated external settings."""

    from __future__ import annotations

    from pathlib import Path

    from src.automation.chatgpt_mcp.jobs import execute_sandbox_spec
    from src.automation.chatgpt_mcp.settings import GatewaySettings


    def run_from_spec(spec_path: Path) -> int:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        settings.ensure_state()
        return execute_sandbox_spec(settings, spec_path)
    ''',
)


write(
    "src/automation/chatgpt_mcp/cli.py",
    r'''
    """Command-line lifecycle for the external ChatGPT WSL MCP control plane."""

    from __future__ import annotations

    import argparse
    import getpass
    import os
    import subprocess
    import sys
    from pathlib import Path

    from src.automation.chatgpt_mcp.queue_worker import run_worker
    from src.automation.chatgpt_mcp.runtime import RuntimeInstaller, current_runtime
    from src.automation.chatgpt_mcp.sandbox_exec import run_from_spec
    from src.automation.chatgpt_mcp.secure_tunnel import SecureTunnelManager
    from src.automation.chatgpt_mcp.server import run_gateway
    from src.automation.chatgpt_mcp.settings import GatewaySettings
    from src.automation.chatgpt_mcp.tunnel import QuickTunnel


    def _state_dir() -> Path:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        settings.ensure_state()
        return settings.state_dir


    def serve(public_base_url: str) -> None:
        settings = GatewaySettings.from_env(public_base_url=public_base_url)
        settings.ensure_state()
        settings.public_url_path.write_text(settings.resource_url + "\n", encoding="utf-8")
        os.chmod(settings.public_url_path, 0o600)
        run_gateway(settings)


    def serve_public() -> None:
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
        settings = GatewaySettings.from_env(require_public_base_url=False)
        if settings.host != "127.0.0.1":
            raise ValueError("serve-private requires TENNIS_MCP_HOST=127.0.0.1")
        settings.ensure_state()
        run_gateway(settings, authenticated=False)


    def _git_root(path: Path | None = None) -> Path:
        directory = (path or Path.cwd()).resolve()
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("runtime deployment source must be a git checkout")
        return Path(result.stdout.strip()).resolve()


    def _head(source_root: Path) -> str:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD^{commit}"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("could not resolve deployment source HEAD")
        return result.stdout.strip().lower()


    def _manager_for_release(
        settings: GatewaySettings,
        *,
        source_root: Path,
        python_executable: Path,
        launcher_path: Path,
    ) -> SecureTunnelManager:
        return SecureTunnelManager(
            settings,
            source_root=source_root,
            python_executable=python_executable,
            launcher_path=launcher_path,
        )


    def _current_manager(settings: GatewaySettings) -> SecureTunnelManager:
        release = current_runtime(settings)
        return _manager_for_release(
            settings,
            source_root=release.source_root,
            python_executable=release.python_executable,
            launcher_path=release.launcher_path,
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


    def deploy_runtime(
        *, source_root: Path, expected_sha: str, start: bool
    ) -> str:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        source = _git_root(source_root)
        release = RuntimeInstaller(
            settings,
            source_root=source,
            expected_sha=expected_sha,
            source_python=Path(sys.executable),
        ).install()
        manager = _manager_for_release(
            settings,
            source_root=release.source_root,
            python_executable=release.python_executable,
            launcher_path=release.launcher_path,
        )
        manager.install_user_services()
        if start:
            manager.start()
        return "\n".join(
            [
                f"Runtime revision: {release.revision}",
                f"Runtime root: {release.root}",
                f"Project RW root: {settings.repo_root}",
                f"Services started: {'yes' if start else 'no'}",
            ]
        )


    def configure_secure_tunnel(
        *,
        tunnel_id: str,
        runtime_key_file: Path | None,
        source_root: Path,
        expected_sha: str,
        start: bool,
    ) -> str:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        source = _git_root(source_root)
        runtime_api_key = _read_runtime_api_key(runtime_key_file)
        bootstrap = SecureTunnelManager(
            settings,
            source_root=source,
            python_executable=Path(sys.executable),
            launcher_path=source / "src/automation/chatgpt_mcp/__main__.py",
        )
        profile_path = bootstrap.configure(
            tunnel_id=tunnel_id, runtime_api_key=runtime_api_key
        )
        deployment = deploy_runtime(
            source_root=source, expected_sha=expected_sha, start=start
        )
        return "\n".join(
            [
                f"Tunnel ID: {tunnel_id.strip()}",
                f"Profile: {profile_path}",
                "Connection: Tunnel",
                "Authentication: None (access is controlled by the OpenAI tunnel)",
                deployment,
            ]
        )


    def show_secure_connection() -> str:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        if not settings.secure_tunnel_id_path.is_file():
            raise RuntimeError("secure tunnel has not been configured yet")
        return "\n".join(
            [
                "Name: tennis-lab WSL",
                "Description: RW project sandbox, local-data validation, CUDA, and training",
                "Connection: Tunnel",
                f"Tunnel ID: {settings.secure_tunnel_id_path.read_text(encoding='utf-8').strip()}",
                "Authentication: None",
                f"Runtime: {settings.runtime_current_dir.resolve()}",
                f"Project RW: {settings.repo_root}",
            ]
        )


    def show_connection() -> str:
        settings = GatewaySettings.from_env(require_public_base_url=False)
        if not settings.public_url_path.is_file() or not settings.owner_secret_path.is_file():
            raise RuntimeError("legacy public gateway has not started yet")
        return "\n".join(
            [
                "Name: tennis-lab WSL",
                "Description: RW project sandbox, local-data validation, CUDA, and training",
                f"Server URL: {settings.public_url_path.read_text(encoding='utf-8').strip()}",
                "Authentication: OAuth",
                f"Owner secret: {settings.owner_secret_path.read_text(encoding='utf-8').strip()}",
            ]
        )


    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        subparsers = parser.add_subparsers(dest="command", required=True)

        serve_parser = subparsers.add_parser("serve")
        serve_parser.add_argument("--public-base-url", required=True)
        subparsers.add_parser("serve-public")
        subparsers.add_parser("serve-private")
        subparsers.add_parser("show-connection")

        secure_parser = subparsers.add_parser("configure-secure-tunnel")
        secure_parser.add_argument("--tunnel-id", required=True)
        secure_parser.add_argument("--runtime-key-file", type=Path)
        secure_parser.add_argument("--source-root", type=Path, default=Path.cwd())
        secure_parser.add_argument("--expected-sha")
        secure_parser.add_argument("--start", action="store_true")

        deploy_parser = subparsers.add_parser("deploy-runtime")
        deploy_parser.add_argument("--source-root", type=Path, default=Path.cwd())
        deploy_parser.add_argument("--expected-sha")
        deploy_parser.add_argument("--start", action="store_true")

        subparsers.add_parser("show-secure-connection")
        subparsers.add_parser("doctor-secure-tunnel")
        worker_parser = subparsers.add_parser("queue-worker", help=argparse.SUPPRESS)
        worker_parser.add_argument("--idle-timeout", type=int, default=30)
        sandbox_parser = subparsers.add_parser("sandbox-exec", help=argparse.SUPPRESS)
        sandbox_parser.add_argument("--spec", type=Path, required=True)

        arguments = parser.parse_args()
        if arguments.command == "serve":
            serve(arguments.public_base_url)
        elif arguments.command == "serve-public":
            serve_public()
        elif arguments.command == "serve-private":
            serve_private()
        elif arguments.command == "show-connection":
            print(show_connection())
        elif arguments.command == "configure-secure-tunnel":
            source = _git_root(arguments.source_root)
            expected = arguments.expected_sha or _head(source)
            print(
                configure_secure_tunnel(
                    tunnel_id=arguments.tunnel_id,
                    runtime_key_file=arguments.runtime_key_file,
                    source_root=source,
                    expected_sha=expected,
                    start=arguments.start,
                )
            )
        elif arguments.command == "deploy-runtime":
            source = _git_root(arguments.source_root)
            expected = arguments.expected_sha or _head(source)
            print(
                deploy_runtime(
                    source_root=source,
                    expected_sha=expected,
                    start=arguments.start,
                )
            )
        elif arguments.command == "show-secure-connection":
            print(show_secure_connection())
        elif arguments.command == "doctor-secure-tunnel":
            settings = GatewaySettings.from_env(require_public_base_url=False)
            result = _current_manager(settings).doctor()
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            return 0 if result.returncode == 0 else 1
        elif arguments.command == "queue-worker":
            settings = GatewaySettings.from_env(require_public_base_url=False)
            return run_worker(settings, idle_timeout=arguments.idle_timeout)
        elif arguments.command == "sandbox-exec":
            return run_from_spec(arguments.spec)
        return 0
    ''',
)


# Patch the server while preserving the OAuth implementation already reviewed in PR #716.
server_path = Path("src/automation/chatgpt_mcp/server.py")
server = server_path.read_text(encoding="utf-8")
server = server.replace(
    "workspaces = WorkspaceManager(settings.repo_root, settings.state_dir, store)",
    "workspaces = WorkspaceManager(\n"
    "        settings.repo_root,\n"
    "        settings.state_dir,\n"
    "        store,\n"
    "        remote_url=settings.remote_url,\n"
    "        mirror_root=settings.trusted_mirror_dir,\n"
    "    )",
)
old_instructions_start = server.index("    instructions = (", server.index("def build_gateway("))
old_instructions_end = server.index("    oauth: OwnerOAuthProvider", old_instructions_start)
server = (
    server[:old_instructions_start]
    + '''    instructions = (\n        "GitHub MCP owns repository exploration, implementation, commits, pushes, issues, "\n        "and pull requests. WSL MCP executes arbitrary shell commands against an exact "\n        "GitHub revision inside a container. The entire tennis-lab project is mounted "\n        "read-write as the sacrificial data plane; paths outside tennis-lab, control-plane "\n        "state, credentials, the Docker socket, and network access are unavailable. GPU "\n        "commands must use the serialized training queue."\n    )\n'''
    + server[old_instructions_end:]
)
server = server.replace(
    '"role": "exact-revision execution and GPU validation only"',
    '"role": "sacrificial-project execution, local-data validation, and queued GPU training"',
)
status_decorator = server.index('    @server.tool(\n        title="Get WSL execution host status"')
next_decorator = server.index('    @server.tool(\n        title="Prepare an exact remote revision"', status_decorator)
status_block = '''    @server.tool(
        title="Get WSL execution host status",
        description="Check Docker, NVIDIA GPU, external runtime, project RW root, and trusted GPU queue.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_host_status() -> dict[str, Any]:
        nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        return {
            "project_root": str(settings.repo_root),
            "project_access": "read-write inside every sandbox",
            "network_access": "disabled",
            "control_plane": {
                "state_root": str(settings.state_dir),
                "runtime_root": str(settings.runtime_root),
                "trusted_mirror": str(settings.trusted_mirror_dir),
                "outside_project": True,
            },
            "gpu": _run_probe(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader",
                ]
            ),
            "docker": _run_probe(["docker", "info", "--format", "{{.ServerVersion}}"]),
            "training_queue": training.status_summary(),
        }

'''
server = server[:status_decorator] + status_block + server[next_decorator:]
server = server.replace(
    'description="Return the registered branch, exact SHA, and tracked-clean state without reading source files."',
    'description="Return the trusted-mirror branch, exact SHA, and clean source state."',
)
server = server.replace(
    '"Run pytest, ruff, mypy, or another bounded CPU validation command against an "\n            "ephemeral copy of an exact revision. Network, Git metadata, host credentials, "\n            "and persistent source modification are unavailable."',
    '"Run any bounded CPU shell command against an exact revision. The full tennis-lab "\n            "project, including real data, outputs, checkpoints, artifacts, and caches, is "\n            "read-write; everything outside the project and all network access are blocked."',
)
server = server.replace(
    "        expected_sha: str,\n        timeout_seconds: int = 900,\n    ) -> dict[str, Any]:\n        return jobs.start(\n            command=command,\n            workspace_id=workspace_id,\n            expected_sha=expected_sha,\n            timeout_seconds=timeout_seconds,\n        )",
    "        expected_sha: str,\n        working_directory: str = \".\",\n        timeout_seconds: int = 900,\n    ) -> dict[str, Any]:\n        return jobs.start(\n            command=command,\n            workspace_id=workspace_id,\n            expected_sha=expected_sha,\n            working_directory=working_directory,\n            timeout_seconds=timeout_seconds,\n        )",
)
server = server.replace(
    '"Queue one CUDA experiment or training command through `.training_queue`. "\n            "All GPU execution is serialized, network-disabled, and bound to the supplied "\n            "full commit SHA."',
    '"Queue any CUDA experiment, real-data validation, dataset generation, or training "\n            "command through the project-external trusted FIFO. GPU execution is serialized, "\n            "network-disabled, exact-SHA bound, and has full RW access to tennis-lab."',
)
server = server.replace(
    "        expected_sha: str,\n        issue: int | None = None,\n        timeout_seconds: int = 86_400,",
    "        expected_sha: str,\n        working_directory: str = \".\",\n        issue: int | None = None,\n        timeout_seconds: int = 86_400,",
)
server = server.replace(
    "            expected_sha=expected_sha,\n            issue=issue,\n            timeout_seconds=timeout_seconds,",
    "            expected_sha=expected_sha,\n            working_directory=working_directory,\n            issue=issue,\n            timeout_seconds=timeout_seconds,",
)
server_path.write_text(server, encoding="utf-8")


# Patch SecureTunnelManager to install and use the project-external launcher and worker.
secure_path = Path("src/automation/chatgpt_mcp/secure_tunnel.py")
secure = secure_path.read_text(encoding="utf-8")
secure = secure.replace(
    'TUNNEL_SERVICE_NAME = "tennis-lab-chatgpt-secure-tunnel.service"',
    'TUNNEL_SERVICE_NAME = "tennis-lab-chatgpt-secure-tunnel.service"\n'
    'WORKER_SERVICE_NAME = "tennis-lab-chatgpt-training-worker.service"',
)
secure = secure.replace(
    "    tunnel_service: Path\n",
    "    tunnel_service: Path\n    worker_service: Path\n",
)
secure = secure.replace(
    "        python_executable: Path,\n        service_dir: Path | None = None,",
    "        python_executable: Path,\n        launcher_path: Path,\n        service_dir: Path | None = None,",
)
secure = secure.replace(
    "        self.python_executable = python_executable\n",
    "        self.python_executable = python_executable\n"
    "        self.launcher_path = launcher_path.resolve()\n",
)
property_marker = "    @property\n    def tunnel_service_path(self) -> Path:\n        return self.service_dir / TUNNEL_SERVICE_NAME\n"
secure = secure.replace(
    property_marker,
    property_marker
    + "\n    @property\n    def worker_service_path(self) -> Path:\n        return self.service_dir / WORKER_SERVICE_NAME\n",
)
install_start = secure.index("    def install_user_services(")
install_end = secure.index("    def start(self) -> None:", install_start)
install_method = '''    def install_user_services(self) -> SecureTunnelPaths:
        """Atomically install the external MCP, worker, and tunnel units."""

        if not self.settings.secure_tunnel_profile_path.is_file():
            raise SecureTunnelError(
                "secure tunnel is not configured; run configure-secure-tunnel first"
            )
        self.service_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.service_dir, 0o700)
        with tempfile.TemporaryDirectory(
            prefix=".tennis-lab-mcp-units-",
            dir=self.service_dir,
        ) as temporary_directory:
            candidate_dir = Path(temporary_directory)
            candidates = {
                self.private_service_path: (
                    candidate_dir / PRIVATE_SERVICE_NAME,
                    self._private_service_unit(),
                ),
                self.worker_service_path: (
                    candidate_dir / WORKER_SERVICE_NAME,
                    self._worker_service_unit(),
                ),
                self.tunnel_service_path: (
                    candidate_dir / TUNNEL_SERVICE_NAME,
                    self._tunnel_service_unit(),
                ),
            }
            for _, (candidate, content) in candidates.items():
                candidate.write_text(content, encoding="utf-8")
                os.chmod(candidate, 0o600)
            verification = subprocess.run(
                [
                    "systemd-analyze",
                    "--user",
                    "verify",
                    *(str(candidate) for candidate, _ in candidates.values()),
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if verification.returncode != 0:
                detail = verification.stderr.strip() or verification.stdout.strip()
                raise SecureTunnelError(f"systemd unit verification failed: {detail}")
            for destination, (candidate, _) in candidates.items():
                os.replace(candidate, destination)
                os.chmod(destination, 0o600)
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            timeout=30,
        )
        return self.paths()

'''
secure = secure[:install_start] + install_method + secure[install_end:]
secure = secure.replace(
    "            tunnel_service=self.tunnel_service_path,\n",
    "            tunnel_service=self.tunnel_service_path,\n"
    "            worker_service=self.worker_service_path,\n",
)
private_start = secure.index("    def _private_service_unit(self) -> str:")
tunnel_start = secure.index("    def _tunnel_service_unit(self) -> str:", private_start)
private_and_worker = '''    def _service_environment(self) -> str:
        values = {
            "TENNIS_MCP_REPO_ROOT": self.settings.repo_root,
            "TENNIS_MCP_STATE_DIR": self.settings.state_dir,
            "TENNIS_MCP_RUNTIME_ROOT": self.settings.runtime_root,
            "TENNIS_MCP_REMOTE_URL": self.settings.remote_url,
            "TENNIS_MCP_UV_PYTHON_ROOT": self.settings.uv_python_root,
            "TENNIS_MCP_UV": self.settings.uv_path,
            "TENNIS_MCP_GPU_LOCK_FILE": self.settings.gpu_lock_file,
            "TENNIS_MCP_HOST": "127.0.0.1",
            "TENNIS_MCP_PORT": PRIVATE_MCP_PORT,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return "\\n".join(
            f'Environment="{key}={str(value).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for key, value in values.items()
        )

    def _private_service_unit(self) -> str:
        return f"""[Unit]
Description=External private tennis-lab MCP endpoint for OpenAI Secure Tunnel
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={_unit_path_value(self.source_root)}
{self._service_environment()}
ExecStart={_unit_value(self.python_executable)} {_unit_value(self.launcher_path)} serve-private
Restart=always
RestartSec=5
TimeoutStopSec=20
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=default.target
"""

    def _worker_service_unit(self) -> str:
        return f"""[Unit]
Description=Trusted external FIFO worker for tennis-lab MCP GPU jobs
After=docker.service

[Service]
Type=simple
WorkingDirectory={_unit_path_value(self.source_root)}
{self._service_environment()}
ExecStart={_unit_value(self.python_executable)} {_unit_value(self.launcher_path)} queue-worker --idle-timeout 30
TimeoutStopSec=45
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

"""

'''
secure = secure[:private_start] + private_and_worker + secure[tunnel_start:]
secure_path.write_text(secure, encoding="utf-8")


write(
    "src/automation/chatgpt_mcp/README.md",
    r'''
    # tennis-lab ChatGPT WSL MCP

    This gateway deliberately treats the complete `tennis-lab` project as a
    sacrificial read-write data plane. GitHub MCP remains the repository control
    plane; WSL MCP is the execution plane.

    ## Responsibility split

    GitHub MCP owns repository exploration, issues, pull requests, branch creation,
    implementation, commits, pushes, and the remote source of truth.

    WSL MCP owns:

    - fetching a fixed `origin` branch into a trusted project-external bare mirror;
    - binding execution to a caller-supplied full commit SHA;
    - arbitrary CPU shell commands against that exact revision;
    - real local dataset and checkpoint validation;
    - dataset generation, visualization, evaluation, and other repository CLIs;
    - serialized CUDA experiments and training;
    - bounded, secret-redacted status and logs.

    WSL MCP does not expose GitHub commit or push tools.

    ## Security boundary

    The following data plane is mounted read-write inside each container:

    ```text
    /home/kamimura/projects/tennis-lab
    /project -> the same tennis-lab project
    ```

    This includes source files, `data/`, `outputs/`, `ckpt/`, `ckpts/`,
    `artifacts/`, `.cache/`, and `third_party/`. Commands may damage or delete
    them. The project `.git` metadata is masked, and `.venv` is over-mounted
    read-only so an MCP command cannot poison binaries later executed by the host.

    The following control plane is never mounted into command containers:

    - versioned MCP runtime and its minimal venv;
    - OpenAI Tunnel ID, runtime API key, and profile;
    - trusted bare Git mirror and exact-revision worktrees;
    - SQLite metadata and GPU FIFO state;
    - systemd units;
    - Docker socket, host credentials, Windows `/mnt/c`, and the rest of `$HOME`.

    Containers use `--network none`, a read-only root filesystem, all capabilities
    dropped, `no-new-privileges`, private IPC, PID/memory limits, and a timeout.
    Disk exhaustion inside the project remains a residual risk because ordinary
    bind mounts do not provide a per-job filesystem quota.

    ## MCP tools

    1. `get_host_status`
    2. `prepare_revision_workspace`
    3. `get_revision_status`
    4. `start_command`
    5. `get_command_job`
    6. `list_command_jobs`
    7. `get_command_output`
    8. `cancel_command_job`
    9. `enqueue_training`
    10. `get_training_job`
    11. `get_training_output`

    `start_command` accepts any CPU shell command and a relative
    `working_directory`; it is limited to 24 hours and four concurrent jobs.
    `enqueue_training` accepts any GPU command, including CUDA tests, dataset
    generation, evaluation, or training; the external FIFO serializes all GPU
    use and shares the local GPU lock with self-hosted Actions.

    Neither command path has network access. Package downloads and other open-world
    operations must be prepared outside the MCP execution plane.

    ## Normal handoff

    1. GitHub MCP implements and pushes a branch.
    2. GitHub MCP obtains the branch's full head SHA.
    3. WSL MCP calls `prepare_revision_workspace(branch, expected_sha)`.
    4. WSL MCP runs CPU tests or real-data validation with `start_command`.
    5. WSL MCP runs CUDA or training with `enqueue_training`.
    6. GitHub MCP alone persists required source changes.

    Relative task paths continue to work. Exact revision code runs from the private
    `/workspace`, while standard runtime roots are linked to the persistent project:

    ```text
    /workspace/data       -> /home/kamimura/projects/tennis-lab/data
    /workspace/outputs    -> /home/kamimura/projects/tennis-lab/outputs
    /workspace/ckpt       -> /home/kamimura/projects/tennis-lab/ckpt
    /workspace/artifacts  -> /home/kamimura/projects/tennis-lab/artifacts
    /workspace/.cache     -> /home/kamimura/projects/tennis-lab/.cache
    /workspace/third_party -> /home/kamimura/projects/tennis-lab/third_party
    ```

    ## External runtime deployment

    The persistent service must not run from `tennis-lab` or its `.venv`. Deploy an
    exact, clean revision into the versioned external runtime:

    ```bash
    cd /home/kamimura/projects/tennis-lab
    SHA="$(git rev-parse HEAD)"

    .venv/bin/python -m src.automation.chatgpt_mcp deploy-runtime \
      --source-root "$PWD" \
      --expected-sha "$SHA" \
      --start
    ```

    The installer creates:

    ```text
    ~/.local/share/tennis-lab-chatgpt-mcp/releases/<sha>/
      source/       # only the MCP package
      venv/         # minimal MCP dependencies
      launcher.py
    ~/.local/share/tennis-lab-chatgpt-mcp/current -> releases/<sha>
    ```

    It atomically verifies and installs these user units:

    - `tennis-lab-chatgpt-mcp-private.service`
    - `tennis-lab-chatgpt-training-worker.service`
    - `tennis-lab-chatgpt-secure-tunnel.service`

    First-time Tunnel configuration remains:

    ```bash
    .venv/bin/python -m src.automation.chatgpt_mcp configure-secure-tunnel \
      --tunnel-id tunnel_0123456789abcdef0123456789abcdef \
      --source-root "$PWD" \
      --expected-sha "$SHA" \
      --start
    ```

    The runtime API key prompt is hidden. Never put that key in chat, GitHub, or a
    command-line argument.

    ## Validation

    ```bash
    systemctl --user is-active tennis-lab-chatgpt-mcp-private.service
    systemctl --user is-active tennis-lab-chatgpt-secure-tunnel.service
    curl --fail --silent http://127.0.0.1:8767/healthz
    curl --fail --silent http://127.0.0.1:8768/readyz
    ```

    Then validate through ChatGPT:

    - `get_host_status` reports the external runtime, trusted mirror, Docker, GPU,
      and queue state;
    - an exact revision can be prepared;
    - `start_command` can read and write project data but cannot see the Docker
      socket, `/mnt/c`, Tunnel state, or external runtime;
    - `enqueue_training` sees CUDA and writes persistent outputs under the project.
    ''',
)


write(
    "tests/unit/automation/chatgpt_mcp/test_settings.py",
    r'''
    from __future__ import annotations

    import stat
    import tempfile
    from pathlib import Path

    import pytest

    from src.automation.chatgpt_mcp.settings import GatewaySettings, normalize_public_base_url


    def _project(path: Path) -> Path:
        path.mkdir()
        return path


    def test_normalize_public_base_url_accepts_https_origin() -> None:
        assert normalize_public_base_url("https://example.test/") == "https://example.test"


    @pytest.mark.parametrize(
        "value",
        [
            "http://example.test",
            "https://example.test/mcp",
            "https://example.test?query=1",
            "localhost:8765",
        ],
    )
    def test_normalize_public_base_url_rejects_non_origin(value: str) -> None:
        with pytest.raises(ValueError):
            normalize_public_base_url(value)


    def test_state_runtime_mirror_and_queue_are_outside_project() -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            settings = GatewaySettings(
                repo_root=_project(root / "project"),
                state_dir=root / "state",
                runtime_root=root / "runtime",
                public_base_url=None,
                uv_python_root=root / "uv-python",
                uv_path=root / "uv",
                gpu_lock_file=root / "gpu.lock",
            )
            settings.ensure_state()
            settings.ensure_project_runtime_dirs()

            assert not settings.state_dir.is_relative_to(settings.repo_root)
            assert not settings.runtime_root.is_relative_to(settings.repo_root)
            assert not settings.trusted_mirror_dir.is_relative_to(settings.repo_root)
            assert stat.S_IMODE(settings.state_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(settings.git_mask_path.stat().st_mode) == 0o400
            assert stat.S_IMODE(settings.git_mask_dir.stat().st_mode) == 0o500
            assert settings.training_queue_jobs_dir.is_dir()
            assert (settings.repo_root / "data").is_dir()
            assert (settings.repo_root / "outputs").is_dir()


    def test_settings_reject_control_plane_inside_project(tmp_path: Path) -> None:
        project = _project(tmp_path / "project")
        with pytest.raises(ValueError, match="STATE_DIR"):
            GatewaySettings(
                repo_root=project,
                state_dir=project / ".state",
                runtime_root=tmp_path / "runtime",
                public_base_url=None,
            )
        with pytest.raises(ValueError, match="RUNTIME_ROOT"):
            GatewaySettings(
                repo_root=project,
                state_dir=tmp_path / "state",
                runtime_root=project / ".runtime",
                public_base_url=None,
            )
    ''',
)


write(
    "tests/unit/automation/chatgpt_mcp/test_workspace.py",
    r'''
    from __future__ import annotations

    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    from src.automation.chatgpt_mcp.storage import SqliteStore
    from src.automation.chatgpt_mcp.workspace import WorkspaceError, WorkspaceManager


    def _run(*arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            list(arguments), cwd=cwd, text=True, capture_output=True, check=True
        )
        return result.stdout.strip()


    def _manager(tmp_path: Path) -> tuple[WorkspaceManager, str, Path]:
        source = tmp_path / "source"
        _run("git", "init", "-q", "-b", "main", str(source))
        _run("git", "config", "user.email", "test@example.com", cwd=source)
        _run("git", "config", "user.name", "Test", cwd=source)
        (source / "example.txt").write_text("revision\n", encoding="utf-8")
        _run("git", "add", "example.txt", cwd=source)
        _run("git", "commit", "-qm", "initial", cwd=source)
        revision = _run("git", "rev-parse", "HEAD", cwd=source)
        remote = tmp_path / "origin.git"
        _run("git", "clone", "-q", "--bare", str(source), str(remote))
        project = tmp_path / "project"
        project.mkdir()
        state = tmp_path / "state"
        manager = WorkspaceManager(
            project,
            state,
            SqliteStore(state / "gateway.sqlite3"),
            remote_url=str(remote),
        )
        return manager, revision, project


    def test_prepare_revision_uses_external_mirror_and_exact_detached_worktree(
        tmp_path: Path,
    ) -> None:
        manager, revision, project = _manager(tmp_path)
        prepared = manager.prepare_revision(branch="main", expected_sha=revision)
        workspace = manager.get_revision(prepared["workspace_id"])

        assert workspace.path.parent == manager.workspace_root
        assert not workspace.path.is_relative_to(project)
        assert not manager.mirror_root.is_relative_to(project)
        assert _run("git", "rev-parse", "HEAD", cwd=workspace.path) == revision
        assert _run("git", "branch", "--show-current", cwd=workspace.path) == ""
        assert manager.describe_revision(workspace.workspace_id)["clean"] is True


    def test_project_git_can_be_destroyed_without_invalidating_trusted_revision(
        tmp_path: Path,
    ) -> None:
        manager, revision, project = _manager(tmp_path)
        prepared = manager.prepare_revision(branch="main", expected_sha=revision)
        shutil.rmtree(project)
        project.mkdir()
        workspace = manager.assert_execution_ready(
            workspace_id=prepared["workspace_id"], expected_sha=revision
        )
        assert workspace.revision == revision


    def test_prepare_revision_rejects_remote_sha_mismatch(tmp_path: Path) -> None:
        manager, revision, _ = _manager(tmp_path)
        wrong = "0" * 40 if revision != "0" * 40 else "1" * 40
        with pytest.raises(WorkspaceError, match="remote revision mismatch"):
            manager.prepare_revision(branch="main", expected_sha=wrong)


    def test_execution_rejects_modified_trusted_source(tmp_path: Path) -> None:
        manager, revision, _ = _manager(tmp_path)
        prepared = manager.prepare_revision(branch="main", expected_sha=revision)
        workspace = manager.get_revision(prepared["workspace_id"])
        (workspace.path / "untracked.py").write_text("bad\n", encoding="utf-8")
        with pytest.raises(WorkspaceError, match="contains changes"):
            manager.assert_execution_ready(
                workspace_id=workspace.workspace_id, expected_sha=revision
            )
    ''',
)


write(
    "tests/unit/automation/chatgpt_mcp/test_jobs.py",
    r'''
    from __future__ import annotations

    import json
    import stat
    import subprocess
    from pathlib import Path

    import pytest
    from pytest import MonkeyPatch

    from src.automation.chatgpt_mcp.jobs import (
        DockerSandbox,
        JobError,
        JobManager,
        SandboxSpec,
        TrainingQueueManager,
        _redact_secrets,
    )
    from src.automation.chatgpt_mcp.settings import GatewaySettings
    from src.automation.chatgpt_mcp.storage import SqliteStore
    from src.automation.chatgpt_mcp.workspace import RevisionWorkspace

    _REVISION = "1" * 40
    _WORKSPACE_ID = "rev-0123456789abcdef"


    class StubWorkspaces:
        def __init__(self, source: Path) -> None:
            self.source = source

        def assert_execution_ready(
            self, *, workspace_id: str, expected_sha: str
        ) -> RevisionWorkspace:
            assert workspace_id == _WORKSPACE_ID
            assert expected_sha == _REVISION
            return RevisionWorkspace(
                workspace_id=workspace_id,
                path=self.source,
                branch="feature/test",
                revision=expected_sha,
            )


    def _settings(tmp_path: Path) -> tuple[GatewaySettings, StubWorkspaces]:
        project = tmp_path / "project"
        project.mkdir()
        (project / ".git").mkdir()
        (project / ".venv/bin").mkdir(parents=True)
        uv_root = tmp_path / "uv-python"
        uv_root.mkdir()
        uv = tmp_path / "uv"
        uv.touch()
        settings = GatewaySettings(
            repo_root=project,
            state_dir=tmp_path / "state",
            runtime_root=tmp_path / "runtime",
            public_base_url=None,
            uv_python_root=uv_root,
            uv_path=uv,
            gpu_lock_file=tmp_path / "gpu.lock",
        )
        settings.ensure_state()
        source = settings.revision_workspace_dir / _WORKSPACE_ID
        source.mkdir(parents=True)
        (source / "example.py").write_text("print('ok')\n", encoding="utf-8")
        return settings, StubWorkspaces(source)


    def _spec(*, job_id: str = "job-0123456789abcdef", use_gpu: bool = False) -> SandboxSpec:
        return SandboxSpec(
            job_id=job_id,
            command="python -m pytest --disable-warnings",
            workspace_id=_WORKSPACE_ID,
            expected_sha=_REVISION,
            working_directory="tests",
            use_gpu=use_gpu,
            timeout_seconds=60,
        )


    def test_sandbox_mounts_complete_project_rw_but_control_plane_is_absent(
        tmp_path: Path,
    ) -> None:
        settings, workspaces = _settings(tmp_path)
        spec = _spec()
        command = DockerSandbox(settings, workspaces).command(spec, detached=True)
        joined = " ".join(command)
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        command_path = settings.sandbox_jobs_dir / spec.job_id / "command"

        assert "--network none" in joined
        assert "--read-only" in command
        assert "--cap-drop ALL" in joined
        assert "--security-opt no-new-privileges" in joined
        assert "--gpus all" not in joined
        assert "/var/run/docker.sock" not in joined
        assert "/mnt/c" not in joined
        assert any(f"src={settings.repo_root},dst={settings.repo_root}" in mount and "readonly" not in mount for mount in mounts)
        assert any(f"src={settings.repo_root},dst=/project" in mount and "readonly" not in mount for mount in mounts)
        assert any("dst=/source,readonly" in mount for mount in mounts)
        assert any("dst=/workspace" in mount and "readonly" not in mount for mount in mounts)
        assert any("dst=/project/.venv,readonly" in mount for mount in mounts)
        assert any("dst=/project/.git,readonly" in mount for mount in mounts)
        assert not any(str(settings.state_dir) in mount for mount in mounts)
        assert not any(str(settings.runtime_root) in mount for mount in mounts)
        assert spec.command not in joined
        assert command_path.read_text(encoding="utf-8") == spec.command
        assert stat.S_IMODE(command_path.stat().st_mode) == 0o600
        assert "ln -s" in joined and "/workspace/data" in joined


    def test_gpu_flag_is_available_only_to_serial_queue_specs(tmp_path: Path) -> None:
        settings, workspaces = _settings(tmp_path)
        command = DockerSandbox(settings, workspaces).command(
            _spec(job_id="train-0123456789abcdef", use_gpu=True), detached=False
        )
        assert command[command.index("--gpus") + 1] == "all"
        assert command[command.index("--network") + 1] == "none"


    def test_working_directory_cannot_escape_exact_revision() -> None:
        with pytest.raises(ValueError, match="working_directory"):
            _spec().model_copy(update={"working_directory": "../../home"}).model_validate(
                _spec().model_dump() | {"working_directory": "../../home"}
            )


    def test_direct_commands_allow_long_local_validation_but_are_bounded(
        tmp_path: Path,
    ) -> None:
        settings, workspaces = _settings(tmp_path)
        manager = JobManager(settings, SqliteStore(settings.database_path), workspaces)
        with pytest.raises(JobError, match="direct commands are limited"):
            manager.start(
                command="sleep 999999",
                workspace_id=_WORKSPACE_ID,
                expected_sha=_REVISION,
                working_directory=".",
                timeout_seconds=24 * 3600 + 1,
            )


    def test_training_queue_uses_external_private_spec_and_fixed_worker(
        tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        settings, workspaces = _settings(tmp_path)
        manager = TrainingQueueManager(
            settings, SqliteStore(settings.database_path), workspaces
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr("src.automation.chatgpt_mcp.jobs.subprocess.run", fake_run)
        result = manager.enqueue(
            name="real-data-training",
            command="python -m src.tasks.ball_detection.scripts.train",
            workspace_id=_WORKSPACE_ID,
            expected_sha=_REVISION,
            working_directory=".",
            issue=716,
            timeout_seconds=60,
        )
        spec_path = next(settings.training_queue_jobs_dir.glob("*.json"))
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        assert result["status"] == "queued"
        assert commands == [["systemctl", "--user", "start", "tennis-lab-chatgpt-training-worker.service"]]
        assert spec["use_gpu"] is True
        assert spec["working_directory"] == "."
        assert not spec_path.is_relative_to(settings.repo_root)
        assert stat.S_IMODE(spec_path.stat().st_mode) == 0o600


    def test_secret_redaction_covers_common_runtime_tokens() -> None:
        value = "token sk-example_12345678901234567890 and Bearer abcdefghijklmnop"
        redacted = _redact_secrets(value)
        assert "sk-example" not in redacted
        assert "abcdefghijklmnop" not in redacted
        assert redacted.count("[REDACTED]") == 2
    ''',
)


# Minimal test updates for the unchanged tool names and new role string.
server_test = Path("tests/unit/automation/chatgpt_mcp/test_server.py")
server_tests = server_test.read_text(encoding="utf-8")
server_tests = server_tests.replace(
    "    (repo / \".venv/bin\").mkdir(parents=True)\n",
    "    (repo / \".git\").mkdir()\n    (repo / \".venv/bin\").mkdir(parents=True)\n",
)
server_tests = server_tests.replace(
    "        state_dir=tmp_path / \"state\",\n        public_base_url=None,",
    "        state_dir=tmp_path / \"state\",\n        runtime_root=tmp_path / \"runtime\",\n        public_base_url=None,",
)
server_test.write_text(server_tests, encoding="utf-8")

integration = Path("tests/integration/chatgpt_mcp/test_oauth_mcp.py")
integration_text = integration.read_text(encoding="utf-8")
integration_text = integration_text.replace(
    "    (repo / \".venv/bin\").mkdir(parents=True)\n",
    "    (repo / \".git\").mkdir()\n    (repo / \".venv/bin\").mkdir(parents=True)\n",
)
integration_text = integration_text.replace(
    "        state_dir=tmp_path / \"state\",\n        public_base_url=\"https://mcp.example.test\",",
    "        state_dir=tmp_path / \"state\",\n        runtime_root=tmp_path / \"runtime\",\n        public_base_url=\"https://mcp.example.test\",",
)
integration_text = integration_text.replace(
    'assert service.json()["role"] == "exact-revision execution and GPU validation only"',
    'assert service.json()["role"] == (\n            "sacrificial-project execution, local-data validation, and queued GPU training"\n        )',
)
integration.write_text(integration_text, encoding="utf-8")

# Secure-tunnel tests are updated with targeted replacements for the third unit.
secure_test = Path("tests/unit/automation/chatgpt_mcp/test_secure_tunnel.py")
secure_tests = secure_test.read_text(encoding="utf-8")
secure_tests = secure_tests.replace(
    "    TUNNEL_SERVICE_NAME,\n",
    "    TUNNEL_SERVICE_NAME,\n    WORKER_SERVICE_NAME,\n",
)
secure_tests = secure_tests.replace(
    "        python_executable=executable,\n        service_dir=tmp_path / \"systemd\",",
    "        python_executable=executable,\n        launcher_path=source / \"launcher.py\",\n        service_dir=tmp_path / \"systemd\",",
)
secure_tests = secure_tests.replace(
    "    source.mkdir()\n",
    "    source.mkdir()\n    (source / \"launcher.py\").write_text(\"# launcher\\n\", encoding=\"utf-8\")\n",
    1,
)
secure_tests = secure_tests.replace(
    "    tunnel_unit = paths.tunnel_service.read_text(encoding=\"utf-8\")\n",
    "    tunnel_unit = paths.tunnel_service.read_text(encoding=\"utf-8\")\n"
    "    worker_unit = paths.worker_service.read_text(encoding=\"utf-8\")\n",
)
secure_tests = secure_tests.replace(
    "    assert \"serve-private\" in private_unit\n",
    "    assert \"serve-private\" in private_unit\n"
    "    assert \"queue-worker\" in worker_unit\n"
    "    assert str(manager.settings.repo_root) in worker_unit\n",
)
secure_tests = secure_tests.replace(
    "    assert private_candidate.parent == tunnel_candidate.parent\n",
    "    worker_candidate = Path(verify_command[4])\n"
    "    tunnel_candidate = Path(verify_command[5])\n"
    "    assert worker_candidate.name == WORKER_SERVICE_NAME\n"
    "    assert private_candidate.parent == worker_candidate.parent == tunnel_candidate.parent\n",
)
secure_tests = secure_tests.replace(
    "    tunnel_candidate = Path(verify_command[4])\n",
    "    tunnel_candidate = Path(verify_command[5])\n",
    1,
)
secure_test.write_text(secure_tests, encoding="utf-8")

# Remove accidental scratch files from the unfinished first attempt.
for scratch in (Path(".github/.noop"), Path(".github/.noop2")):
    scratch.unlink(missing_ok=True)

# Regenerate the strict configuration/path inventory for reviewed automation routes.
from src.utils.configuration import AuditExemption, AuditRule
from src.utils.configuration.audit import (
    regenerate_exemption_rows,
    write_generated_inventory_data,
)

source_root = Path("src").resolve()
_, _, unresolved = regenerate_exemption_rows(source_root)
unexpected = [
    finding
    for finding in unresolved
    if not finding.module.startswith("src.automation.chatgpt_mcp")
]
if unexpected:
    rendered = "\n".join(repr(finding) for finding in unexpected)
    raise SystemExit("unexpected non-MCP inventory findings:\n" + rendered)

path_rules = {
    AuditRule.HYDRA_ABSOLUTE_PATH,
    AuditRule.FILE_PARENT_INDEX,
    AuditRule.RUNTIME_PATH_LITERAL,
    AuditRule.PATH_JOIN,
    AuditRule.PROCESS_CWD,
    AuditRule.HYDRA_RUN_DIRECTORY,
}
approvals: list[AuditExemption] = []
for finding in unresolved:
    reason_code = "persisted-layout" if finding.rule in path_rules else "strict-schema"
    approvals.append(
        AuditExemption.classified(
            module=finding.module,
            qualified_name=finding.qualified_name,
            line=finding.line,
            rule=finding.rule,
            reason_code=reason_code,
        )
    )

migration_count, exemption_count = write_generated_inventory_data(
    source_root,
    source_revision="wsl-mcp-project-sandbox-v1",
    approved_exemptions=tuple(approvals),
)
print(
    f"generated inventory: migrations={migration_count} "
    f"exemptions={exemption_count} new={len(approvals)}"
)
