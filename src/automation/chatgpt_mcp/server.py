"""MCP server for project-bounded local execution and GPU training."""

from __future__ import annotations

import html
import shutil
import subprocess
import time
from collections import defaultdict, deque
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from src.automation.chatgpt_mcp.auth import OwnerOAuthProvider, oauth_scopes
from src.automation.chatgpt_mcp.jobs import JobManager, TrainingQueueManager
from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore
from src.automation.chatgpt_mcp.workspace import WorkspaceManager

_SECURITY_META = {
    "securitySchemes": [
        {"type": "oauth2", "scopes": oauth_scopes()},
    ]
}


class _ConfiguredMCPServer(MCPServer[Any]):
    """MCP 2 server retaining this gateway's fixed HTTP listener contract."""

    _http_options: dict[str, Any]

    def configure_http(
        self,
        *,
        host: str,
        port: int,
        transport_security: TransportSecuritySettings,
    ) -> None:
        self._http_options = {
            "host": host,
            "port": port,
            "streamable_http_path": "/mcp",
            "stateless_http": True,
            "json_response": True,
            "transport_security": transport_security,
        }

    def streamable_http_app(self, **kwargs: Any) -> Starlette:
        options = dict(self._http_options)
        options.pop("port")
        options.update(kwargs)
        return super().streamable_http_app(**options)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        **kwargs: Any,
    ) -> None:
        if transport == "streamable-http":
            options = dict(self._http_options)
            options.update(kwargs)
            kwargs = options
        super().run(transport=transport, **kwargs)


def _annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
    open_world: bool,
) -> ToolAnnotations:
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=open_world,
    )


class ApprovalRateLimiter:
    """Bound repeated owner-secret attempts per forwarded client address."""

    def __init__(self, *, max_attempts: int = 5, window_seconds: int = 900) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _active(self, address: str, now: float) -> deque[float]:
        failures = self._failures[address]
        threshold = now - self.window_seconds
        while failures and failures[0] <= threshold:
            failures.popleft()
        return failures

    def is_blocked(self, address: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else now
        return len(self._active(address, current)) >= self.max_attempts

    def record_failure(self, address: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else now
        self._active(address, current).append(current)

    def clear(self, address: str) -> None:
        self._failures.pop(address, None)


def _client_address(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip")
    if forwarded:
        return str(forwarded[:128])
    if request.client is None:
        return "unknown"
    return str(request.client.host)


def _approval_page(
    *,
    transaction: str,
    client_name: str,
    scopes: list[str],
    error: str | None = None,
) -> HTMLResponse:
    safe_transaction = html.escape(transaction, quote=True)
    safe_client = html.escape(client_name)
    safe_scopes = html.escape(", ".join(scopes))
    error_html = (
        f'<p class="error">{html.escape(error)}</p>' if error is not None else ""
    )
    document = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>tennis-lab MCP authorization</title>
  <style>
    body {{ font: 16px system-ui; max-width: 42rem; margin: 3rem auto; padding: 0 1rem; color: #171717; }}
    form {{ display: grid; gap: 1rem; }}
    input, button {{ font: inherit; padding: .8rem; }}
    button {{ cursor: pointer; }}
    .error {{ color: #b42318; font-weight: 600; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>ChatGPTからWSLへの接続を承認</h1>
  <p>クライアント: <strong>{safe_client}</strong></p>
  <p>権限: <code>{safe_scopes}</code></p>
  <p>この接続はtennis-lab内で任意コマンドとGPU学習を起動できます。</p>
  {error_html}
  <form method="post" action="/oauth/approve" autocomplete="off">
    <input type="hidden" name="transaction" value="{safe_transaction}">
    <label>Owner secret
      <input type="password" name="owner_secret" required autofocus>
    </label>
    <button type="submit">このChatGPT接続を承認</button>
  </form>
</body>
</html>"""
    return HTMLResponse(
        document,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; style-src 'unsafe-inline'; "
                "form-action 'self' https://chatgpt.com; base-uri 'none'; "
                "frame-ancestors 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _run_probe(arguments: list[str], *, timeout: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "output": f"{type(error).__name__}: {error}"}
    return {
        "ok": result.returncode == 0,
        "output": (result.stdout or result.stderr).strip()[:20_000],
    }


def _register_oauth_approval_routes(
    server: MCPServer[Any], oauth: OwnerOAuthProvider
) -> None:
    """Attach the browser approval flow used only by legacy public OAuth mode."""

    limiter = ApprovalRateLimiter()

    @server.custom_route("/oauth/approve", methods=["GET", "POST"])
    async def approve(request: Request) -> Response:
        address = _client_address(request)
        if limiter.is_blocked(address):
            return HTMLResponse(
                "Too many failed authorization attempts. Try again later.",
                status_code=429,
                headers={"Cache-Control": "no-store"},
            )

        if request.method == "GET":
            transaction = request.query_params.get("transaction", "")
            pending = oauth.get_pending_authorization(transaction)
            if pending is None:
                return HTMLResponse(
                    "Authorization request is missing or expired.",
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )
            return _approval_page(
                transaction=transaction,
                client_name=pending.client_name,
                scopes=pending.params.scopes or [],
            )

        form = await request.form()
        transaction = str(form.get("transaction", ""))
        owner_secret = str(form.get("owner_secret", ""))
        pending = oauth.get_pending_authorization(transaction)
        if pending is None:
            return HTMLResponse(
                "Authorization request is missing or expired.",
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )
        try:
            redirect_url = oauth.approve_authorization(transaction, owner_secret)
        except PermissionError:
            limiter.record_failure(address)
            return _approval_page(
                transaction=transaction,
                client_name=pending.client_name,
                scopes=pending.params.scopes or [],
                error="Owner secretが一致しません。",
            )
        limiter.clear(address)
        return RedirectResponse(redirect_url, status_code=303)


def build_gateway(
    settings: GatewaySettings, *, authenticated: bool = True
) -> MCPServer[Any]:
    """Build the public OAuth server or private Secure Tunnel server."""

    settings.ensure_state()
    store = SqliteStore(settings.database_path)
    workspaces = WorkspaceManager(
        settings.trusted_git_dir,
        settings.revision_workspace_dir,
        store,
    )
    jobs = JobManager(settings, store, workspaces)
    training = TrainingQueueManager(settings, store, workspaces)
    instructions = (
        "GitHub MCP exclusively owns repository exploration, source implementation, "
        "branches, commits, pushes, issues, and pull requests. This WSL MCP is the "
        "execution plane. It may fetch one origin branch at an exact SHA and run arbitrary "
        "network-disabled commands inside Docker. /workspace is a private copy of that "
        "revision. /tennis-lab exposes the entire local project read-write, including data, "
        "outputs, checkpoints, artifacts, and caches; all of it may be destroyed. MCP "
        "runtime, trusted venv, tunnel credentials, Git mirror, queue runner, systemd, "
        "Docker socket, Windows mounts, and the rest of the host remain unavailable."
    )

    oauth: OwnerOAuthProvider | None = None
    if authenticated:
        if settings.public_base_url is None:
            raise ValueError("public_base_url is required for authenticated mode")
        oauth = OwnerOAuthProvider(settings, store)
        scopes = oauth_scopes()
        auth_settings = AuthSettings(
            issuer_url=AnyHttpUrl(settings.public_base_url),
            resource_server_url=AnyHttpUrl(settings.resource_url),
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=scopes,
                default_scopes=scopes,
            ),
            required_scopes=scopes,
        )
        public_host = urlsplit(settings.public_base_url).netloc
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                public_host,
                f"127.0.0.1:{settings.port}",
                f"localhost:{settings.port}",
            ],
            allowed_origins=[settings.public_base_url],
        )
        server = _ConfiguredMCPServer(
            name="tennis-lab-wsl",
            instructions=instructions,
            auth_server_provider=oauth,
            auth=auth_settings,
        )
        server.configure_http(
            host=settings.host,
            port=settings.port,
            transport_security=transport_security,
        )
        security_meta = _SECURITY_META
    else:
        if settings.host != "127.0.0.1":
            raise ValueError("private tunnel mode must listen on 127.0.0.1")
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                f"127.0.0.1:{settings.port}",
                f"localhost:{settings.port}",
            ],
            allowed_origins=[],
        )
        server = _ConfiguredMCPServer(
            name="tennis-lab-wsl",
            instructions=instructions,
        )
        server.configure_http(
            host=settings.host,
            port=settings.port,
            transport_security=transport_security,
        )
        security_meta = {}

    @server.custom_route("/", methods=["GET"])
    async def service_document(request: Request) -> Response:
        del request
        return JSONResponse(
            {
                "service": "tennis-lab-wsl-mcp",
                "role": "arbitrary tennis-lab execution, validation, and GPU training",
                "mcp": (
                    settings.resource_url
                    if authenticated
                    else f"http://127.0.0.1:{settings.port}/mcp"
                ),
                "authentication": (
                    "OAuth 2.1 authorization code with PKCE"
                    if authenticated
                    else "OpenAI Secure MCP Tunnel"
                ),
            },
            headers={"Cache-Control": "no-store"},
        )

    @server.custom_route("/healthz", methods=["GET"])
    async def health(request: Request) -> Response:
        del request
        version = (
            settings.runtime_version_path.read_text(encoding="utf-8").strip()
            if settings.runtime_version_path.is_file()
            else "uninstalled"
        )
        return JSONResponse(
            {"status": "ok", "runtime_revision": version},
            headers={"Cache-Control": "no-store"},
        )

    if oauth is not None:
        _register_oauth_approval_routes(server, oauth)

    @server.tool(
        title="Get WSL execution host status",
        description=(
            "Check Docker, NVIDIA GPU, external trusted runtime, Git mirror, and "
            "logical two-slot training queue."
        ),
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_host_status() -> dict[str, Any]:
        nvidia_smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        if settings.trusted_queue_script.is_file():
            queue = subprocess.run(
                ["bash", str(settings.trusted_queue_script), "status"],
                cwd=settings.control_dir,
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": str(settings.runtime_home),
                    "TRAINING_QUEUE_DIR": str(settings.trusted_queue_dir),
                    "TRAINING_QUEUE_LOCK_FILE": str(settings.gpu_lock_file),
                    "TRAINING_QUEUE_PYTHON": str(
                        settings.runtime_venv_root / "bin/python"
                    ),
                },
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            queue_result = {
                "ok": queue.returncode == 0,
                "output": (queue.stdout or queue.stderr).strip()[:20_000],
            }
        else:
            queue_result = {
                "ok": False,
                "output": f"trusted queue runner missing: {settings.trusted_queue_script}",
            }
        return {
            "project_root": str(settings.repo_root),
            "runtime_revision": (
                settings.runtime_version_path.read_text(encoding="utf-8").strip()
                if settings.runtime_version_path.is_file()
                else None
            ),
            "trusted_runtime": settings.runtime_current_dir.is_dir(),
            "trusted_git_mirror": settings.trusted_git_dir.is_dir(),
            "gpu": _run_probe(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,memory.total,memory.used",
                    "--format=csv,noheader",
                ]
            ),
            "docker": _run_probe(["docker", "info", "--format", "{{.ServerVersion}}"]),
            "training_queue": queue_result,
            "gpu_lock_file": str(settings.gpu_lock_file),
        }

    @server.tool(
        title="Describe execution roots and security boundary",
        description=(
            "Show where commands run, which project roots persist, and which host "
            "resources remain unavailable."
        ),
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_execution_layout() -> dict[str, Any]:
        return jobs.sandbox.execution_layout()

    @server.tool(
        title="Prepare an exact remote revision",
        description=(
            "Fetch one branch through the external trusted Git mirror, require its "
            "full SHA to match, and create a detached source worktree. No branch is "
            "created or pushed."
        ),
        annotations=_annotations(
            read_only=False, destructive=False, idempotent=False, open_world=True
        ),
        meta=security_meta,
    )
    def prepare_revision_workspace(branch: str, expected_sha: str) -> dict[str, str]:
        return cast(
            dict[str, str],
            workspaces.prepare_revision(branch=branch, expected_sha=expected_sha),
        )

    @server.tool(
        title="Get exact revision status",
        description=(
            "Return the registered branch, exact SHA, clean state, and available "
            "execution roots without reading project files."
        ),
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_revision_status(workspace_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], workspaces.describe_revision(workspace_id))

    @server.tool(
        title="Start a flexible isolated CPU command",
        description=(
            "Run any network-disabled CPU shell command. Use execution_root=revision "
            "for exact code with persistent project data roots, or project for the "
            "entire local tennis-lab tree read-write. The command cannot reach the "
            "control plane, Docker socket, Windows mounts, or host credentials."
        ),
        annotations=_annotations(
            read_only=False, destructive=True, idempotent=False, open_world=False
        ),
        meta=security_meta,
    )
    def start_command(
        command: str,
        workspace_id: str,
        expected_sha: str,
        execution_root: Literal["revision", "project"] = "revision",
        working_directory: str = ".",
        timeout_seconds: int = 900,
    ) -> dict[str, Any]:
        return jobs.start(
            command=command,
            workspace_id=workspace_id,
            expected_sha=expected_sha,
            execution_root=execution_root,
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
        )

    @server.tool(
        title="Get command job",
        description="Inspect status and exit code for one CPU command container.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_command_job(job_id: str) -> dict[str, Any]:
        return jobs.get(job_id)

    @server.tool(
        title="List command jobs",
        description="List recent CPU command jobs and their current states.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def list_command_jobs(limit: int = 50) -> list[dict[str, Any]]:
        return jobs.list(limit=limit)

    @server.tool(
        title="Read command output",
        description="Read bounded, secret-redacted output from one command container.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_command_output(job_id: str, tail: int = 400) -> dict[str, str]:
        return {"job_id": job_id, "output": jobs.sandbox.logs(job_id, tail=tail)}

    @server.tool(
        title="Cancel command job",
        description="Stop one running CPU command container.",
        annotations=_annotations(
            read_only=False, destructive=True, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def cancel_command_job(job_id: str) -> dict[str, str]:
        return jobs.cancel(job_id)

    @server.tool(
        title="Enqueue flexible GPU or long-running work",
        description=(
            "Queue any network-disabled CUDA experiment, local-data validation, "
            "dataset generation, evaluation, or training command through the trusted "
            "logical two-slot queue. Resource half is coordination metadata only, not "
            "a MIG or VRAM hard cap. Both execution roots remain available."
        ),
        annotations=_annotations(
            read_only=False, destructive=True, idempotent=False, open_world=False
        ),
        meta=security_meta,
    )
    def enqueue_training(
        name: str,
        command: str,
        workspace_id: str,
        expected_sha: str,
        execution_root: Literal["revision", "project"] = "revision",
        working_directory: str = ".",
        issue: int | None = None,
        resource: Literal["half", "all"] = "all",
        timeout_seconds: int = 86_400,
    ) -> dict[str, Any]:
        return training.enqueue(
            name=name,
            command=command,
            workspace_id=workspace_id,
            expected_sha=expected_sha,
            execution_root=execution_root,
            working_directory=working_directory,
            issue=issue,
            timeout_seconds=timeout_seconds,
            resource=resource,
        )

    @server.tool(
        title="Get training job",
        description=(
            "Inspect queue, owned PGID teardown, and container status for one GPU "
            "or long-running job; terminating remains nonterminal and capacity-held."
        ),
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_training_job(job_id: str) -> dict[str, Any]:
        return training.status(job_id)

    @server.tool(
        title="List training jobs",
        description="List recent logical-capacity GPU and long-running jobs.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def list_training_jobs(limit: int = 50) -> list[dict[str, Any]]:
        return training.list(limit=limit)

    @server.tool(
        title="Read training output",
        description="Read bounded, secret-redacted output from the trusted GPU queue.",
        annotations=_annotations(
            read_only=True, destructive=False, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def get_training_output(job_id: str, tail: int = 400) -> dict[str, str]:
        return {"job_id": job_id, "output": training.logs(job_id, tail=tail)}

    @server.tool(
        title="Cancel training job",
        description=(
            "Cancel a queued job or request verified teardown of its running GPU "
            "container; acknowledgement requires the deterministic container to be "
            "observably non-running."
        ),
        annotations=_annotations(
            read_only=False, destructive=True, idempotent=True, open_world=False
        ),
        meta=security_meta,
    )
    def cancel_training_job(job_id: str) -> dict[str, str]:
        return training.cancel(job_id)

    return server


def run_gateway(settings: GatewaySettings, *, authenticated: bool = True) -> None:
    """Start the Streamable HTTP gateway on the configured listener."""

    build_gateway(settings, authenticated=authenticated).run(
        transport="streamable-http"
    )
