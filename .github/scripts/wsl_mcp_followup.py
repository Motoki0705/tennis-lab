"""Apply final security and deployment hardening to the WSL MCP redesign."""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one patch target in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_settings() -> None:
    path = Path("src/automation/chatgpt_mcp/settings.py")
    _replace_once(
        path,
        '''_DEFAULT_ORIGIN_URL = "https://github.com/Motoki0705/tennis-lab.git"\n_ALLOWED_ORIGIN_URLS = {\n    _DEFAULT_ORIGIN_URL,\n    "https://github.com/Motoki0705/tennis-lab",\n    "git@github.com:Motoki0705/tennis-lab.git",\n}\n''',
        '''_DEFAULT_ORIGIN_URL = "https://github.com/Motoki0705/tennis-lab.git"\n_DEFAULT_GPU_LOCK_FILE = Path("/var/lib/tennis-lab-actions/gpu.lock")\n_ALLOWED_ORIGIN_URLS = {\n    _DEFAULT_ORIGIN_URL,\n    "https://github.com/Motoki0705/tennis-lab",\n}\n''',
    )
    _replace_once(
        path,
        '''    control_dir: Path = _DEFAULT_CONTROL_DIR\n    origin_url: str = _DEFAULT_ORIGIN_URL\n    host: str = "127.0.0.1"\n''',
        '''    control_dir: Path = _DEFAULT_CONTROL_DIR\n    origin_url: str = _DEFAULT_ORIGIN_URL\n    gpu_lock_file: Path = _DEFAULT_GPU_LOCK_FILE\n    host: str = "127.0.0.1"\n''',
    )
    _replace_once(
        path,
        '''        control_dir = self.control_dir.resolve()\n        if state_dir == repo_root or state_dir.is_relative_to(repo_root):\n''',
        '''        control_dir = self.control_dir.resolve()\n        gpu_lock_file = self.gpu_lock_file.expanduser()\n        if not gpu_lock_file.is_absolute():\n            raise ValueError("MCP GPU lock file must be an absolute path")\n        gpu_lock_file = gpu_lock_file.resolve()\n        object.__setattr__(self, "repo_root", repo_root)\n        object.__setattr__(self, "state_dir", state_dir)\n        object.__setattr__(self, "control_dir", control_dir)\n        object.__setattr__(self, "gpu_lock_file", gpu_lock_file)\n        if state_dir == repo_root or state_dir.is_relative_to(repo_root):\n''',
    )
    _replace_once(
        path,
        '''        if state_dir == control_dir:\n            raise ValueError("MCP state and control directories must be distinct")\n        if not 1024 <= self.port <= 65535:\n''',
        '''        if state_dir == control_dir:\n            raise ValueError("MCP state and control directories must be distinct")\n        if gpu_lock_file == repo_root or gpu_lock_file.is_relative_to(repo_root):\n            raise ValueError("MCP GPU lock must be outside the destructible project")\n        if not 1024 <= self.port <= 65535:\n''',
    )
    _replace_once(
        path,
        '''        uv_python_root = Path(\n            os.environ.get(\n                "TENNIS_MCP_UV_PYTHON_ROOT",\n                "/home/kamimura/.local/share/uv/python",\n            )\n        ).expanduser()\n\n        return cls(\n''',
        '''        uv_python_root = Path(\n            os.environ.get(\n                "TENNIS_MCP_UV_PYTHON_ROOT",\n                "/home/kamimura/.local/share/uv/python",\n            )\n        ).expanduser()\n        gpu_lock_file = _absolute_path(\n            os.environ.get(\n                "TENNIS_MCP_GPU_LOCK_FILE",\n                str(_DEFAULT_GPU_LOCK_FILE),\n            ),\n            "TENNIS_MCP_GPU_LOCK_FILE",\n        )\n\n        return cls(\n''',
    )
    _replace_once(
        path,
        '''            origin_url=normalize_origin_url(\n                os.environ.get("TENNIS_MCP_ORIGIN_URL", _DEFAULT_ORIGIN_URL)\n            ),\n            host=os.environ.get("TENNIS_MCP_HOST", "127.0.0.1"),\n''',
        '''            origin_url=normalize_origin_url(\n                os.environ.get("TENNIS_MCP_ORIGIN_URL", _DEFAULT_ORIGIN_URL)\n            ),\n            gpu_lock_file=gpu_lock_file,\n            host=os.environ.get("TENNIS_MCP_HOST", "127.0.0.1"),\n''',
    )
    _replace_once(
        path,
        '''    def runtime_releases_dir(self) -> Path:\n        return self.control_dir / "releases"\n\n    @property\n    def runtime_current_dir(self) -> Path:\n''',
        '''    def runtime_releases_dir(self) -> Path:\n        return self.control_dir / "releases"\n\n    @property\n    def runtime_venvs_dir(self) -> Path:\n        return self.control_dir / "venvs"\n\n    @property\n    def runtime_home(self) -> Path:\n        return self.control_dir / "runtime-home"\n\n    @property\n    def runtime_current_dir(self) -> Path:\n''',
    )
    _replace_once(
        path,
        '''            self.control_dir,\n            self.runtime_releases_dir,\n            self.runtime_bin_dir,\n            self.trusted_git_home,\n''',
        '''            self.control_dir,\n            self.runtime_releases_dir,\n            self.runtime_venvs_dir,\n            self.runtime_bin_dir,\n            self.runtime_home,\n            self.trusted_git_home,\n''',
    )


def patch_runtime() -> None:
    path = Path("src/automation/chatgpt_mcp/runtime.py")
    _replace_once(
        path,
        '''_SHA = re.compile(r"^[0-9a-f]{40}$")\n\n\nclass RuntimeInstallError''',
        '''_SHA = re.compile(r"^[0-9a-f]{40}$")\n\n\ndef _validated_sha(value: str) -> str:\n    revision = value.strip().lower()\n    if not _SHA.fullmatch(revision):\n        raise RuntimeInstallError("expected_sha must be a full 40-character commit SHA")\n    return revision\n\n\ndef _origin_identity(value: str) -> str:\n    normalized = value.strip().rstrip("/")\n    if normalized.endswith(".git"):\n        normalized = normalized[:-4]\n    return normalized\n\n\nclass RuntimeInstallError''',
    )
    _replace_once(
        path,
        '''    def install(self, source_root: Path) -> RuntimeInstallResult:\n        """Install one exact source revision and atomically activate it."""\n\n        source = source_root.expanduser().resolve()\n        package = source / "src/automation/chatgpt_mcp"\n''',
        '''    def install(\n        self, source_root: Path, *, expected_sha: str\n    ) -> RuntimeInstallResult:\n        """Install one clean reviewed checkout at the explicitly expected revision."""\n\n        checked_sha = _validated_sha(expected_sha)\n        source = source_root.expanduser().resolve()\n        protected_roots = (\n            self.settings.repo_root,\n            self.settings.state_dir,\n            self.settings.control_dir,\n        )\n        if any(source == root or source.is_relative_to(root) for root in protected_roots):\n            raise RuntimeInstallError(\n                "deployment source must be a separate clean reviewed checkout outside "\n                "tennis-lab, MCP state, and the MCP control plane"\n            )\n        package = source / "src/automation/chatgpt_mcp"\n''',
    )
    _replace_once(
        path,
        '''        revision = _checked(\n            ["git", "-C", str(source), "rev-parse", "HEAD^{commit}"],\n            message="source revision is unavailable",\n        ).lower()\n        if not _SHA.fullmatch(revision):\n            raise RuntimeInstallError("source revision is not a full commit SHA")\n\n        self.settings.ensure_state()\n        self.settings.ensure_control_directories()\n        python_executable = self._install_venv()\n        release_dir = self._install_release(source, revision)\n        installed_queue = self._install_queue_runner(source)\n        self._ensure_trusted_mirror()\n        self._activate_release(release_dir, revision)\n''',
        '''        git_prefix = [\n            "git",\n            "-c",\n            "core.hooksPath=/dev/null",\n            "-c",\n            "core.fsmonitor=false",\n            "-C",\n            str(source),\n        ]\n        revision = _checked(\n            [*git_prefix, "rev-parse", "HEAD^{commit}"],\n            message="source revision is unavailable",\n        ).lower()\n        if revision != checked_sha:\n            raise RuntimeInstallError(\n                f"deployment source is {revision}, expected {checked_sha}"\n            )\n        status = _checked(\n            [*git_prefix, "status", "--porcelain=v1", "--untracked-files=all"],\n            message="deployment source status is unavailable",\n        )\n        if status:\n            raise RuntimeInstallError("deployment source must be completely clean")\n        source_origin = _checked(\n            [*git_prefix, "remote", "get-url", "origin"],\n            message="deployment source origin is unavailable",\n        )\n        if _origin_identity(source_origin) != _origin_identity(self.settings.origin_url):\n            raise RuntimeInstallError(\n                "deployment source origin does not match the fixed tennis-lab origin"\n            )\n\n        self.settings.ensure_state()\n        self.settings.ensure_control_directories()\n        python_executable = self._install_venv()\n        self._ensure_trusted_mirror()\n        _checked(\n            [\n                "git",\n                "--git-dir",\n                str(self.settings.trusted_git_dir),\n                "cat-file",\n                "-e",\n                f"{checked_sha}^{{commit}}",\n            ],\n            env=self._git_environment(),\n            message="expected deployment revision is absent from the trusted mirror",\n        )\n        release_dir = self._install_release(source, revision)\n        installed_queue = self._install_queue_runner(source)\n        self._activate_release(release_dir, revision)\n''',
    )
    _replace_once(
        path,
        '''            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),\n''',
        '''            "PATH": "/usr/bin:/bin",\n''',
    )


def patch_cli() -> None:
    path = Path("src/automation/chatgpt_mcp/cli.py")
    _replace_once(
        path,
        '''def _source_root(value: Path | None) -> Path:\n    return _git_root() if value is None else value.expanduser().resolve()\n\n\ndef install_runtime(source_root: Path | None) -> dict[str, str]:\n    """Install the trusted control plane from one reviewed checkout."""\n\n    settings = GatewaySettings.from_env(require_public_base_url=False)\n    result = RuntimeInstaller(settings).install(_source_root(source_root))\n''',
        '''def _source_root(value: Path | None) -> Path:\n    if value is None:\n        raise ValueError("--source-root is required for trusted runtime deployment")\n    return value.expanduser().resolve()\n\n\ndef install_runtime(source_root: Path | None, expected_sha: str) -> dict[str, str]:\n    """Install the trusted control plane from one reviewed checkout."""\n\n    settings = GatewaySettings.from_env(require_public_base_url=False)\n    result = RuntimeInstaller(settings).install(\n        _source_root(source_root), expected_sha=expected_sha\n    )\n''',
    )
    _replace_once(
        path,
        '''    reuse_existing_key: bool,\n    source_root: Path | None,\n    start: bool,\n''',
        '''    reuse_existing_key: bool,\n    source_root: Path | None,\n    expected_sha: str,\n    start: bool,\n''',
    )
    _replace_once(
        path,
        '''    runtime = RuntimeInstaller(settings).install(_source_root(source_root))\n''',
        '''    runtime = RuntimeInstaller(settings).install(\n        _source_root(source_root), expected_sha=expected_sha\n    )\n''',
    )
    _replace_once(
        path,
        '''def install_user_service(*, source_root: Path | None, start: bool) -> Path:\n''',
        '''def install_user_service(\n    *, source_root: Path | None, expected_sha: str, start: bool\n) -> Path:\n''',
    )
    _replace_once(
        path,
        '''    RuntimeInstaller(settings).install(_source_root(source_root))\n''',
        '''    RuntimeInstaller(settings).install(\n        _source_root(source_root), expected_sha=expected_sha\n    )\n''',
    )
    _replace_once(
        path,
        '''    runtime_parser = subparsers.add_parser("install-runtime")\n    runtime_parser.add_argument("--source-root", type=Path)\n\n    install_parser = subparsers.add_parser("install-user-service")\n    install_parser.add_argument("--source-root", type=Path)\n''',
        '''    runtime_parser = subparsers.add_parser("install-runtime")\n    runtime_parser.add_argument("--source-root", type=Path, required=True)\n    runtime_parser.add_argument("--expected-sha", required=True)\n\n    install_parser = subparsers.add_parser("install-user-service")\n    install_parser.add_argument("--source-root", type=Path, required=True)\n    install_parser.add_argument("--expected-sha", required=True)\n''',
    )
    _replace_once(
        path,
        '''    secure_parser.add_argument("--source-root", type=Path)\n    secure_parser.add_argument("--start", action="store_true")\n''',
        '''    secure_parser.add_argument("--source-root", type=Path, required=True)\n    secure_parser.add_argument("--expected-sha", required=True)\n    secure_parser.add_argument("--start", action="store_true")\n''',
    )
    _replace_once(
        path,
        '''    elif arguments.command == "install-runtime":\n        print(json.dumps(install_runtime(arguments.source_root), indent=2))\n''',
        '''    elif arguments.command == "install-runtime":\n        print(\n            json.dumps(\n                install_runtime(arguments.source_root, arguments.expected_sha),\n                indent=2,\n            )\n        )\n''',
    )
    _replace_once(
        path,
        '''        path = install_user_service(\n            source_root=arguments.source_root,\n            start=arguments.start,\n        )\n''',
        '''        path = install_user_service(\n            source_root=arguments.source_root,\n            expected_sha=arguments.expected_sha,\n            start=arguments.start,\n        )\n''',
    )
    _replace_once(
        path,
        '''                source_root=arguments.source_root,\n                start=arguments.start,\n''',
        '''                source_root=arguments.source_root,\n                expected_sha=arguments.expected_sha,\n                start=arguments.start,\n''',
    )


def patch_jobs() -> None:
    path = Path("src/automation/chatgpt_mcp/jobs.py")
    _replace_once(
        path,
        '''_DIRECT_COMMAND_MAX_SECONDS = 24 * 3600\n_MAX_CONCURRENT_DIRECT_JOBS = 4\n''',
        '''_DIRECT_COMMAND_MAX_SECONDS = 24 * 3600\n_MAX_CONCURRENT_DIRECT_JOBS = 2\n_DIRECT_MEMORY_GB = 24\n_QUEUED_MEMORY_GB = 48\n''',
    )
    _replace_once(
        path,
        '''            "direct_concurrency": _MAX_CONCURRENT_DIRECT_JOBS,\n            "persistent_roots": list(_PERSISTENT_PROJECT_ROOTS),\n''',
        '''            "direct_concurrency": _MAX_CONCURRENT_DIRECT_JOBS,\n            "direct_memory_limit_gb": _DIRECT_MEMORY_GB,\n            "queued_memory_limit_gb": _QUEUED_MEMORY_GB,\n            "persistent_roots": list(_PERSISTENT_PROJECT_ROOTS),\n''',
    )
    _replace_once(
        path,
        '''            "--pids-limit",\n            "4096",\n            "--memory",\n            "48g",\n            "--shm-size",\n            "8g",\n''',
        '''            "--init",\n            "--pids-limit",\n            "4096",\n            "--memory",\n            f"{_QUEUED_MEMORY_GB if spec.use_gpu else _DIRECT_MEMORY_GB}g",\n            "--shm-size",\n            "8g" if spec.use_gpu else "4g",\n''',
    )
    _replace_once(
        path,
        '''            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),\n            "HOME": str(self.settings.control_dir),\n            "TRAINING_QUEUE_DIR": str(self.queue_dir),\n''',
        '''            "PATH": "/usr/bin:/bin",\n            "HOME": str(self.settings.runtime_home),\n            "TRAINING_QUEUE_DIR": str(self.queue_dir),\n            "TRAINING_QUEUE_LOCK_FILE": str(self.settings.gpu_lock_file),\n''',
    )


def patch_server() -> None:
    path = Path("src/automation/chatgpt_mcp/server.py")
    _replace_once(
        path,
        '''                    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),\n                    "HOME": str(settings.control_dir),\n                    "TRAINING_QUEUE_DIR": str(settings.trusted_queue_dir),\n''',
        '''                    "PATH": "/usr/bin:/bin",\n                    "HOME": str(settings.runtime_home),\n                    "TRAINING_QUEUE_DIR": str(settings.trusted_queue_dir),\n                    "TRAINING_QUEUE_LOCK_FILE": str(settings.gpu_lock_file),\n''',
    )
    _replace_once(
        path,
        '''            "training_queue": queue_result,\n        }\n''',
        '''            "training_queue": queue_result,\n            "gpu_lock_file": str(settings.gpu_lock_file),\n        }\n''',
    )


def patch_secure_tunnel() -> None:
    path = Path("src/automation/chatgpt_mcp/secure_tunnel.py")
    _replace_once(
        path,
        '''Environment="PYTHONPATH={self.source_root}"\nEnvironment="TENNIS_MCP_REPO_ROOT={self.settings.repo_root}"\n''',
        '''Environment="PYTHONPATH={self.source_root}"\nEnvironment="HOME={self.settings.runtime_home}"\nEnvironment="PATH=/usr/bin:/bin:/usr/lib/wsl/lib"\nEnvironment="PYTHONNOUSERSITE=1"\nEnvironment="PYTHONDONTWRITEBYTECODE=1"\nEnvironment="GIT_CONFIG_NOSYSTEM=1"\nEnvironment="GIT_CONFIG_GLOBAL=/dev/null"\nEnvironment="TENNIS_MCP_REPO_ROOT={self.settings.repo_root}"\n''',
    )
    _replace_once(
        path,
        '''Environment="TENNIS_MCP_ORIGIN_URL={self.settings.origin_url}"\nEnvironment="TENNIS_MCP_HOST=127.0.0.1"\n''',
        '''Environment="TENNIS_MCP_ORIGIN_URL={self.settings.origin_url}"\nEnvironment="TENNIS_MCP_GPU_LOCK_FILE={self.settings.gpu_lock_file}"\nEnvironment="TENNIS_MCP_HOST=127.0.0.1"\n''',
    )


def rewrite_readme() -> None:
    Path("src/automation/chatgpt_mcp/README.md").write_text(
        '''# tennis-lab ChatGPT WSL MCP\n\nThis gateway gives ChatGPT a broad execution plane inside `tennis-lab` while\nkeeping its host control plane outside the project. GitHub MCP remains the\nrepository control plane.\n\n## Responsibility split\n\nGitHub MCP owns repository exploration, Issues and Pull Requests, branch\ncreation, source implementation, commits, pushes, and remote state.\n\nWSL MCP owns runtime work only:\n\n- fetch one fixed `origin` branch through an external trusted bare mirror;\n- bind every job to a caller-supplied full commit SHA;\n- run arbitrary network-disabled shell commands in Docker;\n- expose the complete local `tennis-lab` project read-write for real data,\n  generated chunks, outputs, checkpoints, artifacts, caches, and experiments;\n- serialize CUDA and long-running work through the external training queue and\n  the same host GPU lock used by local GitHub Actions;\n- return job state and secret-redacted output.\n\nIt does not expose MCP tools for source browsing, patching, committing, or\npushing. A shell command can nevertheless modify or delete anything below the\nlocal `tennis-lab` directory. That destruction is explicitly inside the threat\nmodel and cannot affect GitHub unless GitHub MCP separately persists a change.\n\n## Security boundary\n\nThe destructible zone is the complete project root, including `src/`, `tests/`,\n`data/`, `outputs/`, `ckpt/`, `artifacts/`, `.cache/`, `third_party/`, and all\nother project content. The trusted runtime, versioned venv, tunnel credentials,\nGit mirror, queue runner, systemd units, and durable MCP state live under\n`~/.local/share/tennis-lab-chatgpt-mcp/` and\n`~/.local/state/tennis-lab-chatgpt-mcp/`, outside that tree.\n\nThe container receives:\n\n- `/workspace`: a private read-write copy of the exact remote revision;\n- `/tennis-lab`: the complete local project read-write;\n- standard mutable roots in `/workspace` linked to `/tennis-lab`, including\n  `data`, `outputs`, `ckpt`, `checkpoints`, `artifacts`, `.cache`,\n  `third_party`, and `.training_queue`;\n- the external trusted venv and uv Python runtime read-only;\n- no network, Docker socket, `/mnt/c`, host credentials, systemd, tunnel\n  credentials, queue runner, trusted Git mirror, or MCP runtime source;\n- masked `.git` metadata for both execution roots;\n- a read-only container root filesystem, all capabilities dropped,\n  `no-new-privileges`, private IPC, PID and memory limits, and a bounded timeout.\n\nThe host service imports only the external trusted runtime. Runtime promotion is\naccepted only from a separate, completely clean checkout at an explicitly\nsupplied full SHA whose `origin` matches `Motoki0705/tennis-lab`; the canonical\nread-write project cannot be promoted into the control plane.\n\nBecause all project files are readable, commands can intentionally print project\ndata into MCP logs. Never place API keys, SSH keys, personal credentials, or\nother secrets anywhere below `tennis-lab`. Treat the local project as untrusted\nafter arbitrary execution and do not run its Python or shell code directly on\nthe host until it has been restored or reviewed. Disk exhaustion and kernel or\nDocker vulnerabilities remain residual host risks; keep backups for valuable\ndata and outputs.\n\n## MCP tools\n\n1. `get_host_status`\n2. `get_execution_layout`\n3. `prepare_revision_workspace`\n4. `get_revision_status`\n5. `start_command`\n6. `get_command_job`\n7. `list_command_jobs`\n8. `get_command_output`\n9. `cancel_command_job`\n10. `enqueue_training`\n11. `get_training_job`\n12. `list_training_jobs`\n13. `get_training_output`\n14. `cancel_training_job`\n\n`start_command` accepts any CPU shell command, a relative working directory,\nand one of two roots. `execution_root="revision"` uses exact code with persistent\nproject data/output/checkpoint roots linked in. `execution_root="project"` uses\nthe complete current local project tree. Direct jobs may run for up to 24 hours;\nat most two run concurrently and each is limited to 24 GiB. GPU or heavier work\nuses `enqueue_training`, is serialized, and receives 48 GiB.\n\nNetwork access is intentionally unavailable. Downloads must use a separately\nreviewed workflow rather than an arbitrary MCP command.\n\n## Typical flow\n\n1. GitHub MCP implements and pushes a branch.\n2. GitHub MCP obtains its full head SHA.\n3. WSL MCP calls `prepare_revision_workspace(branch, expected_sha)`.\n4. WSL MCP runs CPU tests, real-data validation, generation, or inspection with\n   `start_command`.\n5. WSL MCP submits CUDA, evaluation, or training with `enqueue_training`.\n6. GitHub MCP alone persists source changes to the remote branch.\n\nExamples:\n\n```text\nstart_command(\n  workspace_id="rev-...",\n  expected_sha="<40 chars>",\n  execution_root="revision",\n  working_directory=".",\n  timeout_seconds=3600,\n  command="python -m pytest -m local_data -q"\n)\n```\n\n```text\nenqueue_training(\n  name="blcs-tracking-chunked",\n  workspace_id="rev-...",\n  expected_sha="<40 chars>",\n  execution_root="revision",\n  working_directory=".",\n  timeout_seconds=86400,\n  command="python -m src.tasks.blcs.scripts.train --config-name train_tracking_chunked"\n)\n```\n\n## Trusted deployment\n\nThe supported deployment route is the self-hosted **Deploy WSL MCP** workflow.\nIt checks out the reviewed `main` revision into the Actions workspace, provisions\na lockfile-keyed venv outside `tennis-lab`, requires an exact clean SHA and fixed\norigin, atomically installs the external runtime, reuses the stored Tunnel ID and\nruntime key, restarts both services, and verifies MCP discovery, real project\nread-write access, host isolation, CPU tests, CUDA, and the serial queue.\n\nDo not run `configure-secure-tunnel --source-root\n/home/kamimura/projects/tennis-lab`; the canonical project is intentionally\nrejected as a deployment source. Manual recovery requires a separate clean\ncheckout and both `--source-root` and `--expected-sha`.\n\nConnector settings remain:\n\n```text\nConnection: Tunnel\nTunnel: tennis-lab WSL\nAuthentication: None\n```\n\nStable services are `tennis-lab-chatgpt-mcp-private.service` and\n`tennis-lab-chatgpt-secure-tunnel.service`. The private MCP endpoint is\n`http://127.0.0.1:8767/mcp`; tunnel readiness is\n`http://127.0.0.1:8768/readyz`. Keep the legacy Quick Tunnel until an actual\nChatGPT Secure Tunnel call succeeds, then disable it.\n''',
        encoding="utf-8",
    )


def rewrite_runtime_tests() -> None:
    Path("tests/unit/automation/chatgpt_mcp/test_runtime.py").write_text(
        '''from __future__ import annotations\n\nimport os\nimport subprocess\nfrom pathlib import Path\n\nimport pytest\n\nfrom src.automation.chatgpt_mcp.runtime import RuntimeInstallError, RuntimeInstaller\nfrom src.automation.chatgpt_mcp.settings import GatewaySettings\n\n\ndef _run(*arguments: str, cwd: Path | None = None) -> str:\n    result = subprocess.run(\n        list(arguments),\n        cwd=cwd,\n        text=True,\n        capture_output=True,\n        check=True,\n    )\n    return result.stdout.strip()\n\n\ndef _source_checkout(tmp_path: Path) -> tuple[Path, Path, str]:\n    source = tmp_path / "reviewed-source"\n    _run("git", "init", "-q", "-b", "main", str(source))\n    _run("git", "config", "user.email", "test@example.com", cwd=source)\n    _run("git", "config", "user.name", "Test", cwd=source)\n\n    package = source / "src/automation/chatgpt_mcp"\n    package.mkdir(parents=True)\n    (source / "src/__init__.py").write_text("", encoding="utf-8")\n    (source / "src/automation/__init__.py").write_text("", encoding="utf-8")\n    (package / "__init__.py").write_text("", encoding="utf-8")\n    (package / "example.py").write_text("VALUE = 1\\n", encoding="utf-8")\n    configuration = source / "src/utils/configuration"\n    configuration.mkdir(parents=True)\n    for module_name in ("errors.py", "schema.py", "paths.py"):\n        (configuration / module_name).write_text("VALUE = 1\\n", encoding="utf-8")\n    queue = source / ".agents/skills/training-queue/scripts/training_queue.sh"\n    queue.parent.mkdir(parents=True)\n    queue.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")\n    os.chmod(queue, 0o700)\n\n    _run("git", "add", "src", ".agents", cwd=source)\n    _run("git", "commit", "-qm", "runtime source", cwd=source)\n    revision = _run("git", "rev-parse", "HEAD", cwd=source)\n    remote = tmp_path / "origin.git"\n    _run("git", "clone", "-q", "--bare", str(source), str(remote))\n    _run("git", "remote", "add", "origin", str(remote), cwd=source)\n    return source, remote, revision\n\n\ndef _settings(tmp_path: Path, remote: Path) -> GatewaySettings:\n    project = tmp_path / "project"\n    project.mkdir()\n    (project / ".git").mkdir()\n    settings = GatewaySettings(\n        repo_root=project,\n        state_dir=tmp_path / "state",\n        control_dir=tmp_path / "control",\n        public_base_url=None,\n        origin_url=str(remote),\n        gpu_lock_file=tmp_path / "gpu.lock",\n        uv_python_root=tmp_path / "uv",\n    )\n    settings.uv_python_root.mkdir()\n    trusted_python = settings.runtime_venv_root / "bin/python"\n    trusted_python.parent.mkdir(parents=True)\n    trusted_python.write_text("#!/usr/bin/env bash\\nexit 0\\n", encoding="utf-8")\n    os.chmod(trusted_python, 0o700)\n    return settings\n\n\ndef test_installer_externalizes_clean_exact_runtime_venv_queue_and_git_mirror(\n    tmp_path: Path,\n) -> None:\n    source, remote, revision = _source_checkout(tmp_path)\n    settings = _settings(tmp_path, remote)\n\n    first = RuntimeInstaller(settings).install(source, expected_sha=revision)\n    second = RuntimeInstaller(settings).install(source, expected_sha=revision)\n\n    assert first.revision == revision\n    assert second.revision == first.revision\n    assert settings.project_venv_link.is_symlink()\n    assert settings.project_venv_link.resolve() == settings.runtime_venv_root\n    assert first.python_executable == settings.runtime_venv_root / "bin/python"\n    assert settings.runtime_current_dir.resolve() == first.release_dir\n    assert first.release_dir.joinpath("src/utils/configuration/paths.py").is_file()\n    assert settings.trusted_queue_script.is_file()\n    assert os.access(settings.trusted_queue_script, os.X_OK)\n    assert _run(\n        "git",\n        "--git-dir",\n        str(settings.trusted_git_dir),\n        "rev-parse",\n        "--is-bare-repository",\n    ) == "true"\n    assert settings.runtime_version_path.read_text(encoding="utf-8").strip() == revision\n\n\ndef test_installer_rejects_canonical_project_as_runtime_source(tmp_path: Path) -> None:\n    source, remote, revision = _source_checkout(tmp_path)\n    settings = _settings(tmp_path, remote)\n    settings = GatewaySettings(\n        repo_root=source,\n        state_dir=settings.state_dir,\n        control_dir=settings.control_dir,\n        public_base_url=None,\n        origin_url=str(remote),\n        gpu_lock_file=tmp_path / "gpu.lock",\n        uv_python_root=settings.uv_python_root,\n    )\n\n    with pytest.raises(RuntimeInstallError, match="separate clean reviewed checkout"):\n        RuntimeInstaller(settings).install(source, expected_sha=revision)\n\n\ndef test_installer_rejects_dirty_or_wrong_revision_source(tmp_path: Path) -> None:\n    source, remote, revision = _source_checkout(tmp_path)\n    settings = _settings(tmp_path, remote)\n\n    with pytest.raises(RuntimeInstallError, match="expected"):\n        RuntimeInstaller(settings).install(source, expected_sha="f" * 40)\n\n    (source / "untracked.txt").write_text("unsafe\\n", encoding="utf-8")\n    with pytest.raises(RuntimeInstallError, match="completely clean"):\n        RuntimeInstaller(settings).install(source, expected_sha=revision)\n''',
        encoding="utf-8",
    )


def patch_tests() -> None:
    jobs = Path("tests/unit/automation/chatgpt_mcp/test_jobs.py")
    _replace_once(
        jobs,
        '''    assert "--pull never" in joined\n    assert "--gpus all" not in joined\n''',
        '''    assert "--pull never" in joined\n    assert "--init" in command\n    assert command[command.index("--memory") + 1] == "24g"\n    assert command[command.index("--shm-size") + 1] == "4g"\n    assert "--gpus all" not in joined\n''',
    )
    _replace_once(
        jobs,
        '''    assert command[command.index("--gpus") + 1] == "all"\n    assert command[command.index("--network") + 1] == "none"\n''',
        '''    assert command[command.index("--gpus") + 1] == "all"\n    assert command[command.index("--memory") + 1] == "48g"\n    assert command[command.index("--shm-size") + 1] == "8g"\n    assert command[command.index("--network") + 1] == "none"\n''',
    )
    _replace_once(
        jobs,
        '''    assert spec["working_directory"] == "src/tasks"\n''',
        '''    assert spec["working_directory"] == "src/tasks"\n    add_environment = queue_commands[0][0]\n    assert add_environment[0] == "bash"\n    assert manager._queue_environment()["TRAINING_QUEUE_LOCK_FILE"] == str(\n        settings.gpu_lock_file\n    )\n''',
    )
    _replace_once(
        jobs,
        '''    assert layout["network"] == "disabled"\n    assert "Docker socket is not mounted" in layout["host_boundaries"]\n''',
        '''    assert layout["network"] == "disabled"\n    assert layout["direct_memory_limit_gb"] == 24\n    assert layout["queued_memory_limit_gb"] == 48\n    assert layout["direct_concurrency"] == 2\n    assert "Docker socket is not mounted" in layout["host_boundaries"]\n''',
    )

    settings_test = Path("tests/unit/automation/chatgpt_mcp/test_settings.py")
    _replace_once(
        settings_test,
        '''        assert stat.S_IMODE(settings.state_dir.stat().st_mode) == 0o700\n''',
        '''        assert stat.S_IMODE(settings.state_dir.stat().st_mode) == 0o700\n        assert settings.gpu_lock_file == Path(\n            "/var/lib/tennis-lab-actions/gpu.lock"\n        )\n''',
    )

    secure_test = Path("tests/unit/automation/chatgpt_mcp/test_secure_tunnel.py")
    _replace_once(
        secure_test,
        '''    assert "TENNIS_MCP_PORT=8767" in private_unit\n''',
        '''    assert "TENNIS_MCP_PORT=8767" in private_unit\n    assert f"HOME={manager.settings.runtime_home}" in private_unit\n    assert "PYTHONNOUSERSITE=1" in private_unit\n    assert f"TENNIS_MCP_GPU_LOCK_FILE={manager.settings.gpu_lock_file}" in private_unit\n''',
    )


def regenerate_inventory() -> None:
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
    approvals: dict[tuple[str, str, int, AuditRule], AuditExemption] = {}
    for finding in unresolved:
        if finding.rule not in path_rules:
            reason_code = "strict-schema"
        elif finding.rule is AuditRule.FILE_PARENT_INDEX:
            reason_code = "code-or-artifact-location"
        else:
            reason_code = "persisted-layout"
        exemption = AuditExemption.classified(
            module=finding.module,
            qualified_name=finding.qualified_name,
            line=finding.line,
            rule=finding.rule,
            reason_code=reason_code,
        )
        approvals.setdefault(
            (
                exemption.module,
                exemption.qualified_name,
                exemption.line,
                exemption.rule,
            ),
            exemption,
        )
    write_generated_inventory_data(
        source_root,
        source_revision="wsl-mcp-project-sandbox-v2",
        approved_exemptions=tuple(approvals.values()),
    )


def apply() -> None:
    patch_settings()
    patch_runtime()
    patch_cli()
    patch_jobs()
    patch_server()
    patch_secure_tunnel()
    rewrite_readme()
    rewrite_runtime_tests()
    patch_tests()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("apply", "inventory"))
    args = parser.parse_args()
    if args.mode == "apply":
        apply()
    else:
        regenerate_inventory()


if __name__ == "__main__":
    main()
