"""Install the MCP control plane outside the destructible tennis-lab tree."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.automation.chatgpt_mcp.settings import GatewaySettings

_SHA = re.compile(r"^[0-9a-f]{40}$")


def _validated_sha(value: str) -> str:
    revision = value.strip().lower()
    if not _SHA.fullmatch(revision):
        raise RuntimeInstallError("expected_sha must be a full 40-character commit SHA")
    return revision


def _origin_identity(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


class RuntimeInstallError(RuntimeError):
    """Raised when the trusted MCP runtime cannot be installed atomically."""


@dataclass(frozen=True)
class RuntimeInstallResult:
    """Installed control-plane locations and exact source revision."""

    revision: str
    release_dir: Path
    current_dir: Path
    python_executable: Path
    trusted_git_dir: Path
    queue_script: Path

    def public_dict(self) -> dict[str, str]:
        return {
            "revision": self.revision,
            "release_dir": str(self.release_dir),
            "current_dir": str(self.current_dir),
            "python_executable": str(self.python_executable),
            "trusted_git_dir": str(self.trusted_git_dir),
            "queue_script": str(self.queue_script),
        }


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _checked(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    message: str,
    timeout: int = 300,
) -> str:
    result = _run(arguments, cwd=cwd, env=env, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeInstallError(
            result.stderr.strip() or result.stdout.strip() or message
        )
    return result.stdout.strip()


class RuntimeInstaller:
    """Copy code, queue runner, venv, and Git mirror into trusted storage."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings

    def install(self, source_root: Path, *, expected_sha: str) -> RuntimeInstallResult:
        """Install one clean reviewed checkout at the explicitly expected revision."""

        checked_sha = _validated_sha(expected_sha)
        source = source_root.expanduser().resolve()
        protected_roots = (
            self.settings.repo_root,
            self.settings.state_dir,
            self.settings.control_dir,
        )
        if any(
            source == root or source.is_relative_to(root) for root in protected_roots
        ):
            raise RuntimeInstallError(
                "deployment source must be a separate clean reviewed checkout outside "
                "tennis-lab, MCP state, and the MCP control plane"
            )
        package = source / "src/automation/chatgpt_mcp"
        queue_script = (
            source / ".agents/skills/training-queue/scripts/training_queue.sh"
        )
        if not package.is_dir() or not queue_script.is_file():
            raise RuntimeInstallError(
                f"source root is not a complete tennis-lab checkout: {source}"
            )

        git_prefix = [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(source),
        ]
        revision = _checked(
            [*git_prefix, "rev-parse", "HEAD^{commit}"],
            message="source revision is unavailable",
        ).lower()
        if revision != checked_sha:
            raise RuntimeInstallError(
                f"deployment source is {revision}, expected {checked_sha}"
            )
        status = _checked(
            [*git_prefix, "status", "--porcelain=v1", "--untracked-files=all"],
            message="deployment source status is unavailable",
        )
        if status:
            raise RuntimeInstallError("deployment source must be completely clean")
        source_origin = _checked(
            [*git_prefix, "remote", "get-url", "origin"],
            message="deployment source origin is unavailable",
        )
        if _origin_identity(source_origin) != _origin_identity(
            self.settings.origin_url
        ):
            raise RuntimeInstallError(
                "deployment source origin does not match the fixed tennis-lab origin"
            )

        self.settings.ensure_state()
        self.settings.ensure_control_directories()
        python_executable = self._install_venv()
        self._ensure_trusted_mirror()
        _checked(
            [
                "git",
                "--git-dir",
                str(self.settings.trusted_git_dir),
                "cat-file",
                "-e",
                f"{checked_sha}^{{commit}}",
            ],
            env=self._git_environment(),
            message="expected deployment revision is absent from the trusted mirror",
        )
        release_dir = self._install_release(source, revision)
        installed_queue = self._install_queue_runner(source)
        self._activate_release(release_dir, revision)

        return RuntimeInstallResult(
            revision=revision,
            release_dir=release_dir,
            current_dir=self.settings.runtime_current_dir,
            python_executable=python_executable,
            trusted_git_dir=self.settings.trusted_git_dir,
            queue_script=installed_queue,
        )

    def _install_venv(self) -> Path:
        target = self.settings.runtime_venv_root
        link = self.settings.project_venv_link
        if not target.is_dir():
            raise RuntimeInstallError(
                "trusted virtual environment must be provisioned outside tennis-lab "
                f"before runtime installation: {target}"
            )
        python_executable = target / "bin/python"
        if not python_executable.exists():
            raise RuntimeInstallError(
                f"trusted virtual environment has no Python: {python_executable}"
            )
        self._ensure_project_venv_link(link, target)
        return python_executable

    @staticmethod
    def _ensure_project_venv_link(link: Path, target: Path) -> None:
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        os.symlink(target, link, target_is_directory=True)

    def _install_release(self, source: Path, revision: str) -> Path:
        release = self.settings.runtime_releases_dir / revision
        if release.is_dir():
            return release
        if release.exists():
            raise RuntimeInstallError(
                f"runtime release path is not a directory: {release}"
            )

        with tempfile.TemporaryDirectory(
            prefix=f".{revision[:12]}.",
            dir=self.settings.runtime_releases_dir,
        ) as temporary:
            candidate = Path(temporary)
            (candidate / "src/automation").mkdir(mode=0o700, parents=True)
            for init_file in (
                source / "src/__init__.py",
                source / "src/automation/__init__.py",
            ):
                destination = candidate / init_file.relative_to(source)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copy2(init_file, destination)
            shutil.copytree(
                source / "src/automation/chatgpt_mcp",
                candidate / "src/automation/chatgpt_mcp",
                dirs_exist_ok=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            configuration = candidate / "src/utils/configuration"
            configuration.mkdir(mode=0o700, parents=True)
            (candidate / "src/utils/__init__.py").write_text(
                '"""Minimal trusted runtime utilities."""\n',
                encoding="utf-8",
            )
            for module_name in ("errors.py", "schema.py", "paths.py"):
                shutil.copy2(
                    source / "src/utils/configuration" / module_name,
                    configuration / module_name,
                )
            (configuration / "__init__.py").write_text(
                "from src.utils.configuration.paths import (\n"
                "    BoundaryPathField, NonHydraPathBoundary, PathDirection, PathKind,\n"
                "    PathResolver, PathRole, RuntimePathRoots,\n"
                ")\n"
                "__all__ = [\n"
                "    'BoundaryPathField', 'NonHydraPathBoundary', 'PathDirection',\n"
                "    'PathKind', 'PathResolver', 'PathRole', 'RuntimePathRoots',\n"
                "]\n",
                encoding="utf-8",
            )
            os.replace(candidate, release)
        return release

    def _install_queue_runner(self, source: Path) -> Path:
        destination = self.settings.trusted_queue_script
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_script = (
            source / ".agents/skills/training-queue/scripts/training_queue.sh"
        )
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source_script, temporary)
        os.chmod(temporary, 0o700)
        os.replace(temporary, destination)

        prune_source = source_script.with_name("prune_ckpts.py")
        if prune_source.is_file():
            prune_destination = destination.with_name("prune_ckpts.py")
            prune_temporary = prune_destination.with_name(
                f".{prune_destination.name}.tmp"
            )
            shutil.copy2(prune_source, prune_temporary)
            os.chmod(prune_temporary, 0o700)
            os.replace(prune_temporary, prune_destination)
        return destination

    def _git_environment(self) -> dict[str, str]:
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.settings.trusted_git_home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "1",
        }
        return environment

    def _ensure_trusted_mirror(self) -> None:
        git_dir = self.settings.trusted_git_dir
        environment = self._git_environment()
        if not git_dir.exists():
            _checked(
                ["git", "init", "--bare", str(git_dir)],
                env=environment,
                message="trusted bare repository initialization failed",
            )
            _checked(
                [
                    "git",
                    "--git-dir",
                    str(git_dir),
                    "remote",
                    "add",
                    "origin",
                    self.settings.origin_url,
                ],
                env=environment,
                message="trusted origin creation failed",
            )
        elif not git_dir.is_dir():
            raise RuntimeInstallError(f"trusted Git path is not a directory: {git_dir}")

        bare = _checked(
            ["git", "--git-dir", str(git_dir), "rev-parse", "--is-bare-repository"],
            env=environment,
            message="trusted Git repository validation failed",
        )
        if bare != "true":
            raise RuntimeInstallError("trusted Git repository must be bare")
        _checked(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "remote",
                "set-url",
                "origin",
                self.settings.origin_url,
            ],
            env=environment,
            message="trusted origin validation failed",
        )
        _checked(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "config",
                "remote.origin.fetch",
                "+refs/heads/*:refs/remotes/origin/*",
            ],
            env=environment,
            message="trusted fetch policy configuration failed",
        )
        _checked(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "fetch",
                "--force",
                "--prune",
                "--no-tags",
                "--no-recurse-submodules",
                "origin",
            ],
            env=environment,
            message="trusted origin fetch failed",
            timeout=600,
        )

    def _activate_release(self, release: Path, revision: str) -> None:
        current = self.settings.runtime_current_dir
        if current.exists() and not current.is_symlink():
            raise RuntimeInstallError(
                f"runtime current path must be a symlink: {current}"
            )
        temporary = current.with_name(".current.tmp")
        temporary.unlink(missing_ok=True)
        relative_target = release.relative_to(self.settings.control_dir)
        os.symlink(relative_target, temporary, target_is_directory=True)
        os.replace(temporary, current)

        version_temp = self.settings.runtime_version_path.with_name(
            ".runtime-version.tmp"
        )
        version_temp.write_text(revision + "\n", encoding="utf-8")
        os.chmod(version_temp, 0o600)
        os.replace(version_temp, self.settings.runtime_version_path)
