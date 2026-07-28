"""Fail-closed subprocess adapter for the independently managed NHT runtime."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.synthetic_data_generation.dataset.pipeline import PipelineCommand

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class NhtRuntime:
    """Pinned filesystem and Git identity of one NHT runtime."""

    repository: Path
    python: Path
    expected_commit: str
    require_clean: bool = True

    def __post_init__(self) -> None:
        repository = self.repository.resolve()
        # Preserve the venv interpreter path. Resolving its symlink selects the
        # base interpreter and silently drops the isolated site-packages.
        python = Path(os.path.abspath(self.python))
        if not repository.is_dir():
            raise FileNotFoundError(f"NHT repository does not exist: {repository}")
        if not python.is_file():
            raise FileNotFoundError(f"NHT Python does not exist: {python}")
        commit = self.expected_commit.lower()
        if _COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError("NHT expected_commit must be a full Git commit.")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "python", python)
        object.__setattr__(self, "expected_commit", commit)


@dataclass(frozen=True)
class ProcessResult:
    """Captured result of one isolated Python module invocation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def raise_for_status(self) -> None:
        """Raise with captured diagnostics when the worker failed."""
        if self.returncode != 0:
            raise RuntimeError(
                "NHT worker failed with exit code "
                f"{self.returncode}.\nstdout:\n{self.stdout}\nstderr:\n{self.stderr}"
            )


class NhtProcessBackend:
    """Execute project-owned render workers inside a pinned NHT environment."""

    def __init__(self, *, project_root: Path, runtime: NhtRuntime) -> None:
        project_root = project_root.resolve()
        if not (project_root / "src" / "synthetic_data_generation").is_dir():
            raise ValueError(
                f"Not a tennis-lab project root for NHT execution: {project_root}"
            )
        self._project_root = project_root
        self._runtime = runtime

    @property
    def runtime(self) -> NhtRuntime:
        """Return the validated NHT runtime configuration."""
        return self._runtime

    def verify_runtime(self) -> None:
        """Verify the exact NHT commit and optional clean-tree invariant."""
        head = subprocess.check_output(
            ["git", "-C", str(self._runtime.repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if head != self._runtime.expected_commit:
            raise RuntimeError(
                f"NHT commit differs: {head} != {self._runtime.expected_commit}."
            )
        if self._runtime.require_clean:
            status = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(self._runtime.repository),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            )
            if status.strip():
                raise RuntimeError("NHT repository contains tracked modifications.")

    def command_for(self, command: PipelineCommand) -> tuple[str, ...]:
        """Build the shell-free subprocess command for one NHT stage."""
        if command.runtime != "nht":
            raise ValueError(
                f"NHT backend cannot execute {command.runtime!r} stage "
                f"{command.stage!r}."
            )
        return (
            str(self._runtime.python),
            "-m",
            command.module,
            *command.arguments,
        )

    def run(self, command: PipelineCommand) -> ProcessResult:
        """Verify the runtime and synchronously execute one NHT worker."""
        self.verify_runtime()
        argv = self.command_for(command)
        environment = os.environ.copy()
        previous_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(self._project_root)
            if not previous_pythonpath
            else f"{self._project_root}{os.pathsep}{previous_pythonpath}"
        )
        completed = subprocess.run(
            argv,
            cwd=self._project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        return ProcessResult(
            command=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
