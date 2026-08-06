"""Canonical scene workspace, locking, staging, and atomic publication."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import (
    AbstractContextManager,
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from src.utils.configuration import PathResolver, PathRole

from .stages import BY_STAGE, Stage, descendants


def remove_path(path: Path) -> None:
    """Remove one resolved owned path without following the final symlink."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


@contextmanager
def capture_log(path: Path) -> Iterator[None]:
    """Capture Python and native subprocess stdout/stderr for one attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    with path.open("w") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


@dataclass(frozen=True, slots=True)
class SceneWorkspace:
    """All fixed paths for exactly one path-safe scene ID."""

    scene_id: str
    root: Path

    @classmethod
    def resolve(cls, resolver: PathResolver, scene_id: str) -> Self:
        if (
            not scene_id
            or scene_id in {".", ".."}
            or "/" in scene_id
            or "\\" in scene_id
        ):
            raise ValueError("scene_id must be a non-empty path-safe identifier")
        return cls(
            scene_id=scene_id,
            root=resolver.resolve(
                PathRole.DATA, "synthetic_data_generation/scenes", scene_id
            ),
        )

    def path(self, relative: Path | str) -> Path:
        path = (self.root / relative).resolve(strict=False)
        if path == self.root or not path.is_relative_to(self.root):
            raise ValueError(f"Path escapes scene workspace: {relative}")
        return path

    def staging(self, stage: Stage) -> Path:
        path = self.path(Path(".staging") / stage.value)
        remove_path(path)
        path.mkdir(parents=True)
        return path

    def invalidate(self, from_stage: Stage) -> None:
        """Unpublish local outputs before execution.

        The external reconstruction workspace is intentionally not traversed.
        Only its public export/report boundary is removed; NHT owns and reuses
        its frames, SfM model and checkpoint internals.
        """
        invalidated = descendants(from_stage, include_self=True)
        for stage in invalidated:
            if stage is Stage.RECONSTRUCTION:
                remove_path(self.path("reconstruction/export"))
                remove_path(self.path("reconstruction/reconstruction-report.json"))
                continue
            remove_path(self.path(BY_STAGE[stage].owned_path))
        dataset_stages = {
            Stage.COURT_DATASET,
            Stage.BLCS_DATASET,
            Stage.PLCS_DATASET,
        }
        datasets_root = self.path("datasets")
        if dataset_stages.issubset(invalidated) and datasets_root.is_dir():
            remove_path(datasets_root)

    def publish(self, stage: Stage, staging: Path) -> None:
        owned = BY_STAGE[stage].owned_path
        source = staging / owned
        if not source.exists():
            raise RuntimeError(f"Stage did not produce owned path: {owned}")
        destination = self.path(owned)
        destination.parent.mkdir(parents=True, exist_ok=True)
        remove_path(destination)
        source.replace(destination)
        remove_path(staging)
        parent = self.path(".staging")
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def cleanup_staging(self) -> None:
        remove_path(self.path(".staging"))


class WorkspaceLock(AbstractContextManager["WorkspaceLock"]):
    """Exclusive process lock with dead-owner recovery."""

    def __init__(self, workspace: SceneWorkspace):
        self.workspace = workspace
        self.path = workspace.path(".pipeline.lock")
        self.acquired = False
        self.recovered = False

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def __enter__(self) -> Self:
        self.workspace.root.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                try:
                    pid = int(json.loads(self.path.read_text())["pid"])
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pid = -1
                if pid > 0 and self._alive(pid):
                    raise RuntimeError(
                        f"Scene workspace is locked by live process {pid}: "
                        f"{self.workspace.root}"
                    ) from None
                self.path.unlink(missing_ok=True)
                self.recovered = True
                continue
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    {
                        "schema": "tennis_scene_workspace_lock_v1",
                        "pid": os.getpid(),
                        "scene_id": self.workspace.scene_id,
                    },
                    stream,
                )
                stream.write("\n")
            self.acquired = True
            return self
        raise RuntimeError("Could not acquire scene workspace lock")

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False
