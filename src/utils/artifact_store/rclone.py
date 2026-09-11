"""Headless rclone-backed artifact persistence."""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from contextlib import suppress
from pathlib import Path, PurePosixPath

from .contracts import ArtifactStore, ArtifactStoreError

_REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class RcloneArtifactStore(ArtifactStore):
    """Mirror a local run tree through an existing rclone OAuth configuration."""

    def __init__(
        self,
        local_root: Path,
        *,
        remote: str,
        remote_root: str,
        config_path: Path,
        timeout_seconds: int = 3600,
    ) -> None:
        super().__init__(local_root)
        if not _REMOTE_RE.fullmatch(remote):
            raise ArtifactStoreError(f"invalid rclone remote name: {remote!r}")
        parts = remote_root.split("/")
        candidate = PurePosixPath(remote_root)
        if (
            not remote_root
            or remote_root != remote_root.strip()
            or candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in parts)
            or ":" in remote_root
            or "\\" in remote_root
        ):
            raise ArtifactStoreError(
                "remote_root must be a normalized non-empty POSIX relative path"
            )
        resolved_config = config_path.expanduser().resolve(strict=False)
        if not resolved_config.is_file() or resolved_config.is_symlink():
            raise ArtifactStoreError(
                f"rclone configuration is not a regular file: {resolved_config}"
            )
        if os.name == "posix" and resolved_config.stat().st_mode & 0o077:
            raise ArtifactStoreError(
                f"rclone configuration must not be accessible by group/other: {resolved_config}"
            )
        if timeout_seconds <= 0:
            raise ArtifactStoreError("rclone timeout_seconds must be positive")
        self.remote = remote
        self.remote_root = candidate.as_posix()
        self.config_path = resolved_config
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return True

    @property
    def remote_uri(self) -> str:
        return f"{self.remote}:{self.remote_root}"

    def _remote_file(self, path: Path) -> str:
        relative = self.relative_path(path).as_posix()
        return f"{self.remote_uri}/{relative}"

    def _run(self, *arguments: str, timeout: int | None = None) -> None:
        command = ["rclone", "--config", str(self.config_path), *arguments]
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds if timeout is None else timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ArtifactStoreError(
                f"rclone command could not complete: {arguments[0]}"
            ) from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
            raise ArtifactStoreError(
                f"rclone {arguments[0]} failed with exit code {result.returncode}: {detail}"
            )

    def publish_file(self, path: Path) -> None:
        destination = self._remote_file(path)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError(f"artifact is not a regular file: {path}")
        temporary = f"{destination}.uploading-{uuid.uuid4().hex}"
        self._run("copyto", str(path), temporary)
        try:
            self._run("moveto", temporary, destination, timeout=900)
        except BaseException:
            with suppress(ArtifactStoreError):
                self._run("deletefile", temporary, timeout=300)
            raise

    def fetch_file(self, path: Path) -> None:
        source = self._remote_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.downloading-{uuid.uuid4().hex}")
        try:
            self._run("copyto", source, str(temporary))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def remove_file(self, path: Path) -> None:
        self._run("deletefile", self._remote_file(path), timeout=300)

    def sync_tree(self) -> None:
        if not self.local_root.is_dir() or self.local_root.is_symlink():
            raise ArtifactStoreError(
                f"local artifact root is not a regular directory: {self.local_root}"
            )
        self._run("copy", str(self.local_root), self.remote_uri)
