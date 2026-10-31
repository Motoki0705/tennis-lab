"""Disposable model checkouts and integrity checks for the trusted controller."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


class WorkspaceError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TrustedCheckout:
    def __init__(self, package: Path, run_dir: Path) -> None:
        self.root = package.parents[2].resolve()
        self.package = package.relative_to(self.root)
        self.run_dir = run_dir.resolve()
        if self.run_dir.is_relative_to(self.root):
            raise WorkspaceError("private_directory_inside_checkout")
        if (
            Path(os.fsdecode(self.git("rev-parse", "--show-toplevel")).strip())
            != self.root
        ):
            raise WorkspaceError("invalid_controller_checkout")
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        # A model must start from the same committed code that performs checks.
        self.git("diff", "--quiet", "--no-ext-diff", "--no-textconv", "HEAD", "--")
        self.baseline = self.snapshot()

    def git(self, *args: str) -> bytes:
        return subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(self.root),
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout

    def snapshot(self) -> tuple[bytes, bytes, list[tuple[bytes, int, bytes]], bytes]:
        index = self.git("ls-files", "--stage", "-z")
        files = []
        for entry in index.split(b"\0"):
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            if metadata.startswith(b"160000 "):
                continue  # Uninitialized submodules are not controller files.
            path = self.root / os.fsdecode(name)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                digest = hashlib.sha256(os.fsencode(os.readlink(path))).digest()
            elif stat.S_ISREG(mode):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").digest()
            else:
                raise WorkspaceError("controller_checkout_changed")
            files.append((name, mode, digest))
        # Include ignored additions too; the initial files are a baseline, not
        # a blanket exemption for ignored paths. Do not rely on git stat caches.
        return (
            self.git("rev-parse", "HEAD"),
            index,
            files,
            self.git("ls-files", "--others", "-z"),
        )

    def committed(self, relative: str) -> bytes:
        path = self.package / relative
        payload = self.git("show", f"{self.head}:{path.as_posix()}")
        if (self.root / path).read_bytes() != payload:
            raise WorkspaceError("controller_asset_not_committed")
        return payload

    def verify(self, expected: Mapping[Path, bytes]) -> None:
        try:
            if self.snapshot() != self.baseline:
                raise WorkspaceError("controller_checkout_changed")
            if any(path.read_bytes() != payload for path, payload in expected.items()):
                raise WorkspaceError("model_input_changed")
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkspaceError("controller_integrity_check_failed") from exc

    @contextmanager
    def disposable(self) -> Iterator[Path]:
        # Each turn starts clean: a writer cannot plant context for a verifier
        # or a repair. History/remote are shared, working files are independent.
        with tempfile.TemporaryDirectory(
            prefix="issue-rewrite-", dir=self.run_dir.parent
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            try:
                self.git("worktree", "add", "--detach", str(checkout), self.head)
                yield checkout
            finally:
                # --force removes model edits and untracked files on every
                # exit, including schema errors and integrity check failures.
                # Remove only this registration; never prune other worktrees.
                try:
                    self.git("worktree", "remove", "--force", str(checkout))
                except (OSError, subprocess.SubprocessError) as exc:
                    raise WorkspaceError("model_workspace_cleanup_failed") from exc
