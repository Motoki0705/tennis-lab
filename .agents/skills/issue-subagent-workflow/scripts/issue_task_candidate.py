"""Deterministic candidate-content fingerprints."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

FINGERPRINT_PREFIX = "sha256:"
EXCLUDED_PREFIXES = (".codex/tasks/",)


def _git(
    root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if check and completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise ValueError(message or f"git {' '.join(args)} failed")
    return completed


def repository_root(task_dir: Path) -> Path:
    completed = _git(task_dir, "rev-parse", "--show-toplevel", check=False)
    if completed.returncode == 0:
        return Path(completed.stdout.decode().strip()).resolve()
    return task_dir.parent.resolve()


def current_revision(task_dir: Path) -> str:
    root = repository_root(task_dir)
    completed = _git(root, "rev-parse", "HEAD", check=False)
    if completed.returncode == 0:
        return completed.stdout.decode().strip()
    return "WORKTREE"


def initial_base_revision(task_dir: Path) -> str:
    revision = current_revision(task_dir)
    return "" if revision == "WORKTREE" else revision


def _excluded(path: str) -> bool:
    normalized = path.replace(os.sep, "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def _changed_paths(root: Path, base_revision: str) -> list[str]:
    if not base_revision:
        return []
    tracked = _git(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        base_revision,
        "--",
        ".",
    ).stdout.split(b"\0")
    untracked = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout.split(b"\0")
    paths = {
        item.decode(errors="surrogateescape") for item in (*tracked, *untracked) if item
    }
    return sorted(path for path in paths if not _excluded(path))


def _revision_changed_paths(root: Path, base_revision: str, revision: str) -> list[str]:
    return [
        relative
        for relative in _revision_path_inventory(root, base_revision, revision)
        if not _excluded(relative)
    ]


def _revision_path_inventory(
    root: Path,
    base_revision: str,
    revision: str,
) -> list[str]:
    output = _git(
        root,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        base_revision,
        revision,
        "--",
        ".",
    ).stdout.split(b"\0")
    return sorted(item.decode(errors="surrogateescape") for item in output if item)


def _worktree_entry(root: Path, relative: str) -> tuple[str, bytes]:
    path = root / relative
    if path.is_symlink():
        return "120000", os.readlink(path).encode(errors="surrogateescape")
    if path.is_file():
        mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
        return mode, path.read_bytes()
    if path.is_dir():
        completed = _git(path, "rev-parse", "HEAD", check=False)
        if completed.returncode == 0:
            return "160000", completed.stdout.strip()
    return "deleted", b""


def _revision_entry(
    root: Path,
    revision: str,
    relative: str,
) -> tuple[str, bytes]:
    tree = _git(root, "ls-tree", revision, "--", relative, check=False)
    if tree.returncode != 0 or not tree.stdout.strip():
        return "deleted", b""
    line = tree.stdout.splitlines()[0]
    metadata, _, _ = line.partition(b"\t")
    parts = metadata.split()
    if len(parts) != 3:
        raise ValueError(f"unexpected git ls-tree output for {relative}")
    mode = parts[0].decode()
    object_sha = parts[2].decode()
    if mode == "160000":
        return mode, object_sha.encode()
    blob = _git(root, "show", f"{revision}:{relative}")
    return mode, blob.stdout


def _fingerprint(entries: list[tuple[str, str, bytes]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"issue-workflow-candidate-v2\0")
    for relative, mode, content in entries:
        digest.update(relative.encode(errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(mode.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")
    return FINGERPRINT_PREFIX + digest.hexdigest()


def _fallback_entries(root: Path, task_dir: Path) -> list[tuple[str, str, bytes]]:
    entries: list[tuple[str, str, bytes]] = []
    task_resolved = task_dir.resolve()
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if resolved == task_resolved or task_resolved in resolved.parents:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".git/") or _excluded(relative):
            continue
        if path.is_file() or path.is_symlink():
            mode, content = _worktree_entry(root, relative)
            entries.append((relative, mode, content))
    return entries


def changed_paths(
    task_dir: Path,
    state: dict[str, Any] | None = None,
) -> list[str]:
    """Return the complete current candidate path list."""
    root = repository_root(task_dir)
    base_revision = str((state or {}).get("base_revision", ""))
    if base_revision:
        return _changed_paths(root, base_revision)
    return [relative for relative, _, _ in _fallback_entries(root, task_dir)]


def revision_changed_paths(
    task_dir: Path,
    state: dict[str, Any],
    revision: str,
) -> list[str]:
    """Return every path changed between the frozen base and one revision."""
    root = repository_root(task_dir)
    base_revision = str(state.get("base_revision", ""))
    if not base_revision or revision == "WORKTREE":
        return changed_paths(task_dir, state)
    return _revision_changed_paths(root, base_revision, revision)


def revision_path_inventory(
    task_dir: Path,
    base_revision: str,
    revision: str,
) -> list[str]:
    """Return the unfiltered no-renames path inventory for one revision range."""
    if not base_revision:
        raise ValueError("revision path inventory requires a base revision")
    if revision == "WORKTREE":
        raise ValueError("revision path inventory requires a committed revision")
    return _revision_path_inventory(
        repository_root(task_dir),
        base_revision,
        revision,
    )


def compute_candidate_fingerprint(
    task_dir: Path,
    state: dict[str, Any] | None = None,
) -> str:
    root = repository_root(task_dir)
    base_revision = str((state or {}).get("base_revision", ""))
    if base_revision:
        entries = [
            (relative, *_worktree_entry(root, relative))
            for relative in _changed_paths(root, base_revision)
        ]
        return _fingerprint(entries)
    return _fingerprint(_fallback_entries(root, task_dir))


def compute_revision_fingerprint(
    task_dir: Path,
    state: dict[str, Any],
    revision: str,
) -> str:
    root = repository_root(task_dir)
    base_revision = str(state.get("base_revision", ""))
    if not base_revision or revision == "WORKTREE":
        return compute_candidate_fingerprint(task_dir, state)
    entries = [
        (relative, *_revision_entry(root, revision, relative))
        for relative in _revision_changed_paths(root, base_revision, revision)
    ]
    return _fingerprint(entries)


def candidate_metadata(path: Path) -> str:
    import re

    pattern = re.compile(
        rf"(?m)^- Candidate SHA-256: `({re.escape(FINGERPRINT_PREFIX)}[0-9a-f]{{64}})`\s*$"
    )
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path.name} does not record a valid Candidate SHA-256")
    return match.group(1)
