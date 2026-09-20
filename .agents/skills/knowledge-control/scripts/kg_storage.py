"""Task-scoped storage and serialized node registration (no generated index)."""
from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from kg_lib import dump_frontmatter, load_nodes, nodes_dir

TASK_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def check_identity(meta: dict[str, Any]) -> None:
    if not TASK_RE.fullmatch(str(meta.get("task", ""))):
        raise ValueError("task is required: lowercase letters, digits and underscores")
    if not re.fullmatch(r"(?:run|group)-[a-z0-9]+(?:-[a-z0-9]+)*", str(meta.get("id", ""))):
        raise ValueError("id must be run-<slug> or group-<slug> (lowercase kebab-case)")
    if not str(meta["id"]).startswith(f"{meta.get('type')}-"):
        raise ValueError("id prefix must match type")


@contextmanager
def registration_lock() -> Iterator[None]:
    # Lock the directory inode: no stale lockfiles or unlink/recreate race.
    base = nodes_dir()
    base.mkdir(parents=True, exist_ok=True)
    fd = os.open(base, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def prepare_node(meta: dict[str, Any], force: bool = False) -> Path:
    """Called under registration_lock; force preserves identity and sequence."""
    check_identity(meta)
    nodes = load_nodes()
    existing = [n for n in nodes if n.id == meta["id"]]
    if existing:
        if len(existing) != 1 or not force:
            raise ValueError(f"{meta['id']} already exists (use --force for an intentional replacement)")
        old = existing[0]
        if old.meta["task"] != meta["task"]:
            raise ValueError("--force cannot change a node's task")
        meta["sequence"] = old.meta["sequence"]
        meta["recorded_at"] = old.meta["recorded_at"]
        return old.path
    meta["sequence"] = 1 + max((n.meta["sequence"] for n in nodes if n.meta["task"] == meta["task"]), default=0)
    meta["recorded_at"] = date.today().isoformat()
    meta.setdefault("papers", [])
    if meta["sequence"] > 999999:
        raise ValueError("task sequence exhausted")
    return nodes_dir() / meta["task"] / f"{meta['sequence']:06d}-{meta['id']}.md"


def write_node(path: Path, meta: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers see either the old or the complete new document.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(f"---\n{dump_frontmatter(meta)}---\n\n{body.rstrip()}\n", encoding="utf-8")
    temporary.replace(path)


def save_node(meta: dict[str, Any], body: str, force: bool = False) -> Path:
    with registration_lock():
        path = prepare_node(meta, force)
        write_node(path, meta, body)
    return path
