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

from kg_lib import Node, dump_frontmatter, load_nodes, nodes_dir
from kg_schema import ID_RE, TASK_RE


def check_identity(meta: dict[str, Any]) -> None:
    if not TASK_RE.fullmatch(str(meta.get("task", ""))):
        raise ValueError("task is required: lowercase letters, digits and underscores")
    if not ID_RE.fullmatch(str(meta.get("id", ""))):
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


def read_counter(directory: Path) -> int:
    """The allocator state survives node deletion; never reconstruct it silently."""
    if not directory.exists():
        return 0
    path = directory / ".sequence"
    if not path.is_file():
        raise ValueError(f"{directory.name}: missing .sequence allocator state")
    raw = path.read_text().strip()
    if not re.fullmatch(r"[1-9][0-9]{0,5}", raw):
        raise ValueError(f"{directory.name}: invalid .sequence allocator state")
    return int(raw)


def validate_counters(nodes: list[Node]) -> list[str]:
    errors = []
    for directory in sorted(nodes_dir().glob("*")):
        if not directory.is_dir():
            continue
        if not TASK_RE.fullmatch(directory.name):
            errors.append(f"{directory}: invalid task directory")
        try:
            counter = read_counter(directory)
            maximum = max((n.meta["sequence"] for n in nodes if n.meta.get("task") == directory.name and type(n.meta.get("sequence")) is int), default=0)
            if counter < maximum:
                errors.append(f"{directory.name}: .sequence {counter} is below existing node {maximum}")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


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
    directory = nodes_dir() / meta["task"]
    counter = read_counter(directory)
    maximum = max((n.meta["sequence"] for n in nodes if n.meta["task"] == meta["task"]), default=0)
    if counter < maximum:
        raise ValueError(f"{meta['task']}: .sequence is below existing nodes; reconcile allocator state")
    meta["sequence"] = counter + 1
    meta["recorded_at"] = date.today().isoformat()
    meta.setdefault("papers", [])
    if meta["sequence"] > 999999:
        raise ValueError("task sequence exhausted")
    # Reserve before writing the node. An interrupted write leaves a gap, never
    # a reused number. This state is git-managed along with the new node.
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".sequence.tmp"
    temporary.write_text(f"{meta['sequence']}\n")
    temporary.replace(directory / ".sequence")
    return directory / f"{meta['sequence']:06d}-{meta['id']}.md"


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
