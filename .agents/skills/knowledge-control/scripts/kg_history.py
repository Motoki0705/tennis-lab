"""Read the Git base of a knowledge change without checking out another tree."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from kg_lib import Node, git_toplevel, nodes_dir, parse_node
from kg_storage import read_counter

# Library files (relative to the knowledge root) whose base text change rules compare.
TRACKED_FILES = ("summary.md", "registry/deployments.yaml")


@dataclass
class BaseSnapshot:
    """Node identities, allocator state and selected files at the base commit."""

    ref: str
    sha: str
    counters: dict[str, int] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    files: dict[str, str | None] = field(default_factory=dict)


def load_base(base_ref: str) -> BaseSnapshot:
    # Compare with the history of the checkout holding the library (KNOWLEDGE_DIR may be another clone).
    root = git_toplevel(nodes_dir().parent)

    def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        if check and proc.returncode:
            raise ValueError(f"cannot read history base {base_ref}: {proc.stderr.strip()}")
        return proc

    sha = git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}").stdout.strip()
    snapshot = BaseSnapshot(ref=base_ref, sha=sha)
    prefix = nodes_dir().resolve().relative_to(root)
    paths = git("ls-tree", "-r", "--name-only", "-z", sha, "--", str(prefix)).stdout.split("\0")
    for raw in filter(None, paths):
        path = Path(raw)
        relative = path.relative_to(prefix)
        # Flat pre-migration nodes have no allocation contract to compare.
        if len(relative.parts) != 2:
            continue
        if path.name == ".sequence":
            snapshot.counters[path.parent.name] = int(git("show", f"{sha}:{raw}").stdout.strip())
        elif path.suffix == ".md":
            node = parse_node(root / path, text=git("show", f"{sha}:{raw}").stdout)
            snapshot.nodes[node.id] = node
    library = prefix.parent
    for name in TRACKED_FILES:
        spec = f"{sha}:{(library / name).as_posix()}"
        exists = git("cat-file", "-e", spec, check=False).returncode == 0
        snapshot.files[name] = git("show", spec).stdout if exists else None
    return snapshot


def validate_identity(nodes: list[Node], base: BaseSnapshot) -> list[str]:
    errors = []
    for task, before in base.counters.items():
        directory = nodes_dir() / task
        if not directory.is_dir() or read_counter(directory) < before:
            errors.append(f"{task}: .sequence must retain at least base allocation {before}, even after deleting nodes")
    for node in nodes:
        before_node = base.nodes.get(node.id)
        if before_node:
            for key in ("task", "sequence", "recorded_at"):
                before_value, after_value = before_node.meta.get(key), node.meta.get(key)
                # SafeLoader parses unquoted dates as date objects; quoting an
                # unchanged calendar day is a formatting change, not a new identity.
                if key == "recorded_at":
                    before_value, after_value = str(before_value), str(after_value)
                if before_value != after_value:
                    errors.append(f"{node.id}: {key} is immutable relative to {base.ref}")
        else:
            node_task, sequence = node.meta.get("task"), node.meta.get("sequence")
            if isinstance(node_task, str) and type(sequence) is int and sequence <= base.counters.get(node_task, 0):
                errors.append(f"{node.id}: new sequence must exceed base allocation {base.counters[node_task]} for {node_task}")
    return errors


def validate_history(nodes: list[Node], base_ref: str) -> list[str]:
    return validate_identity(nodes, load_base(base_ref))
