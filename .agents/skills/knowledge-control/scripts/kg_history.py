"""Compare allocator history with a Git base without checking out another tree."""
from __future__ import annotations

import subprocess
from pathlib import Path

from kg_lib import Node, nodes_dir, parse_node, repo_root
from kg_storage import read_counter


def validate_history(nodes: list[Node], base_ref: str) -> list[str]:
    root = repo_root()

    def git(*args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        if proc.returncode:
            raise ValueError(f"cannot read history base {base_ref}: {proc.stderr.strip()}")
        return proc.stdout

    sha = git("rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}").strip()
    prefix = nodes_dir().relative_to(root)
    paths = git("ls-tree", "-r", "--name-only", "-z", sha, "--", str(prefix)).split("\0")
    counters: dict[str, int] = {}
    previous: dict[str, Node] = {}
    for raw in filter(None, paths):
        path = Path(raw)
        relative = path.relative_to(prefix)
        # Flat pre-migration nodes have no allocation contract to compare.
        if len(relative.parts) != 2:
            continue
        if path.name == ".sequence":
            counters[path.parent.name] = int(git("show", f"{sha}:{raw}").strip())
        elif path.suffix == ".md":
            node = parse_node(root / path, text=git("show", f"{sha}:{raw}"))
            previous[node.id] = node

    errors = []
    for task, before in counters.items():
        directory = nodes_dir() / task
        if not directory.is_dir() or read_counter(directory) < before:
            errors.append(f"{task}: .sequence must retain at least base allocation {before}, even after deleting nodes")
    for node in nodes:
        before_node = previous.get(node.id)
        if before_node:
            for key in ("task", "sequence", "recorded_at"):
                before_value, after_value = before_node.meta.get(key), node.meta.get(key)
                # SafeLoader parses unquoted dates as date objects; quoting an
                # unchanged calendar day is a formatting change, not a new identity.
                if key == "recorded_at":
                    before_value, after_value = str(before_value), str(after_value)
                if before_value != after_value:
                    errors.append(f"{node.id}: {key} is immutable relative to {base_ref}")
        else:
            task, sequence = node.meta.get("task"), node.meta.get("sequence")
            if isinstance(task, str) and type(sequence) is int and sequence <= counters.get(task, 0):
                errors.append(f"{node.id}: new sequence must exceed base allocation {counters[task]} for {task}")
    return errors
