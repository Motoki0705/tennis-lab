#!/usr/bin/env python
"""Migrate legacy flat nodes once; dry-run by default. Bundles stay byte-identical."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from kg_lib import nodes_dir, parse_node, repo_root


def classify(node_id: str, tags: list[str]) -> str:
    """Initial repository taxonomy only; --task-map handles other collections."""
    if node_id == "group-i593-physics-prior":
        return "blcs"
    if node_id.startswith("run-tennis-scene-"):
        return "tennis_scene"
    if node_id.startswith("run-i618-b00-") or node_id.startswith("run-i618-blcs-b00-") or node_id == "run-i618-renderer-port-cpu-reference-v1" or node_id == "run-i878-b00-rendered-line-projection":
        return "synthetic_data_generation"
    normalized = {t.replace("-", "_") for t in tags}
    for task in ("slcs", "ball_detection", "court_detection", "plcs", "blcs"):
        if task in normalized:
            return task
    if "court" in normalized or "court" in node_id:
        return "court_detection"
    for task in ("plcs", "blcs"):
        if task in node_id:
            return task
    raise ValueError(f"No unambiguous task for {node_id}; supply --task-map JSON")


def migrate(write: bool, overrides: dict[str, str]) -> dict[Path, Path]:
    base, root = nodes_dir(), repo_root()
    nodes = [parse_node(p) for p in base.glob("*.md")]
    if not nodes:
        return {}
    if any(base.glob("*/*.md")):
        raise ValueError("mixed legacy/task layout; finish or revert the partial migration first")
    records = []
    for n in nodes:
        added = []
        if not n.meta.get("date"):
            added = subprocess.check_output(["git", "log", "--follow", "--diff-filter=A", "--format=%aI", "--", str(n.path)], cwd=root, text=True).splitlines()
        day = str(n.meta.get("date") or (added[-1][:10] if added else ""))
        if not day:
            raise ValueError(f"{n.id}: no date or Git creation date")
        task = overrides.get(n.id) or classify(n.id, n.meta.get("tags", []))
        if not re.fullmatch(r"[a-z][a-z0-9_]*", task):
            raise ValueError(f"invalid task {task}")
        records.append((task, day, n.id, n))
    counts: dict[str, int] = {}
    mapping: dict[Path, Path] = {}
    text_by_path: dict[Path, str] = {}
    for task, day, _, n in sorted(records, key=lambda r: r[:3]):
        counts[task] = counts.get(task, 0) + 1
        seq = counts[task]
        target = base / task / f"{seq:06d}-{n.id}.md"
        mapping[n.path.resolve()] = target.resolve()
        source = "experiment_date" if n.meta.get("date") else "git_added"
        addition = f"task: {task}\nsequence: {seq}\nrecorded_at: {day}\ndate_source: {source}\npapers: []\n"
        text_by_path[n.path.resolve()] = n.path.read_text().replace("---\n", "---\n" + addition, 1)
    # Rewrite maintained Markdown links relative to their NEW document location.
    # Repro snapshots are historical evidence and must never be rewritten.
    tracked = subprocess.check_output(["git", "ls-files", "-z", "*.md"], cwd=root).decode().split("\0")
    for rel in filter(None, tracked):
        if rel.startswith(("knowledge/runs/", "third_party/")):
            continue
        old = (root / rel).resolve()
        if not old.is_file():
            continue
        new = mapping.get(old, old)
        original = text_by_path.get(old, old.read_text())
        def replace(match: re.Match[str], old: Path = old, new: Path = new) -> str:
            raw = match.group(1)
            if re.match(r"[a-z]+:|/|#", raw):
                return match.group(0)
            url, sep, anchor = raw.partition("#")
            dest = (old.parent / url).resolve()
            if dest not in mapping and old == new:
                return match.group(0)
            if not dest.exists() and dest not in mapping:
                return match.group(0)
            target = mapping.get(dest, dest)
            return "](" + os.path.relpath(target, new.parent) + (sep + anchor if sep else "") + ")"
        updated = re.sub(r"\]\(([^\s)]+)\)", replace, original)
        if write and (updated != old.read_text() or old != new):
            new.parent.mkdir(parents=True, exist_ok=True)
            new.write_text(updated)
            if old != new:
                old.unlink()
    if write:
        for task, count in counts.items():
            (base / task / ".sequence").write_text(f"{count}\n")
    print(json.dumps({"nodes": len(nodes), "tasks": counts, "write": write}, ensure_ascii=False))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--task-map", type=Path, help="JSON mapping of node id to explicit task")
    args = parser.parse_args()
    migrate(args.write, json.loads(args.task_map.read_text()) if args.task_map else {})


if __name__ == "__main__":
    main()
