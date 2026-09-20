"""Shared helpers for the knowledge-control graph.

The knowledge graph is a set of Markdown files under ``knowledge/nodes/``: one
file per node. Each file starts with a YAML frontmatter block delimited by
``---`` lines, followed by a Markdown body (the human-readable 考察 / findings).

This module is intentionally dependency-light (only PyYAML) so that any provider
agent can run it from the repo root with ``.venv/bin/python``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from kg_schema import (
    ID_RE,
    NODE_TYPES,
    PROVIDERS,
    STATUSES,
    TASK_RE,
    has_prose,
    iso_date,
    json_value,
    nonempty_text,
)

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class UniqueLoader(yaml.SafeLoader):
    """Never silently discard duplicate keys or accept recursive YAML aliases."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported; write values explicitly")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict:
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in keys:
                raise ValueError(f"duplicate YAML key '{key}'")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def repo_root() -> Path:
    """Walk up from this file until a directory containing ``.git`` is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback: four levels up (.agents/skills/knowledge-control/scripts -> root)
    return here.parents[4]


def nodes_dir() -> Path:
    import os

    override = os.environ.get("KNOWLEDGE_DIR")
    base = Path(override) if override else repo_root() / "knowledge"
    return base / "nodes"


@dataclass
class Node:
    id: str
    type: str
    meta: dict[str, Any]
    body: str
    path: Path

    @property
    def title(self) -> str:
        return str(self.meta.get("title", self.id))

    @property
    def parents(self) -> list[str]:
        value = self.meta.get("parents")
        return [str(x) for x in value] if isinstance(value, list) else []

    @property
    def members(self) -> list[str]:
        value = self.meta.get("members")
        return [str(x) for x in value] if isinstance(value, list) else []

    @property
    def relations(self) -> list[dict[str, Any]]:
        rels = self.meta.get("relations") or []
        return [r for r in rels if isinstance(r, dict)] if isinstance(rels, list) else []


def parse_node(path: Path, *, text: str | None = None) -> Node:
    text = path.read_text(encoding="utf-8") if text is None else text
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path.name}: missing or malformed YAML frontmatter")
    try:
        meta = yaml.load(m.group(1), Loader=UniqueLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: frontmatter is not a mapping")
    body = m.group(2).strip()
    node_id = str(meta.get("id", ""))
    node_type = str(meta.get("type", ""))
    return Node(id=node_id, type=node_type, meta=meta, body=body, path=path)


def load_nodes(directory: Path | None = None) -> list[Node]:
    directory = directory or nodes_dir()
    if not directory.exists():
        return []
    nodes = [parse_node(p) for p in sorted(directory.rglob("*.md"))]
    return sorted(nodes, key=lambda n: (str(n.meta.get("task", "")), n.meta.get("sequence", 0) if type(n.meta.get("sequence")) is int else 0, n.id))


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(nodes: list[Node]) -> ValidationResult:
    res = ValidationResult()
    ids: dict[str, Path] = {}
    sequences: set[tuple[str, int]] = set()

    for n in nodes:
        loc = n.path.name
        if not isinstance(n.meta.get("id"), str) or not ID_RE.fullmatch(n.id):
            res.errors.append(f"{loc}: invalid id '{n.id}' (use lowercase a-z0-9-)")
        if n.id in ids:
            res.errors.append(f"{loc}: duplicate id '{n.id}' (also in {ids[n.id].name})")
        ids[n.id] = n.path
        task, sequence = n.meta.get("task"), n.meta.get("sequence")
        if not isinstance(task, str) or not TASK_RE.fullmatch(task):
            res.errors.append(f"{loc}: invalid or missing task")
        if type(sequence) is not int or not 1 <= sequence <= 999999:
            res.errors.append(f"{loc}: sequence must be an integer in 1..999999")
        else:
            key = (str(task), sequence)
            if key in sequences:
                res.errors.append(f"{loc}: duplicate task sequence {key}")
            sequences.add(key)
            if n.path != nodes_dir() / str(task) / f"{sequence:06d}-{n.id}.md":
                res.errors.append(f"{loc}: expected nodes/{task}/{sequence:06d}-{n.id}.md")
        if not n.id.startswith(f"{n.type}-"):
            res.errors.append(f"{loc}: id prefix must match type")
        for field_name in ("date", "recorded_at"):
            value = n.meta.get(field_name)
            if value is None and field_name == "date":
                continue
            if not iso_date(value):
                res.errors.append(f"{loc}: {field_name} must be an ISO date")
        for field_name in ("parents", "members", "tags", "papers"):
            value = n.meta.get(field_name, [])
            if not isinstance(value, list) or any(not nonempty_text(x) for x in value):
                res.errors.append(f"{loc}: {field_name} must be a list of nonempty strings")
            elif len(value) != len(set(value)):
                res.errors.append(f"{loc}: duplicate {field_name} entries")
        rels = n.meta.get("relations", [])
        if not isinstance(rels, list) or any(not isinstance(r, dict) for r in rels):
            res.errors.append(f"{loc}: relations must be a list of mappings")
        if n.type not in NODE_TYPES:
            res.errors.append(f"{loc}: type must be one of {sorted(NODE_TYPES)}, got '{n.type}'")
        if not nonempty_text(n.meta.get("title")):
            res.errors.append(f"{loc}: title must be a nonempty string")
        for field_name in ("config", "metrics", "repro", "artifacts"):
            value = n.meta.get(field_name, {})
            if not isinstance(value, dict) or not json_value(value):
                res.errors.append(f"{loc}: {field_name} must be a mapping of JSON-compatible values (no NaN/Infinity)")
        issue = n.meta.get("issue")
        if issue is not None:
            issues = issue if isinstance(issue, list) else [issue]
            if not issues or any(type(i) is not int or i <= 0 for i in issues):
                res.errors.append(f"{loc}: issue must be a positive integer or a nonempty list of positive integers")
        for field_name, allowed in (("status", STATUSES), ("provider", PROVIDERS)):
            value = n.meta.get(field_name)
            if value is not None and (not isinstance(value, str) or value not in allowed):
                res.errors.append(f"{loc}: {field_name} must be one of {sorted(allowed)}")
        source = n.meta.get("date_source")
        if source is not None and source not in ("experiment_date", "git_added"):
            res.errors.append(f"{loc}: date_source must be experiment_date or git_added")
        if source == "experiment_date" and n.meta.get("date") is None:
            res.errors.append(f"{loc}: experiment_date requires date")
        if not has_prose(n.body):
            res.errors.append(f"{loc}: write findings/summary text; headings and comments alone are unfinished")

        if n.type == "run":
            _validate_run(n, res)
        elif n.type == "group" and not n.members:
            res.errors.append(f"{loc}: group node has no 'members'")

    known = set(ids)
    for n in nodes:
        loc = n.path.name
        for p in n.parents:
            if p not in known:
                res.errors.append(f"{loc}: parent '{p}' does not exist")
            if p == n.id:
                res.errors.append(f"{loc}: self parent is not allowed")
        if n.type == "group":
            for mem in n.members:
                if mem not in known:
                    res.errors.append(f"{loc}: member '{mem}' does not exist")
                if mem == n.id:
                    res.errors.append(f"{loc}: self member is not allowed")
        if n.type == "run" and n.members:
            res.errors.append(f"{loc}: members are only allowed on group nodes")
        for rel in n.relations:
            to = str(rel.get("to", ""))
            if to not in known:
                res.errors.append(f"{loc}: relation target '{to}' does not exist")
            if not nonempty_text(rel.get("to")) or not nonempty_text(rel.get("rel")):
                res.errors.append(f"{loc}: relation requires nonempty string to and rel")
            if to == n.id:
                res.errors.append(f"{loc}: self relation is not allowed")
    for field_name in ("parents", "members"):
        _validate_acyclic(nodes, field_name, res)
    return res


def _validate_acyclic(nodes: list[Node], field_name: str, res: ValidationResult) -> None:
    # Kahn's algorithm avoids recursion limits on long experimental lineages.
    graph = {n.id: set(getattr(n, field_name)) for n in nodes}
    incoming = dict.fromkeys(graph, 0)
    for targets in graph.values():
        for target in targets & graph.keys():
            incoming[target] += 1
    ready = [key for key, count in incoming.items() if count == 0]
    seen = 0
    while ready:
        key = ready.pop()
        seen += 1
        for target in graph[key] & graph.keys():
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if seen != len(graph):
        res.errors.append(f"{field_name}: cycle detected; baseline ancestry and group nesting must be acyclic")


def _validate_run(n: Node, res: ValidationResult) -> None:
    loc = n.path.name
    if "issue" not in n.meta:
        res.warnings.append(f"{loc}: run node has no 'issue'")
    # issue #533: when a run records its git-tracked reproducibility bundle, that
    # directory must actually exist (it holds repro.sh / run.json / pred_test.npz).
    artifacts = n.meta.get("artifacts") or {}
    run_dir = artifacts.get("run_dir") if isinstance(artifacts, dict) else None
    if run_dir not in (None, ""):
        expected = (nodes_dir().parent / "runs" / n.id).resolve()
        if not isinstance(run_dir, str) or (repo_root() / str(run_dir)).resolve() != expected or not expected.is_dir():
            res.errors.append(f"{loc}: artifacts.run_dir must point to the existing runs/{n.id} directory")


def dump_frontmatter(meta: dict[str, Any]) -> str:
    """Serialize frontmatter with stable key ordering for nice git diffs."""
    order = [
        "id", "type", "task", "sequence", "recorded_at", "date_source", "title", "issue", "provider", "session", "date", "status",
        "config", "metrics", "repro", "artifacts", "members", "parents",
        "relations", "papers", "tags",
    ]
    ordered = {k: meta[k] for k in order if k in meta}
    for k, v in meta.items():
        if k not in ordered:
            ordered[k] = v
    return yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, default_flow_style=False)


def queue_dir() -> Path:
    """Honor the shared training queue even when invoked from a worktree."""
    import os
    import subprocess

    override = os.environ.get("TRAINING_QUEUE_DIR")
    if override:
        return Path(override).resolve()
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=repo_root(), text=True).strip()
    return (repo_root() / common).resolve().parent / ".training_queue"


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())
