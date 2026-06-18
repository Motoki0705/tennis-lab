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

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
NODE_TYPES = {"run", "group"}
STATUSES = {"done", "failed", "running", "planned"}
PROVIDERS = {"claude", "codex", "gemini", "human", "other"}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


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
        return [str(x) for x in (self.meta.get("parents") or [])]

    @property
    def members(self) -> list[str]:
        return [str(x) for x in (self.meta.get("members") or [])]

    @property
    def relations(self) -> list[dict[str, Any]]:
        rels = self.meta.get("relations") or []
        return [r for r in rels if isinstance(r, dict)]


def parse_node(path: Path) -> Node:
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"{path.name}: missing or malformed YAML frontmatter")
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: frontmatter is not a mapping")
    body = m.group(2).strip()
    node_id = str(meta.get("id", path.stem))
    node_type = str(meta.get("type", ""))
    return Node(id=node_id, type=node_type, meta=meta, body=body, path=path)


def load_nodes(directory: Path | None = None) -> list[Node]:
    directory = directory or nodes_dir()
    if not directory.exists():
        return []
    nodes = [parse_node(p) for p in sorted(directory.glob("*.md"))]
    return nodes


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

    for n in nodes:
        loc = n.path.name
        if not ID_RE.match(n.id):
            res.errors.append(f"{loc}: invalid id '{n.id}' (use lowercase a-z0-9-)")
        if n.id in ids:
            res.errors.append(f"{loc}: duplicate id '{n.id}' (also in {ids[n.id].name})")
        ids[n.id] = n.path
        if n.path.stem != n.id:
            res.warnings.append(f"{loc}: filename does not match id '{n.id}'")
        if n.type not in NODE_TYPES:
            res.errors.append(f"{loc}: type must be one of {sorted(NODE_TYPES)}, got '{n.type}'")
        if not n.meta.get("title"):
            res.errors.append(f"{loc}: missing 'title'")

        if n.type == "run":
            _validate_run(n, res)
        elif n.type == "group":
            if not n.members:
                res.errors.append(f"{loc}: group node has no 'members'")

    known = set(ids)
    for n in nodes:
        loc = n.path.name
        for p in n.parents:
            if p not in known:
                res.errors.append(f"{loc}: parent '{p}' does not exist")
        if n.type == "group":
            for mem in n.members:
                if mem not in known:
                    res.errors.append(f"{loc}: member '{mem}' does not exist")
        for rel in n.relations:
            to = str(rel.get("to", ""))
            if to not in known:
                res.errors.append(f"{loc}: relation target '{to}' does not exist")
            if not rel.get("rel"):
                res.warnings.append(f"{loc}: relation to '{to}' has no 'rel' label")
    return res


def _validate_run(n: Node, res: ValidationResult) -> None:
    loc = n.path.name
    if "issue" not in n.meta:
        res.warnings.append(f"{loc}: run node has no 'issue'")
    status = n.meta.get("status")
    if status is not None and str(status) not in STATUSES:
        res.errors.append(f"{loc}: status must be one of {sorted(STATUSES)}, got '{status}'")
    provider = n.meta.get("provider")
    if provider is not None and str(provider) not in PROVIDERS:
        res.warnings.append(f"{loc}: provider '{provider}' not in {sorted(PROVIDERS)}")
    if not n.body:
        res.warnings.append(f"{loc}: empty body (no 考察/findings written yet)")


def dump_frontmatter(meta: dict[str, Any]) -> str:
    """Serialize frontmatter with stable key ordering for nice git diffs."""
    order = [
        "id", "type", "title", "issue", "provider", "date", "status",
        "config", "metrics", "artifacts", "members", "parents", "relations", "tags",
    ]
    ordered = {k: meta[k] for k in order if k in meta}
    for k, v in meta.items():
        if k not in ordered:
            ordered[k] = v
    return yaml.safe_dump(ordered, allow_unicode=True, sort_keys=False, default_flow_style=False)
