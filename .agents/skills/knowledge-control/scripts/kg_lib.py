"""Shared helpers for the git-managed knowledge graph.

Formal graph nodes live under ``knowledge/nodes/``.  The graph contains both
internal experimental evidence (``run`` / ``group``) and reviewed literature
knowledge (``paper`` / ``proposal``).  Raw hourly literature discoveries are
not graph nodes; they are validated separately by the literature-radar skill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
NODE_TYPES = {"run", "group", "paper", "proposal"}
RUN_STATUSES = {"done", "failed", "running", "planned"}
PAPER_STATUSES = {"reviewed", "superseded", "withdrawn"}
PROPOSAL_STATUSES = {
    "candidate",
    "ready",
    "issue-open",
    "testing",
    "supported",
    "refuted",
    "inconclusive",
    "adopted",
}
PROPOSAL_EVIDENCE_STATUSES = {"supported", "refuted", "inconclusive", "adopted"}
EVIDENCE_LEVELS = {"abstract", "fulltext", "fulltext-code", "fulltext-code-data"}
PAPER_EVIDENCE_LEVELS = EVIDENCE_LEVELS - {"abstract"}
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


def knowledge_root() -> Path:
    import os

    override = os.environ.get("KNOWLEDGE_DIR")
    return Path(override) if override else repo_root() / "knowledge"


def nodes_dir() -> Path:
    return knowledge_root() / "nodes"


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
        return _as_string_list(self.meta.get("parents"))

    @property
    def members(self) -> list[str]:
        return _as_string_list(self.meta.get("members"))

    @property
    def relations(self) -> list[dict[str, Any]]:
        rels = self.meta.get("relations") or []
        return [r for r in rels if isinstance(r, dict)]

    @property
    def evidence_runs(self) -> list[str]:
        return _as_string_list(self.meta.get("evidence_runs"))


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def parse_node(path: Path) -> Node:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{path.name}: missing or malformed YAML frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path.name}: frontmatter is not a mapping")
    body = match.group(2).strip()
    node_id = str(meta.get("id", path.stem))
    node_type = str(meta.get("type", ""))
    return Node(id=node_id, type=node_type, meta=meta, body=body, path=path)


def load_nodes(directory: Path | None = None) -> list[Node]:
    directory = directory or nodes_dir()
    if not directory.exists():
        return []
    return [parse_node(path) for path in sorted(directory.glob("*.md"))]


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

    for node in nodes:
        loc = node.path.name
        if not ID_RE.match(node.id):
            res.errors.append(f"{loc}: invalid id '{node.id}' (use lowercase a-z0-9-)")
        if node.id in ids:
            res.errors.append(
                f"{loc}: duplicate id '{node.id}' (also in {ids[node.id].name})"
            )
        ids[node.id] = node.path
        if node.path.stem != node.id:
            res.warnings.append(f"{loc}: filename does not match id '{node.id}'")
        if node.type not in NODE_TYPES:
            res.errors.append(
                f"{loc}: type must be one of {sorted(NODE_TYPES)}, got '{node.type}'"
            )
        if not node.meta.get("title"):
            res.errors.append(f"{loc}: missing 'title'")

        if node.type == "run":
            _validate_run(node, res)
        elif node.type == "group":
            _validate_group(node, res)
        elif node.type == "paper":
            _validate_paper(node, res)
        elif node.type == "proposal":
            _validate_proposal(node, res)

    known = set(ids)
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        loc = node.path.name
        for parent in node.parents:
            if parent not in known:
                res.errors.append(f"{loc}: parent '{parent}' does not exist")
        if node.type == "group":
            for member in node.members:
                if member not in known:
                    res.errors.append(f"{loc}: member '{member}' does not exist")
        for relation in node.relations:
            target = str(relation.get("to", ""))
            if target not in known:
                res.errors.append(f"{loc}: relation target '{target}' does not exist")
            if not relation.get("rel"):
                res.warnings.append(f"{loc}: relation to '{target}' has no 'rel' label")

        if node.type == "proposal":
            _validate_proposal_links(node, by_id, res)

    return res


def _validate_run(node: Node, res: ValidationResult) -> None:
    loc = node.path.name
    if "issue" not in node.meta:
        res.warnings.append(f"{loc}: run node has no 'issue'")
    status = node.meta.get("status")
    if status is not None and str(status) not in RUN_STATUSES:
        res.errors.append(
            f"{loc}: run status must be one of {sorted(RUN_STATUSES)}, got '{status}'"
        )
    provider = node.meta.get("provider")
    if provider is not None and str(provider) not in PROVIDERS:
        res.warnings.append(f"{loc}: provider '{provider}' not in {sorted(PROVIDERS)}")

    artifacts = node.meta.get("artifacts") or {}
    run_dir = artifacts.get("run_dir") if isinstance(artifacts, dict) else None
    if run_dir and not (repo_root() / str(run_dir)).exists():
        res.errors.append(f"{loc}: artifacts.run_dir '{run_dir}' does not exist")
    if not node.body:
        res.warnings.append(f"{loc}: empty body (no 考察/findings written yet)")


def _validate_group(node: Node, res: ValidationResult) -> None:
    if not node.members:
        res.errors.append(f"{node.path.name}: group node has no 'members'")


def _validate_paper(node: Node, res: ValidationResult) -> None:
    loc = node.path.name
    status = str(node.meta.get("status", ""))
    if status not in PAPER_STATUSES:
        res.errors.append(
            f"{loc}: paper status must be one of {sorted(PAPER_STATUSES)}, got '{status}'"
        )

    external_ids = node.meta.get("external_ids")
    has_external_id = isinstance(external_ids, dict) and any(
        str(external_ids.get(key, "")).strip()
        for key in ("doi", "arxiv", "openreview")
    )
    if not has_external_id and not node.id.startswith("paper-title-"):
        res.errors.append(
            f"{loc}: paper requires an external id unless it uses a paper-title-* id"
        )

    evidence_level = str(node.meta.get("evidence_level", ""))
    if evidence_level not in PAPER_EVIDENCE_LEVELS:
        res.errors.append(
            f"{loc}: paper evidence_level must be one of "
            f"{sorted(PAPER_EVIDENCE_LEVELS)}, "
            f"got '{evidence_level}'"
        )

    _require_nonempty_string_list(node, "tasks", res)
    _validate_repo_paths(node, res)

    sources = node.meta.get("sources")
    if not isinstance(sources, list) or not sources:
        res.errors.append(f"{loc}: paper requires a non-empty 'sources' list")
    elif not any(
        isinstance(source, dict) and str(source.get("url", "")).startswith("http")
        for source in sources
    ):
        res.errors.append(f"{loc}: paper sources must contain at least one URL")

    if not node.body:
        res.errors.append(f"{loc}: paper review body must not be empty")


def _validate_proposal(node: Node, res: ValidationResult) -> None:
    loc = node.path.name
    status = str(node.meta.get("status", ""))
    if status not in PROPOSAL_STATUSES:
        res.errors.append(
            f"{loc}: proposal status must be one of {sorted(PROPOSAL_STATUSES)}, "
            f"got '{status}'"
        )

    if not str(node.meta.get("task", "")).strip():
        res.errors.append(f"{loc}: proposal requires 'task'")
    _validate_repo_paths(node, res)

    hypothesis = node.meta.get("hypothesis")
    if not isinstance(hypothesis, dict):
        res.errors.append(f"{loc}: proposal requires a 'hypothesis' mapping")
    else:
        for key in ("statement", "expected_effect", "failure_condition"):
            if not str(hypothesis.get(key, "")).strip():
                res.errors.append(f"{loc}: hypothesis.{key} is required")

    evaluation = node.meta.get("evaluation")
    if not isinstance(evaluation, dict):
        res.errors.append(f"{loc}: proposal requires an 'evaluation' mapping")
    else:
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            res.errors.append(f"{loc}: evaluation.metrics must be a non-empty list")
        if not str(evaluation.get("acceptance", "")).strip():
            res.errors.append(f"{loc}: evaluation.acceptance is required")
        seeds = evaluation.get("seeds")
        if not isinstance(seeds, int) or seeds < 1:
            res.errors.append(f"{loc}: evaluation.seeds must be an integer >= 1")
        baseline_nodes = evaluation.get("baseline_nodes")
        if status != "candidate" and (
            not isinstance(baseline_nodes, list) or not baseline_nodes
        ):
            res.errors.append(
                f"{loc}: non-candidate proposal requires evaluation.baseline_nodes"
            )

    if status in {"issue-open", "testing", *PROPOSAL_EVIDENCE_STATUSES} and "issue" not in node.meta:
        res.errors.append(f"{loc}: proposal status '{status}' requires 'issue'")
    if status in PROPOSAL_EVIDENCE_STATUSES and not node.evidence_runs:
        res.errors.append(
            f"{loc}: proposal status '{status}' requires non-empty 'evidence_runs'"
        )
    if not node.body:
        res.errors.append(f"{loc}: proposal body must not be empty")


def _validate_proposal_links(
    node: Node, by_id: dict[str, Node], res: ValidationResult
) -> None:
    loc = node.path.name
    derived_from = [
        str(relation.get("to", ""))
        for relation in node.relations
        if relation.get("rel") == "derived-from"
    ]
    if not derived_from:
        res.errors.append(f"{loc}: proposal requires a 'derived-from' paper relation")
    for target in derived_from:
        target_node = by_id.get(target)
        if target_node is not None and target_node.type != "paper":
            res.errors.append(
                f"{loc}: derived-from target '{target}' must be a paper node"
            )

    for run_id in node.evidence_runs:
        target_node = by_id.get(run_id)
        if target_node is None:
            res.errors.append(f"{loc}: evidence run '{run_id}' does not exist")
        elif target_node.type != "run":
            res.errors.append(f"{loc}: evidence run '{run_id}' must be a run node")

    evaluation = node.meta.get("evaluation")
    if isinstance(evaluation, dict):
        baseline_nodes = evaluation.get("baseline_nodes")
        if isinstance(baseline_nodes, list):
            for baseline_id in baseline_nodes:
                baseline = str(baseline_id)
                if baseline not in by_id:
                    res.errors.append(
                        f"{loc}: baseline node '{baseline}' does not exist"
                    )


def _require_nonempty_string_list(
    node: Node, key: str, res: ValidationResult
) -> None:
    value = node.meta.get(key)
    if not isinstance(value, list) or not any(str(item).strip() for item in value):
        res.errors.append(f"{node.path.name}: '{key}' must be a non-empty list")


def _validate_repo_paths(node: Node, res: ValidationResult) -> None:
    value = node.meta.get("repo_paths")
    if not isinstance(value, list) or not value:
        res.errors.append(f"{node.path.name}: 'repo_paths' must be a non-empty list")
        return
    for item in value:
        path = Path(str(item))
        if path.is_absolute() or ".." in path.parts:
            res.errors.append(
                f"{node.path.name}: repo path '{item}' must be repository-relative"
            )


def dump_frontmatter(meta: dict[str, Any]) -> str:
    """Serialize frontmatter with stable key ordering for reviewable diffs."""
    order = [
        "id",
        "type",
        "title",
        "issue",
        "provider",
        "curator",
        "session",
        "date",
        "status",
        "external_ids",
        "published_at",
        "reviewed_at",
        "evidence_level",
        "task",
        "tasks",
        "repo_paths",
        "sources",
        "hypothesis",
        "evaluation",
        "evidence_runs",
        "config",
        "metrics",
        "repro",
        "artifacts",
        "members",
        "parents",
        "relations",
        "tags",
    ]
    ordered = {key: meta[key] for key in order if key in meta}
    for key, value in meta.items():
        if key not in ordered:
            ordered[key] = value
    return yaml.safe_dump(
        ordered,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
