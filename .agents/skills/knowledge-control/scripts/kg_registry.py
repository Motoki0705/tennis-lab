"""Controlled vocabularies of the library: tasks, relation types and deployments.

The registry lives in ``knowledge/registry/`` (``KNOWLEDGE_DIR`` aware). Every
file is required: an absent or malformed registry is an error, never an empty
vocabulary. Paths inside it are relative to the directory holding the library. The deployment registry ties each checkpoint selected by a tracked
config (for example the standard scene pipeline) to the nodes or source document
that justify it, so a changed checkpoint cannot leave the library stale.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from kg_lib import Node, UniqueLoader, nodes_dir
from kg_schema import TASK_RE, iso_date, nonempty_text

SCHEMA_VERSION = 1
TASK_STATUSES = {"active", "retired"}
TASK_FIELDS = {"status", "module", "note", "successor", "since"}
RELATION_RE = re.compile(r"^[a-z][a-z_]*$")
RELATION_FIELDS = {"description", "note_required"}
NODE_RELATION_FIELDS = {"to", "rel", "note"}
DEPLOYMENT_FIELDS = {"config", "key", "value", "task", "evidence", "papers", "source", "since", "note"}
CHECKPOINT_KEY = re.compile(r"(?:^|_)checkpoint$")


def registry_dir() -> Path:
    return nodes_dir().parent / "registry"


def _load(name: str, keys: set[str]) -> dict[str, Any]:
    path = registry_dir() / name
    if not path.is_file():
        raise ValueError(f"{path}: missing registry file (see knowledge/README.md#レジストリ)")
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION or set(data) != keys:
        raise ValueError(f"{path}: expected exactly the keys {sorted(keys)} with schema_version {SCHEMA_VERSION}")
    return data


def project_root() -> Path:
    """Registry paths are relative to the directory holding the library (the repository root)."""
    return nodes_dir().parent.parent


def _repo_file(value: Any) -> Path | None:
    """Resolve a project-relative path without leaving the project."""
    if not nonempty_text(value) or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        return None
    path = project_root() / value
    return path if path.exists() else None


def load_tasks() -> tuple[dict[str, dict[str, Any]], list[str]]:
    tasks = _load("tasks.yaml", {"schema_version", "tasks"})["tasks"]
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("registry/tasks.yaml: tasks must be a nonempty mapping")
    errors = []
    for name, entry in tasks.items():
        loc = f"registry/tasks.yaml: {name}"
        if not TASK_RE.fullmatch(name):
            errors.append(f"{loc}: task names are lowercase snake_case")
        if not isinstance(entry, dict) or not set(entry) <= TASK_FIELDS:
            errors.append(f"{loc}: entry must be a mapping of {sorted(TASK_FIELDS)}")
            continue
        status, module = entry.get("status"), entry.get("module")
        if status not in TASK_STATUSES:
            errors.append(f"{loc}: status must be one of {sorted(TASK_STATUSES)}")
        if module is not None and _repo_file(module) is None:
            errors.append(f"{loc}: module must be an existing repository-relative path")
        if (module is None or status == "retired") and not nonempty_text(entry.get("note")):
            errors.append(f"{loc}: a retired task or a task without module needs a note")
        if "note" in entry and not nonempty_text(entry["note"]):
            errors.append(f"{loc}: note must be nonempty text")
        successor = entry.get("successor")
        if successor is not None and (status != "retired" or successor == name or successor not in tasks):
            errors.append(f"{loc}: successor must name another registered task of a retired task")
        if "since" in entry and not iso_date(entry["since"]):
            errors.append(f"{loc}: since must be an ISO date")
    return tasks, errors


def load_relations() -> tuple[dict[str, dict[str, Any]], list[str]]:
    relations = _load("relations.yaml", {"schema_version", "relations"})["relations"]
    if not isinstance(relations, dict) or not relations:
        raise ValueError("registry/relations.yaml: relations must be a nonempty mapping")
    errors = []
    for name, entry in relations.items():
        loc = f"registry/relations.yaml: {name}"
        if not RELATION_RE.fullmatch(name):
            errors.append(f"{loc}: relation names are lowercase snake_case")
        if not isinstance(entry, dict) or not set(entry) <= RELATION_FIELDS or not nonempty_text(entry.get("description")):
            errors.append(f"{loc}: entry needs a description and may set note_required")
        elif "note_required" in entry and type(entry["note_required"]) is not bool:
            errors.append(f"{loc}: note_required must be true or false")
    return relations, errors


def config_checkpoints(config: Path) -> dict[str, str]:
    """Dotted keys named ``checkpoint`` / ``*_checkpoint`` with string values."""
    data = yaml.safe_load(config.read_text(encoding="utf-8"))

    def walk(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
        if isinstance(value, dict):
            for key, child in value.items():
                dotted = f"{prefix}.{key}" if prefix else str(key)
                yield dotted, child
                yield from walk(child, dotted)

    return {key: value for key, value in walk(data, "") if CHECKPOINT_KEY.search(key.rsplit(".", 1)[-1]) and isinstance(value, str)}


def validate_deployments(known_nodes: set[str], known_papers: set[str], tasks: dict[str, dict[str, Any]]) -> list[str]:
    data = _load("deployments.yaml", {"schema_version", "configs", "deployments"})
    configs, entries = data["configs"], data["deployments"]
    errors: list[str] = []
    if not isinstance(configs, list) or any(not nonempty_text(c) for c in configs) or len(set(configs)) != len(configs):
        raise ValueError("registry/deployments.yaml: configs must be a list of unique repository-relative paths")
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError("registry/deployments.yaml: deployments must be a list of mappings")
    discovered: dict[tuple[str, str], str] = {}
    for config in configs:
        path = _repo_file(config)
        if path is None or not path.is_file():
            errors.append(f"registry/deployments.yaml: tracked config {config} does not exist")
            continue
        discovered.update({(config, key): value for key, value in config_checkpoints(path).items()})
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (str(entry.get("config")), str(entry.get("key")))
        loc = f"registry/deployments.yaml: {key[0]} {key[1]}"
        if not set(entry) <= DEPLOYMENT_FIELDS or not all(nonempty_text(entry.get(f)) for f in ("config", "key", "value", "task")):
            errors.append(f"{loc}: entry needs config/key/value/task and may set {sorted(DEPLOYMENT_FIELDS - {'config', 'key', 'value', 'task'})}")
            continue
        if key in seen:
            errors.append(f"{loc}: duplicate deployment entry")
        seen.add(key)
        if key[0] not in configs:
            errors.append(f"{loc}: config is not listed in configs")
        elif key not in discovered:
            errors.append(f"{loc}: the config has no such checkpoint key; remove or move the entry")
        elif discovered[key] != entry["value"]:
            errors.append(f"{loc}: config selects '{discovered[key]}' but the registry records '{entry['value']}'; update the entry, its evidence and summary.md")
        if entry["task"] not in tasks:
            errors.append(f"{loc}: task '{entry['task']}' is not registered")
        evidence, source = entry.get("evidence", []), entry.get("source")
        if not isinstance(evidence, list) or any(not nonempty_text(e) for e in evidence) or len(set(evidence)) != len(evidence):
            errors.append(f"{loc}: evidence must be a list of unique node IDs")
            evidence = []
        errors.extend(f"{loc}: evidence node '{e}' does not exist" for e in evidence if e not in known_nodes)
        papers = entry.get("papers", [])
        if not isinstance(papers, list) or any(p not in known_papers for p in papers) or len(set(papers)) != len(papers):
            errors.append(f"{loc}: papers must be a list of unique registered paper IDs")
        if source is not None:
            document = _repo_file(source)
            if document is None or not document.is_file():
                errors.append(f"{loc}: source must be an existing repository-relative document")
            elif PurePosixPath(entry["value"]).name not in document.read_text(encoding="utf-8"):
                errors.append(f"{loc}: source {source} does not mention {PurePosixPath(entry['value']).name}")
        if not evidence and source is None:
            errors.append(f"{loc}: cite evidence nodes, or a source document for external weights")
        if "since" in entry and not iso_date(entry["since"]):
            errors.append(f"{loc}: since must be an ISO date")
        if "note" in entry and not nonempty_text(entry["note"]):
            errors.append(f"{loc}: note must be nonempty text")
    for missing in sorted(discovered.keys() - seen):
        errors.append(f"registry/deployments.yaml: {missing[0]} {missing[1]} selects '{discovered[missing]}' without a deployment entry")
    return errors


def validate_registry(nodes: list[Node], papers: list[Node]) -> list[str]:
    """Check registries and every node/paper reference to them."""
    tasks, errors = load_tasks()
    for directory in sorted(p for p in nodes_dir().iterdir() if p.is_dir()):
        if directory.name not in tasks:
            errors.append(f"nodes/{directory.name}: task is not registered in registry/tasks.yaml")
    for paper in papers:
        values = paper.meta.get("tasks")
        for task in values if isinstance(values, list) else []:
            if isinstance(task, str) and task not in tasks:
                errors.append(f"{paper.path.parent.name}: paper task '{task}' is not registered in registry/tasks.yaml")
    relations, relation_errors = load_relations()
    errors.extend(relation_errors)
    allowed = ", ".join(sorted(relations))
    for node in nodes:
        for relation in node.relations:
            loc = f"{node.path.name}: relation to '{relation.get('to')}'"
            rel, note = relation.get("rel"), relation.get("note")
            if not set(relation) <= NODE_RELATION_FIELDS:
                errors.append(f"{loc}: relations allow only {sorted(NODE_RELATION_FIELDS)}")
            if isinstance(rel, str) and rel not in relations:
                errors.append(f"{loc}: unknown rel '{rel}'; use one of {allowed} (registry/relations.yaml)")
            if "note" in relation and not nonempty_text(note):
                errors.append(f"{loc}: note must be nonempty text")
            elif isinstance(rel, str) and relations.get(rel, {}).get("note_required") and note is None:
                errors.append(f"{loc}: rel '{rel}' requires a note describing the relation")
    errors.extend(validate_deployments({n.id for n in nodes}, {p.id for p in papers}, tasks))
    return errors


def require_registered_task(task: str) -> None:
    """Registration fails early instead of creating an unregistered topic."""
    tasks, errors = load_tasks()
    if errors:
        raise ValueError("; ".join(errors))
    if task not in tasks:
        raise ValueError(f"task '{task}' is not registered; add it to {registry_dir() / 'tasks.yaml'} first")
    if tasks[task]["status"] == "retired":
        successor = tasks[task].get("successor")
        hint = f"; record new evidence under '{successor}'" if successor else ""
        raise ValueError(f"task '{task}' is retired{hint}")
