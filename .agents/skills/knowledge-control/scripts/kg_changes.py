"""Rules for what a branch adds or changes relative to its Git base.

These checks only judge the change itself, so pre-policy nodes never produce
permanent warnings. Without a base ref they cannot run; kg_validate.py says so.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import yaml
from kg_history import BaseSnapshot
from kg_lib import Node, nodes_dir
from kg_schema import SUMMARY_IMPACTS

# Absolute paths under a user's home directory identify one machine and account.
HOME_PATH = re.compile(r"(?<![\w.~-])/(?:home|Users)/[^/\s]+/|(?<![\w.~-])/root/")


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _current(name: str) -> str | None:
    path = nodes_dir().parent / name
    return path.read_text(encoding="utf-8") if path.is_file() else None


def validate_added_nodes(nodes: list[Node], base: BaseSnapshot, tasks: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Metadata rules for nodes that did not exist at the base commit."""
    errors: list[str] = []
    warnings: list[str] = []
    for node in nodes:
        if node.id in base.nodes:
            continue
        loc = node.path.name
        entry = tasks.get(str(node.meta.get("task")), {})
        if entry.get("status") == "retired":
            successor = entry.get("successor")
            errors.append(f"{loc}: task '{node.meta.get('task')}' is retired; record new evidence under {successor or 'an active task'}")
        if node.type == "run" and "issue" not in node.meta:
            warnings.append(f"{loc}: new run node has no 'issue'")
        for value in _strings(node.meta.get("artifacts", {})):
            if HOME_PATH.search(value):
                errors.append(f"{loc}: artifacts must not record a home-directory path ({value}); use a checkout-relative, ~/ or drive path")
    return errors, warnings


def validate_summary_review(nodes: list[Node], base: BaseSnapshot) -> list[str]:
    """Each added or status-changed node declares whether it changed summary.md."""
    errors = []
    reviewed = [n for n in nodes if n.id not in base.nodes or n.meta.get("status") != base.nodes[n.id].meta.get("status")]
    for node in reviewed:
        if node.meta.get("summary_impact") not in SUMMARY_IMPACTS:
            errors.append(f"{node.path.name}: added or status-changed node must declare summary_impact: updated or none after reviewing summary.md")
    summary_changed = _current("summary.md") != base.files["summary.md"]
    updated = sorted(n.id for n in reviewed if n.meta.get("summary_impact") == "updated")
    if updated and not summary_changed:
        errors.append(f"summary.md is unchanged relative to {base.ref} although {', '.join(updated)} declare summary_impact: updated")
    before, after = base.files["registry/deployments.yaml"], _current("registry/deployments.yaml")
    deployments_changed = (yaml.safe_load(before) if before else None) != (yaml.safe_load(after) if after else None)
    if deployments_changed and not summary_changed:
        errors.append(f"registry/deployments.yaml changed relative to {base.ref}; describe the deployment change in summary.md")
    return errors
