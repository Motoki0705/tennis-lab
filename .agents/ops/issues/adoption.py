"""Durable provenance for triage tasks, alongside the single-source proposal index."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .format import (
    CHILD_REQUIRED,
    ProposalSource,
    _render_body,
    _scan_markdown,
    parse_proposal_source,
    proposal_marker,
    validate_acceptance,
)

PREFIX = "<!-- ops-report-task:v1 "


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("duplicate task metadata keys")
    return dict(pairs)


def task_marker(plan: str, node: str, bindings: list[dict[str, Any]]) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", plan) or not re.fullmatch(
        r"[a-z][a-z0-9-]{0,63}", node
    ):
        raise ValueError("invalid triage plan/node identity")
    keys = []
    for binding in bindings:
        if set(binding) != {"source", "proposal_sha256"}:
            raise ValueError("invalid proposal binding")
        raw = binding["source"]
        if not isinstance(raw, dict) or set(raw) != {
            "report_number",
            "proposal_id",
            "option",
        }:
            raise ValueError("invalid proposal binding source")
        source = ProposalSource(**raw)
        if not re.fullmatch(r"[0-9a-f]{64}", binding["proposal_sha256"]):
            raise ValueError("invalid proposal content digest")
        keys.append((source.report_number, source.proposal_id))
    if not bindings or keys != sorted(set(keys)):
        raise ValueError("task bindings must be nonempty, unique and sorted")
    data = {"plan_sha256": plan, "node": node, "bindings": bindings}
    encoded = (
        json.dumps(data, sort_keys=True, ensure_ascii=True)
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
    return PREFIX + encoded + " -->"


def parse_task_metadata(body: str) -> dict[str, Any] | None:
    _, markers = _scan_markdown(body, prefix="<!-- ops-report-task:")
    if not markers:
        return None
    source = parse_proposal_source(body)
    index, line = markers[0]
    if (
        len(markers) != 1
        or not line.startswith(PREFIX)
        or index != (1 if source else 0)
        or not line.endswith(" -->")
    ):
        raise ValueError("task metadata must immediately follow proposal metadata")
    data = json.loads(line[len(PREFIX) : -4], object_pairs_hook=_unique)
    if not isinstance(data, dict) or set(data) != {"plan_sha256", "node", "bindings"}:
        raise ValueError("invalid task metadata")
    if task_marker(data["plan_sha256"], data["node"], data["bindings"]) != line:
        raise ValueError("noncanonical task metadata")
    if source is not None and {
        "report_number": source.report_number,
        "proposal_id": source.proposal_id,
        "option": source.option,
    } not in [b["source"] for b in data["bindings"]]:
        raise ValueError("proposal index source is absent from task bindings")
    return data


def bind_task(
    body: str,
    *,
    plan: str,
    node: str,
    bindings: list[dict[str, Any]],
    source: ProposalSource | None = None,
) -> str:
    """One proposal owns one index anchor; shared parents keep context bindings."""
    if parse_task_metadata(body) is not None:
        raise ValueError("task already has triage provenance")
    previous = parse_proposal_source(body)
    if previous is not None and previous != source:
        raise ValueError("cannot replace an existing proposal source")
    if previous is not None:
        body = body.split("\n", 1)[1].lstrip("\n")
    prefix = proposal_marker(source) + "\n" if source is not None else ""
    return prefix + task_marker(plan, node, bindings) + "\n\n" + body


def render_standalone_body(
    sections: Mapping[str, str], *, source: ProposalSource | None = None
) -> str:
    """A concrete task has the child task's requirements without a fake parent."""
    if any(not sections.get(name, "").strip() for name in CHILD_REQUIRED):
        raise ValueError("standalone task requires the task sections")
    validate_acceptance(sections["Acceptance checklist"])
    body: str = _render_body(
        sections, is_parent=False, parent_number=None, source=source
    )
    return body
