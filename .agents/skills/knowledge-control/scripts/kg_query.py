#!/usr/bin/env python
"""Query the knowledge library without reading every node.

Subcommands (all accept ``--json`` for machine-readable output):

  find         filter nodes by task, type, status, tag, issue, paper, date, text or metric key
  show ID      one node with its parents, children, groups, relations in both directions,
               papers and the deployments that cite it (``--body`` adds the findings)
  metrics      metric keys with counts, or with ``--key`` a value table for comparison
  tasks        registered tasks with node counts and latest dates
  deployments  checkpoints selected by the tracked pipeline configs and their evidence

Dates filter on the experiment ``date``, or on ``recorded_at`` when the date is
unknown. Output is read-only; nothing in the library is modified.

Examples:
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_query.py find --task court_detection --since 2026-10-20
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_query.py show run-court-scene-diversity-6000-v1
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_query.py metrics --task blcs --key position_error_m
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Any

import yaml
from kg_lib import Node, load_nodes, nodes_dir, repo_root
from kg_papers import load_papers
from kg_registry import load_tasks, registry_dir
from kg_schema import iso_date


def day(node: Node) -> str:
    """Experiment date, or the registration date when the experiment date is unknown."""
    return str(node.meta.get("date") or node.meta.get("recorded_at") or "")


def issues(node: Node) -> list[int]:
    value = node.meta.get("issue")
    return value if isinstance(value, list) else [value] if isinstance(value, int) else []


def location(node: Node) -> str:
    try:
        return node.path.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(node.path)


def summary(node: Node) -> dict[str, Any]:
    return {
        "id": node.id, "type": node.type, "task": node.meta.get("task"), "sequence": node.meta.get("sequence"),
        "date": day(node), "status": node.meta.get("status"), "title": node.title, "path": location(node),
    }


def select(nodes: list[Node], args: argparse.Namespace) -> list[Node]:
    pattern = re.compile(args.text, re.IGNORECASE) if args.text else None
    chosen = []
    for node in nodes:
        tags = node.meta.get("tags") or []
        metrics = node.meta.get("metrics") or {}
        if args.task and node.meta.get("task") not in args.task:
            continue
        if args.type and node.type != args.type:
            continue
        if args.status and node.meta.get("status") not in args.status:
            continue
        if args.tag and not set(args.tag) <= set(tags):
            continue
        if args.issue is not None and args.issue not in issues(node):
            continue
        if args.paper and args.paper not in (node.meta.get("papers") or []):
            continue
        if args.since and day(node) < args.since or args.until and day(node) > args.until:
            continue
        if args.metric and args.metric not in metrics:
            continue
        if pattern and not (pattern.search(node.title) or pattern.search(node.body)):
            continue
        chosen.append(node)
    return sorted(chosen, key=lambda n: (day(n), str(n.meta.get("task")), n.meta.get("sequence") or 0), reverse=not args.oldest)


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in [header, *rows]]
    return "\n".join(lines)


def emit(args: argparse.Namespace, data: Any, text: str) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else text)


def cmd_find(args: argparse.Namespace, nodes: list[Node]) -> int:
    chosen = select(nodes, args)
    shown = chosen[: args.limit]
    rows = [[day(n), str(n.meta.get("task")), str(n.meta.get("status") or "-"), n.id, n.title[:60]] for n in shown]
    text = table(rows, ["date", "task", "status", "id", "title"]) + f"\n\n{len(shown)} of {len(chosen)} matching nodes"
    emit(args, {"total": len(chosen), "nodes": [summary(n) for n in shown]}, text)
    return 0


def cmd_show(args: argparse.Namespace, nodes: list[Node]) -> int:
    by_id = {n.id: n for n in nodes}
    node = by_id.get(args.id)
    if node is None:
        print(f"ERROR: no node '{args.id}'", file=sys.stderr)
        return 1
    deployments = [d for d in _deployments() if node.id in (d.get("evidence") or [])]
    data = summary(node) | {
        "issue": node.meta.get("issue"), "provider": node.meta.get("provider"), "tags": node.meta.get("tags") or [],
        "papers": node.meta.get("papers") or [], "metrics": node.meta.get("metrics") or {},
        "parents": node.parents, "children": [n.id for n in nodes if node.id in n.parents],
        "members": node.members, "groups": [n.id for n in nodes if node.id in n.members],
        "relations_out": node.relations,
        "relations_in": [{"from": n.id, **{k: v for k, v in r.items() if k != "to"}} for n in nodes for r in n.relations if r.get("to") == node.id],
        "deployments": [f"{d['config']} {d['key']}" for d in deployments],
    }
    if args.body:
        data["body"] = node.body
    lines = [f"{node.id}  [{data['task']} #{data['sequence']}, {node.type}, {data['status'] or '-'}, {data['date'] or 'date unknown'}]", node.title, data["path"]]
    for key in ("issue", "provider", "tags", "papers", "parents", "children", "members", "groups", "deployments"):
        if data[key]:
            lines.append(f"{key}: {', '.join(map(str, data[key])) if isinstance(data[key], list) else data[key]}")
    for relation in data["relations_out"]:
        lines.append(f"-> {relation.get('rel')} {relation.get('to')}" + (f"  ({relation['note']})" if relation.get("note") else ""))
    for relation in data["relations_in"]:
        lines.append(f"<- {relation.get('rel')} {relation['from']}" + (f"  ({relation['note']})" if relation.get("note") else ""))
    if data["metrics"]:
        lines.append("metrics: " + ", ".join(f"{k}={v}" for k, v in data["metrics"].items()))
    if args.body:
        lines += ["", node.body]
    emit(args, data, "\n".join(lines))
    return 0


def cmd_metrics(args: argparse.Namespace, nodes: list[Node]) -> int:
    chosen = select(nodes, args)
    if not args.key:
        counts = Counter(key for n in chosen for key in (n.meta.get("metrics") or {}))
        rows = [[str(count), key] for key, count in counts.most_common(args.limit)]
        emit(args, dict(counts.most_common(args.limit)), table(rows, ["nodes", "metric"]) + f"\n\n{len(counts)} metric keys in {len(chosen)} nodes")
        return 0
    having = [n for n in chosen if any(k in (n.meta.get("metrics") or {}) for k in args.key)]
    shown = having[: args.limit]
    values = [{"id": n.id, "date": day(n), "task": n.meta.get("task"), **{k: (n.meta.get("metrics") or {}).get(k) for k in args.key}} for n in shown]
    rows = [[v["date"], str(v["task"]), v["id"], *("-" if v[k] is None else str(v[k]) for k in args.key)] for v in values]
    text = table(rows, ["date", "task", "id", *args.key]) + f"\n\n{len(shown)} of {len(having)} nodes; values are only comparable under the same split, metric definition and protocol (see each node)"
    emit(args, {"total": len(having), "rows": values}, text)
    return 0


def cmd_tasks(args: argparse.Namespace, nodes: list[Node]) -> int:
    tasks, errors = load_tasks()
    if errors:
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 1
    rows, data = [], []
    for name, entry in sorted(tasks.items()):
        members = [n for n in nodes if n.meta.get("task") == name]
        latest = max((day(n) for n in members), default="")
        where = entry.get("module") or (f"-> {entry['successor']}" if entry.get("successor") else "-")
        rows.append([name, entry["status"], str(len(members)), latest or "-", where])
        data.append({"task": name, **entry, "nodes": len(members), "latest": latest})
    emit(args, data, table(rows, ["task", "status", "nodes", "latest", "module/successor"]))
    return 0


def _deployments() -> list[dict[str, Any]]:
    path = registry_dir() / "deployments.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    return list(data.get("deployments") or []) if isinstance(data, dict) else []


def cmd_deployments(args: argparse.Namespace, nodes: list[Node]) -> int:
    by_id = {n.id: n for n in nodes}
    entries = _deployments()
    rows = [[f"{e.get('key')}", str(e.get("value")), str(e.get("task")), ", ".join(e.get("evidence") or []) or str(e.get("source") or "-")] for e in entries]
    data = [e | {"evidence_titles": {i: by_id[i].title for i in e.get("evidence") or [] if i in by_id}} for e in entries]
    emit(args, data, table(rows, ["key", "checkpoint", "task", "evidence/source"]))
    return 0


def main() -> int:
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--task", nargs="+", help="one or more task names")
    filters.add_argument("--type", choices=("run", "group"))
    filters.add_argument("--status", nargs="+", choices=("done", "failed", "running", "planned"))
    filters.add_argument("--tag", nargs="+", help="nodes carrying all of these tags")
    filters.add_argument("--issue", type=int)
    filters.add_argument("--paper", help="paper ID cited by the node")
    filters.add_argument("--since", help="YYYY-MM-DD, inclusive")
    filters.add_argument("--until", help="YYYY-MM-DD, inclusive")
    filters.add_argument("--text", help="case-insensitive regular expression over title and findings")
    filters.add_argument("--metric", help="nodes whose metrics contain this key")
    filters.add_argument("--oldest", action="store_true", help="oldest first (default: newest first)")
    filters.add_argument("--limit", type=int, default=50)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print JSON")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("find", parents=[filters, common], help="filter nodes")
    show = sub.add_parser("show", parents=[common], help="one node and its neighbourhood")
    show.add_argument("id")
    show.add_argument("--body", action="store_true", help="include the findings text")
    metrics = sub.add_parser("metrics", parents=[filters, common], help="metric keys or a value table")
    metrics.add_argument("--key", nargs="+", help="metric keys to tabulate")
    sub.add_parser("tasks", parents=[common], help="registered tasks")
    sub.add_parser("deployments", parents=[common], help="deployed checkpoints and evidence")
    args = parser.parse_args()
    for name in ("since", "until"):
        value = getattr(args, name, None)
        if value is not None and not iso_date(value):
            parser.error(f"--{name} must be YYYY-MM-DD")
    if getattr(args, "text", None):
        try:
            re.compile(args.text)
        except re.error as exc:
            parser.error(f"--text is not a valid regular expression: {exc}")
    if not nodes_dir().is_dir():
        parser.error(f"{nodes_dir()}: missing nodes directory")
    try:
        nodes = load_nodes()
        load_papers()  # Fail loudly on a malformed library instead of answering from part of it.
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    handlers = {"find": cmd_find, "show": cmd_show, "metrics": cmd_metrics, "tasks": cmd_tasks, "deployments": cmd_deployments}
    return handlers[args.command](args, nodes)


if __name__ == "__main__":
    raise SystemExit(main())
