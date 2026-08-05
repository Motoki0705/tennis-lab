#!/usr/bin/env python
"""Scaffold a formal knowledge node under ``knowledge/nodes/``.

Supported node types:
- ``run``: one completed/planned experiment run
- ``group``: a collection of related runs
- ``paper``: a reviewed external paper
- ``proposal``: a repository-specific hypothesis derived from paper nodes
"""

from __future__ import annotations

import argparse
from datetime import date as _date
from typing import Any

from kg_lib import (
    ID_RE,
    NODE_TYPES,
    PAPER_EVIDENCE_LEVELS,
    PAPER_STATUSES,
    PROPOSAL_STATUSES,
    RUN_STATUSES,
    dump_frontmatter,
    nodes_dir,
)


def _external_ids(values: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {"doi": None, "arxiv": None, "openreview": None}
    for value in values:
        if "=" not in value:
            raise ValueError("--external-id must use key=value")
        key, raw = value.split("=", 1)
        key = key.strip().lower()
        if key not in result:
            raise ValueError("--external-id key must be doi, arxiv, or openreview")
        result[key] = raw.strip() or None
    return result


def build_meta(args: argparse.Namespace) -> dict[str, Any]:
    meta: dict[str, Any] = {"id": args.id, "type": args.type, "title": args.title}
    if args.issue is not None:
        meta["issue"] = args.issue

    if args.type == "run":
        meta.update(
            {
                "provider": args.provider or "claude",
                "date": args.date or _date.today().isoformat(),
                "status": args.status or "done",
                "config": {"model": "", "loss": "", "data": ""},
                "metrics": {},
                "artifacts": {"log": "", "output_dir": ""},
                "parents": list(args.parents),
                "relations": [],
            }
        )
    elif args.type == "group":
        meta.update({"members": list(args.members), "parents": list(args.parents)})
    elif args.type == "paper":
        meta.update(
            {
                "curator": args.curator or "human",
                "date": args.date or _date.today().isoformat(),
                "status": args.status or "reviewed",
                "external_ids": _external_ids(args.external_id),
                "published_at": args.published_at,
                "reviewed_at": args.date or _date.today().isoformat(),
                "evidence_level": args.evidence_level or "fulltext",
                "tasks": list(args.task),
                "repo_paths": list(args.repo_path),
                "sources": [{"kind": "paper", "url": url} for url in args.source],
                "relations": [],
            }
        )
    elif args.type == "proposal":
        task = args.task[0] if args.task else ""
        meta.update(
            {
                "curator": args.curator or "human",
                "date": args.date or _date.today().isoformat(),
                "status": args.status or "candidate",
                "task": task,
                "repo_paths": list(args.repo_path),
                "hypothesis": {
                    "statement": "",
                    "expected_effect": "",
                    "failure_condition": "",
                },
                "evaluation": {
                    "metrics": list(args.metric),
                    "baseline_nodes": list(args.baseline),
                    "seeds": args.seeds,
                    "acceptance": "",
                },
                "evidence_runs": [],
                "parents": list(args.parents),
                "relations": [
                    {"to": paper_id, "rel": "derived-from"}
                    for paper_id in args.paper
                ],
            }
        )

    meta["tags"] = list(args.tags)
    return meta


def body_template(node_type: str) -> str:
    if node_type == "run":
        return (
            "## 考察 / Findings\n\n"
            "### 要約\n\n"
            "### アーキテクチャ詳細\n\n"
            "### メトリクスの解釈\n\n"
            "### アーキテクチャ⇄メトリクスの因果考察\n\n"
            "### 既存実験との比較\n\n"
            "### 次に有効な実験\n"
        )
    if node_type == "group":
        return "## まとめ\n"
    if node_type == "paper":
        return (
            "## 要約\n\n"
            "## 主要な主張と根拠\n\n"
            "## tennis-labへの適用可能性\n\n"
            "## 制約・失敗条件\n\n"
            "## コード・データ・ライセンス\n"
        )
    return (
        "## 背景\n\n"
        "## 現行実装との差分\n\n"
        "## 最小検証\n\n"
        "## 比較対象\n\n"
        "## 合格条件と停止条件\n\n"
        "## リスク\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", required=True, choices=sorted(NODE_TYPES))
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--issue", type=int)
    parser.add_argument("--provider")
    parser.add_argument("--curator")
    parser.add_argument("--date")
    parser.add_argument("--status")
    parser.add_argument("--published-at")
    parser.add_argument("--evidence-level", choices=sorted(PAPER_EVIDENCE_LEVELS))
    parser.add_argument("--external-id", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--repo-path", action="append", default=[])
    parser.add_argument("--paper", action="append", default=[])
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--parents", nargs="*", default=[])
    parser.add_argument("--members", nargs="*", default=[])
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--force", action="store_true", help="overwrite if file exists")
    args = parser.parse_args()

    if not ID_RE.match(args.id):
        parser.error(f"invalid id '{args.id}' (use lowercase a-z0-9-)")
    if args.type == "run" and args.status and args.status not in RUN_STATUSES:
        parser.error(f"run status must be one of {sorted(RUN_STATUSES)}")
    if args.type == "paper":
        if args.status and args.status not in PAPER_STATUSES:
            parser.error(f"paper status must be one of {sorted(PAPER_STATUSES)}")
        if not args.external_id or not args.task or not args.repo_path or not args.source:
            parser.error(
                "paper requires --external-id, --task, --repo-path, and --source"
            )
    if args.type == "proposal":
        if args.status and args.status not in PROPOSAL_STATUSES:
            parser.error(f"proposal status must be one of {sorted(PROPOSAL_STATUSES)}")
        if len(args.task) != 1 or not args.repo_path or not args.paper or not args.metric:
            parser.error(
                "proposal requires exactly one --task and at least one --repo-path, "
                "--paper, and --metric"
            )

    output = nodes_dir() / f"{args.id}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        parser.error(f"{output} already exists (use --force to overwrite)")

    try:
        meta = build_meta(args)
    except ValueError as exc:
        parser.error(str(exc))
    body = body_template(args.type)
    output.write_text(
        f"---\n{dump_frontmatter(meta)}---\n\n{body}", encoding="utf-8"
    )
    print(f"created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
