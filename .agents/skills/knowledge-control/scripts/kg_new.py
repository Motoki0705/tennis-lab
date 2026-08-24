#!/usr/bin/env python
"""Scaffold a new knowledge node (run or group) under ``knowledge/nodes/``.

Examples:
    # A run node
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_new.py \
        --type run --id run-i521-base-vel --title "velocity loss baseline" \
        --issue 521 --provider claude --status done \
        --parents run-i520-canon-none

    # A group node
    .venv/bin/python .agents/skills/knowledge-control/scripts/kg_new.py \
        --type group --id group-i521-velocity --title "角速度 canonical loss (#521)" \
        --issue 521 --members run-i521-base-vel run-i521-ex10-vel

The frontmatter is written with placeholders; fill in metrics/config and the
考察 body, then run kg_validate.py.
"""

from __future__ import annotations

import argparse
from datetime import date as _date

from kg_lib import ID_RE, NODE_TYPES, dump_frontmatter, nodes_dir


def build_meta(args: argparse.Namespace) -> dict:
    meta: dict = {"id": args.id, "type": args.type, "title": args.title}
    if args.issue is not None:
        meta["issue"] = args.issue
    if args.type == "run":
        meta["provider"] = args.provider or "claude"
        meta["date"] = args.date or _date.today().isoformat()
        meta["status"] = args.status or "done"
        meta["config"] = {"model": "", "loss": "", "data": ""}
        meta["metrics"] = {}
        meta["artifacts"] = {"log": "", "output_dir": ""}
        meta["parents"] = list(args.parents or [])
        meta["relations"] = []
    else:
        meta["members"] = list(args.members or [])
        meta["parents"] = list(args.parents or [])
    meta["tags"] = list(args.tags or [])
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--type", required=True, choices=sorted(NODE_TYPES))
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--issue", type=int)
    p.add_argument("--provider")
    p.add_argument("--date")
    p.add_argument("--status")
    p.add_argument("--parents", nargs="*", default=[])
    p.add_argument("--members", nargs="*", default=[])
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--force", action="store_true", help="overwrite if file exists")
    args = p.parse_args()

    if not ID_RE.match(args.id):
        p.error(f"invalid id '{args.id}' (use lowercase a-z0-9-)")

    out = nodes_dir() / f"{args.id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        p.error(f"{out} already exists (use --force to overwrite)")

    meta = build_meta(args)
    body = (
        "## 考察 / Findings\n\n"
        "<!-- このノード(=1 run)の結果と考察を書く。"
        " 主要 metrics は frontmatter にも転記すること。 -->\n"
    )
    out.write_text(f"---\n{dump_frontmatter(meta)}---\n\n{body}", encoding="utf-8")
    print(f"created {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
