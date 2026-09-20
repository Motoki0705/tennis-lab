#!/usr/bin/env python
"""Register a locally available research PDF and its source metadata."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kg_lib import Node, ValidationResult, dump_frontmatter, nodes_dir, parse_node

PAPER_RE = re.compile(r"^paper-[0-9]{4}-[a-z0-9]+(?:-[a-z0-9]+)*$")


def papers_dir() -> Path:
    return nodes_dir().parent / "Papers"


def load_papers() -> list[Node]:
    return [parse_node(p) for p in sorted(papers_dir().glob("*/paper.md"))]


def validate_papers(nodes: list[Node]) -> ValidationResult:
    result = ValidationResult()
    papers = load_papers()
    known: set[str] = set()
    for paper in papers:
        meta = paper.meta
        loc = str(paper.path)
        if not PAPER_RE.fullmatch(paper.id) or paper.id != paper.path.parent.name:
            result.errors.append(f"{loc}: expected Papers/paper-YYYY-slug/paper.md")
        if paper.id in known:
            result.errors.append(f"{loc}: duplicate paper id")
        known.add(paper.id)
        if paper.type != "paper" or not meta.get("title"):
            result.errors.append(f"{loc}: paper type/title required")
        if type(meta.get("year")) is not int or not paper.id.startswith(f"paper-{meta.get('year')}-"):
            result.errors.append(f"{loc}: year must match id")
        for key in ("authors", "tasks"):
            value = meta.get(key)
            if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
                result.errors.append(f"{loc}: nonempty {key} list required")
        tasks = meta.get("tasks")
        if isinstance(tasks, list) and any(not re.fullmatch(r"[a-z][a-z0-9_]*", str(t)) for t in tasks):
            result.errors.append(f"{loc}: invalid task slug")
        for key in ("source", "license"):
            url = urlparse(str(meta.get(key, "")))
            if url.scheme not in {"http", "https"} or not url.netloc:
                result.errors.append(f"{loc}: {key} must be an HTTP(S) URL")
        pdf = paper.path.parent / "paper.pdf"
        if meta.get("pdf") != "paper.pdf" or not pdf.is_file():
            result.errors.append(f"{loc}: local paper.pdf required")
        else:
            data = pdf.read_bytes()
            if not data.startswith(b"%PDF-"):
                result.errors.append(f"{loc}: not a PDF")
            if meta.get("sha256") != hashlib.sha256(data).hexdigest():
                result.errors.append(f"{loc}: PDF sha256 mismatch")
    for node in nodes:
        refs = node.meta.get("papers", [])
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, str) or ref not in known:
                    result.errors.append(f"{node.path.name}: unknown paper {ref}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--authors", nargs="+", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--source", required=True, help="version-specific source URL")
    parser.add_argument("--license", required=True, help="verified redistribution license URL")
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    if not PAPER_RE.fullmatch(args.id):
        parser.error("id must be paper-YYYY-kebab-slug")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", t) for t in args.tasks):
        parser.error("invalid task slug")
    for value in (args.source, args.license):
        url = urlparse(value)
        if url.scheme not in {"http", "https"} or not url.netloc:
            parser.error("source/license must be HTTP(S) URLs")
    data = args.pdf.read_bytes()
    if not data.startswith(b"%PDF-"):
        parser.error("input is not a PDF")
    target = papers_dir() / args.id
    meta: dict[str, Any] = dict(id=args.id, type="paper", title=args.title, year=int(args.id.split("-")[1]), authors=args.authors, tasks=args.tasks, source=args.source, license=args.license, pdf="paper.pdf", sha256=hashlib.sha256(data).hexdigest())
    target.mkdir(parents=True, exist_ok=False)
    try:
        (target / "paper.pdf").write_bytes(data)
        (target / "paper.md").write_text(f"---\n{dump_frontmatter(meta)}---\n\n## 研究の要点\n\n## このプロジェクトとの関係\n\n## 検証したい仮説・適用限界\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(target)
        raise
    print(target / "paper.md")


if __name__ == "__main__":
    main()
