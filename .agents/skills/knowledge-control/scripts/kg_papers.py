#!/usr/bin/env python
"""Register a locally available research PDF and its source metadata."""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

from kg_lib import Node, ValidationResult, dump_frontmatter, nodes_dir, parse_node
from kg_schema import TASK_RE, has_prose, http_url, nonempty_text

PAPER_RE = re.compile(r"^paper-[1-9][0-9]{3}-[a-z0-9]+(?:-[a-z0-9]+)*$")


def papers_dir() -> Path:
    return nodes_dir().parent / "Papers"


def load_papers() -> list[Node]:
    for path in sorted(papers_dir().iterdir()) if papers_dir().exists() else []:
        if path.name == "README.md" and path.is_file():
            continue
        if path.is_symlink() or not path.is_dir() or not PAPER_RE.fullmatch(path.name):
            raise ValueError(f"{path}: expected Papers/paper-YYYY-slug/ directory")
        if not (path / "paper.md").is_file():
            raise ValueError(f"{path}: missing paper.md (orphan paper directory/PDF)")
        if any(p.name not in {"paper.md", "paper.pdf"} or p.is_symlink() or not p.is_file() for p in path.iterdir()):
            raise ValueError(f"{path}: only regular paper.md and paper.pdf files are allowed")
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
        if paper.type != "paper" or not nonempty_text(meta.get("title")):
            result.errors.append(f"{loc}: paper type/title required")
        if type(meta.get("year")) is not int or not paper.id.startswith(f"paper-{meta.get('year')}-"):
            result.errors.append(f"{loc}: year must match id")
        for key in ("authors", "tasks"):
            value = meta.get(key)
            if not isinstance(value, list) or not value or any(not nonempty_text(x) for x in value):
                result.errors.append(f"{loc}: nonempty {key} list required")
            elif len(value) != len(set(value)):
                result.errors.append(f"{loc}: duplicate {key} entries")
        tasks = meta.get("tasks")
        if isinstance(tasks, list) and any(not TASK_RE.fullmatch(str(t)) for t in tasks):
            result.errors.append(f"{loc}: invalid task slug")
        for key in ("source", "license"):
            if not http_url(meta.get(key)):
                result.errors.append(f"{loc}: {key} must be an HTTP(S) URL")
        if not has_prose(paper.body):
            result.errors.append(f"{loc}: write a research note; headings and comments alone are unfinished")
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
    if any(not TASK_RE.fullmatch(t) for t in args.tasks):
        parser.error("invalid task slug")
    for value in (args.source, args.license):
        if not http_url(value):
            parser.error("source/license must be HTTP(S) URLs")
    if not nonempty_text(args.title) or any(not nonempty_text(a) for a in args.authors):
        parser.error("title and authors must be nonempty strings")
    if len(set(args.authors)) != len(args.authors) or len(set(args.tasks)) != len(args.tasks):
        parser.error("authors/tasks must not contain duplicates")
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
