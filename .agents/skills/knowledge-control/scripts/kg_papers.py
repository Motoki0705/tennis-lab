#!/usr/bin/env python
"""Register a human-selected paper as a document, manually or from a citation search."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from kg_citations import CitationLibrary, load_citations, validate_library
from kg_identifiers import (
    external_id_index,
    normalize_external_ids,
    normalize_identifier,
    resolve_identity,
)
from kg_lib import Node, ValidationResult, dump_frontmatter, nodes_dir, parse_node
from kg_registry import load_tasks
from kg_schema import TASK_RE, http_url, nonempty_text

PAPER_RE = re.compile(r"^paper-[1-9][0-9]{3}-[a-z0-9]+(?:-[a-z0-9]+)*$")


def papers_dir() -> Path:
    directory: Path = nodes_dir()
    return directory.parent / "Papers"


def load_papers() -> list[Node]:
    directory = papers_dir()
    if directory.is_symlink():
        raise ValueError(f"{directory}: symlinks are not allowed")
    for path in sorted(directory.iterdir()) if directory.exists() else []:
        if path.name == "README.md" and path.is_file() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_dir() or not PAPER_RE.fullmatch(path.name):
            raise ValueError(f"{path}: expected Papers/paper-YYYY-slug/ directory")
        if not (path / "paper.md").is_file():
            raise ValueError(f"{path}: missing paper.md")
        if any(p.name != "paper.md" or p.is_symlink() or not p.is_file() for p in path.iterdir()):
            raise ValueError(f"{path}: only regular paper.md is allowed")
    return [parse_node(p) for p in sorted(directory.glob("*/paper.md"))]


def validate_papers(nodes: list[Node], papers: list[Node] | None = None) -> ValidationResult:
    result = ValidationResult()
    papers = load_papers() if papers is None else papers
    known: set[str] = set()
    sources: set[str] = set()
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
        if not http_url(meta.get("source")):
            result.errors.append(f"{loc}: source must be an HTTP(S) URL")
        else:
            source = normalize_identifier("url", meta["source"])
            if source in sources:
                result.errors.append(f"{loc}: duplicate literature source {source}")
            sources.add(source)
        for key in ("summary", "reason"):
            if not nonempty_text(meta.get(key)):
                result.errors.append(f"{loc}: nonempty {key} required")
        for key in ("pdf", "sha256", "license", "citation_id", "search_id"):
            if key in meta:
                result.errors.append(f"{loc}: {key} is not allowed in document-only Papers")
        values = meta.get("external_ids")
        if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()):
            result.errors.append(f"{loc}: external_ids must be a string mapping")
        else:
            try:
                if normalize_external_ids(values) != values:
                    raise ValueError("external_ids must use normalized values")
            except ValueError as exc:
                result.errors.append(f"{loc}: {exc}")
    if not result.errors:
        try:
            external_id_index((p.id, p.meta["external_ids"]) for p in papers)
        except ValueError as exc:
            result.errors.append(str(exc))
    for node in nodes:
        refs = node.meta.get("papers", [])
        if isinstance(refs, list):
            for ref in refs:
                if not isinstance(ref, str) or ref not in known:
                    result.errors.append(f"{node.path.name}: unknown paper {ref}")
    return result


def discovery_reason(library: CitationLibrary, search_id: str, paper_id: str) -> str:
    if search_id not in library.searches:
        raise ValueError(f"unknown citation search: {search_id}")
    search = library.searches[search_id]
    reached = {p for g in search.generations for p in g.paper_ids}
    if paper_id not in reached:
        raise ValueError(f"{paper_id} was not reached by {search_id}")
    paths = []
    for generation in search.generations:
        for source_id in generation.expanded_ids:
            source = library.papers[source_id]
            for ref in source.references:
                if ref.target_id == paper_id:
                    paths.append(f"引用元 {source.title} ({source_id}) / {ref.ref_id}; 役割 {', '.join(ref.roles) or ref.decision_status}; 根拠 {'; '.join(ref.evidence) or '記載なし'}; 判断 {ref.reason}")
    prefix = f"探索 {search_id}（起点 {search.root_id}、指定役割 {', '.join(search.roles)}、最大深さ {search.max_depth}）から採録。"
    return prefix + ("\n" + "\n".join(paths) if paths else " 起点論文として指定された一次資料を確認。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--title")
    parser.add_argument("--authors", nargs="+")
    parser.add_argument("--source", help="version-specific primary source HTTP(S) URL")
    parser.add_argument("--summary", help="short research summary; required for manual registration")
    parser.add_argument("--reason", help="research question/adoption reason; appended to discovery path for search registration")
    parser.add_argument("--external-id", action="append", default=[], metavar="NAMESPACE=VALUE")
    parser.add_argument("--from-search", help="search ID to record the discovery path from")
    parser.add_argument("--citation-id", help="citation document to adopt from that search")
    args = parser.parse_args()
    if not PAPER_RE.fullmatch(args.id):
        parser.error("id must be paper-YYYY-kebab-slug")
    if any(not TASK_RE.fullmatch(t) for t in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must contain unique valid task slugs")
    if bool(args.from_search) != bool(args.citation_id):
        parser.error("--from-search and --citation-id must be supplied together")
    try:
        tasks, task_errors = load_tasks()
        if task_errors:
            raise ValueError("; ".join(task_errors))
        if any(t not in tasks for t in args.tasks):
            raise ValueError(f"tasks {args.tasks} are not registered in registry/tasks.yaml")
        if any(tasks[t].get("status") == "retired" for t in args.tasks):
            raise ValueError("cannot register a paper to a retired task")
        existing = load_papers()
        validation = validate_papers([], existing)
        if validation.errors:
            raise ValueError("; ".join(validation.errors))
        values: dict[str, str] = {}
        for item in args.external_id:
            namespace, sep, value = item.partition("=")
            if not sep or namespace in values:
                raise ValueError("--external-id requires unique NAMESPACE=VALUE entries")
            values[namespace] = value
        values = normalize_external_ids(values)
        if args.from_search:
            library = load_citations()
            errors = validate_library(library, existing)
            if errors:
                raise ValueError("; ".join(errors))
            reason = discovery_reason(library, args.from_search, args.citation_id)
            paper = library.papers[args.citation_id]
            if paper.year is not None and int(args.id.split("-")[1]) != paper.year:
                raise ValueError("Papers id year must match the citation paper year")
            for namespace, value in paper.external_ids.items():
                if namespace in values and values[namespace] != value:
                    raise ValueError(f"conflicting {namespace} alias for selected citation paper")
                values[namespace] = value
            title = args.title if args.title is not None else paper.title
            authors = args.authors if args.authors is not None else paper.authors
            source = args.source if args.source is not None else paper.source
            summary = args.summary if args.summary is not None else paper.summary
            if args.reason is not None:
                reason += "\n" + args.reason
        else:
            title, authors, source, summary, reason = args.title, args.authors, args.source, args.summary, args.reason
        if not nonempty_text(title) or not authors or any(not nonempty_text(a) for a in authors) or len(set(authors)) != len(authors):
            raise ValueError("nonempty title and unique authors required (supply --title/--authors if unknown)")
        if not http_url(source) or not nonempty_text(summary) or not nonempty_text(reason):
            raise ValueError("primary source URL, nonempty summary and reason required")
        match = resolve_identity(values, external_id_index((p.id, p.meta["external_ids"]) for p in existing))
        if match:
            raise ValueError(f"duplicate literature: already registered as {match}")
        if any(normalize_identifier("url", p.meta["source"]) == normalize_identifier("url", source) for p in existing):
            raise ValueError("duplicate literature source: already registered")
        # Aliases supplied by the human must not connect distinct citation docs.
        library = load_citations()
        alias_match = resolve_identity(values, external_id_index((p.id, p.external_ids) for p in library.papers.values()))
        if args.citation_id and alias_match != args.citation_id:
            raise ValueError("external IDs disagree with selected citation paper")
        prospective = Node(args.id, "paper", {"external_ids": values}, "", Path())
        errors = validate_library(library, [*existing, prospective])
        if errors:
            raise ValueError("; ".join(errors))
        target = papers_dir() / args.id
        meta: dict[str, Any] = dict(id=args.id, type="paper", title=title, year=int(args.id.split("-")[1]), authors=authors, tasks=args.tasks, source=source, summary=summary, reason=reason, external_ids=values)
        target.mkdir(parents=True, exist_ok=False)
        try:
            (target / "paper.md").write_text(f"---\n{dump_frontmatter(meta)}---\n", encoding="utf-8")
        except BaseException:
            shutil.rmtree(target)
            raise
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(target / "paper.md")


if __name__ == "__main__":
    main()
