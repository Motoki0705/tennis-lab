#!/usr/bin/env python3
"""Explore selected backward-citation roles, checkpointing each completed paper.

Example: kg_explore.py arxiv:2311.05237 --role comparison --max-depth 2
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from kg_citation_reader import AgyError, CitationReader
from kg_citations import (
    ROLES,
    CitationPaper,
    CitationSearch,
    Generation,
    Reference,
    Role,
    load_citations,
    search_issues,
    serialize_document,
    validate_library,
    write_document,
)
from kg_identifiers import (
    citation_id,
    external_id_index,
    initial_identity_key,
    normalize_external_ids,
    resolve_identity,
)
from kg_lib import nodes_dir
from kg_literature import (
    Bibliography,
    Literature,
    TextUnavailable,
    input_identifiers,
    primary_url,
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ExplorationResult:
    search_id: str
    elapsed_seconds: float
    agy_calls: int
    newly_read: int
    reused: int
    read_papers: int
    fetch_failed: int
    unresolved: int
    unclassifiable: int
    known_pending: int
    status: str


class Explorer:
    def __init__(self, directory: Path, literature: Literature, reader: CitationReader,
                 *, agy_timeout: float = 1800, time_limit: float = 28800,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not math.isfinite(time_limit) or not 0 < time_limit <= 28800:
            raise ValueError("time_limit must be positive and at most 28800 seconds (8 hours)")
        if not math.isfinite(agy_timeout) or agy_timeout <= 0:
            raise ValueError("agy_timeout must be finite and positive")
        self.directory = directory
        self.literature = literature
        self.reader = reader
        self.agy_timeout = agy_timeout
        self.time_limit = time_limit
        self.clock = clock
        self.library = load_citations(directory)
        self._validate()
        self.locations: dict[str, Bibliography] = {}

    def _validate(self) -> None:
        errors = validate_library(self.library)
        if errors:
            raise ValueError("invalid citation library: " + "; ".join(errors))

    def _upsert(self, bibliography: Bibliography) -> CitationPaper:
        ids = normalize_external_ids(bibliography.external_ids)
        index = external_id_index((p.id, p.external_ids) for p in self.library.papers.values())
        found = resolve_identity(ids, index)
        if found is not None:
            paper = self.library.papers[found]
            paper.external_ids.update(ids)
        else:
            key = initial_identity_key(ids)
            paper = CitationPaper(citation_id(key), key, bibliography.title or bibliography.source,
                                  bibliography.year, bibliography.authors, ids, bibliography.source,
                                  "", "not_requested", "", [])
            self.library.papers[paper.id] = paper
        self.locations[paper.id] = bibliography
        return paper

    def _persist_papers(self, previous: dict[str, CitationPaper]) -> None:
        self._validate()
        # Targets first: a newly stored source never points to a nonexistent file.
        changed = [p for p in self.library.papers.values() if p.id not in previous or p != previous[p.id]]
        for paper in sorted(changed, key=lambda p: bool(p.references)):
            write_document(paper, self.directory)

    def _read(self, paper: CitationPaper, remaining: float) -> bool:
        deadline = self.clock() + remaining
        print(f"reading {paper.id}: {paper.title}", file=sys.stderr, flush=True)
        previous = copy.deepcopy(self.library.papers)
        bibliography = self.locations.get(paper.id) or Bibliography(paper.title, paper.external_ids, paper.source, paper.year, paper.authors)
        try:
            full_text = self.literature.full_text(bibliography)
        except TextUnavailable as exc:
            paper.text_status, paper.text_reason = "failed", str(exc)
            self._persist_papers(previous)
            return False
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise BudgetExceeded("execution reached its time limit during full-text retrieval")
        output, note = self.reader.read(bibliography, full_text, min(remaining, self.agy_timeout))
        print(f"{paper.id}: {len(output['references'])} bibliography rows; identifying references", file=sys.stderr, flush=True)
        try:
            # Location metadata can expose DOI/arXiv aliases of an OpenAlex root.
            self._upsert(bibliography)
            refs: list[Reference] = []
            for item in output["references"]:
                if self.clock() >= deadline:
                    raise BudgetExceeded("execution reached its time limit during reference identification")
                target, identity_reason = self.literature.identify(item)
                target_id = self._upsert(target).id if target is not None else None
                refs.append(Reference(item["ref_id"], item["citation"], target_id,
                                      "resolved" if target_id else "unresolved", item["roles"],
                                      item["decision_status"], item["reason"] + "\nIdentification: " + identity_reason,
                                      item["evidence"]))
            paper.title, paper.year, paper.authors = output["title"], output["year"], output["authors"]
            paper.summary, paper.references = output["summary"], refs
            paper.text_status, paper.source = "available", full_text.source
            paper.text_reason = "; ".join(full_text.attempts)
            paper.body = note
            self._persist_papers(previous)
            print(f"saved {paper.id}", file=sys.stderr, flush=True)
        except (Exception, KeyboardInterrupt):
            self.library.papers = previous
            raise
        return True

    def explore(self, root: str, roles: list[Role], max_depth: int, search_id: str) -> ExplorationResult:
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if not roles or len(set(roles)) != len(roles) or not set(roles) <= ROLES:
            raise ValueError("roles must be a nonempty, unique subset of the five citation roles")
        started = self.clock()
        calls_before = self.reader.calls
        ids = input_identifiers(root)
        previous = copy.deepcopy(self.library.papers)
        source = root if root.startswith(("https://", "http://")) else primary_url(ids)
        paper = self._upsert(Bibliography(root, ids, source))
        search = self.library.searches.get(search_id)
        if search is not None:
            if (search.root_id, set(search.roles), search.max_depth) != (paper.id, set(roles), max_depth):
                raise ValueError("search-id already belongs to different exploration conditions")
        else:
            search = CitationSearch(search_id, datetime.now(UTC).isoformat(), paper.id, roles, max_depth,
                                    [Generation(0, [paper.id], [])], [], status="interrupted", stop_reason="execution in progress")
        serialize_document(search)  # Validate the requested ID before writing anything.
        self._persist_papers(previous)
        self.library.searches[search.id] = search
        newly_read: set[str] = set()
        reused: set[str] = set()
        history = search.body

        def snapshot(reason: str = "execution in progress", *, complete: bool = False) -> ExplorationResult:
            expanded = [p for generation in search.generations for p in generation.expanded_ids]
            search.issues = sorted(search_issues(expanded, self.library.papers), key=lambda p: (p.paper_id, p.ref_id or ""))
            search.status, search.stop_reason = ("complete", "") if complete else ("interrupted", reason)
            selected = [self.library.papers[p] for p in expanded]
            pending = {p for generation in search.generations if generation.depth < max_depth
                       for p in generation.paper_ids if p not in generation.expanded_ids}
            # Discovered next-generation papers are a lower bound while a generation is partial.
            for generation in search.generations:
                if generation.depth + 1 < max_depth:
                    pending.update(ref.target_id for p in generation.expanded_ids
                                   for ref in self.library.papers[p].references
                                   if ref.target_id and ref.decision_status == "classified" and set(ref.roles) & set(roles))
            pending -= set(expanded)
            result = ExplorationResult(search.id, self.clock() - started, self.reader.calls - calls_before,
                                       len(newly_read), len(reused), sum(p.text_status == "available" for p in selected),
                                       sum(p.text_status == "failed" for p in selected),
                                       sum(r.identification_status == "unresolved" for p in selected for r in p.references),
                                       sum(r.decision_status == "unclassifiable" for p in selected for r in p.references),
                                       len(pending), search.status)
            report = asdict(result)
            search.body = history + "\nAttempt report (seconds are for this invocation):\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n"
            self._validate()
            write_document(search, self.directory)
            return result

        seen: set[str] = set()
        current = [paper.id]
        # Rebuild traversal from saved paper decisions, rather than appending edges.
        search.generations = [Generation(0, current, [])]
        snapshot()
        try:
            for depth in range(max_depth + 1):
                generation = search.generations[-1]
                seen.update(current)
                if depth == max_depth:
                    break
                for paper_id in current:
                    paper = self.library.papers[paper_id]
                    remaining = self.time_limit - (self.clock() - started)
                    if remaining <= 0:
                        raise BudgetExceeded("execution reached its time limit")
                    if paper.text_status == "not_requested":
                        if self._read(paper, remaining):
                            newly_read.add(paper_id)
                    else:
                        reused.add(paper_id)
                    generation.expanded_ids.append(paper_id)
                    snapshot()
                following = sorted({ref.target_id for p in current for ref in self.library.papers[p].references
                                    if ref.target_id is not None and ref.decision_status == "classified"
                                    and set(ref.roles) & set(roles)} - seen)
                if not following:
                    break
                current = following
                search.generations.append(Generation(depth + 1, current, []))
                snapshot()
        except (Exception, KeyboardInterrupt) as exc:
            reason = f"agy:{exc.reason}" if isinstance(exc, AgyError) else ("interrupted by operator" if isinstance(exc, KeyboardInterrupt) else str(exc))
            snapshot(reason)
            raise
        return snapshot(complete=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="DOI, arXiv/OpenAlex ID, or primary-source HTTP(S) URL")
    parser.add_argument("--role", action="append", choices=sorted(ROLES), required=True, help="repeat to follow multiple roles")
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--search-id", default="search-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8], help="reuse the same ID to resume; conditions must match")
    parser.add_argument("--agy", type=Path, default=Path.home() / ".local/bin/agy", help="citation-profile pinned executable")
    parser.add_argument("--agy-timeout", type=float, default=1800)
    parser.add_argument("--time-limit", type=float, default=28800, help="seconds for this invocation, <= 28800")
    parser.add_argument("--run-dir", type=Path, help="private agy evidence, outside knowledge/")
    args = parser.parse_args(argv)
    directory = nodes_dir().parent / "citation_map"
    run_dir = args.run_dir or Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "tennis-lab-citations" / args.search_id
    try:
        if run_dir.resolve().is_relative_to(nodes_dir().parent.resolve()):
            raise ValueError("run-dir must be outside knowledge storage")
        lock_key = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
        import tempfile
        with (Path(tempfile.gettempdir()) / f"citation-explore-{lock_key}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another exploration is writing this citation library") from exc
            reader = CitationReader(args.agy, run_dir)
            explorer = Explorer(directory, Literature(), reader, agy_timeout=args.agy_timeout, time_limit=args.time_limit)
            result = asdict(explorer.explore(args.root, cast(list[Role], args.role), args.max_depth, args.search_id))
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (AgyError, ValueError, OSError, BudgetExceeded) as exc:
        reason = f"agy:{exc.reason}; see private run-dir evidence" if isinstance(exc, AgyError) else str(exc)
        print(f"error: {reason}; search={args.search_id}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"interrupted; resume with --search-id {args.search_id}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
