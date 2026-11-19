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
from typing import Any, cast

from kg_citation_reader import AgyError, CitationReader
from kg_citations import (
    ROLES,
    CitationPaper,
    CitationSearch,
    Generation,
    IdentificationStatus,
    Reference,
    ReferenceLookup,
    Role,
    identification_pending,
    legacy_lookup_failure,
    legacy_search_result,
    load_citations,
    rebuild_search,
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
    reference_identifiers,
)
from kg_openalex import IncompleteSearch, LookupUnavailable, OpenAlexClient


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
    pending_references: int
    unverified_references: int
    identified_papers: list[str]
    refreshed_fetch_failures: list[str]


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
        self.previous_scopes = {s.id: {p for g in s.generations for p in g.expanded_ids} for s in self.library.searches.values()}
        self.locations: dict[str, Bibliography] = {}
        self.identified: set[str] = set()
        self.refreshed: set[str] = set()
        self.legacy_failed: set[str] = set()
        self.legacy_identifications: set[str] = set()
        previous = copy.deepcopy(self.library.papers)
        migrate = False
        for paper in self.library.papers.values():
            legacy_search = [r for r in paper.references if legacy_search_result(r)]
            if legacy_search:
                recovered = self.reader.recovered_lookups(paper)
                for ref in legacy_search:
                    old_target = ref.target_id
                    ref.target_id, ref.identification_status = None, "unverified"
                    ref.reason = ref.reason.split("\nIdentification:", 1)[0] + f"\nIdentification: uniqueness unverified; legacy top-5 search; previous target={old_target}"
                    if ref.lookup is None and ref.ref_id in recovered:
                        ref.lookup = ReferenceLookup(**recovered[ref.ref_id])
                paper.body += "\nLegacy top-5 search identities invalidated; reading, citations and classifications retained.\n"
                migrate = True
            if any(legacy_lookup_failure(ref) for ref in paper.references):
                self.legacy_identifications.add(paper.id)
                recovered = self.reader.recovered_lookups(paper)
                for ref in paper.references:
                    if ref.identification_status == "unresolved":
                        ref.identification_status = "pending"
                        ref.reason = ref.reason.split("\nIdentification:", 1)[0] + "\nIdentification: pending; source identification invalidated after transient failures"
                        if ref.lookup is None and ref.ref_id in recovered:
                            ref.lookup = ReferenceLookup(**recovered[ref.ref_id])
                paper.body += "\nLegacy identification results invalidated; original reading and classifications retained.\n"
                migrate = True
            if paper.text_status == "failed" and not any(marker in paper.text_reason for marker in ("Attempted full-text candidate:", "Rejected full-text candidate:")):
                self.legacy_failed.add(paper.id)
                paper.text_status = "not_requested"
                paper.text_reason = "Refreshing legacy full-text candidate diagnostics."
                migrate = True
            migrate |= identification_pending(paper)
        if migrate:
            self._persist_papers(previous)
        else:
            self._validate()

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
        for search in self.library.searches.values():
            rebuild_search(search, self.library.papers)
        self._validate()
        # Targets first: a newly stored source never points to a nonexistent file.
        changed = [p for p in self.library.papers.values() if p.id not in previous or p != previous[p.id]]
        for paper in sorted(changed, key=lambda p: bool(p.references)):
            write_document(paper, self.directory)
        for search in self.library.searches.values():
            write_document(search, self.directory)

    def _read(self, paper: CitationPaper, remaining: float) -> bool:
        deadline = self.clock() + remaining
        print(f"reading {paper.id}: {paper.title}", file=sys.stderr, flush=True)
        previous = copy.deepcopy(self.library.papers)
        bibliography = self.locations.get(paper.id) or Bibliography(paper.title, paper.external_ids, paper.source, paper.year, paper.authors)
        try:
            full_text = self.literature.full_text(bibliography)
        except TextUnavailable as exc:
            paper.text_status = "failed"
            paper.text_reason = str(exc)
            if not any(marker in paper.text_reason for marker in ("Attempted full-text candidate:", "Rejected full-text candidate:")):
                paper.text_reason = f"Attempted full-text candidate: {bibliography.source}; {exc}"
            if paper.id in self.legacy_failed:
                self.refreshed.add(paper.id)
            self._persist_papers(previous)
            return False
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise BudgetExceeded("execution reached its time limit during full-text retrieval")
        output, note = self.reader.read(bibliography, full_text, min(remaining, self.agy_timeout))
        self._upsert(bibliography)
        paper.title, paper.year, paper.authors = output["title"], output["year"], output["authors"]
        paper.summary = output["summary"]
        paper.references = []
        for item in output["references"]:
            ids = reference_identifiers(item["citation"], item["links"])
            target = self._upsert(Bibliography(item["title"] or primary_url(ids), ids, primary_url(ids), item["year"], item["authors"])) if ids else None
            state: IdentificationStatus = "resolved" if target is not None else "pending"
            reason = "explicit external ID in reference" if target is not None else "pending"
            paper.references.append(Reference(item["ref_id"], item["citation"], target.id if target else None,
                                              state, item["roles"], item["decision_status"],
                                              item["reason"] + "\nIdentification: " + reason, item["evidence"],
                                              ReferenceLookup(item["title"], item["year"], item["authors"], item["links"])))
        paper.text_status, paper.source = "available", full_text.source
        paper.text_reason, paper.body = "; ".join(full_text.attempts), note
        if paper.id in self.legacy_failed:
            self.refreshed.add(paper.id)
        self._persist_papers(previous)
        print(f"saved reading {paper.id}; {len(paper.references)} references pending identification", file=sys.stderr, flush=True)
        return True

    def _identify(self, paper: CitationPaper, remaining: float) -> None:
        deadline = self.clock() + remaining
        previous = copy.deepcopy(self.library.papers)
        bibliography = Bibliography(paper.title, paper.external_ids, paper.source, paper.year, paper.authors)
        print(f"identifying references for {paper.id}: {paper.title}", file=sys.stderr, flush=True)
        try:
            # Explicit bibliography IDs do not require an available index/search.
            for ref in paper.references:
                if ref.identification_status not in {"pending", "unverified"}:
                    continue
                links = ref.lookup.links if ref.lookup else []
                if reference_identifiers(ref.citation, links):
                    query = asdict(ref.lookup) if ref.lookup else {"title": "", "year": None, "authors": [], "links": []}
                    target, reason = self.literature.identify({"citation": ref.citation, **query}, candidates=[])
                    if target is None:
                        raise ValueError("an explicit identifier did not resolve")
                    ref.target_id = self._upsert(target).id
                    ref.identification_status = "resolved"
                    ref.reason = ref.reason.split("\nIdentification:", 1)[0] + "\nIdentification: " + reason
            if not identification_pending(paper):
                self.identified.add(paper.id)
                self._persist_papers(previous)
                return
            candidates = self.literature.reference_candidates(bibliography)
            for ref in paper.references:
                if ref.identification_status not in {"pending", "unverified"}:
                    continue
                if self.clock() >= deadline:
                    raise BudgetExceeded("execution reached its time limit during reference identification")
                # The original bibliography is durable, including after a process
                # restart. Candidate titles/years/author names are matched to it.
                query = asdict(ref.lookup) if ref.lookup else {"title": "", "year": None, "authors": [], "links": []}
                item: dict[str, Any] = {"citation": ref.citation, **query}
                try:
                    target, reason = self.literature.identify(item, candidates=candidates)
                except IncompleteSearch as exc:
                    ref.target_id, ref.identification_status = None, "unverified"
                    ref.reason = ref.reason.split("\nIdentification:", 1)[0] + "\nIdentification: " + str(exc)
                    raise
                except LookupUnavailable as exc:
                    ref.target_id, ref.identification_status = None, "pending"
                    ref.reason = ref.reason.split("\nIdentification:", 1)[0] + "\nIdentification: pending; " + str(exc)
                    raise
                ref.target_id = self._upsert(target).id if target is not None else None
                ref.identification_status = "resolved" if target is not None else "unresolved"
                ref.reason = ref.reason.split("\nIdentification:", 1)[0] + "\nIdentification: " + reason
        except (Exception, KeyboardInterrupt):
            # Successful rows survive; the failed/unattempted rows remain pending.
            self._persist_papers(previous)
            raise
        self.identified.add(paper.id)
        self._persist_papers(previous)
        print(f"saved {paper.id}", file=sys.stderr, flush=True)

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
        self.library.searches[search.id] = search
        self._persist_papers(previous)
        newly_read: set[str] = set()
        cached = {p.id for p in self.library.papers.values() if p.text_status != "not_requested"}
        cached |= self.legacy_failed
        history = search.body

        def snapshot(reason: str = "") -> ExplorationResult:
            for saved in self.library.searches.values():
                rebuild_search(saved, self.library.papers)
            if reason:
                search.status, search.stop_reason = "interrupted", reason
            # Count the known scope even when an incomplete source prevents its
            # expansion. These edges are evidence, not permission to read ahead.
            required: set[str] = set()
            frontier = {search.root_id}
            for _ in range(max_depth):
                required.update(frontier)
                frontier = {r.target_id for p in frontier for r in self.library.papers[p].references
                            if r.target_id is not None and r.decision_status == "classified"
                            and set(r.roles) & set(search.roles)} - required
            selected = [self.library.papers[p] for p in required]
            pending = [p for p in selected if p.text_status == "not_requested" or identification_pending(p)]
            result = ExplorationResult(search.id, self.clock() - started, self.reader.calls - calls_before,
                                       len(newly_read), len(cached & required), sum(p.text_status == "available" for p in selected),
                                       sum(p.text_status == "failed" for p in selected),
                                       sum(r.identification_status == "unresolved" for p in selected for r in p.references),
                                       sum(r.decision_status == "unclassifiable" for p in selected for r in p.references),
                                       len(pending), search.status,
                                       sum(r.identification_status == "pending" for p in selected for r in p.references),
                                       sum(r.identification_status == "unverified" for p in selected for r in p.references),
                                       sorted(self.identified), sorted(self.refreshed))
            report = asdict(result)
            search.body = history + "\nAttempt report (seconds are for this invocation):\n\n```json\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n"
            self._validate()
            for saved in self.library.searches.values():
                write_document(saved, self.directory)
            return result

        snapshot()
        active_paper = paper.id
        try:
            # Repair already-read records from this search before retrying its
            # root. A missing root index/search quota must not prevent independent
            # repairs of the old, explicitly recorded expansion scope.
            scope = self.previous_scopes.get(search.id, set())
            repair = sorted(((self.legacy_identifications | self.legacy_failed) & scope) - {search.root_id})
            for paper_id in repair:
                active_paper = paper_id
                old = self.library.papers[paper_id]
                remaining = self.time_limit - (self.clock() - started)
                if remaining <= 0:
                    raise BudgetExceeded("execution reached its time limit")
                if old.text_status == "not_requested" and self._read(old, remaining):
                    newly_read.add(old.id)
                if identification_pending(old):
                    self._identify(old, self.time_limit - (self.clock() - started))
                snapshot()
            while True:
                unfinished = [p for generation in search.generations if generation.depth < max_depth
                              for p in generation.paper_ids if self.library.papers[p].text_status == "not_requested"
                              or identification_pending(self.library.papers[p])]
                if not unfinished:
                    break
                active_paper = unfinished[0]
                paper = self.library.papers[active_paper]
                remaining = self.time_limit - (self.clock() - started)
                if remaining <= 0:
                    raise BudgetExceeded("execution reached its time limit")
                if paper.text_status == "not_requested":
                    if self._read(paper, remaining):
                        newly_read.add(paper.id)
                elif identification_pending(paper):
                    self._identify(paper, remaining)
                snapshot()
        except (Exception, KeyboardInterrupt) as exc:
            reason = f"agy:{exc.reason}" if isinstance(exc, AgyError) else ("interrupted by operator" if isinstance(exc, KeyboardInterrupt) else str(exc))
            reason += "; paper=" + active_paper
            snapshot(reason)
            raise
        return snapshot()


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
            literature = Literature(client=OpenAlexClient(state_file=run_dir / "openalex-throttle.json"))
            literature.client.deadline = time.monotonic() + args.time_limit
            explorer = Explorer(directory, literature, reader, agy_timeout=args.agy_timeout, time_limit=args.time_limit)
            result = asdict(explorer.explore(args.root, cast(list[Role], args.role), args.max_depth, args.search_id))
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (AgyError, LookupUnavailable, ValueError, OSError, BudgetExceeded) as exc:
        reason = f"agy:{exc.reason}; see private run-dir evidence" if isinstance(exc, AgyError) else str(exc)
        print(f"error: {reason}; search={args.search_id}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"interrupted; resume with --search-id {args.search_id}", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
