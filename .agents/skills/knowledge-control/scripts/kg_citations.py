"""Typed Markdown contract for citation readers, writers and the library validator.

Local parsing rejects malformed documents. ``validate_library`` additionally
checks identities, references, search generations and Papers joins. No model API
or network access is used here.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from kg_identifiers import (
    citation_id,
    external_id_index,
    identity_keys,
    normalize_external_ids,
    resolve_identity,
)
from kg_lib import Node, UniqueLoader, dump_frontmatter, nodes_dir
from kg_schema import http_url

Role = Literal["foundation", "component", "comparison", "related", "benchmark"]
TextStatus = Literal["available", "failed", "not_requested"]
IdentificationStatus = Literal["resolved", "unresolved", "pending", "unverified"]
DecisionStatus = Literal["classified", "unclassifiable", "pending"]
ROLES = {"foundation", "component", "comparison", "related", "benchmark"}
TEXT_STATUSES = {"available", "failed", "not_requested"}
IDENTIFICATION_STATUSES = {"resolved", "unresolved", "pending", "unverified"}
DECISION_STATUSES = {"classified", "unclassifiable", "pending"}
CITATION_RE = re.compile(r"^cite-[0-9a-f]{16}$")
SEARCH_RE = re.compile(r"^search-[a-z0-9]+(?:-[a-z0-9]+)*$")
DOCUMENT_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)(.*)$", re.DOTALL)


@dataclass
class ReferenceLookup:
    """Query inputs parsed from the citing bibliography, not target metadata."""
    title: str
    year: int | None
    authors: list[str]
    links: list[str]


@dataclass
class Reference:
    ref_id: str
    citation: str
    target_id: str | None
    identification_status: IdentificationStatus
    roles: list[Role]
    decision_status: DecisionStatus
    reason: str
    evidence: list[str]
    lookup: ReferenceLookup | None = None


@dataclass
class CitationPaper:
    id: str
    identity_key: str
    title: str
    year: int | None
    authors: list[str]
    external_ids: dict[str, str]
    source: str
    summary: str
    text_status: TextStatus
    text_reason: str
    references: list[Reference]
    body: str = ""
    schema_version: int = 1
    type: Literal["citation_paper"] = "citation_paper"


@dataclass
class Generation:
    depth: int
    paper_ids: list[str]
    expanded_ids: list[str]


@dataclass(frozen=True)
class IssuePointer:
    paper_id: str
    ref_id: str | None


@dataclass
class CitationSearch:
    id: str
    created_at: str
    root_id: str
    roles: list[Role]
    max_depth: int
    generations: list[Generation]
    issues: list[IssuePointer]
    body: str = ""
    schema_version: int = 1
    type: Literal["citation_search"] = "citation_search"
    status: Literal["complete", "interrupted"] = "complete"
    stop_reason: str = ""


Document = CitationPaper | CitationSearch


@dataclass
class CitationLibrary:
    papers: dict[str, CitationPaper] = field(default_factory=dict)
    searches: dict[str, CitationSearch] = field(default_factory=dict)


def _mapping(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{context}: expected keys {sorted(keys)}")
    return value


def _text(value: Any, context: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{context}: expected {'a string' if empty else 'nonempty text'}")
    return value


def _strings(value: Any, context: str, *, empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not empty and not value):
        raise ValueError(f"{context}: expected {'a' if empty else 'a nonempty'} list")
    result = [_text(x, context) for x in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{context}: duplicate entries")
    return result


def _choice(value: Any, choices: set[str], context: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{context}: expected one of {sorted(choices)}")
    return value


def _roles(value: Any, *, empty: bool = True) -> list[Role]:
    return [cast(Role, _choice(x, ROLES, "roles")) for x in _strings(value, "roles", empty=empty)]


def _integer(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return value


def _paper_id(value: Any) -> str:
    if not isinstance(value, str) or not CITATION_RE.fullmatch(value):
        raise ValueError(f"invalid citation paper id: {value}")
    return value


def _ref_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"ref-[0-9]+", value):
        raise ValueError(f"invalid reference id: {value}")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected a list")
    return value


def _reference(value: Any) -> Reference:
    if not isinstance(value, dict):
        raise ValueError("reference: expected a mapping")
    d = dict(value)
    query = d.pop("lookup", None)
    lookup = None
    if query is not None:
        q = _mapping(query, {"title", "year", "authors", "links"}, "reference lookup")
        if q["year"] is not None and (type(q["year"]) is not int or not 1000 <= q["year"] <= 9999):
            raise ValueError("reference lookup year must be 1000..9999 or null")
        links = _strings(q["links"], "reference lookup links")
        if any(not http_url(link) for link in links):
            raise ValueError("reference lookup links require HTTP(S) URLs")
        lookup = ReferenceLookup(_text(q["title"], "reference lookup title", empty=True), q["year"], _strings(q["authors"], "reference lookup authors"), links)
    _mapping(d, {"ref_id", "citation", "target_id", "identification_status", "roles", "decision_status", "reason", "evidence"}, "reference")
    target = None if d["target_id"] is None else _paper_id(d["target_id"])
    identification = cast(IdentificationStatus, _choice(d["identification_status"], IDENTIFICATION_STATUSES, "identification_status"))
    decision = cast(DecisionStatus, _choice(d["decision_status"], DECISION_STATUSES, "decision_status"))
    roles = _roles(d["roles"])
    evidence = _strings(d["evidence"], "evidence")
    if (identification == "resolved") != (target is not None):
        raise ValueError("identification_status and target_id disagree")
    if decision == "classified" and (not roles or not evidence):
        raise ValueError("classified reference requires roles and evidence")
    if decision != "classified" and roles:
        raise ValueError("unclassifiable/pending reference must have empty roles")
    return Reference(_ref_id(d["ref_id"]), _text(d["citation"], "citation"), target, identification, roles, decision, _text(d["reason"], "reason"), evidence, lookup)


def _paper(d: dict[str, Any], body: str) -> CitationPaper:
    _mapping(d, {"schema_version", "type", "id", "identity_key", "title", "year", "authors", "external_ids", "source", "summary", "text_status", "text_reason", "references"}, "citation_paper")
    values = d["external_ids"]
    if not isinstance(values, dict) or not values or any(not isinstance(k, str) or not isinstance(v, str) for k, v in values.items()):
        raise ValueError("external_ids: nonempty string mapping required")
    normalized = normalize_external_ids(values)
    if normalized != values:
        raise ValueError("external_ids must use normalized values")
    key = _text(d["identity_key"], "identity_key")
    if key not in identity_keys(normalized) or _paper_id(d["id"]) != citation_id(key):
        raise ValueError("citation id/identity_key must match an external ID")
    year = d["year"]
    if year is not None and (type(year) is not int or not 1000 <= year <= 9999):
        raise ValueError("year: expected 1000..9999 or null")
    if not http_url(d["source"]):
        raise ValueError("source: HTTP(S) URL required")
    status = cast(TextStatus, _choice(d["text_status"], TEXT_STATUSES, "text_status"))
    summary = _text(d["summary"], "summary", empty=status != "available")
    reason = _text(d["text_reason"], "text_reason", empty=status != "failed")
    refs = [_reference(x) for x in _list(d["references"], "references")]
    if status != "available" and refs:
        raise ValueError("failed/not_requested paper cannot have references")
    if len({r.ref_id for r in refs}) != len(refs):
        raise ValueError("duplicate reference id")
    return CitationPaper(d["id"], key, _text(d["title"], "title"), year, _strings(d["authors"], "authors"), normalized, d["source"], summary, status, reason, refs, body)


def _search(d: dict[str, Any], body: str) -> CitationSearch:
    d = {"status": "complete", "stop_reason": "", **d}
    _mapping(d, {"schema_version", "type", "id", "created_at", "root_id", "roles", "max_depth", "generations", "issues", "status", "stop_reason"}, "citation_search")
    search_id = _text(d["id"], "id")
    if not SEARCH_RE.fullmatch(search_id):
        raise ValueError("search id must be search-kebab-slug")
    created = _text(d["created_at"], "created_at")
    try:
        stamp = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise ValueError("created_at must be an ISO 8601 timestamp with timezone") from exc
    generations: list[Generation] = []
    for value in _list(d["generations"], "generations"):
        g = _mapping(value, {"depth", "paper_ids", "expanded_ids"}, "generation")
        papers = [_paper_id(x) for x in _strings(g["paper_ids"], "paper_ids", empty=False)]
        expanded = [_paper_id(x) for x in _strings(g["expanded_ids"], "expanded_ids")]
        generations.append(Generation(_integer(g["depth"], "depth"), papers, expanded))
    if not generations:
        raise ValueError("generations: root generation required")
    issues: list[IssuePointer] = []
    for value in _list(d["issues"], "issues"):
        p = _mapping(value, {"paper_id", "ref_id"}, "issue pointer")
        issues.append(IssuePointer(_paper_id(p["paper_id"]), None if p["ref_id"] is None else _ref_id(p["ref_id"])))
    if len(set(issues)) != len(issues):
        raise ValueError("duplicate issue pointer")
    status = cast(Literal["complete", "interrupted"], _choice(d["status"], {"complete", "interrupted"}, "search status"))
    stop_reason = _text(d["stop_reason"], "stop_reason", empty=status == "complete")
    if status == "complete" and stop_reason:
        raise ValueError("complete search must have an empty stop_reason")
    return CitationSearch(search_id, created, _paper_id(d["root_id"]), _roles(d["roles"], empty=False), _integer(d["max_depth"], "max_depth"), generations, issues, body, status=status, stop_reason=stop_reason)


def parse_document(text: str) -> Document:
    match = DOCUMENT_RE.match(text)
    if not match:
        raise ValueError("citation document requires YAML frontmatter")
    try:
        d = yaml.load(match.group(1), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid citation YAML: {exc}") from exc
    if not isinstance(d, dict) or type(d.get("schema_version")) is not int or d["schema_version"] != 1:
        raise ValueError("citation schema_version must be 1")
    if d.get("type") == "citation_paper":
        return _paper(d, match.group(2))
    if d.get("type") == "citation_search":
        return _search(d, match.group(2))
    raise ValueError("type must be citation_paper or citation_search")


def serialize_document(document: Document) -> str:
    data = asdict(document)
    if isinstance(document, CitationPaper):
        for ref in data["references"]:
            if ref["lookup"] is None:
                del ref["lookup"]
    if isinstance(document, CitationSearch) and document.status == "complete":
        del data["status"], data["stop_reason"]
    body = data.pop("body")
    text = f"---\n{dump_frontmatter(data)}---\n{body}"
    parse_document(text)  # Dataclass callers get the same validation as readers.
    return text


def write_document(document: Document, directory: Path | None = None) -> Path:
    """Write one locally validated document; call validate_library on the batch."""
    text = serialize_document(document)
    directory = directory if directory is not None else nodes_dir().parent / "citation_map"
    folder = directory / ("papers" if isinstance(document, CitationPaper) else "searches")
    path = folder / f"{document.id}.md"
    if directory.is_symlink() or folder.is_symlink() or path.is_symlink():
        raise ValueError("symlinks are not allowed in citation storage")
    folder.mkdir(parents=True, exist_ok=True)
    # A killed reader must leave either the previous document or its replacement.
    fd, temporary = tempfile.mkstemp(prefix=".citation-", dir=directory.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_citations(directory: Path | None = None) -> CitationLibrary:
    directory = directory if directory is not None else nodes_dir().parent / "citation_map"
    library = CitationLibrary()
    if directory.is_symlink():
        raise ValueError(f"{directory}: symlinks are not allowed in citation storage")
    if not directory.exists():
        return library
    if not directory.is_dir():
        raise ValueError(f"{directory}: expected citation_map directory")
    for item in directory.iterdir():
        if item.name == "README.md" and item.is_file() and not item.is_symlink():
            continue
        if item.name not in {"papers", "searches"} or item.is_symlink() or not item.is_dir():
            raise ValueError(f"{item}: expected papers/ or searches/ directory")
        for path in sorted(item.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix != ".md":
                raise ValueError(f"{path}: only regular Markdown citation documents are allowed")
            try:
                doc = parse_document(path.read_text(encoding="utf-8"))
                expected = "papers" if isinstance(doc, CitationPaper) else "searches"
                collection = library.papers if isinstance(doc, CitationPaper) else library.searches
                if doc.id in collection:
                    raise ValueError(f"duplicate document id: {doc.id}")
                if item.name != expected or path.stem != doc.id:
                    raise ValueError("document type/id must match path")
                if isinstance(doc, CitationPaper):
                    library.papers[doc.id] = doc
                else:
                    library.searches[doc.id] = doc
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
    return library


def search_issues(expanded_ids: list[str], papers: dict[str, CitationPaper]) -> set[IssuePointer]:
    issues: set[IssuePointer] = set()
    for paper_id in expanded_ids:
        paper = papers[paper_id]
        if paper.text_status == "failed":
            issues.add(IssuePointer(paper_id, None))
        for ref in paper.references:
            if ref.identification_status != "resolved" or ref.decision_status != "classified":
                issues.add(IssuePointer(paper_id, ref.ref_id))
    return issues


def legacy_lookup_failure(ref: Reference) -> bool:
    """Pre-fix CLI outputs incorrectly finalized provider failures as unresolved."""
    return ref.identification_status == "unresolved" and "Identification: reference lookup failed:" in ref.reason


def legacy_search_result(ref: Reference) -> bool:
    return ref.identification_status in {"resolved", "unresolved"} and (
        "Identification: unique exact title/year match in OpenAlex" in ref.reason
        or "Identification: OpenAlex global_search" in ref.reason and "cursor_complete" not in ref.reason)


def identification_pending(paper: CitationPaper) -> bool:
    return any(ref.identification_status in {"pending", "unverified"} or legacy_lookup_failure(ref) for ref in paper.references)


def rebuild_search(search: CitationSearch, papers: dict[str, CitationPaper]) -> None:
    """Reconcile every affected search after shared paper identities change."""
    current = [search.root_id]
    seen: set[str] = set()
    generations: list[Generation] = []
    reason = ""
    for depth in range(search.max_depth + 1):
        seen.update(current)
        expanded = [] if depth == search.max_depth else [p for p in current
                    if papers[p].text_status != "not_requested" and not identification_pending(papers[p])]
        generations.append(Generation(depth, current, expanded))
        if depth == search.max_depth:
            break
        if len(expanded) != len(current):
            waiting = [p for p in current if p not in expanded]
            reason = "Paper processing incomplete: " + ", ".join(waiting)
            break
        following = sorted({ref.target_id for p in expanded for ref in papers[p].references
                            if ref.target_id is not None and ref.decision_status == "classified"
                            and set(ref.roles) & set(search.roles)} - seen)
        if not following:
            break
        current = following
    search.generations = generations
    expanded_all = [p for generation in generations for p in generation.expanded_ids]
    search.issues = sorted(search_issues(expanded_all, papers), key=lambda p: (p.paper_id, p.ref_id or ""))
    search.status, search.stop_reason = ("interrupted", reason) if reason else ("complete", "")


def _validate_search(search: CitationSearch, papers: dict[str, CitationPaper]) -> list[str]:
    errors: list[str] = []
    expected = {search.root_id}
    seen: set[str] = set()
    expanded_all: list[str] = []
    for depth, generation in enumerate(search.generations):
        reached = set(generation.paper_ids)
        expanded = set(generation.expanded_ids)
        if generation.depth != depth or depth > search.max_depth:
            errors.append("generations must have consecutive depths within max_depth")
        if reached != expected or seen & reached:
            errors.append(f"generation {depth}: paper_ids disagree with selected-role traversal")
        partial_last = search.status == "interrupted" and depth == len(search.generations) - 1 and depth < search.max_depth
        if (partial_last and not expanded <= reached) or (not partial_last and expanded != (reached if depth < search.max_depth else set())):
            errors.append(f"generation {depth}: expanded_ids disagree with max_depth")
        missing = (reached | expanded) - papers.keys()
        if missing:
            errors.append(f"unknown citation papers: {sorted(missing)}")
        seen.update(reached)
        expected = set()
        expanded_all.extend(expanded & papers.keys())
        for paper_id in expanded & papers.keys():
            paper = papers[paper_id]
            if paper.text_status == "not_requested":
                errors.append(f"expanded paper {paper_id} has text_status not_requested")
            if identification_pending(paper):
                errors.append(f"expanded paper {paper_id} has incomplete reference identification")
            expected.update(ref.target_id for ref in paper.references if ref.target_id is not None and ref.decision_status == "classified" and set(ref.roles) & set(search.roles))
        expected -= seen
    if expected and search.status == "complete":
        errors.append("search is missing a selected-role generation")
    if set(search.issues) != search_issues(expanded_all, papers):
        errors.append("search issues disagree with expanded paper/reference states")
    return errors


def papers_join(library: CitationLibrary, registered: list[Node]) -> dict[str, str]:
    """Compute citation ID -> Papers ID; persist neither direction of this join."""
    index = external_id_index((p.id, p.meta.get("external_ids", {})) for p in registered)
    joins: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for paper in library.papers.values():
        match = resolve_identity(paper.external_ids, index)
        if match is not None:
            if match in reverse:
                raise ValueError(f"duplicate literature through Papers {match}: {reverse[match]} and {paper.id}")
            joins[paper.id] = match
            reverse[match] = paper.id
    return joins


def validate_library(library: CitationLibrary, registered: list[Node] | None = None) -> list[str]:
    errors: list[str] = []
    for collection in (library.papers, library.searches):
        for key, document in collection.items():
            if key != document.id:
                errors.append(f"document index key {key} disagrees with id {document.id}")
            try:
                serialize_document(document)
            except ValueError as exc:
                errors.append(f"{document.id}: {exc}")
    if errors:
        return errors
    try:
        external_id_index((p.id, p.external_ids) for p in library.papers.values())
        if registered is not None:
            papers_join(library, registered)
    except ValueError as exc:
        errors.append(str(exc))
    for paper in library.papers.values():
        for ref in paper.references:
            if legacy_lookup_failure(ref):
                errors.append(f"{paper.id}/{ref.ref_id}: transient lookup failure must be pending, not unresolved")
            if legacy_search_result(ref):
                errors.append(f"{paper.id}/{ref.ref_id}: legacy partial search must be unverified until rechecked")
            if ref.target_id is not None and ref.target_id not in library.papers:
                errors.append(f"{paper.id}/{ref.ref_id}: unknown target_id {ref.target_id}")
    for search in library.searches.values():
        errors.extend(f"{search.id}: {e}" for e in _validate_search(search, library.papers))
    return errors
