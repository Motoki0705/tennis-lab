"""Search arXiv, Semantic Scholar, and OpenAlex with one normalized output.

Adapted from ARIS (https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep,
revision 3b19a22dc5a64c983d2eafd6c2c94000fadc4f8d, MIT License; see
../LICENSE.ARIS.txt) as vendored in tennis-lab commit 57ae07631:
``tools/arxiv_fetch.py``, ``tools/semantic_scholar_fetch.py`` and
``tools/openalex_fetch.py``. Changes: one CLI and record schema, standard library
only, injectable HTTP for tests, and no silent fallbacks (no curl rescue, no
command aliases, no partial results when a requested source fails).

Python 3.11+, standard library only. Network failures exit non-zero.

Examples::

    PY=.venv/bin/python; S=.agents/skills/research-loop/scripts/lit_search.py
    $PY $S search "structure from motion broadcast video" --source arxiv --max 5
    $PY $S search "ball trajectory 3D" --source s2 --source openalex --year 2020- --format markdown
    $PY $S arxiv-pdf 2008.04524 --out /tmp/paper.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

USER_AGENT = "tennis-lab-research-loop/1.0 (+https://github.com/Motoki0705/tennis-lab)"
ARXIV_API = "https://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
S2_FIELDS = "paperId,title,abstract,year,venue,url,openAccessPdf,authors,externalIds,citationCount"
SOURCES = ("arxiv", "s2", "openalex")
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MIN_PDF_BYTES = 10_240
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_FIELD = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all):")
_ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})(?:v\d+)?$")
_YEAR = re.compile(r"^(\d{4})?(-)?(\d{4})?$")


class LitSearchError(RuntimeError):
    """A requested source could not be queried or returned unusable data."""


# (url, headers, timeout_seconds) -> response body. Raises urllib errors.
Fetch = Callable[[str, dict[str, str], float], bytes]


def urllib_fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body: bytes = response.read()
    return body


@dataclass(frozen=True)
class Paper:
    source: str
    id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str | None
    url: str
    pdf_url: str | None
    doi: str | None
    arxiv_id: str | None
    citations: int | None


@dataclass(frozen=True)
class YearRange:
    start: int | None
    end: int | None

    @classmethod
    def parse(cls, text: str) -> YearRange:
        match = _YEAR.fullmatch(text.strip())
        if not match or not (match.group(1) or match.group(3)):
            raise ValueError(f"year must be YYYY, YYYY-, -YYYY or YYYY-YYYY: {text!r}")
        start = int(match.group(1)) if match.group(1) else None
        end = int(match.group(3)) if match.group(3) else None
        if not match.group(2):
            end = start
        if start is not None and end is not None and start > end:
            raise ValueError(f"year range is reversed: {text!r}")
        return cls(start, end)


def get(
    url: str,
    *,
    source: str,
    fetch: Fetch,
    headers: dict[str, str] | None = None,
    retries: int = 2,
    backoff: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """GET with bounded retry on rate-limit/server errors; every retry is logged."""
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(retries + 1):
        try:
            return fetch(url, merged, 30.0)
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt < retries:
                delay = backoff * (attempt + 1)
                print(
                    f"[{source}] HTTP {exc.code}; retry {attempt + 1}/{retries} in {delay:.0f}s",
                    file=sys.stderr,
                )
                sleep(delay)
                continue
            raise LitSearchError(f"{source}: HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LitSearchError(f"{source}: network error for {url}: {exc}") from exc
    raise AssertionError("unreachable")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _json(body: bytes, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LitSearchError(f"{source}: response is not JSON: {body[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise LitSearchError(
            f"{source}: expected a JSON object, got {type(payload).__name__}"
        )
    return payload


def _required(value: Any, source: str, field: str) -> str:
    text = _text(value)
    if text is None:
        raise LitSearchError(f"{source}: result without {field}: {value!r}")
    return text


# --- arXiv -----------------------------------------------------------------


def arxiv_query(query: str, years: YearRange | None) -> str:
    query = query.strip()
    if _ARXIV_FIELD.search(query):
        expr = query
    else:
        terms = re.findall(r'"[^"]+"|\S+', query)
        expr = " AND ".join(f"all:{term}" for term in terms)
    if years is not None:
        start = f"{years.start or 1991}01010000"
        end = f"{years.end or 9999}12312359"
        expr = f"({expr}) AND submittedDate:[{start} TO {end}]"
    return expr


def search_arxiv(
    query: str, max_results: int, years: YearRange | None, fetch: Fetch
) -> list[Paper]:
    bare = query.strip().removeprefix("id:")
    if _ARXIV_ID.fullmatch(bare):
        if years is not None:
            raise ValueError("--year cannot be combined with an arXiv ID lookup")
        params: dict[str, Any] = {"id_list": bare}
    else:
        params = {
            "search_query": arxiv_query(query, years),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    body = get(
        f"{ARXIV_API}?{urllib.parse.urlencode(params)}", source="arxiv", fetch=fetch
    )
    if body.strip() == b"Rate exceeded.":
        raise LitSearchError("arxiv: rate limited (body 'Rate exceeded.')")
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise LitSearchError(
            f"arxiv: response is not Atom XML: {body[:200]!r}"
        ) from exc
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = _required(entry.findtext(f"{_ATOM}id"), "arxiv", "id")
        if "/api/errors" in raw_id:
            raise LitSearchError(
                f"arxiv: API error: {_text(entry.findtext(f'{_ATOM}summary'))}"
            )
        arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/", 1)[-1])
        published = entry.findtext(f"{_ATOM}published") or ""
        doi = entry.findtext("{http://arxiv.org/schemas/atom}doi")
        venue = entry.findtext("{http://arxiv.org/schemas/atom}journal_ref")
        papers.append(
            Paper(
                source="arxiv",
                id=arxiv_id,
                title=_required(entry.findtext(f"{_ATOM}title"), "arxiv", "title"),
                authors=[
                    name
                    for author in entry.findall(f"{_ATOM}author")
                    if (name := _text(author.findtext(f"{_ATOM}name")))
                ],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=_text(venue),
                abstract=_text(entry.findtext(f"{_ATOM}summary")),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                doi=_text(doi),
                arxiv_id=arxiv_id,
                citations=None,
            )
        )
    return papers[:max_results]


def download_arxiv_pdf(arxiv_id: str, out: Path, fetch: Fetch) -> Path:
    """Save an arXiv PDF (e.g. for knowledge-control's kg_papers.py --pdf)."""
    clean = re.sub(
        r"v\d+$", "", arxiv_id.strip().removeprefix("id:").split("/abs/")[-1]
    )
    if not _ARXIV_ID.fullmatch(clean):
        raise ValueError(f"not an arXiv ID: {arxiv_id!r}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    data = get(f"https://arxiv.org/pdf/{clean}", source="arxiv", fetch=fetch)
    if len(data) < MIN_PDF_BYTES or b"%PDF-" not in data[:1024]:
        raise LitSearchError(f"arxiv: {clean} did not return a PDF ({len(data)} bytes)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


# --- Semantic Scholar ------------------------------------------------------


def search_s2(
    query: str, max_results: int, years: YearRange | None, fetch: Fetch
) -> list[Paper]:
    params: dict[str, Any] = {"query": query, "limit": max_results, "fields": S2_FIELDS}
    if years is not None:
        params["year"] = (
            f"{years.start or ''}-{years.end or ''}"
            if years.start != years.end
            else str(years.start)
        )
    headers = {"Accept": "application/json"}
    if key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip():
        headers["x-api-key"] = key
    payload = _json(
        get(
            f"{S2_API}?{urllib.parse.urlencode(params)}",
            source="s2",
            fetch=fetch,
            headers=headers,
        ),
        "s2",
    )
    if "data" not in payload:
        raise LitSearchError(f"s2: response has no 'data': {payload}")
    papers = []
    for item in payload["data"] or []:
        external = item.get("externalIds") or {}
        pdf = item.get("openAccessPdf") or {}
        paper_id = _required(item.get("paperId"), "s2", "paperId")
        papers.append(
            Paper(
                source="s2",
                id=paper_id,
                title=_required(item.get("title"), "s2", "title"),
                authors=[
                    name
                    for a in item.get("authors") or []
                    if (name := _text(a.get("name")))
                ],
                year=item.get("year"),
                venue=_text(item.get("venue")),
                abstract=_text(item.get("abstract")),
                url=_text(item.get("url"))
                or f"https://www.semanticscholar.org/paper/{paper_id}",
                pdf_url=_text(pdf.get("url")),
                doi=_text(external.get("DOI")),
                arxiv_id=_text(external.get("ArXiv")),
                citations=item.get("citationCount"),
            )
        )
    return papers


# --- OpenAlex --------------------------------------------------------------


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    words = sorted(
        (pos, word) for word, positions in inverted.items() for pos in positions
    )
    return " ".join(word for _, word in words)


def search_openalex(
    query: str, max_results: int, years: YearRange | None, fetch: Fetch
) -> list[Paper]:
    params: dict[str, Any] = {"search": query, "per_page": min(max_results, 200)}
    filters = []
    if years is not None and years.start is not None:
        filters.append(f"from_publication_date:{years.start}-01-01")
    if years is not None and years.end is not None:
        filters.append(f"to_publication_date:{years.end}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
    if email := os.environ.get("OPENALEX_EMAIL", "").strip():
        params["mailto"] = email
    payload = _json(
        get(
            f"{OPENALEX_API}?{urllib.parse.urlencode(params)}",
            source="openalex",
            fetch=fetch,
        ),
        "openalex",
    )
    if "results" not in payload:
        raise LitSearchError(f"openalex: response has no 'results': {payload}")
    papers = []
    for work in payload["results"] or []:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        oa = work.get("open_access") or {}
        ids = work.get("ids") or {}
        doi = _text(work.get("doi"))
        url = _required(work.get("id"), "openalex", "id")
        arxiv_url = _text(ids.get("arxiv"))
        papers.append(
            Paper(
                source="openalex",
                id=url.rsplit("/", 1)[-1],
                title=_required(
                    work.get("display_name") or work.get("title"), "openalex", "title"
                ),
                authors=[
                    name
                    for a in work.get("authorships") or []
                    if (name := _text((a.get("author") or {}).get("display_name")))
                ],
                year=work.get("publication_year"),
                venue=_text(source.get("display_name")),
                abstract=_openalex_abstract(work.get("abstract_inverted_index")),
                url=url,
                pdf_url=_text(oa.get("oa_url")),
                doi=doi.removeprefix("https://doi.org/") if doi else None,
                arxiv_id=arxiv_url.rsplit("/", 1)[-1] if arxiv_url else None,
                citations=work.get("cited_by_count"),
            )
        )
    return papers[:max_results]


SEARCHERS: dict[str, Callable[[str, int, YearRange | None, Fetch], list[Paper]]] = {
    "arxiv": search_arxiv,
    "s2": search_s2,
    "openalex": search_openalex,
}


def search(
    query: str,
    sources: Sequence[str],
    max_results: int,
    years: YearRange | None = None,
    fetch: Fetch = urllib_fetch,
) -> list[Paper]:
    """Query every requested source; any failure aborts the whole search."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results < 1:
        raise ValueError("--max must be >= 1")
    unknown = sorted(set(sources) - set(SEARCHERS))
    if unknown or not sources:
        raise ValueError(
            f"sources must be a non-empty subset of {SOURCES}, got {list(sources)}"
        )
    results: list[Paper] = []
    for source in dict.fromkeys(sources):
        results.extend(SEARCHERS[source](query, max_results, years, fetch))
    return results


def to_markdown(papers: Sequence[Paper]) -> str:
    lines = [
        "| source | year | title | venue | cites | link |",
        "|---|---|---|---|---|---|",
    ]
    for p in papers:
        title = p.title.replace("|", "\\|")
        venue = (p.venue or "").replace("|", "\\|")
        cites = "" if p.citations is None else str(p.citations)
        lines.append(
            f"| {p.source} | {p.year or ''} | {title} | {venue} | {cites} | {p.url} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="search one or more sources")
    s.add_argument("query")
    s.add_argument(
        "--source", action="append", choices=SOURCES, required=True, help="repeatable"
    )
    s.add_argument(
        "--max", type=int, default=10, help="results per source (default 10)"
    )
    s.add_argument(
        "--year", type=YearRange.parse, help="YYYY, YYYY-, -YYYY or YYYY-YYYY"
    )
    s.add_argument("--format", choices=("json", "markdown"), default="json")
    d = sub.add_parser("arxiv-pdf", help="download one arXiv PDF")
    d.add_argument("arxiv_id")
    d.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None, fetch: Fetch = urllib_fetch) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "search":
            papers = search(args.query, args.source, args.max, args.year, fetch)
            if args.format == "markdown":
                print(to_markdown(papers))
            else:
                print(
                    json.dumps(
                        [asdict(p) for p in papers], ensure_ascii=False, indent=2
                    )
                )
        else:
            print(download_arxiv_pdf(args.arxiv_id, args.out, fetch))
    except (LitSearchError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
