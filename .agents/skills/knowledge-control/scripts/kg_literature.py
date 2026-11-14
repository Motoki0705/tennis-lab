"""Public full text and conservative reference identity resolution.

Downloads live only in TemporaryDirectory; the caller persists bibliographic
metadata and reading decisions, never this module's FullText objects.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from kg_identifiers import normalize_identifier
from kg_openalex import LookupUnavailable, OpenAlexClient
from kg_reference_match import authors, select_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research-loop/scripts"))
from lit_search import (  # noqa: E402
    Fetch,
    LitSearchError,
    get,
    urllib_fetch,
)


@dataclass
class Bibliography:
    title: str
    external_ids: dict[str, str]
    source: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    pdf_urls: list[str] = field(default_factory=list)
    location_notes: list[str] = field(default_factory=list)


@dataclass
class FullText:
    text: str
    source: str
    format: str
    attempts: list[str] = field(default_factory=list)
    reference_count: int | None = None


class TextUnavailable(RuntimeError):
    pass


class MetadataConflict(ValueError):
    """A provider assigned contradictory identifiers to one work."""


def input_identifiers(value: str) -> dict[str, str]:
    """Accept external IDs and primary URLs, including arXiv HTML URLs."""
    value = value.strip()
    arxiv_url = re.sub(r"^(https?://(?:www\.)?arxiv\.org)/html/", r"\1/abs/", value, flags=re.I)
    if urlsplit(arxiv_url).hostname in {"arxiv.org", "www.arxiv.org"}:
        arxiv_url = arxiv_url.split("?", 1)[0].split("#", 1)[0]
    for namespace, candidate in (("doi", value), ("arxiv", arxiv_url), ("openalex", re.sub(r"^openalex:", "", value, flags=re.I))):
        try:
            return {namespace: normalize_identifier(namespace, candidate)}
        except ValueError:
            continue
    return {"url": normalize_identifier("url", value)}


def primary_url(ids: dict[str, str]) -> str:
    if "arxiv" in ids:
        return f"https://arxiv.org/abs/{ids['arxiv']}"
    if "doi" in ids:
        return f"https://doi.org/{ids['doi']}"
    if "url" in ids:
        return ids["url"]
    return f"https://openalex.org/{ids['openalex']}"


class PaperHTML(HTMLParser):
    """Preserve headings, reference anchors and link targets for the reader."""

    def __init__(self, raw: str, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.parts: list[str] = []
        self.meta: dict[str, list[str]] = {}
        self.links: list[str] = []
        self.skip = 0
        self.has_article = False
        self.bibliography_ids: list[str] = []
        self.feed(raw)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag in {"script", "style", "nav", "footer"}:
            self.skip += 1
        if tag == "meta":
            name = a.get("name") or a.get("property") or ""
            if a.get("content"):
                self.meta.setdefault(name.lower(), []).append(str(a["content"]))
        if tag == "article":
            self.has_article = True
        if tag == "li" and "ltx_bibitem" in (a.get("class") or "").split() and a.get("id"):
            self.bibliography_ids.append(str(a["id"]))
        if self.skip:
            return
        if tag in {"p", "div", "li", "tr", "br", "section", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")
        if a.get("id"):
            self.parts.append(f" [anchor:{a['id']}] ")
        if tag == "a" and a.get("href"):
            link = urljoin(self.url, str(a["href"]))
            if urlsplit(link).scheme in {"https", "http"}:
                self.links.append(link)
                self.parts.append(f" [link:{link}] ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer"} and self.skip:
            self.skip -= 1
        if not self.skip and tag in {"p", "li", "tr", "section", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self.parts).splitlines() if line.strip())


def reference_identifiers(citation: str, links: list[str]) -> dict[str, str]:
    """Use explicit identifiers, rejecting contradictory IDs rather than guessing."""
    found: dict[str, set[str]] = {}
    for link in links:
        if link.startswith(("http://", "https://")):
            ids = input_identifiers(link)
            if "url" not in ids:
                for namespace, value in ids.items():
                    found.setdefault(namespace, set()).add(value)
    for match in re.finditer(r"\b10\.[0-9]{4,9}/[^\s<>\[\]{}]+", citation, re.I):
        found.setdefault("doi", set()).add(normalize_identifier("doi", match[0].rstrip(".,;")))
    for match in re.finditer(r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf|html)/)([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z.-]+/[0-9]{7}(?:v[0-9]+)?)", citation, re.I):
        found.setdefault("arxiv", set()).add(normalize_identifier("arxiv", match[1]))
    if any(len(values) != 1 for values in found.values()):
        raise ValueError("reference contains conflicting external identifiers")
    if found:
        return {namespace: next(iter(values)) for namespace, values in found.items()}
    urls = {normalize_identifier("url", value.rstrip(".,;")) for value in [*links, *re.findall(r"https?://[^\s<>\[\]{}]+", citation)]}
    if len(urls) == 1:
        return {"url": next(iter(urls))}
    return {}


class Literature:
    def __init__(self, fetch: Fetch = urllib_fetch, *, client: OpenAlexClient | None = None) -> None:
        self.fetch = fetch
        self.client = client or OpenAlexClient(fetch)

    def _get(self, url: str) -> bytes:
        body: bytes = get(url, source="citation", fetch=self.fetch)
        return body

    def openalex(self, identifier: str) -> Bibliography:
        work = self.client.work(identifier)
        if work is None:
            raise LookupUnavailable("work_not_found")
        return self._bibliography(work)

    @staticmethod
    def _bibliography(work: dict[str, Any]) -> Bibliography:
        ids = {"openalex": normalize_identifier("openalex", work["id"])}
        if work.get("doi"):
            ids["doi"] = normalize_identifier("doi", work["doi"])
        pdfs: list[str] = []
        landings: list[str] = []
        notes = [f"OpenAlex best_oa_location PDF: {(work.get('best_oa_location') or {}).get('pdf_url') or 'none'}"]
        locations = [("best_oa_location", work.get("best_oa_location")), ("primary_location", work.get("primary_location"))]
        locations += [(f"locations[{i}]", loc) for i, loc in enumerate(work.get("locations") or [])]
        for label, loc in locations:
            if not loc:
                continue
            notes.append(f"OpenAlex {label}: PDF={loc.get('pdf_url') or 'none'}; landing={loc.get('landing_page_url') or 'none'}")
            for key, target in (("pdf_url", pdfs), ("landing_page_url", landings)):
                if loc.get(key):
                    target.append(loc[key])
                    extracted = input_identifiers(loc[key])
                    if "arxiv" in extracted:
                        if "arxiv" in ids and ids["arxiv"] != extracted["arxiv"]:
                            raise MetadataConflict("conflicting arXiv locations in OpenAlex metadata")
                        ids["arxiv"] = extracted["arxiv"]
        source = primary_url(ids) if "arxiv" in ids or "doi" in ids else next(iter(landings), primary_url(ids))
        return Bibliography(work["display_name"], ids, source, work.get("publication_year"),
                            authors(work), list(dict.fromkeys(pdfs)), notes)

    def reference_candidates(self, source: Bibliography) -> list[dict[str, Any]] | None:
        """None means source unindexed; [] means an indexed empty reference list."""
        ids = source.external_ids
        identifier = ids.get("openalex")
        if identifier is None and "doi" in ids:
            identifier = "https://doi.org/" + ids["doi"]
        if identifier is None and "arxiv" in ids:
            identifier = "https://doi.org/10.48550/arxiv." + ids["arxiv"]
        if identifier is None:
            return None
        work = self.client.work(identifier)
        if work is None:
            return None
        references = work.get("referenced_works")
        if not isinstance(references, list) or any(not isinstance(x, str) for x in references):
            raise LookupUnavailable("invalid_referenced_works")
        result = []
        for target in dict.fromkeys(references):
            candidate = self.client.work(normalize_identifier("openalex", target))
            if candidate is not None:  # A 404 is a definitive absent index record.
                result.append(candidate)
        print(f"OpenAlex {normalize_identifier('openalex', work['id'])}: {len(result)} referenced-work candidates", file=sys.stderr, flush=True)
        return result

    def identify(self, reference: dict[str, Any], *, candidates: list[dict[str, Any]] | None = None) -> tuple[Bibliography | None, str]:
        ids = reference_identifiers(reference["citation"], reference["links"])
        if ids:
            return Bibliography(reference.get("title") or primary_url(ids), ids, primary_url(ids), reference.get("year"), reference.get("authors", [])), "explicit external ID in reference"
        scope = "referenced_works"
        if not candidates:
            scope = "global_search"
            candidates = self.client.search(reference.get("title") or reference["citation"])
        match, reason = select_reference(reference, candidates)
        if match is None:
            return None, f"OpenAlex {scope} ({len(candidates)} candidates): {reason}"
        try:
            bibliography = self._bibliography(match)
        except MetadataConflict:
            return None, f"OpenAlex {scope}: MetadataConflict (conflicting identifiers)"
        return bibliography, f"OpenAlex {scope}: {reason}"

    def full_text(self, bibliography: Bibliography) -> FullText:
        ids = bibliography.external_ids
        candidates: list[str] = []
        attempts: list[str] = []
        if "arxiv" in ids:
            version = re.search(r"(?:abs|pdf|html)/([^?#]+)", bibliography.source)
            arxiv = version[1].removesuffix(".pdf") if version and "arxiv.org/" in bibliography.source else ids["arxiv"]
            candidates += [f"https://arxiv.org/html/{arxiv}", f"https://arxiv.org/pdf/{arxiv}"]
        candidates += bibliography.pdf_urls + [bibliography.source]
        attempts += bibliography.location_notes
        metadata_done = False
        visited: set[str] = set()
        while candidates or not metadata_done:
            if not candidates:
                metadata_done = True
                key = ids.get("openalex") or (f"https://doi.org/{ids['doi']}" if "doi" in ids else None)
                if key is None:
                    continue
                work = self.client.work(key)
                if work is None:
                    attempts.append("OpenAlex source record: not indexed (HTTP 404)")
                    continue
                enriched = self._bibliography(work)
                attempts.extend(enriched.location_notes)
                for namespace, value in enriched.external_ids.items():
                    if namespace in ids and ids[namespace] != value:
                        raise MetadataConflict(f"conflicting {namespace} in OpenAlex full-text metadata")
                bibliography.external_ids = {**ids, **enriched.external_ids}
                candidates.extend(enriched.pdf_urls)
                if "arxiv" in enriched.external_ids:
                    arxiv = enriched.external_ids["arxiv"]
                    candidates += [f"https://arxiv.org/html/{arxiv}", f"https://arxiv.org/pdf/{arxiv}"]
                candidates.append(enriched.source)
                continue
            url = candidates.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if urlsplit(url).hostname in {"openalex.org", "www.openalex.org", "api.openalex.org"}:
                attempts.append(f"{url}: metadata record, not full text")
                continue
            try:
                attempts.append(f"Attempted full-text candidate: {url}")
                raw = self._get(url)
                if b"%PDF-" in raw[:1024]:
                    text = self._pdf_text(raw)
                    return FullText(text, url, "pdf", attempts)
                html = PaperHTML(raw.decode("utf-8"), url)
                # Abstract/catalogue pages must never masquerade as full text.
                if html.has_article and re.search(r"\b(references|bibliography)\b", html.text, re.I) and len(html.text) > 1500:
                    return FullText(html.text, url, "html", attempts, len(html.bibliography_ids) or None)
                pdfs = html.meta.get("citation_pdf_url", [])
                pdfs += [link for link in html.links if urlsplit(link).path.lower().endswith(".pdf")]
                candidates.extend(urljoin(url, link) for link in pdfs)
                attempts.append(f"{url}: landing/abstract HTML, not full text")
            except (LitSearchError, UnicodeError, TextUnavailable) as exc:
                attempts.append(f"{url}: {exc}")
        raise TextUnavailable("; ".join(attempts) or "no public full-text location")

    @staticmethod
    def _pdf_text(raw: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="citation-pdf-") as name:
            pdf = Path(name) / "paper.pdf"
            pdf.write_bytes(raw)
            try:
                result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, timeout=60, check=True)
                pages = result.stdout.decode("utf-8").split("\f")
            except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
                raise TextUnavailable(f"PDF text extraction failed: {type(exc).__name__}") from exc
            text = "\n".join(f"[Page {i}]\n{page.strip()}" for i, page in enumerate(pages, 1) if page.strip())
            if len(text) < 500:
                raise TextUnavailable("PDF contains insufficient extractable text (OCR is not performed)")
            return text
