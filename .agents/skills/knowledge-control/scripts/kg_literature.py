"""Public full text and conservative reference identity resolution.

Downloads live only in TemporaryDirectory; the caller persists bibliographic
metadata and reading decisions, never this module's FullText objects.
"""
from __future__ import annotations

import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research-loop/scripts"))
from lit_search import (  # noqa: E402
    Fetch,
    LitSearchError,
    get,
    search_openalex,
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


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"\w+", title.casefold()))


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
    return {namespace: next(iter(values)) for namespace, values in found.items()}


class Literature:
    def __init__(self, fetch: Fetch = urllib_fetch) -> None:
        self.fetch = fetch

    def _get(self, url: str) -> bytes:
        body: bytes = get(url, source="citation", fetch=self.fetch)
        return body

    def openalex(self, identifier: str) -> Bibliography:
        work = json.loads(self._get(f"https://api.openalex.org/works/{identifier}"))
        ids = {"openalex": normalize_identifier("openalex", work["id"])}
        if work.get("doi"):
            ids["doi"] = normalize_identifier("doi", work["doi"])
        pdfs: list[str] = []
        landings: list[str] = []
        for loc in [work.get("best_oa_location"), work.get("primary_location"), *(work.get("locations") or [])]:
            if not loc:
                continue
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
                            [a["author"]["display_name"] for a in work.get("authorships", [])], list(dict.fromkeys(pdfs)))

    def identify(self, reference: dict[str, Any]) -> tuple[Bibliography | None, str]:
        ids = reference_identifiers(reference["citation"], reference["links"])
        if ids:
            return Bibliography(reference["title"], ids, primary_url(ids), reference["year"], reference["authors"]), "explicit external ID in reference"
        if not reference["title"].strip():
            return None, "reference has no identifiable title or external ID"
        try:
            matches = search_openalex(reference["title"], 5, None, self.fetch)
            exact = [p for p in matches if _title_key(p.title) == _title_key(reference["title"])
                     and (reference["year"] is None or p.year == reference["year"])]
            if len(exact) == 1:
                return self.openalex(exact[0].id), "unique exact title/year match in OpenAlex"
            return None, f"OpenAlex exact title/year matches: {len(exact)}; identity left unresolved"
        except (LitSearchError, KeyError, TypeError, ValueError) as exc:
            # lit_search includes request URLs, including optional mailto from
            # the caller's environment. Public documents need the failure class,
            # not private contact/configuration query parameters.
            status = re.search(r"\bHTTP ([0-9]{3})\b", str(exc))
            detail = f"HTTP {status[1]}" if status else type(exc).__name__
            return None, f"reference lookup failed: OpenAlex {detail}"

    def full_text(self, bibliography: Bibliography) -> FullText:
        ids = bibliography.external_ids
        candidates: list[str] = []
        attempts: list[str] = []
        if "arxiv" in ids:
            version = re.search(r"(?:abs|pdf|html)/([^?#]+)", bibliography.source)
            arxiv = version[1].removesuffix(".pdf") if version and "arxiv.org/" in bibliography.source else ids["arxiv"]
            candidates += [f"https://arxiv.org/html/{arxiv}", f"https://arxiv.org/pdf/{arxiv}"]
        candidates += bibliography.pdf_urls + [bibliography.source]
        if "openalex" in ids or "doi" in ids:
            key = ids.get("openalex") or f"https://doi.org/{ids['doi']}"
            try:
                enriched = self.openalex(key)
                for namespace, value in enriched.external_ids.items():
                    if namespace in ids and ids[namespace] != value:
                        raise ValueError(f"conflicting {namespace} in OpenAlex full-text metadata")
                bibliography.external_ids = {**ids, **enriched.external_ids}
                candidates += enriched.pdf_urls
                if "arxiv" in enriched.external_ids:
                    arxiv = enriched.external_ids["arxiv"]
                    candidates += [f"https://arxiv.org/html/{arxiv}", f"https://arxiv.org/pdf/{arxiv}"]
                candidates.append(enriched.source)
            except (LitSearchError, KeyError, TypeError, json.JSONDecodeError) as exc:
                attempts.append(f"OpenAlex full-text location lookup: {exc}")
        visited: set[str] = set()
        while candidates:
            url = candidates.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
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
