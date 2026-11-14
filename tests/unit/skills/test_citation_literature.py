"""Identity precision, full-text gating, and temporary PDF lifetime."""
from __future__ import annotations

import importlib
import json
import subprocess
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/skills/knowledge-control/scripts"))
    return importlib.import_module("kg_literature")


@pytest.mark.parametrize("value,expected", [
    ("https://arxiv.org/html/2311.05237v2", {"arxiv": "2311.05237"}),
    ("https://arxiv.org/pdf/2311.05237v2.pdf?download=1#page=2", {"arxiv": "2311.05237"}),
    ("https://doi.org/10.5555/EXAMPLE", {"doi": "10.5555/example"}),
    ("openalex:W123", {"openalex": "W123"}),
    ("not-an-id", None),
    ("https://openalex.org/W123", {"openalex": "W123"}),
    ("W123", {"openalex": "W123"}),
    ("https://Example.org/paper#refs", {"url": "https://example.org/paper"}),
])
def test_input_identifiers(api: Any, value: str, expected: dict[str, str] | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            api.input_identifiers(value)
    else:
        assert api.input_identifiers(value) == expected


def test_explicit_ids_and_conflicting_reference_ids(api: Any) -> None:
    assert api.reference_identifiers("A. arXiv:2311.05237v1. doi:10.5555/EXAMPLE.", []) == {"arxiv": "2311.05237", "doi": "10.5555/example"}
    with pytest.raises(ValueError, match="conflicting"):
        api.reference_identifiers("doi:10.5555/one", ["https://doi.org/10.5555/two"])


@pytest.mark.parametrize("variant,resolved", [("exact", True), ("near", False), ("ambiguous", False), ("year", False)])
def test_title_lookup_resolves_only_unique_exact_title_and_year(api: Any, variant: str, resolved: bool) -> None:
    work = {"id": "https://openalex.org/W123", "display_name": "An Exact Title", "publication_year": 2023,
            "doi": "https://doi.org/10.5555/exact", "authorships": [{"author": {"display_name": "Alice Smith"}}], "primary_location": {},
            "best_oa_location": {"landing_page_url": "https://arxiv.org/abs/2311.05237", "pdf_url": "https://arxiv.org/pdf/2311.05237"}}
    if variant == "near":
        work["display_name"] = "A Different Title"
    elif variant == "year":
        work["publication_year"] = 2020

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        if "/works?" in url:
            works = [work, {**work, "id": "https://openalex.org/W124"}] if variant == "ambiguous" else [work]
            return json.dumps({"meta": {"count": len(works), "next_cursor": None}, "results": works}).encode()
        return json.dumps(work).encode()

    found, reason = api.Literature(fetch).identify({"citation": "Alice Smith. An Exact Title. 2023.", "title": "An Exact Title", "authors": ["Alice Smith"], "year": 2023, "links": []})
    assert (found is not None) == resolved
    assert reason
    if found:
        assert found.external_ids == {"doi": "10.5555/exact", "openalex": "W123", "arxiv": "2311.05237"}
        assert found.source == "https://arxiv.org/abs/2311.05237"


def test_html_preserves_locations_links_and_rejects_abstract_only(api: Any) -> None:
    full = '<article><h1>A</h1><h2 id="sec3">3 Experiments</h2><p>' + "Full paper body. " * 120 + '</p><h2>References</h2><li id="bib1" class="ltx_bibitem"><a href="https://doi.org/10.5555/example">Example</a></li></article>'

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        return full.encode() if "/html/" in url else b"<html><h1>A</h1><h2>Abstract</h2>Not the full text</html>"

    literature = api.Literature(fetch)
    text = literature.full_text(api.Bibliography("A", {"arxiv": "2311.05237"}, "https://arxiv.org/abs/2311.05237v2"))
    assert text.source == "https://arxiv.org/html/2311.05237v2"
    assert "[anchor:sec3]" in text.text and "[anchor:bib1]" in text.text
    assert "https://doi.org/10.5555/example" in text.text
    assert text.reference_count == 1
    with pytest.raises(api.TextUnavailable, match="not full text"):
        literature.full_text(api.Bibliography("A", {"url": "https://example.org/a"}, "https://example.org/a"))


def test_public_lookup_failure_keeps_status_without_environment_contact(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENALEX_EMAIL", "private-contact@example.invalid")

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        assert "mailto=private-contact%40example.invalid" in url
        raise HTTPError(url, 403, "forbidden", Message(), None)

    with pytest.raises(api.LookupUnavailable) as caught:
        api.Literature(fetch).identify({"citation": "Author. Exact Title.", "title": "Exact Title", "year": None, "authors": [], "links": []})
    reason = str(caught.value)
    assert "HTTP_403" in reason
    assert "private-contact" not in reason and "mailto" not in reason and "?" not in reason


def test_conflicting_provider_locations_are_unresolved_instead_of_last_wins(api: Any) -> None:
    work = {"id": "https://openalex.org/W123", "display_name": "Exact Title", "publication_year": 2023,
            "doi": "https://doi.org/10.5555/exact", "authorships": [{"author": {"display_name": "Alice Smith"}}],
            "locations": [{"landing_page_url": "https://arxiv.org/abs/2204.01899"},
                          {"landing_page_url": "https://arxiv.org/abs/1902.07304"}]}

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        return json.dumps({"meta": {"count": 1, "next_cursor": None}, "results": [work]} if "/works?" in url else work).encode()

    found, reason = api.Literature(fetch).identify({"citation": "Alice Smith. Exact Title.", "title": "Exact Title", "year": 2023, "authors": ["Alice Smith"], "links": []})
    assert found is None and "MetadataConflict" in reason


def test_pdf_fallback_uses_pages_and_removes_temporary_file(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Path] = []

    def extract(argv: list[str], **kwargs: Any) -> Any:
        pdf = Path(argv[2])
        assert pdf.read_bytes().startswith(b"%PDF-")
        captured.append(pdf)
        return subprocess.CompletedProcess(argv, 0, ("A\nAbstract\n" + "First page. " * 60 + "\fSecond page. " * 40).encode(), b"")

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        if "/html/" in url:
            raise HTTPError(url, 404, "no HTML", Message(), None)
        return b"%PDF-1.5 temporary test payload"

    monkeypatch.setattr(api.subprocess, "run", extract)
    text = api.Literature(fetch).full_text(api.Bibliography("A", {"arxiv": "2311.05237"}, "https://arxiv.org/abs/2311.05237"))
    assert text.format == "pdf" and "[Page 1]" in text.text and "[Page 2]" in text.text
    assert any("HTTP 404" in attempt for attempt in text.attempts)
    assert captured and all(not path.exists() for path in captured)


@pytest.mark.parametrize("with_pdf", [False, True])
def test_failed_fetch_records_all_oa_locations_and_attempted_candidates(api: Any, with_pdf: bool) -> None:
    work = {"id": "https://openalex.org/W1", "display_name": "Closed paper", "doi": "https://doi.org/10.5555/closed",
            "authorships": [], "best_oa_location": None, "locations": [{"pdf_url": "https://example.org/closed.pdf" if with_pdf else None,
                                                                         "landing_page_url": "https://example.org/landing"}]}
    fetched: list[str] = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        fetched.append(url)
        if url.startswith("https://api.openalex.org/"):
            return json.dumps(work).encode()
        if url.endswith(".pdf"):
            raise HTTPError(url, 404, "missing", Message(), None)
        return b"<html>Abstract/landing only</html>"

    with pytest.raises(api.TextUnavailable) as caught:
        api.Literature(fetch).full_text(api.Bibliography("Closed paper", {"openalex": "W1"}, "https://example.org/landing"))
    reason = str(caught.value)
    assert "best_oa_location PDF: none" in reason and "locations[0]" in reason
    assert "Attempted full-text candidate: https://example.org/landing" in reason
    if with_pdf:
        assert "https://example.org/closed.pdf" in fetched
        assert "Attempted full-text candidate: https://example.org/closed.pdf" in reason and "HTTP 404" in reason


def test_landing_page_pdf_anchors_cannot_substitute_another_paper(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[str] = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        fetched.append(url)
        if url.endswith(".pdf"):
            return b"%PDF-1.5 another paper"
        return b'<meta name="citation_title" content="Target paper with unique title"><h1>Abstract</h1><a href="reference-b.pdf">Reference B</a><a href="related-b.pdf">Related paper</a><a href="supplement.pdf">Supplementary material</a>'

    monkeypatch.setattr(api.Literature, "_pdf_text", staticmethod(lambda raw: "Unrelated paper B\n" + "Body of B. " * 100))
    lib = api.Literature(fetch)
    with pytest.raises(api.TextUnavailable, match="unverified PDF anchors ignored"):
        lib.full_text(api.Bibliography("Target paper with unique title", {"url": "https://example.org/a"}, "https://example.org/a"))
    assert fetched == ["https://example.org/a"]


@pytest.mark.parametrize("variant", ["correct", "wrong_title", "short_title", "supplement_text", "supplement_url", "non_http"])
def test_citation_pdf_metadata_requires_the_target_title(api: Any, monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    fetched: list[str] = []
    pdf_url = "supplement.pdf" if variant == "supplement_url" else "paper.pdf"
    if variant == "non_http":
        pdf_url = "file:///private/paper.pdf"
    target_title = "Paper A" if variant == "short_title" else "Target paper with unique title"

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        fetched.append(url)
        if url.endswith(".pdf"):
            return b"%PDF-1.5 payload"
        return f'<meta name="citation_title" content="{target_title}"><meta name="citation_pdf_url" content="{pdf_url}">'.encode()

    title = "Another paper with a different title" if variant == "wrong_title" else "Target paper with\nunique title"
    if variant == "short_title":
        title = "Paper B"
    if variant == "supplement_text":
        title = "Supplemental material\n" + title
    monkeypatch.setattr(api.Literature, "_pdf_text", staticmethod(lambda raw: "[Page 1]\n" + title + "\nAuthors\nAbstract\n" + "Body. " * 100))
    lib = api.Literature(fetch)
    bib = api.Bibliography("https://example.org/a", {"url": "https://example.org/a"}, "https://example.org/a")
    if variant == "correct":
        text = lib.full_text(bib)
        assert text.source == "https://example.org/paper.pdf"
        assert "Identity verified: citation_pdf_url" in " ".join(text.attempts)
    else:
        with pytest.raises(api.TextUnavailable, match="identity unverified|candidate rejected"):
            lib.full_text(bib)
    if variant in {"supplement_url", "non_http"}:
        assert fetched == ["https://example.org/a"]


def test_article_heading_mismatch_is_not_hidden_by_page_metadata(api: Any) -> None:
    html = '<meta name="citation_title" content="Target paper with unique title"><article><h1>Another paper title</h1><p>' + "Other paper body. " * 120 + '</p><h2>References</h2></article>'
    lib = api.Literature(lambda url, headers, timeout: html.encode())
    with pytest.raises(api.TextUnavailable, match="article heading differs"):
        lib.full_text(api.Bibliography("Target paper with unique title", {"url": "https://example.org/a"}, "https://example.org/a"))


def test_primary_source_cannot_override_a_different_external_id(api: Any) -> None:
    def forbidden(*args: Any) -> bytes:
        raise AssertionError("conflicting source must not be fetched")

    with pytest.raises(api.TextUnavailable, match="primary URL external ID conflicts"):
        api.Literature(forbidden).full_text(api.Bibliography("A target", {"arxiv": "2311.05237"}, "https://arxiv.org/html/1902.07304"))


@pytest.mark.parametrize("doi", ["10.5555/b", "not-a-doi"])
def test_landing_metadata_for_another_doi_cannot_supply_the_pdf(api: Any, monkeypatch: pytest.MonkeyPatch, doi: str) -> None:
    fetched: list[str] = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        fetched.append(url)
        return f'<meta name="citation_title" content="Another paper"><meta name="citation_doi" content="{doi}"><meta name="citation_pdf_url" content="b.pdf">'.encode()

    lib = api.Literature(fetch)
    monkeypatch.setattr(lib.client, "work", lambda identifier: None)
    with pytest.raises(api.TextUnavailable, match="citation_doi"):
        lib.full_text(api.Bibliography("doi:10.5555/a", {"doi": "10.5555/a"}, "https://example.org/a"))
    assert fetched == ["https://example.org/a"]
