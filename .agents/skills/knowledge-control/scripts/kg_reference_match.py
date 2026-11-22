"""Conservative title/year/first-author matching against a queried candidate set."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def words(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text).casefold().replace("&", " and ")
    clean = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.findall(r"\w+", clean)


def authors(work: dict[str, Any]) -> list[str]:
    return [a["author"]["display_name"] for a in work.get("authorships") or []
            if isinstance(a, dict) and isinstance(a.get("author"), dict)
            and isinstance(a["author"].get("display_name"), str)]


def _surname(text: str, title: str = "") -> str:
    first = re.split(r",|;|\s+and\s+|\s+et\s+al", text, maxsplit=1, flags=re.I)[0]
    normalized = " " + " ".join(words(first)) + " "
    if title and " " + title + " " in normalized:
        normalized = normalized.split(" " + title + " ", 1)[0]
    tokens = normalized.split()
    while tokens and (tokens[-1].isdigit() or tokens[-1] in {"et", "al", "jr", "ii", "iii"}):
        tokens.pop()
    return tokens[-1] if tokens else ""


def select_reference(reference: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    citation = " " + " ".join(words(reference["citation"])) + " "
    title = " ".join(words(reference.get("title") or ""))
    years = {reference["year"]} if reference.get("year") is not None else {
        int(y) for y in re.findall(r"\b((?:19|20)\d{2})[a-z]?\b", reference["citation"])}
    title_matches = []
    for work in candidates:
        candidate_title = " ".join(words(work.get("display_name") or work.get("title") or ""))
        if candidate_title and ((title and title == candidate_title) or (not title and " " + candidate_title + " " in citation)):
            title_matches.append(work)
    if not title_matches:
        return None, "no_title_match"
    dated = [w for w in title_matches if w.get("publication_year") in years]
    if not dated:
        return None, "year_mismatch_or_missing"
    matched: dict[str, dict[str, Any]] = {}
    for work in dated:
        names = authors(work)
        if not names:
            continue
        surname = _surname(names[0])
        if not surname:
            continue
        if reference.get("authors"):
            agrees = _surname(reference["authors"][0]) == surname
        else:
            candidate_title = " ".join(words(work.get("display_name") or work.get("title") or ""))
            agrees = _surname(reference["citation"], candidate_title) == surname
        if agrees:
            matched[work["id"]] = work
    if not matched:
        return None, "author_mismatch_or_missing"
    if len(matched) != 1:
        return None, f"ambiguous ({len(matched)} matches)"
    return next(iter(matched.values())), "unique_title_year_first_author_match"
