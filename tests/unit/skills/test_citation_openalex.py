"""Provider pacing, failure semantics and conservative bibliography matching."""
from __future__ import annotations

import importlib
import json
from email.message import Message
from email.utils import formatdate
from http.client import IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/skills/knowledge-control/scripts"))
    return importlib.import_module("kg_openalex")


@pytest.mark.parametrize("retry_after", ["4", formatdate(1004, usegmt=True)])
def test_retry_after_pacing_backoff_and_success_cache(api: Any, retry_after: str) -> None:
    clock = Clock()
    starts: list[float] = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        starts.append(clock.now)
        if len(starts) <= 2:
            response = Message()
            if len(starts) == 1:
                response["Retry-After"] = retry_after
            raise HTTPError(url, 429 if len(starts) == 1 else 503, "temporary", response, None)
        return b'{"id":"https://openalex.org/W1","display_name":"Paper"}'

    client = api.OpenAlexClient(fetch, clock=clock.time, wall_clock=lambda: 1000 + clock.now, sleep=clock.sleep)
    assert client.work("W1")["display_name"] == "Paper"
    client.work("W1")
    client.work("W2")
    assert starts == [0, 4, 8, 9]  # Cache hit is free; a different request is paced.


@pytest.mark.parametrize("failure", [429, 500, 503, "network", "incomplete", "json"])
def test_exhaustion_is_an_error_without_contacts_or_credentials(api: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: int | str) -> None:
    clock = Clock()
    starts: list[float] = []
    monkeypatch.setenv("OPENALEX_EMAIL", "private-contact@example.invalid")
    monkeypatch.setenv("OPENALEX_API_KEY", "private-key")

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        assert "mailto=private-contact%40example.invalid" in url
        assert headers["Authorization"] == "Bearer private-key"
        starts.append(clock.now)
        if failure == "network":
            raise URLError("private-contact@example.invalid")
        if failure == "incomplete":
            raise IncompleteRead(b"private-key", 100)
        if failure == "json":
            return b"private-contact@example.invalid private-key"
        assert isinstance(failure, int)
        raise HTTPError(url, failure, "private-key", Message(), None)

    client = api.OpenAlexClient(fetch, clock=clock.time, sleep=clock.sleep)
    with pytest.raises(api.LookupUnavailable) as caught:
        client.work("W1")
    assert starts == [0, 2, 6, 14]
    diagnostics = str(caught.value) + capsys.readouterr().err + str(client.cache)
    assert "retries_exhausted" in diagnostics
    assert not any(secret in diagnostics for secret in ("private-contact", "private-key", "mailto="))


def test_retry_after_above_bound_stops_without_an_early_retry(api: Any) -> None:
    clock = Clock()
    starts = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        starts.append(clock.now)
        response = Message()
        response["Retry-After"] = "600"
        raise HTTPError(url, 429, "limited", response, None)

    client = api.OpenAlexClient(fetch, clock=clock.time, sleep=clock.sleep)
    with pytest.raises(api.LookupUnavailable, match="retry_after_exceeds_limit"):
        client.work("W1")
    assert starts == [0] and clock.now == 0


def test_deadline_during_backoff_stays_incomplete(api: Any) -> None:
    clock = Clock()

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        raise HTTPError(url, 503, "temporary", Message(), None)

    client = api.OpenAlexClient(fetch, clock=clock.time, sleep=clock.sleep, deadline=1)
    with pytest.raises(api.LookupUnavailable, match="deadline_exceeded"):
        client.work("W1")
    assert clock.now == 0


def test_retry_after_survives_restart_and_allows_singleton_requests(api: Any, tmp_path: Path) -> None:
    clock = Clock()
    requests: list[str] = []
    state = tmp_path / "private/throttle.json"

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        requests.append(url)
        if "/works?" in url:
            response = Message()
            response["Retry-After"] = "600"
            raise HTTPError(url, 429, "limited", response, None)
        return json.dumps(work("W1")).encode()

    def client() -> Any:
        return api.OpenAlexClient(fetch, clock=clock.time, wall_clock=clock.time, sleep=clock.sleep, state_file=state)

    with pytest.raises(api.LookupUnavailable, match="retry_after_exceeds_limit"):
        client().search("Title")
    resumed = client()
    with pytest.raises(api.LookupUnavailable, match="query_cooldown"):
        resumed.search("A different title")
    assert len(requests) == 1 and clock.now == 0
    assert resumed.work("W1")["display_name"] == "An Exact Title"
    assert len(requests) == 2
    assert state.stat().st_mode & 0o777 == 0o600
    saved = json.loads(state.read_text())
    assert set(saved) == {"principal", "not_before"}
    clock.now = 600
    with pytest.raises(api.LookupUnavailable, match="retry_after_exceeds_limit"):
        client().search("Title")
    assert len(requests) == 3


def work(identifier: str, title: str = "An Exact Title", year: int | None = 2023, author: str = "Alice Smith") -> dict[str, Any]:
    return {"id": "https://openalex.org/" + identifier, "display_name": title,
            "publication_year": year, "authorships": [{"author": {"display_name": author}}]}


@pytest.mark.parametrize("case,expected", [
    ("unique", "W1"), ("punctuation", "W1"), ("legacy", "W1"),
    ("wrong_year", None), ("wrong_author", None), ("coauthor_only", None),
    ("wrong_title", None), ("ambiguous", None), ("missing_year", None), ("missing_authors", None),
])
def test_candidate_matching_requires_title_year_and_first_author(api: Any, case: str, expected: str | None) -> None:
    match = importlib.import_module("kg_reference_match")
    reference = {"citation": "[2] Alice Smith, Bob Jones. An Exact Title. 2023.", "title": "An Exact Title", "year": 2023, "authors": ["Alice Smith"]}
    candidates = [work("W1")]
    if case == "punctuation":
        reference["title"] = "AN EXACT-TITLE!"
    elif case in {"legacy", "coauthor_only"}:
        reference.update(title="", year=None, authors=[])
        if case == "coauthor_only":
            candidates = [work("W1", author="Bob Jones")]
    elif case == "wrong_year":
        candidates = [work("W1", year=2022)]
    elif case == "wrong_author":
        candidates = [work("W1", author="Bob Jones")]
    elif case == "wrong_title":
        candidates = [work("W1", title="A Similar Title")]
    elif case == "ambiguous":
        candidates.append(work("W2"))
    elif case == "missing_year":
        candidates = [work("W1", year=None)]
    elif case == "missing_authors":
        candidates[0]["authorships"] = []
    chosen, reason = match.select_reference(reference, candidates)
    assert (chosen["id"].split("/")[-1] if chosen else None) == expected
    assert reason


def test_source_referenced_works_are_used_without_title_search(api: Any) -> None:
    literature = importlib.import_module("kg_literature")
    clock = Clock()
    urls: list[str] = []
    records = {"W9": {**work("W9", title="Source"), "referenced_works": ["https://openalex.org/W1"]},
               "W1": {**work("W1"), "doi": "https://doi.org/10.5555/one"}}

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        assert "/works?" not in url
        urls.append(url)
        return json.dumps(records[url.rsplit("/", 1)[-1]]).encode()

    client = api.OpenAlexClient(fetch, clock=clock.time, sleep=clock.sleep)
    source = literature.Bibliography("Source", {"openalex": "W9"}, "https://example.org/source")
    lib = literature.Literature(fetch, client=client)
    candidates = lib.reference_candidates(source)
    reference: dict[str, Any] = {"citation": "Alice Smith. An Exact Title. 2023.", "title": "", "authors": [], "year": None, "links": []}
    identified, reason = lib.identify(reference, candidates=candidates)
    assert identified.external_ids == {"openalex": "W1", "doi": "10.5555/one"}
    assert "referenced_works" in reason and len(urls) == 2


def test_empty_source_reference_index_uses_search_and_propagates_quota(api: Any) -> None:
    literature = importlib.import_module("kg_literature")
    clock = Clock()
    urls: list[str] = []

    def fetch(url: str, headers: dict[str, str], timeout: float) -> bytes:
        urls.append(url)
        if "/works?" in url:
            raise HTTPError(url, 429, "limited", Message(), None)
        return json.dumps({**work("W9"), "referenced_works": []}).encode()

    client = api.OpenAlexClient(fetch, attempts=1, clock=clock.time, sleep=clock.sleep)
    lib = literature.Literature(fetch, client=client)
    source = literature.Bibliography("Source", {"openalex": "W9"}, "https://example.org/source")
    reference = {"citation": "Alice Smith. An Exact Title. 2023.", "title": "An Exact Title", "authors": ["Alice Smith"], "year": 2023, "links": []}
    with pytest.raises(api.LookupUnavailable, match="HTTP_429"):
        lib.identify(reference, candidates=lib.reference_candidates(source))
    assert len(urls) == 2 and "/works?" in urls[-1]
