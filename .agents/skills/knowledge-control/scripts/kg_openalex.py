"""Paced, bounded OpenAlex requests; transport failures never mean no match."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from kg_reference_match import words

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research-loop/scripts"))
from lit_search import Fetch, urllib_fetch  # noqa: E402


class LookupUnavailable(RuntimeError):
    """Safe diagnostics only: URLs, response bodies and contact data stay private."""

    def __init__(self, reason: str) -> None:
        super().__init__("OpenAlex " + reason)
        self.reason = reason


class WorkNotFound(LookupUnavailable):
    pass


class IncompleteSearch(ValueError):
    """A successful response did not establish complete candidate coverage."""


class OpenAlexClient:
    def __init__(self, fetch: Fetch = urllib_fetch, *, interval: float = 1.0,
                 attempts: int = 4, base_backoff: float = 2.0, max_wait: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 deadline: float | None = None, state_file: Path | None = None) -> None:
        if not math.isfinite(interval) or interval <= 0 or type(attempts) is not int or attempts < 1:
            raise ValueError("OpenAlex interval and attempt limit must be positive")
        if not 0 < base_backoff <= max_wait or not math.isfinite(max_wait):
            raise ValueError("invalid OpenAlex backoff limits")
        self.transport, self.interval, self.attempts = fetch, interval, attempts
        self.base_backoff, self.max_wait = base_backoff, max_wait
        self.clock, self.wall_clock, self.sleep = clock, wall_clock, sleep
        self.deadline = deadline
        self.next_start = -math.inf
        self.cache: dict[str, bytes] = {}
        self.state_file = state_file
        self.principal = hashlib.sha256(os.environ.get("OPENALEX_API_KEY", "").strip().encode()).hexdigest()
        self.not_before: dict[str, float] = {}
        if state_file is not None and state_file.exists():
            try:
                saved = json.loads(state_file.read_text())
                if saved["principal"] == self.principal:
                    self.not_before = saved["not_before"]
                    if not isinstance(self.not_before, dict) or any(k not in {"query", "singleton"} or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in self.not_before.items()):
                        raise ValueError("invalid cooldown")
            except (OSError, ValueError, KeyError, TypeError):
                raise LookupUnavailable("invalid_private_throttle_state") from None

    def remember_retry(self, scope: str, seconds: float) -> None:
        """Honor a server Retry-After across CLI restarts, without public secrets."""
        if scope not in {"query", "singleton"} or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid Retry-After scope or duration")
        self.not_before[scope] = max(self.not_before.get(scope, 0), self.wall_clock() + seconds)
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".openalex-", dir=self.state_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump({"principal": self.principal, "not_before": self.not_before}, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.state_file)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _pause(self, seconds: float) -> None:
        if self.deadline is not None and self.clock() + seconds >= self.deadline:
            raise LookupUnavailable("deadline_exceeded")
        if seconds > 0:
            self.sleep(seconds)

    def _retry_after(self, value: str | None) -> float:
        if value is None:
            return 0.0
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(value).timestamp() - self.wall_clock()
            except (ValueError, TypeError, OverflowError):
                raise LookupUnavailable("invalid_retry_after") from None
        if not math.isfinite(seconds):
            raise LookupUnavailable("invalid_retry_after")
        return max(0.0, seconds)

    def fetch(self, url: str, headers: dict[str, str], timeout: float) -> bytes:
        parts = urlsplit(url)
        if parts.hostname != "api.openalex.org" or parts.scheme != "https":
            raise ValueError("OpenAlex client accepts only its HTTPS API")
        # Cache keys and diagnostics exclude caller credentials/contact values.
        public = [(k, v) for k, v in parse_qsl(parts.query) if k not in {"mailto", "api_key"}]
        key = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(public), ""))
        if key in self.cache:
            return self.cache[key]
        scope = "query" if parts.path.rstrip("/") == "/works" else "singleton"
        params = list(public)
        if contact := os.environ.get("OPENALEX_EMAIL", "").strip():
            params.append(("mailto", contact))
        request_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), ""))
        request_headers = {**headers, "User-Agent": "tennis-lab-citation-explorer/1.0"}
        if api_key := os.environ.get("OPENALEX_API_KEY", "").strip():
            request_headers["Authorization"] = "Bearer " + api_key
        for attempt in range(self.attempts):
            cooldown = max(0.0, self.not_before.get(scope, 0) - self.wall_clock())
            if cooldown > self.max_wait:
                raise LookupUnavailable(f"{scope}_cooldown; retry_after_exceeds_limit ({cooldown:.0f}s)")
            self._pause(cooldown)
            self._pause(max(0.0, self.next_start - self.clock()))
            remaining = timeout if self.deadline is None else min(timeout, self.deadline - self.clock())
            if remaining <= 0:
                raise LookupUnavailable("deadline_exceeded")
            self.next_start = self.clock() + self.interval
            retry_after = 0.0
            try:
                raw: bytes = self.transport(request_url, request_headers, remaining)
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("expected object")
                self.cache[key] = raw
                return raw
            except HTTPError as exc:
                status = exc.code
                if status == 404:
                    raise WorkNotFound("HTTP_404") from None
                if status != 429 and not 500 <= status <= 599:
                    raise LookupUnavailable(f"HTTP_{status}") from None
                reason = f"HTTP_{status}"
                retry_after = self._retry_after(exc.headers.get("Retry-After"))
                if retry_after:
                    self.remember_retry(scope, retry_after)
            except (URLError, TimeoutError, OSError, HTTPException):
                reason = "transport_error"
            except (ValueError, UnicodeError):
                reason = "invalid_response"
            if attempt + 1 == self.attempts:
                raise LookupUnavailable(f"{reason}; retries_exhausted ({self.attempts} attempts)") from None
            delay = max(retry_after, self.base_backoff * 2 ** attempt)
            if delay > self.max_wait:
                raise LookupUnavailable(f"{reason}; retry_after_exceeds_limit ({delay:.0f}s)") from None
            print(f"OpenAlex {reason}; retry {attempt + 1}/{self.attempts - 1} after {delay:g}s", file=sys.stderr, flush=True)
            self._pause(delay)
        raise AssertionError("unreachable")

    def work(self, identifier: str) -> dict[str, Any] | None:
        try:
            raw = self.fetch(f"https://api.openalex.org/works/{identifier}", {}, 30.0)
        except WorkNotFound:
            return None
        result: dict[str, Any] = json.loads(raw)
        if not isinstance(result.get("id"), str) or not isinstance(result.get("display_name"), str):
            raise LookupUnavailable("invalid_work_response")
        return result

    def search(self, query: str, *, title_only: bool = True) -> list[dict[str, Any]]:
        """Exhaust the cursor; no subset is returned as a complete search."""
        cursor = "*"
        visited: set[str] = set()
        found: dict[str, dict[str, Any]] = {}
        total: int | None = None
        while True:
            visited.add(cursor)
            # Bibliographic text is data, not OpenAlex Boolean/field syntax.
            params = {"search.title" if title_only else "search": " ".join(words(query)), "per_page": 100, "cursor": cursor}
            url = "https://api.openalex.org/works?" + urlencode(params)
            payload = json.loads(self.fetch(url, {}, 30.0))
            results, meta = payload.get("results"), payload.get("meta")
            if not isinstance(results, list) or any(not isinstance(w, dict) or not isinstance(w.get("id"), str) for w in results):
                raise IncompleteSearch("OpenAlex incomplete_search: invalid_results")
            if not isinstance(meta, dict) or type(meta.get("count")) is not int or meta["count"] < 0 or "next_cursor" not in meta:
                raise IncompleteSearch("OpenAlex incomplete_search: missing_pagination_metadata")
            if total is not None and total != meta["count"]:
                raise IncompleteSearch("OpenAlex incomplete_search: count_changed")
            total = meta["count"]
            for work in results:
                if work["id"] in found:
                    raise IncompleteSearch("OpenAlex incomplete_search: repeated_work")
                found[work["id"]] = work
            following = meta["next_cursor"]
            if following is None:
                if len(found) != total:
                    raise IncompleteSearch("OpenAlex incomplete_search: count_mismatch")
                return list(found.values())
            if not isinstance(following, str) or not following or following in visited or not results or len(found) > total:
                raise IncompleteSearch("OpenAlex incomplete_search: cursor_did_not_progress")
            cursor = following
