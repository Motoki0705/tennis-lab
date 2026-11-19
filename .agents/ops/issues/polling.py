"""Polling results and GitHub's documented rate-limit backoff policy.

This layer computes deadlines; its caller owns sleeping and persistence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Literal

from .transport import HTTPResponse

MAX_SECONDARY_ATTEMPTS = 5


@dataclass(frozen=True)
class RateLimit:
    kind: Literal["primary", "secondary"]
    resume_at: float
    reason: str
    secondary_attempt: int = 0


@dataclass(frozen=True)
class GETResponse:
    status: int
    headers: dict[str, str]
    body: object
    etag_supported: bool | None
    rate_limit: RateLimit | None = None

    @property
    def from_cache(self) -> bool:
        return self.status == 304


@dataclass(frozen=True)
class JobLog:
    """A missing log always has a reason; signed URLs never enter diagnostics."""

    status: int | None
    body: str | None
    reason: str | None = None
    rate_limit: RateLimit | None = None


def _retry_after(value: str, now: float) -> float:
    if value.isdecimal():
        return now + int(value)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is not None:
            return max(now, date.timestamp())
    except (ValueError, TypeError, OverflowError):
        pass
    raise ValueError("invalid GitHub Retry-After header")


def rate_limit(
    response: HTTPResponse, *, now: float, secondary_attempt: int
) -> RateLimit | None:
    """403 permissions stay errors; 429, explicit signals and messages wait.

    https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api#handle-rate-limit-errors-appropriately
    """
    if response.status not in {403, 429}:
        return None
    if not math.isfinite(now) or secondary_attempt < 1:
        raise ValueError("rate-limit clock/attempt is invalid")
    headers = response.headers
    try:
        data = json.loads(response.body)
        message = (
            str(data.get("message", "")).casefold() if isinstance(data, dict) else ""
        )
    except ValueError:
        message = ""
    primary = (
        headers.get("x-ratelimit-remaining") == "0"
        or "api rate limit exceeded" in message
    )
    if not (
        primary
        or response.status == 429
        or "retry-after" in headers
        or "secondary rate limit" in message
        or "abuse detection" in message
        or "api rate limit exceeded" in message
    ):
        return None
    retry = (
        _retry_after(headers["retry-after"], now) if "retry-after" in headers else None
    )
    if primary:
        reset = headers.get("x-ratelimit-reset", "")
        if not reset and retry is not None:
            return RateLimit("primary", retry, "retry-after")
        if not reset.isdecimal():
            raise ValueError("primary GitHub rate limit has no valid x-ratelimit-reset")
        return RateLimit(
            "primary", max(now, int(reset), retry or now), "x-ratelimit-reset"
        )
    if secondary_attempt > MAX_SECONDARY_ATTEMPTS:
        raise ValueError("secondary GitHub rate limit retry limit exhausted")
    # A nonzero remaining count's primary reset is not a secondary deadline.
    if retry is not None:
        return RateLimit("secondary", retry, "retry-after", secondary_attempt)
    return RateLimit(
        "secondary",
        now + 60 * 2 ** (secondary_attempt - 1),
        "secondary_exponential_backoff",
        secondary_attempt,
    )
