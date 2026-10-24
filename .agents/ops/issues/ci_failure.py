"""CI source anchors share proposal provenance's parsing and uniqueness rules."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .format import first_source_marker, index_sources

PREFIX = "<!-- ci-failure:"


def ci_failure_marker(signature: str) -> str:
    if not isinstance(signature, str) or not re.fullmatch(
        r"ci-failure-v1:[0-9a-f]{64}", signature
    ):
        raise ValueError("invalid CI failure signature")
    return f"{PREFIX} {signature} -->"


def parse_ci_failure_source(body: str) -> str | None:
    marker = first_source_marker(body, PREFIX)
    if marker is None:
        return None
    signature = marker.removeprefix(PREFIX + " ").removesuffix(" -->")
    if ci_failure_marker(signature) != marker:
        raise ValueError("noncanonical CI failure metadata")
    return signature


def index_ci_failures(bodies: Iterable[tuple[int, str]]) -> dict[str, int]:
    return index_sources(bodies, parse_ci_failure_source)
