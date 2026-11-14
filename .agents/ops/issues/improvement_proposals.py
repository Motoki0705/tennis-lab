"""Immutable provenance for proposal generations and grouped extractions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from .format import _scan_markdown
from .improvement import ImprovementSource, positive


@dataclass(frozen=True)
class Proposal:
    parent: int
    generation: int

    def __post_init__(self) -> None:
        positive(self.parent)
        positive(self.generation)

    def marker(self) -> str:
        return f"<!-- parent-improvement-proposal:v1 parent={self.parent} generation={self.generation} -->"


@dataclass(frozen=True)
class Extraction:
    parent: int
    proposal: int
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        positive(self.parent)
        positive(self.proposal)
        if not self.keys or tuple(sorted(set(self.keys))) != self.keys:
            raise ValueError(
                "extraction candidates must be nonempty, sorted and unique"
            )
        for key in self.keys:
            ImprovementSource(self.parent, key)

    def marker(self) -> str:
        ids = ",".join(
            ImprovementSource(self.parent, k).candidate_id for k in self.keys
        )
        return f"<!-- parent-improvement-extraction:v1 parent={self.parent} proposal={self.proposal} candidates={ids} -->"


def _marker(body: str, kind: str) -> str | None:
    _, markers = _scan_markdown(body, prefix=f"<!-- parent-improvement-{kind}:")
    if not markers:
        return None
    if len(markers) != 1 or markers[0][0] != 0:
        raise ValueError(f"improvement {kind} source must occur once on the first line")
    return cast(str, markers[0][1])


def parse_proposal(body: str) -> Proposal | None:
    line = _marker(body, "proposal")
    if line is None:
        return None
    match = re.fullmatch(
        r"<!-- parent-improvement-proposal:v1 parent=([1-9][0-9]*) generation=([1-9][0-9]*) -->",
        line,
    )
    if match is None:
        raise ValueError("invalid improvement proposal provenance")
    return Proposal(int(match[1]), int(match[2]))


def parse_extraction(body: str) -> Extraction | None:
    line = _marker(body, "extraction")
    if line is None:
        return None
    match = re.fullmatch(
        r"<!-- parent-improvement-extraction:v1 parent=([1-9][0-9]*) proposal=([1-9][0-9]*) candidates=(.+) -->",
        line,
    )
    if match is None:
        raise ValueError("invalid improvement extraction provenance")
    parent = int(match[1])
    prefix = f"I-{parent}-"
    ids = match[3].split(",")
    if any(not value.startswith(prefix) for value in ids):
        raise ValueError("extraction candidate belongs to another parent")
    result = Extraction(
        parent, int(match[2]), tuple(value[len(prefix) :] for value in ids)
    )
    if result.marker() != line:
        raise ValueError("noncanonical improvement extraction provenance")
    return result
