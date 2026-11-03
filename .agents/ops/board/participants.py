"""The reserved human has no parent registration or name assignment."""

from __future__ import annotations

from dataclasses import dataclass

from .names import Assignment


@dataclass(frozen=True)
class Human:
    id: None = None
    name: str = "human"
    repo: None = None
    parent: None = None


HUMAN = Human()
Participant = Assignment | Human
