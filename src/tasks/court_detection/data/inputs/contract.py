"""Input-layer protocol for source-specific Court schemas."""

from __future__ import annotations

from typing import Protocol

from src.tasks.court_detection.data.contracts import (
    CourtInputSpec,
    CourtRawSample,
    CourtSampleRecord,
    CourtSourceSplit,
)


class CourtInput(Protocol):
    @property
    def spec(self) -> CourtInputSpec: ...

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]: ...

    def load(self, record: CourtSampleRecord) -> CourtRawSample: ...


__all__ = ["CourtInput"]
