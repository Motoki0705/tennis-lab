"""Base contracts for raw web dataset parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from src.tasks.ball_detection.data.components.web.data_access_layer.writer import (
    WebFrameRecord,
)


@dataclass(frozen=True)
class ParsedSource:
    """One logical source and a lazy stream of normalized frames."""

    name: str
    records: Callable[[], Iterator[WebFrameRecord]]


class WebDatasetParser(ABC):
    """Convert one raw dataset format into normalized frame records."""

    @abstractmethod
    def sources(self) -> Iterable[ParsedSource]:
        """Return independently reported logical source streams."""


__all__ = [
    "ParsedSource",
    "WebDatasetParser",
]
