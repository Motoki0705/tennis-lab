"""Backend enum used by operator dispatch."""

from __future__ import annotations

from enum import StrEnum


class OpBackend(StrEnum):
    """Execution backend choices for custom ops."""

    CUDA = "cuda"
    CPU = "cpu"
    PYTORCH = "pytorch"
