"""Expected reconstruction failures, distinct from programming/contract errors."""

from __future__ import annotations

from typing import Any


class ReconstructionUnavailable(RuntimeError):
    def __init__(self, reason: str, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        self.reason = reason
        self.diagnostics = {} if diagnostics is None else diagnostics
        super().__init__(message)
