"""Typed semantic gates for reusing completed scene-stage publications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RequiredOutputsReusablePublicationValidator:
    """Preserve the established reuse policy for non-PLCS stages."""

    def validate(self, owner_path: Path) -> None:
        """Require the existing fixed owner after required-output validation."""
        if not owner_path.is_dir() or owner_path.is_symlink():
            raise ValueError("Reusable stage owner must be an ordinary directory.")


__all__ = ["RequiredOutputsReusablePublicationValidator"]
