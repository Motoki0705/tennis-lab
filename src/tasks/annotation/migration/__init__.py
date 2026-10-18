"""Legacy schema adapters; kept separate from the normal annotation workflow."""

from .legacy import LegacyReader, LegacySelection, LegacySource

__all__ = ["LegacyReader", "LegacySelection", "LegacySource"]
