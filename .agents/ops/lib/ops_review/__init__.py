"""Owned import path for ops review code, independent of other review packages."""

from pathlib import Path

# Keep the existing CLI/source directory and Python's normal import locking.
__path__ = [str(Path(__file__).resolve().parents[2] / "review")]
