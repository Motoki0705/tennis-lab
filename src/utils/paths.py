"""Project-root authority used to bootstrap the typed runtime path contract.

Consolidates the ``Path(__file__).resolve().parents[N]`` idiom that was
re-implemented across tasks. Runtime code must construct
:class:`src.utils.configuration.RuntimePathRoots`
and use its resolver. This module owns only the repository location needed to
bootstrap that explicit contract.
"""

from __future__ import annotations

from pathlib import Path

# ``src/utils/paths.py`` -> parents[0] = src/utils, [1] = src, [2] = repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

__all__ = ["PROJECT_ROOT"]
