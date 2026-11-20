"""Path rules shared by publication and the CI workflow's standard-library check."""

from __future__ import annotations

import re
from typing import Final

SHARED_LAYERS = ("workflow", "decision", "environment", "gotcha")
MEMORY_DIR: Final = ".agents/memory"
SUMMARY_PATH = f"{MEMORY_DIR}/SUMMARY.md"
INDEX_NAME = "index.md"
DEDICATED_AREA = "ops-report"
_LAYER_DIRS = frozenset(f"{MEMORY_DIR}/{layer}" for layer in SHARED_LAYERS)
_FILENAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.md")


def is_entry_filename(filename: str) -> bool:
    return filename != INDEX_NAME and _FILENAME.fullmatch(filename) is not None


def is_shared_path(path: str) -> bool:
    if path == SUMMARY_PATH:
        return True
    directory, _, filename = path.rpartition("/")
    return directory in _LAYER_DIRS and (
        filename == INDEX_NAME or is_entry_filename(filename)
    )
