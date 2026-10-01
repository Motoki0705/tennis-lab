from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.tasks.player_detection.chat_root import (
    SyntheticRoot,
    make_synthetic_root,
)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> SyntheticRoot:
    return make_synthetic_root(tmp_path)
