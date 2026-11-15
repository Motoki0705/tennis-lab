"""The board accepts identifier lists while blocking private path text."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def board_safety(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return SimpleNamespace(
        public=importlib.import_module("board.safety").public,
        error=importlib.import_module("board.errors").BoardError,
    )


@pytest.mark.parametrize(
    "text",
    [
        "`cli.py`（`_state_dir`／`_issue_lock` 付近）",
        "`start`/`stop`",
        "（`start`／`stop`/`status`）",
        "`start/stop` と 観測/補間",
    ],
)
def test_public_accepts_identifier_separators(
    board_safety: SimpleNamespace, text: str
) -> None:
    board_safety.public(text, {}, {})


@pytest.mark.parametrize(
    "text",
    [
        "`/home/user/x`",
        "`%2Fhome%2Fuser%2Fx`",
        "`／ｈｏｍｅ／user／x`",
        "保存先：/資料/x",
        "保存先：／資料/x",
        "`~/.ssh`",
        "`_state_dir`／home/user/x",
        "`_state_dir`／`/資料/x`",
        "`_state_dir`／`user`/file",
        "`_state_dir`／`user`%2Ffile",
        "`_state_dir`%2F`user`",
        "`_state_dir`／`_issue_lock` を参照（`/home/user/x`）",
    ],
)
def test_public_rejects_paths_next_to_or_inside_code(
    board_safety: SimpleNamespace, text: str
) -> None:
    with pytest.raises(board_safety.error, match="private path"):
        board_safety.public(text, {}, {})
