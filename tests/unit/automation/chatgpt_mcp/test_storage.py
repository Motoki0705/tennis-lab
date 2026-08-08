from __future__ import annotations

import time
from pathlib import Path

from src.automation.chatgpt_mcp.storage import SqliteStore


def test_store_get_pop_and_expiry(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.sqlite3")
    store.put("jobs", "active", {"value": 1}, expires_at=time.time() + 60)
    store.put("jobs", "expired", {"value": 2}, expires_at=time.time() - 1)

    assert store.get("jobs", "active") == {"value": 1}
    assert store.get("jobs", "expired") is None
    assert store.pop("jobs", "active") == {"value": 1}
    assert store.pop("jobs", "active") is None


def test_store_rejects_unknown_table(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.sqlite3")

    try:
        store.put("arbitrary_table", "key", {})
    except ValueError as error:
        assert "unsupported storage table" in str(error)
    else:
        raise AssertionError("unknown table was accepted")
