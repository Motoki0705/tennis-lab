"""Unit tests for the optional queue reproduction path boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.base.training.repro import (
    QueueReproDirError,
    resolve_queue_repro_dir,
)

pytestmark = pytest.mark.unit


def test_queue_repro_dir_absence_preserves_non_queue_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)

    assert resolve_queue_repro_dir() is None


def test_queue_repro_dir_returns_resolved_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_dir = tmp_path / "queue" / "repro" / "job-1"
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_dir))

    assert resolve_queue_repro_dir() == repro_dir.resolve()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "relative/repro",
        " /tmp/repro",
        "/tmp/repro\n",
        "/",
        "/tmp/../repro",
    ],
)
def test_queue_repro_dir_rejects_unsafe_values(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENNIS_REPRO_DIR", value)

    with pytest.raises(QueueReproDirError):
        resolve_queue_repro_dir()


def test_queue_repro_dir_rejects_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repro_file = tmp_path / "not-a-directory"
    repro_file.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro_file))

    with pytest.raises(QueueReproDirError, match="directory"):
        resolve_queue_repro_dir()
