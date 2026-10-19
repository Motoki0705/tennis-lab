"""Shared readers must distinguish deliberate expiry from missing/corrupt history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.training_queue import (
    RETENTION_NS,
    expired_completion,
    job_label,
    latest_named_job,
    read_completion,
)


@pytest.mark.parametrize(
    "damage", ["json", "identity", "source", "time", "premature", "symlink"]
)
def test_invalid_completion_never_authorizes_expiry(
    tmp_path: Path, damage: str
) -> None:
    path = tmp_path / "control/completions/123_4_name.json"
    path.parent.mkdir(parents=True)
    record = dict(
        version=1,
        job="123_4_name.job",
        state="done",
        finished_at_ns=123,
        retention_started_at_ns=123,
        time_source="recorded",
        expired_at_ns=123 + RETENTION_NS + 1,
    )
    if damage == "identity":
        record["job"] = "other.job"
    elif damage == "source":
        record["time_source"] = "mtime"
    elif damage == "time":
        record["finished_at_ns"] = "123"
    elif damage == "premature":
        record["expired_at_ns"] = 123 + RETENTION_NS
    path.write_text("{" if damage == "json" else json.dumps(record))
    if damage == "symlink":
        target = path.with_suffix(".saved")
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(ValueError):
        expired_completion(tmp_path, "123_4_name.job")


def test_missing_history_is_not_expiry_and_traversal_is_rejected(
    tmp_path: Path,
) -> None:
    assert read_completion(tmp_path, "123.job") is None
    assert expired_completion(tmp_path, "123.job") is None
    with pytest.raises(ValueError, match="invalid published"):
        read_completion(tmp_path, "../123.job")


def test_new_and_legacy_names_share_chronological_selection(tmp_path: Path) -> None:
    legacy = "1792367999999999999_1_readable_name"
    newest = "0-8207631999999999999_2026-10-19T00-00-00-000000000Z_2_readable_name"
    assert job_label(newest) == job_label(legacy) == "readable_name"
    repro = tmp_path / "repro"
    repro.mkdir()
    (repro / legacy).mkdir()
    (repro / newest).mkdir()
    assert latest_named_job(tmp_path, "readable_name") == newest + ".job"


def test_expired_completion_cannot_override_active_identity(tmp_path: Path) -> None:
    path = tmp_path / "control/completions/123.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            dict(
                version=1,
                job="123.job",
                state="done",
                finished_at_ns=1,
                retention_started_at_ns=1,
                time_source="recorded",
                expired_at_ns=RETENTION_NS + 2,
            )
        )
    )
    (tmp_path / "running").mkdir()
    (tmp_path / "running/123.job").touch()
    with pytest.raises(ValueError, match="conflicts with an active"):
        expired_completion(tmp_path, "123.job")
