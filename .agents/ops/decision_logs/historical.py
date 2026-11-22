"""Bounded, explicitly sourced historical observations; no summary inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worker_logs.journal import decode, require

from .journal import Journal
from .projection import validate_public


def records() -> list[dict[str, Any]]:
    data = decode(Path(__file__).with_name("historical_observations.json").read_bytes())
    require(
        set(data) == {"version", "description", "records"}
        and type(data["version"]) is int
        and data["version"] == 1,
        "record_version",
    )
    require(isinstance(data["records"], list), "record_schema")
    return [validate_public(value)["record"] for value in data["records"]]


def capture(journal: Journal) -> list[str]:
    """Explicitly append the bundled facts for this namespace, idempotently.

    This entry never prepares, publishes, fetches, or changes parent ownership.
    Original v1 records and private historical runtime data are not reconstructed.
    """
    return [
        journal.capture("historical-source:" + row["id"], row)["id"]
        for row in records()
        if row["scope"]["repo"].casefold() == journal.repo.casefold()
        and row["scope"]["parent"] == journal.parent
    ]
