"""An ambiguous remote mutation must never turn into a second issue."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def create(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("issues.creation").create_issue_once


def test_lost_response_is_recovered_without_rewriting_renamed_closed_issue(
    create: Any,
) -> None:
    durable: dict[str, Any] = {}
    receipt = {"draft": None, "attempted": False, "number": None}
    draft = {"title": "proposal", "body": "immutable source", "labels": ["needs-human"]}

    class GitHub:
        repo = "o/r"
        calls = 0

        def write(
            self, method: str, endpoint: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            assert durable == {"draft": draft, "attempted": True, "number": None}
            assert (method, endpoint, payload) == ("POST", "repos/o/r/issues", draft)
            self.calls += 1
            raise OSError("response lost after remote commit")

    def save() -> None:
        durable.update(copy.deepcopy(receipt))

    gh = GitHub()
    with pytest.raises(OSError):
        create(gh, draft=draft, receipt=receipt, matches=[], save=save)
    receipt = copy.deepcopy(durable)  # A new process reads the fixed receipt.
    with pytest.raises(ValueError, match="outcome is unknown"):
        create(gh, draft=draft, receipt=receipt, matches=[], save=save)
    existing = {"number": 42, "title": "renamed", "body": "edited", "state": "closed"}
    assert (
        create(gh, draft=draft, receipt=receipt, matches=[existing], save=save)
        == existing
    )
    assert durable["number"] == 42 and gh.calls == 1
    create(gh, draft=draft, receipt=receipt, matches=[existing], save=save)
    assert gh.calls == 1


def test_failed_receipt_save_never_sends_post(create: Any) -> None:
    class GitHub:
        repo = "o/r"

        def write(self, *args: Any) -> Any:
            pytest.fail("POST before durable receipt")

    def save() -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        create(
            GitHub(),
            draft={"title": "x", "body": "x", "labels": []},
            receipt={"draft": None, "attempted": False, "number": None},
            matches=[],
            save=save,
        )


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "number", "draft", "boolean"]
)
def test_corrupt_or_conflicting_receipt_never_creates(create: Any, fault: str) -> None:
    class GitHub:
        repo = "o/r"

        def write(self, *args: Any) -> Any:
            pytest.fail("conflict must not create")

    draft = {"title": "x", "body": "source", "labels": []}
    receipt = {"draft": copy.deepcopy(draft), "attempted": True, "number": 42}
    matches = [{"number": 42}]
    if fault == "duplicate":
        matches.append({"number": 43})
    elif fault == "missing":
        matches = []
    elif fault == "number":
        matches = [{"number": 43}]
    elif fault == "draft":
        draft["body"] = "another purpose"
    else:
        receipt["number"] = True
    with pytest.raises(ValueError):
        create(
            GitHub(), draft=draft, receipt=receipt, matches=matches, save=lambda: None
        )
