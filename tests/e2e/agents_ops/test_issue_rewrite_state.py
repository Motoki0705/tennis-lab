"""The shared state API and real gh HTTP adapter, with offline responses."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
ENABLED = "2026-10-30T12:00:00Z"
PARENT = ["issue:parent"]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("issues.rewrite_state")


@pytest.mark.parametrize(
    ("labels", "created", "enabled", "expected"),
    [
        (PARENT, ENABLED, None, "not_target"),
        ([], ENABLED, ENABLED, "not_target"),
        (["issue:child"], ENABLED, ENABLED, "not_target"),
        (PARENT, "2026-10-30T11:59:59Z", ENABLED, "not_target"),
        (PARENT, ENABLED, ENABLED, "pending"),
        (PARENT, "2026-10-30T12:00:01Z", ENABLED, "pending"),
        (PARENT + ["rewrite:done"], ENABLED, ENABLED, "done"),
        (PARENT + ["rewrite:failed"], ENABLED, ENABLED, "failed"),
        (PARENT + ["rewrite:skipped"], ENABLED, ENABLED, "skipped"),
        (PARENT + ["issue:child", "enhancement"], ENABLED, ENABLED, "pending"),
        (["rewrite:done"], ENABLED, ENABLED, "not_target"),
        (PARENT + ["rewrite:failed"], ENABLED, None, "not_target"),
        (PARENT + ["rewrite:done"], "2026-10-30T11:59:59Z", ENABLED, "not_target"),
    ],
)
def test_rewrite_state_boundaries(
    api: Any, labels: list[str], created: str, enabled: str | None, expected: str
) -> None:
    assert api.rewrite_state(labels, created, enabled) == expected


@pytest.mark.parametrize(
    "labels",
    [
        ["rewrite:done", "rewrite:failed"],
        ["rewrite:done", "rewrite:skipped"],
        ["rewrite:failed", "rewrite:skipped"],
        ["rewrite:done", "rewrite:failed", "rewrite:skipped"],
    ],
)
def test_conflicting_final_labels_are_always_errors(
    api: Any, labels: list[str]
) -> None:
    with pytest.raises(ValueError, match="conflicting"):
        api.rewrite_state(labels, ENABLED, None)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "2026-10-30",
        "2026-10-30T12:00:00+00:00",
        "2026-10-30T12:00:00.000Z",
        "2026-02-30T12:00:00Z",
        "2026-10-30T24:00:00Z",
    ],
)
def test_invalid_timestamps_do_not_disable_rewrite(api: Any, value: str) -> None:
    with pytest.raises(ValueError):
        api.rewrite_state(PARENT, ENABLED, value)
    with pytest.raises(ValueError):
        api.rewrite_state(PARENT, value, ENABLED)


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, {"name": "ISSUE_REWRITE_ENABLED_SINCE", "value": ENABLED}, ENABLED),
        (404, {"message": "Not Found"}, None),
        (403, {"message": "Forbidden"}, "error"),
        (500, {"message": "Server Error"}, "error"),
        (200, {"name": "ISSUE_REWRITE_ENABLED_SINCE", "value": ""}, "error"),
        (200, {"name": "ISSUE_REWRITE_ENABLED_SINCE", "value": None}, "error"),
        (200, {"name": "other", "value": ENABLED}, "error"),
    ],
)
def test_enabled_since_uses_http_status_and_validates_response(
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: Any,
    expected: str | None,
) -> None:
    client = importlib.import_module("issues.cli").GitHub("o/r")
    calls = []

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            0 if status == 200 else 1,
            f"HTTP/2.0 {status} response\n\n{json.dumps(payload)}",
            "HTTP failure" if status != 200 else "",
        )

    monkeypatch.setattr(client, "_run", run)
    if expected == "error":
        with pytest.raises(ValueError):
            api.read_enabled_since(client)
    else:
        assert api.read_enabled_since(client) == expected
    assert len(calls) == 1
    assert calls[0][-1] == "repos/o/r/actions/variables/ISSUE_REWRITE_ENABLED_SINCE"
    assert calls[0][:4] == ("api", "--include", "--method", "GET")


def test_network_failure_without_http_status_is_not_disabled(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = importlib.import_module("issues.cli").GitHub("o/r")
    monkeypatch.setattr(
        client,
        "_run",
        lambda *a: subprocess.CompletedProcess(
            a, 1, "", "network failure (404 in proxy text)"
        ),
    )
    with pytest.raises(ValueError, match="missing HTTP response"):
        api.read_enabled_since(client)


def test_package_exports_module_and_workflow_constants(api: Any) -> None:
    package = importlib.import_module("issues")
    assert package.rewrite_state is api
    assert package.read_enabled_since is api.read_enabled_since
    assert {
        "rewrite:done",
        "rewrite:failed",
        "rewrite:skipped",
    } == api.REWRITE_FINAL_LABELS
