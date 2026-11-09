"""Bound GET transport retries without replaying HTTP errors or mutations."""

from __future__ import annotations

import importlib
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ENDPOINT = "repos/o/r/issues/12/sub_issues?per_page=100&page=1"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("issues.cli")


def response(status: int, body: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        [],
        int(status != 200),
        f"HTTP/2.0 {status} test\r\nContent-Type: application/json\r\n\r\n{json.dumps(body)}",
        f"gh: error (HTTP {status})" if status != 200 else "",
    )


def disconnected(
    detail: str = "error connecting to api.github.com",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 1, "", detail)


def scripted_get(
    monkeypatch: pytest.MonkeyPatch,
    api: ModuleType,
    responses: Sequence[subprocess.CompletedProcess[str] | subprocess.TimeoutExpired],
) -> tuple[list[tuple[str, ...]], list[float]]:
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []
    pending = iter(responses)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        result = next(pending)
        if isinstance(result, subprocess.TimeoutExpired):
            raise result
        return result

    monkeypatch.setattr(api.GitHub, "_run", staticmethod(run))
    monkeypatch.setattr(api, "time", SimpleNamespace(sleep=sleeps.append))
    return calls, sleeps


@pytest.mark.parametrize("failures", [1, 2])
def test_transient_disconnect_recovers_the_actual_response(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    children = [{"number": 13}, {"number": 21}]
    calls, sleeps = scripted_get(
        monkeypatch, api, [disconnected()] * failures + [response(200, children)]
    )
    assert api.GitHub("o/r").get(ENDPOINT) == children
    assert len(calls) == failures + 1
    assert sleeps == [1.0, 2.0][:failures]
    assert all(
        call[-1] == ENDPOINT and call[call.index("--method") + 1] == "GET"
        for call in calls
    )


def test_persistent_disconnect_reports_endpoint_attempts_and_last_error(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [
            disconnected(),
            disconnected("TLS connection lost"),
            disconnected("final connection error"),
        ],
    )
    with pytest.raises(api.GitHubTransportError) as failed:
        api.GitHub("o/r").get(ENDPOINT, missing_parent=True)
    assert failed.value.endpoint == ENDPOINT
    assert failed.value.attempts == 3
    assert failed.value.error_kind == "missing_http_response_status"
    assert ENDPOINT in str(failed.value)
    assert "attempts=3" in str(failed.value)
    assert "last_error_kind=missing_http_response_status" in str(failed.value)
    assert "final connection error" in str(failed.value)
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_http_errors_are_not_transport_retries(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch, api, [response(status, {"message": "rejected"})]
    )
    with pytest.raises(api.GitHubHTTPError) as failed:
        api.GitHub("o/r").get(ENDPOINT)
    assert failed.value.status == status
    assert failed.value.attempts == 1
    assert len(calls) == 1 and sleeps == []


def test_rate_limit_after_disconnect_stops_at_the_http_response(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [disconnected(), response(429, {"message": "rate limit exceeded"})],
    )
    with pytest.raises(api.GitHubHTTPError) as failed:
        api.GitHub("o/r").get(ENDPOINT)
    assert failed.value.status == 429 and failed.value.attempts == 2
    assert "last_error_kind=http_429" in str(failed.value)
    assert len(calls) == 2 and sleeps == [1.0]


@pytest.mark.parametrize(
    "stdout", ["HTTP/2.0 200\r\n", "HTTP/2.0 200\n\nnot-json", "HTTP/2.0 200\n\nnull"]
)
def test_invalid_success_response_is_not_a_transport_retry(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch, api, [subprocess.CompletedProcess([], 0, stdout, "")]
    )
    with pytest.raises(ValueError):
        api.GitHub("o/r").get(ENDPOINT)
    assert len(calls) == 1 and sleeps == []


def test_timeout_without_http_response_can_recover(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [subprocess.TimeoutExpired(["gh", "api", ENDPOINT], 60), response(200, [])],
    )
    assert api.GitHub("o/r").get(ENDPOINT) == []
    assert len(calls) == 2 and sleeps == [1.0]


def test_successful_exit_without_http_status_is_not_a_connection_retry(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch, api, [subprocess.CompletedProcess([], 0, "[]", "")]
    )
    with pytest.raises(ValueError, match="last_error_kind=invalid_http_response"):
        api.GitHub("o/r").get(ENDPOINT)
    assert len(calls) == 1 and sleeps == []


def test_timeout_with_http_response_is_not_retried(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [
            subprocess.TimeoutExpired(
                ["gh", "api", ENDPOINT], 60, output=b"HTTP/2.0 429\r\n"
            )
        ],
    )
    with pytest.raises(api.GitHubHTTPError) as failed:
        api.GitHub("o/r").get(ENDPOINT)
    assert failed.value.status == 429 and failed.value.attempts == 1
    assert len(calls) == 1 and sleeps == []


def test_invalid_body_after_disconnect_keeps_request_diagnostics(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [
            disconnected(),
            subprocess.CompletedProcess([], 0, "HTTP/2.0 200\n\ntruncated", ""),
        ],
    )
    with pytest.raises(ValueError) as failed:
        api.GitHub("o/r").get(ENDPOINT)
    assert ENDPOINT in str(failed.value) and "attempts=2" in str(failed.value)
    assert "last_error_kind=invalid_json_response" in str(failed.value)
    assert len(calls) == 2 and sleeps == [1.0]


def test_exhausted_timeouts_record_the_last_failure_kind(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [
            disconnected(),
            subprocess.TimeoutExpired(["gh", "api", ENDPOINT], 60),
            subprocess.TimeoutExpired(["gh", "api", ENDPOINT], 60),
        ],
    )
    with pytest.raises(api.GitHubTransportError) as failed:
        api.GitHub("o/r").get(ENDPOINT)
    assert failed.value.attempts == 3
    assert failed.value.error_kind == "timeout_without_http_response"
    assert "last_error_kind=timeout_without_http_response" in str(failed.value)
    assert "60s" in str(failed.value)
    assert len(calls) == 3 and sleeps == [1.0, 2.0]


@pytest.mark.parametrize("body", ["", "not-json", "[]"])
def test_invalid_parent_404_body_keeps_the_http_failure(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    calls, sleeps = scripted_get(
        monkeypatch,
        api,
        [subprocess.CompletedProcess([], 1, "HTTP/2.0 404\n\n" + body, "Not Found")],
    )
    with pytest.raises(api.GitHubHTTPError) as failed:
        api.GitHub("o/r").get("repos/o/r/issues/12/parent", missing_parent=True)
    assert failed.value.status == 404
    assert len(calls) == 1 and sleeps == []


@pytest.mark.parametrize("method", ["POST", "PATCH", "PUT", "DELETE"])
def test_mutations_are_never_replayed_on_disconnect(
    api: ModuleType, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    parent_api = importlib.import_module("parent.github")
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert json.loads(str(kwargs["input"])) == {"body": "only once"}
        return disconnected()

    monkeypatch.setattr(parent_api.subprocess, "run", run)
    monkeypatch.setattr(api, "time", SimpleNamespace(sleep=sleeps.append))
    monkeypatch.setattr(
        parent_api,
        "time",
        SimpleNamespace(monotonic=time.monotonic, sleep=sleeps.append),
    )
    with pytest.raises(ValueError, match="error connecting"):
        parent_api.GitHub("o/r").write(
            method, "repos/o/r/issues/12/comments", {"body": "only once"}
        )
    assert len(calls) == 1 and sleeps == []
    assert calls[0][calls[0].index("--method") + 1] == method
