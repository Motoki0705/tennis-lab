"""Opt-in polling contracts against scripted HTTP, without real sleeps/API use."""

from __future__ import annotations

import importlib
import io
import json
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

ENDPOINT = "repos/o/r/actions/runs?head_sha=abc&page=1"
TOKEN = "conditional-test-secret"


def reply(status: int, body: object = "", **headers: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    response.headers.update(headers)
    return response


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    monkeypatch.setenv("GH_TOKEN", TOKEN)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_HOST", raising=False)
    pending: deque[requests.Response | Exception] = deque()
    sent: list[requests.PreparedRequest] = []

    def send(
        session: requests.Session, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        sent.append(request)
        result = pending.popleft()
        if isinstance(result, Exception):
            raise result
        result.request = request
        return result

    monkeypatch.setattr(requests.Session, "send", send)
    yield SimpleNamespace(
        reader=importlib.import_module("issues.cli"),
        transport=importlib.import_module("issues.transport"),
        metrics=importlib.import_module("parent.metrics"),
        pending=pending,
        sent=sent,
    )
    assert not pending, "scenario did not consume every scripted HTTP response"


def test_etag_revalidation_preserves_current_headers_and_isolates_mutable_body(
    api: Any,
) -> None:
    api.pending.extend(
        [
            reply(200, {"runs": [17]}, ETag='W/"v1"'),
            reply(
                304,
                ETag='W/"v1"',
                **{"X-RateLimit-Remaining": "41", "X-Poll-Interval": "90"},
            ),
            reply(200, {"runs": [23]}, ETag='"v2"'),
            reply(304),
        ]
    )
    gh = api.reader.GitHub("o/r")
    first = gh.get_conditional(ENDPOINT)
    first.body["runs"].append(999)
    cached = gh.get_conditional(ENDPOINT)
    assert cached.body == {"runs": [17]}
    assert cached.status == 304 and cached.from_cache and cached.etag_supported
    assert cached.headers == {
        "etag": 'W/"v1"',
        "x-ratelimit-remaining": "41",
        "x-poll-interval": "90",
    }
    cached.body["runs"].clear()
    assert gh.get_conditional(ENDPOINT).body == {"runs": [23]}
    assert gh.get_conditional(ENDPOINT).body == {"runs": [23]}
    assert [r.headers.get("If-None-Match") for r in api.sent] == [
        None,
        'W/"v1"',
        'W/"v1"',
        '"v2"',
    ]
    assert all(r.headers["Authorization"] == f"Bearer {TOKEN}" for r in api.sent)


def test_raw_response_retains_the_polling_headers_and_body(api: Any) -> None:
    api.pending.append(
        reply(
            200,
            {"id": 7},
            **{
                "ETag": '"v1"',
                "X-RateLimit-Limit": "5000",
                "X-RateLimit-Used": "8",
                "X-RateLimit-Remaining": "4992",
                "X-RateLimit-Reset": "1700",
                "X-RateLimit-Resource": "core",
                "X-Poll-Interval": "90",
                "Retry-After": "120",
                "Set-Cookie": "private=cookie",
            },
        )
    )
    result = api.transport.HTTPClient().request_response(
        "GET", ENDPOINT, headers={"If-None-Match": '"old"'}
    )
    assert result.status == 200 and json.loads(result.body) == {"id": 7}
    assert result.headers == {
        "etag": '"v1"',
        "x-ratelimit-limit": "5000",
        "x-ratelimit-used": "8",
        "x-ratelimit-remaining": "4992",
        "x-ratelimit-reset": "1700",
        "x-ratelimit-resource": "core",
        "x-poll-interval": "90",
        "retry-after": "120",
    }
    assert api.sent[0].headers["If-None-Match"] == '"old"'


@pytest.mark.parametrize("body", ["", "truncated", "null"])
def test_invalid_200_is_not_cached_as_a_success(api: Any, body: str) -> None:
    api.pending.extend([reply(200, body, ETag='"bad"'), reply(304)])
    gh = api.reader.GitHub("o/r")
    with pytest.raises(ValueError, match="invalid JSON|unexpected null"):
        gh.get_conditional(ENDPOINT)
    with pytest.raises(ValueError, match="no saved response"):
        gh.get_conditional(ENDPOINT)
    assert "If-None-Match" not in api.sent[-1].headers


def test_body_failure_after_status_does_not_return_cached_success(api: Any) -> None:
    class BrokenResponse(requests.Response):
        @property
        def content(self) -> bytes:
            raise requests.ConnectionError("sensitive transport details")

    broken = BrokenResponse()
    broken.status_code = 200
    broken.raw = io.BytesIO()
    broken.headers["ETag"] = '"broken"'
    api.pending.extend([reply(200, [7], ETag='"good"'), broken, reply(304)])
    gh = api.reader.GitHub("o/r")
    gh.get_conditional(ENDPOINT)
    with pytest.raises(api.reader.GitHubHTTPError, match="body failed after HTTP 200"):
        gh.get_conditional(ENDPOINT)
    assert gh.get_conditional(ENDPOINT).body == [7]
    assert api.sent[-1].headers["If-None-Match"] == '"good"'


@pytest.mark.parametrize("scope", ["no_saved_response", "query", "credential", "host"])
def test_304_never_reuses_another_url_or_credential(
    api: Any, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    gh = api.reader.GitHub("o/r")
    if scope != "no_saved_response":
        api.pending.append(reply(200, {"private": 11}, ETag='"private"'))
        gh.get_conditional(ENDPOINT)
    endpoint = ENDPOINT
    if scope == "query":
        endpoint = ENDPOINT.replace("page=1", "page=2")
    elif scope == "credential":
        gh.close()
        monkeypatch.setenv("GH_TOKEN", "another-authentication")
        gh._http = api.transport.HTTPClient()
    elif scope == "host":
        monkeypatch.setattr(api.transport, "_API_URL", "https://other.example/api/")
    api.pending.append(reply(304))
    with pytest.raises(ValueError, match="304 has no saved response"):
        gh.get_conditional(endpoint)
    assert "If-None-Match" not in api.sent[-1].headers


def test_missing_etag_is_reported_and_evicts_the_old_validator(api: Any) -> None:
    api.pending.extend(
        [
            reply(200, []),
            reply(200, []),
            reply(200, [1], ETag='"once"'),
            reply(200, [2]),
            reply(304),
        ]
    )
    gh = api.reader.GitHub("o/r")
    with api.metrics.command("poll", None) as row:
        assert not gh.get_conditional(ENDPOINT).etag_supported
        assert not gh.get_conditional(ENDPOINT).etag_supported
        assert gh.get_conditional(ENDPOINT).etag_supported
        assert not gh.get_conditional(ENDPOINT).etag_supported
        with pytest.raises(ValueError, match="no saved response"):
            gh.get_conditional(ENDPOINT)
    assert [r.headers.get("If-None-Match") for r in api.sent] == [
        None,
        None,
        None,
        '"once"',
        None,
    ]
    assert row["github_get_without_etag"] == 3


@pytest.mark.parametrize(
    "status,headers,message,kind,deadline",
    [
        (
            403,
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700"},
            "API rate limit exceeded",
            "primary",
            1700,
        ),
        (
            429,
            {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1700",
                "Retry-After": "900",
            },
            "limited",
            "primary",
            1900,
        ),
        (
            403,
            {"X-RateLimit-Remaining": "0", "Retry-After": "120"},
            "limited",
            "primary",
            1120,
        ),
        (
            403,
            {"X-RateLimit-Reset": "1700"},
            "API rate limit exceeded for user",
            "primary",
            1700,
        ),
        (
            403,
            {"Retry-After": "120"},
            "You have exceeded a secondary rate limit",
            "secondary",
            1120,
        ),
        (429, {"Retry-After": "30"}, "limited", "secondary", 1030),
        (403, {}, "You have exceeded a secondary rate limit", "secondary", 1060),
        (403, {}, "abuse detection mechanism", "secondary", 1060),
        (429, {}, "too many requests", "secondary", 1060),
        (
            403,
            {"Retry-After": "Thu, 01 Jan 1970 00:20:00 GMT"},
            "limited",
            "secondary",
            1200,
        ),
        (
            403,
            {"X-RateLimit-Remaining": "400", "X-RateLimit-Reset": "9000"},
            "secondary rate limit",
            "secondary",
            1060,
        ),
    ],
)
def test_rate_limit_classification_and_deadlines(
    api: Any,
    status: int,
    headers: dict[str, str],
    message: str,
    kind: str,
    deadline: int,
) -> None:
    api.pending.append(reply(status, {"message": message}, **headers))
    result = api.reader.GitHub("o/r").get_conditional(ENDPOINT, clock=lambda: 1000)
    assert result.status == status and result.rate_limit.kind == kind
    assert result.rate_limit.resume_at == deadline
    assert len(api.sent) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 201, 204, 302, 500])
def test_permissions_and_unknown_statuses_raise_without_wait_or_retry(
    api: Any, status: int
) -> None:
    api.pending.append(
        reply(status, {"message": "Resource not accessible by integration"})
    )
    with pytest.raises(api.reader.GitHubHTTPError) as error:
        api.reader.GitHub("o/r").get_conditional(ENDPOINT)
    assert error.value.status == status and len(api.sent) == 1


@pytest.mark.parametrize(
    "headers,match",
    [
        ({"X-RateLimit-Remaining": "0"}, "no valid x-ratelimit-reset"),
        (
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "invalid"},
            "no valid x-ratelimit-reset",
        ),
        ({"Retry-After": "tomorrow"}, "invalid GitHub Retry-After"),
    ],
)
def test_invalid_wait_metadata_is_an_explicit_error(
    api: Any, headers: dict[str, str], match: str
) -> None:
    api.pending.append(reply(403, {"message": "rate limit"}, **headers))
    with pytest.raises(ValueError, match=match):
        api.reader.GitHub("o/r").get_conditional(ENDPOINT, clock=lambda: 1000)


@pytest.mark.parametrize("success", [200, 304])
def test_secondary_backoff_is_shared_across_endpoints_and_resets_after_success(
    api: Any,
    success: int,
) -> None:
    gh = api.reader.GitHub("o/r")
    api.pending.append(reply(200, [], ETag='"v1"'))
    gh.get_conditional(ENDPOINT)
    api.pending.extend(
        [
            reply(429),
            reply(429),
            reply(success, [] if success == 200 else ""),
            reply(429),
        ]
    )
    deadlines = [
        gh.get_conditional(ENDPOINT, clock=lambda: 1000).rate_limit.resume_at,
        gh.get_conditional(
            "repos/o/r/pulls/17", clock=lambda: 1000
        ).rate_limit.resume_at,
    ]
    gh.get_conditional(ENDPOINT)
    deadlines.append(
        gh.get_conditional(ENDPOINT, clock=lambda: 1000).rate_limit.resume_at
    )
    assert deadlines == [1060, 1120, 1060]


def test_repeated_secondary_limits_stop_after_five_deadlines(api: Any) -> None:
    api.pending.extend(reply(429) for _ in range(6))
    gh = api.reader.GitHub("o/r")
    deadlines = [
        gh.get_conditional(ENDPOINT, clock=lambda: 1000).rate_limit.resume_at
        for _ in range(5)
    ]
    assert deadlines == [1060, 1120, 1240, 1480, 1960]
    with pytest.raises(ValueError, match="retry limit exhausted"):
        gh.get_conditional(ENDPOINT, clock=lambda: 1000)


def test_polling_connection_failure_is_explicit_and_not_replayed(api: Any) -> None:
    api.pending.append(requests.ConnectionError("sensitive transport details"))
    with pytest.raises(api.reader.GitHubTransportError) as error:
        api.reader.GitHub("o/r").get_conditional(ENDPOINT)
    assert error.value.attempts == 1 and len(api.sent) == 1
    assert "sensitive transport details" not in str(error.value)


def test_legacy_get_stays_unconditional_and_keeps_304_as_an_error(api: Any) -> None:
    api.pending.extend([reply(200, [], ETag='"v1"'), reply(200, [2]), reply(304)])
    gh = api.reader.GitHub("o/r")
    gh.get_conditional(ENDPOINT)
    assert gh.get(ENDPOINT) == [2]
    with pytest.raises(api.reader.GitHubHTTPError) as error:
        gh.get(ENDPOINT)
    assert error.value.status == 304
    assert all("If-None-Match" not in r.headers for r in api.sent)


def test_metrics_file_counts_actual_get_200_and_304_without_private_data(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api.pending.extend(
        [reply(200, [TOKEN], ETag='"v1"'), reply(304), reply(403, {"message": TOKEN})]
    )
    metric_file = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("PARENT_METRICS_FILE", str(metric_file))
    gh = api.reader.GitHub("o/r")
    with api.metrics.command("poll", Path(os.environ["PARENT_METRICS_FILE"])):
        assert gh.get_conditional(ENDPOINT).body == ["[REDACTED]"]
        gh.get_conditional(ENDPOINT)
        with pytest.raises(api.reader.GitHubHTTPError):
            gh.get_conditional(ENDPOINT)
    row = json.loads(metric_file.read_text())
    assert (
        row["github_get_requests"],
        row["github_get_200"],
        row["github_get_304"],
        row["github_get_403"],
    ) == (3, 1, 1, 1)
    assert (
        TOKEN not in metric_file.read_text() and ENDPOINT not in metric_file.read_text()
    )


def test_job_log_download_uses_a_separate_unauthenticated_session(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc = tmp_path / "netrc"
    netrc.write_text("machine logs.example login ambient password wrong-token\n")
    monkeypatch.setenv("NETRC", str(netrc))
    api.pending.extend(
        [
            reply(200, []),
            reply(302, Location="https://logs.example/job?signature=private-url"),
            reply(200, f"step failed\n{TOKEN}\n"),
        ]
    )
    gh = api.reader.GitHub("o/r")
    gh.get_conditional(ENDPOINT)
    gh._http._session.cookies.set("private", "cookie", domain="logs.example")
    result = gh.job_log(71)
    assert result.body == "step failed\n[REDACTED]\n" and result.reason is None
    assert api.sent[-2].headers["Authorization"] == f"Bearer {TOKEN}"
    assert (
        "Authorization" not in api.sent[-1].headers
        and "Cookie" not in api.sent[-1].headers
    )
    assert api.sent[-1].headers["Accept"] == "text/plain"
    assert "private-url" not in repr(result)


@pytest.mark.parametrize(
    "failure",
    [
        "permission",
        "expired",
        "missing",
        "redirect_again",
        "empty",
        "no_location",
        "unsafe_url",
        "disconnect",
    ],
)
def test_missing_job_logs_include_a_reason_and_no_signed_url(
    api: Any, failure: str
) -> None:
    redirect = reply(302, Location="https://logs.example/job?signature=private-url")
    if failure == "permission":
        api.pending.append(reply(403, {"message": "Resource not accessible"}))
    elif failure == "no_location":
        api.pending.append(reply(302))
    elif failure == "unsafe_url":
        api.pending.append(reply(302, Location="http://logs.example/private-url"))
    else:
        api.pending.append(redirect)
        api.pending.append(
            {
                "expired": reply(403, "expired private-url"),
                "missing": reply(404),
                "redirect_again": reply(
                    302, Location="https://other.example/private-url"
                ),
                "empty": reply(200),
                "disconnect": requests.ConnectionError("private-url"),
            }[failure]
        )
    result = api.reader.GitHub("o/r").job_log(71)
    assert result.body is None and result.reason
    assert "private-url" not in result.reason
    expected = {
        "permission": 403,
        "expired": 403,
        "missing": 404,
        "redirect_again": 302,
        "empty": 200,
        "no_location": 302,
        "unsafe_url": 302,
        "disconnect": None,
    }
    assert result.status == expected[failure]


def test_job_log_rate_limit_returns_a_deadline_before_any_download(api: Any) -> None:
    api.pending.append(reply(429, {"message": "limited"}, **{"Retry-After": "90"}))
    result = api.reader.GitHub("o/r").job_log(71, clock=lambda: 1000)
    assert result.body is None and result.reason == "secondary_rate_limit"
    assert result.rate_limit.resume_at == 1090 and len(api.sent) == 1
