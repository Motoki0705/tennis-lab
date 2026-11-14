"""Real sockets for pooled REST/GraphQL, failure boundaries and credentials."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    transport = importlib.import_module("issues.transport")
    transport._auth_token.cache_clear()
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("GH_TOKEN", "secret-primary-token")
    return SimpleNamespace(
        transport=transport,
        reader=importlib.import_module("issues.cli"),
        parent=importlib.import_module("parent.github"),
        metrics=importlib.import_module("parent.metrics"),
    )


@contextmanager
def server(
    api: Any, monkeypatch: pytest.MonkeyPatch, replies: list[bytes | None]
) -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def respond(self) -> None:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            calls.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "port": self.client_address[1],
                    "headers": dict(self.headers),
                    "body": self.rfile.read(int(self.headers.get("Content-Length", 0))),
                }
            )
            raw = replies.pop(0)
            if raw is None:
                self.close_connection = True
                return
            self.connection.sendall(raw)
            if self.command == "CONNECT" or "Content-Length: 999" in raw.decode(
                errors="replace"
            ):
                self.close_connection = True

        do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = do_CONNECT = respond

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=http.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    monkeypatch.setattr(
        api.transport, "_API_URL", f"http://127.0.0.1:{http.server_port}/"
    )
    try:
        yield calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=5)
        assert not replies, "the scenario did not exercise every response"


def response(body: object, status: int = 200) -> bytes:
    raw = json.dumps(body).encode()
    return (
        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw
    )


def test_rest_graphql_and_detached_share_one_connection_and_token(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login ambient password wrong-token\n")
    monkeypatch.setenv("NETRC", str(netrc))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an API request started a subprocess")

    monkeypatch.setattr(api.transport.subprocess, "run", forbidden)
    body = '日本語\nquote: " \\ literal \\u001b'
    with server(
        api,
        monkeypatch,
        [
            response({"login": "operator"}),
            response({"body": body}, 201),
            response({"data": {"viewer": {"login": "operator"}}}),
            response([]),
        ],
    ) as calls:
        gh = api.parent.GitHub("o/r")
        try:
            assert gh.get("user") == {"login": "operator"}
            assert gh.write("POST", "repos/o/r/issues/1/comments", {"body": body}) == {
                "body": body
            }
            assert gh.graphql("query { viewer { login } }", {}) == {
                "viewer": {"login": "operator"}
            }
            assert gh.detached().get("repos/o/r/issues/1/comments") == []
        finally:
            gh.close()
    assert len({c["port"] for c in calls}) == 1
    assert [c["method"] for c in calls] == ["GET", "POST", "POST", "GET"]
    assert all(
        c["headers"]["Authorization"] == "Bearer secret-primary-token" for c in calls
    )
    assert json.loads(calls[1]["body"]) == {"body": body}


@pytest.mark.parametrize(
    "fault",
    [
        "disconnect",
        "truncated_200",
        "truncated_429",
        "headers_timeout_200",
        "headers_timeout_429",
    ],
)
def test_status_receipt_ends_socket_retries(
    api: Any, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(api.reader, "time", SimpleNamespace(sleep=sleeps.append))
    code = 429 if fault.endswith("429") else 200
    raw = (
        None
        if fault == "disconnect"
        else f"HTTP/1.1 {code} Test\r\nContent-Length: 999\r\n\r\n".encode()
    )
    if fault.startswith("headers_timeout"):
        monkeypatch.setattr(api.transport, "_TIMEOUT", 0.2)
        raw = f"HTTP/1.1 {code} Test\r\nX-Incomplete: ".encode()
    replies: list[bytes | None] = [None, response([])] if raw is None else [raw]
    with server(api, monkeypatch, replies) as calls:
        gh = api.reader.GitHub("o/r")
        try:
            if fault == "disconnect":
                assert gh.get("user") == []
            else:
                with pytest.raises(api.reader.GitHubHTTPError) as error:
                    gh.get("user")
                assert error.value.status == code and error.value.attempts == 1
        finally:
            gh.close()
    assert len(calls) == (2 if fault == "disconnect" else 1)
    assert sleeps == ([1.0] if fault == "disconnect" else [])


@pytest.mark.parametrize("operation", ["GET", "PATCH", "graphql"])
def test_tokens_are_redacted_from_responses_errors_tracebacks_and_metrics(
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    operation: str,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "secret-secondary-token")
    message = "secret-primary-token secret-secondary-token\x1b[31m"
    payload = (
        {"errors": [{"message": message}]}
        if operation == "graphql"
        else {"message": message}
    )
    metric_file = tmp_path / "metrics.jsonl"
    with server(
        api, monkeypatch, [response(payload, 200 if operation == "graphql" else 403)]
    ):
        gh = api.parent.GitHub("o/r")
        try:
            with api.metrics.command("secret-test", metric_file):
                with pytest.raises(ValueError) as error:
                    if operation == "GET":
                        gh.get("user")
                    elif operation == "graphql":
                        gh.graphql("query {}", {})
                    else:
                        gh.write("PATCH", "repos/o/r/issues/1", {})
                print("".join(traceback.format_exception(error.value)))
        finally:
            gh.close()
    captured = capsys.readouterr()
    output = captured.out + captured.err + metric_file.read_text()
    assert "secret-primary-token" not in output
    assert "secret-secondary-token" not in output
    assert "[REDACTED]" in output
    assert "\x1b" not in output


@pytest.mark.parametrize("credential", ["GH_TOKEN", "GITHUB_TOKEN", "gh_auth"])
def test_credentials_resolve_once_with_environment_precedence(
    api: Any, monkeypatch: pytest.MonkeyPatch, credential: str
) -> None:
    monkeypatch.delenv("GH_TOKEN")
    lookups: list[list[str]] = []

    def lookup(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        lookups.append(args)
        return subprocess.CompletedProcess(args, 0, "secret-auth-token\n", "")

    monkeypatch.setattr(api.transport.subprocess, "run", lookup)
    if credential != "gh_auth":
        monkeypatch.setenv(credential, "secret-chosen-token")
    if credential == "GH_TOKEN":
        monkeypatch.setenv("GITHUB_TOKEN", "secret-ignored-token")
    with server(api, monkeypatch, [response([]), response([])]) as calls:
        for _ in range(2):
            gh = api.reader.GitHub("o/r")
            try:
                assert gh.get("user") == []
            finally:
                gh.close()
    assert len(lookups) == (1 if credential == "gh_auth" else 0)
    expected = "secret-auth-token" if credential == "gh_auth" else "secret-chosen-token"
    assert all(c["headers"]["Authorization"] == "Bearer " + expected for c in calls)


def test_auth_lookup_failure_never_includes_credential_output(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.setattr(
        api.transport.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 1, "secret stdout", "secret stderr"
        ),
    )
    gh = api.reader.GitHub("o/r")
    with pytest.raises(ValueError, match="token lookup failed") as error:
        gh.get("user")
    assert "secret" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/user",
        "//example.com/user",
        "../user",
        "repos/../user",
        "user\nAuthorization: secret",
        "user#fragment",
    ],
)
def test_absolute_and_unsafe_endpoints_fail_before_sending_credentials(
    api: Any, endpoint: str
) -> None:
    with pytest.raises(ValueError, match="relative API path"):
        api.reader.GitHub("o/r").get(endpoint)


@pytest.mark.parametrize("operation", ["POST", "PATCH", "PUT", "DELETE", "graphql"])
def test_socket_disconnect_never_replays_a_mutation(
    api: Any, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    with server(api, monkeypatch, [None]) as calls:
        gh = api.parent.GitHub("o/r")
        try:
            with pytest.raises(ValueError, match="failed"):
                if operation == "graphql":
                    gh.graphql("query {}", {})
                else:
                    gh.write(operation, "repos/o/r/issues/1", {"body": "once"})
        finally:
            gh.close()
    assert len(calls) == 1


def test_redirect_is_not_followed(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"HTTP/1.1 302 Redirect\r\nLocation: https://example.invalid/steal\r\nContent-Length: 0\r\n\r\n"
    with server(api, monkeypatch, [raw]) as calls:
        gh = api.reader.GitHub("o/r")
        try:
            with pytest.raises(api.reader.GitHubHTTPError) as error:
                gh.get("user")
            assert error.value.status == 302
        finally:
            gh.close()
    assert len(calls) == 1


def test_proxy_tunnel_status_does_not_hide_an_api_transport_disconnect(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(api.reader, "time", SimpleNamespace(sleep=sleeps.append))
    with server(api, monkeypatch, [response({})] * 3) as calls:
        proxy = api.transport._API_URL
        for name in ("HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(name, proxy)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")
        monkeypatch.setattr(api.transport, "_API_URL", "https://api.github.com/")
        gh = api.reader.GitHub("o/r")
        try:
            with pytest.raises(api.reader.GitHubTransportError) as error:
                gh.get("user")
            assert error.value.attempts == 3
        finally:
            gh.close()
    assert len(calls) == 3 and sleeps == [1.0, 2.0]
    assert all(
        c["method"] == "CONNECT" and "Authorization" not in c["headers"] for c in calls
    )


def test_json_unicode_escapes_cannot_echo_a_token(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b'{"message":"\\u0073ecret-primary-token"}'
    raw = f"HTTP/1.1 200 Test\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
    with server(api, monkeypatch, [raw]):
        gh = api.reader.GitHub("o/r")
        try:
            assert gh.get("user") == {"message": "[REDACTED]"}
        finally:
            gh.close()
