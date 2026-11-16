"""Real sockets for pooled REST/GraphQL, failure boundaries and credentials."""

from __future__ import annotations

import importlib
import json
import socket
import subprocess
import sys
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
    monkeypatch.setattr(transport, "_logins", {})
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


def response(body: object, status: int = 200, /, **headers: str) -> bytes:
    raw = json.dumps(body).encode()
    metadata = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    return (
        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(raw)}\r\n{metadata}\r\n".encode()
        + raw
    )


def test_conditional_parent_reads_revalidate_on_the_wire_with_current_headers(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = importlib.import_module("parent.state").Store(tmp_path / "registry")
    endpoint = "repos/o/r/pulls/17"
    with server(
        api,
        monkeypatch,
        [
            response({"revision": 0}),
            response({"revision": 1}, ETag='"v1"', **{"X-RateLimit-Remaining": "49"}),
            b'HTTP/1.1 304 Not Modified\r\nETag: "v1"\r\nX-RateLimit-Remaining: 48\r\nX-Poll-Interval: 90\r\n\r\n',
        ],
    ) as calls:
        gh = api.parent.GitHub("o/r", store=store)
        gh.parent = 1
        try:
            with pytest.raises(ValueError, match="active registry transaction"):
                gh.get_conditional(endpoint)
            with api.metrics.command("conditional", None) as row, store.locked():
                assert gh.get(endpoint) == {"revision": 0}
                assert gh.get_conditional(endpoint).body == {"revision": 1}
                current = gh.get_conditional(endpoint)
                assert current.status == 304 and current.body == {"revision": 1}
                assert current.headers["x-ratelimit-remaining"] == "48"
                assert current.headers["x-poll-interval"] == "90"
                assert gh.get(endpoint) == {"revision": 0}
                with pytest.raises(ValueError, match="lock to be released"):
                    gh.detached().get_conditional(endpoint)
        finally:
            gh.close()
    assert len({c["port"] for c in calls}) == 1
    assert [c["headers"].get("If-None-Match") for c in calls] == [None, None, '"v1"']
    assert (
        row["github_get_requests"],
        row["github_get_200"],
        row["github_get_304"],
    ) == (3, 2, 1)
    assert row["github_requests"] == 3 and row["cache_hits"] == 1


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


def test_login_is_reused_across_clients_transactions_and_detached_reads(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = importlib.import_module("parent.state").Store(tmp_path / "registry")
    with server(api, monkeypatch, [response({"login": "operator"})]) as calls:
        gh = api.parent.GitHub("o/r", store=store)
        other = api.parent.GitHub("other/repo")
        try:
            with store.locked():
                assert gh.authenticated_login() == "operator"
                with gh.io_scope():
                    assert gh.detached().authenticated_login() == "operator"
                with pytest.raises(ValueError, match="lock to be released"):
                    gh.detached().authenticated_login()
            with pytest.raises(ValueError, match="active registry transaction"):
                gh.authenticated_login()
            with store.locked():
                assert gh.authenticated_login() == "operator"
            assert other.authenticated_login() == "operator"
        finally:
            gh.close()
            other.close()
    assert len(calls) == 1 and calls[0]["path"] == "/user"


@pytest.fixture
def auth_registry(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, Path]:
    """Run the actual token subprocess against a real, independently probed lock."""
    store = importlib.import_module("parent.state").Store(tmp_path / "registry")
    store.focus("o/r#1")
    with store.locked():
        store.parents["o/r#1"] = {
            "session": "operator-session",
            "status": "active",
            "generation": 1,
            "tasks": {},
            "reviews": [],
        }
        store.save()
    executable = tmp_path / "gh"
    executable.write_text(
        f"#!{sys.executable}\n"
        + r"""
import fcntl, json, os, pathlib, sys
directory = pathlib.Path(os.environ["AUTH_TEST_REGISTRY"])
acquired = False
with (directory / "registry.lock").open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        acquired = True
        if os.environ.get("AUTH_TEST_OUTCOME") == "changed":
            path = directory / "registry.json"
            data = json.loads(path.read_text())
            data["parents"]["o/r#1"]["generation"] += 1
            path.write_text(json.dumps(data))
pathlib.Path(os.environ["AUTH_TEST_TRACE"]).write_text(json.dumps({
    "acquired": acquired, "args": sys.argv[1:]
}))
if os.environ.get("AUTH_TEST_OUTCOME") == "failure":
    print("secret-auth-token")
    print("secret-auth-stderr", file=sys.stderr)
    sys.exit(1)
print("secret-auth-token")
"""
    )
    executable.chmod(0o700)
    trace = tmp_path / "auth-trace.json"
    monkeypatch.delenv("GH_TOKEN")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("AUTH_TEST_REGISTRY", str(store.directory))
    monkeypatch.setenv("AUTH_TEST_TRACE", str(trace))
    return store, trace


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_token_subprocess_releases_registry_lock_and_hides_output(
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    auth_registry: tuple[Any, Path],
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    store, trace = auth_registry
    monkeypatch.setenv("AUTH_TEST_OUTCOME", outcome)
    replies: list[bytes | None] = (
        [response({"login": "operator"})] if outcome == "success" else []
    )
    with server(api, monkeypatch, replies) as calls:
        gh = api.parent.GitHub("o/r", store=store)
        try:
            with store.locked():
                if outcome == "success":
                    assert gh.authenticated_login() == "operator"
                else:
                    with pytest.raises(
                        ValueError, match="token lookup failed"
                    ) as error:
                        gh.authenticated_login()
                    assert "secret" not in "".join(
                        traceback.format_exception(error.value)
                    )
                store.save()
        finally:
            gh.close()
    assert json.loads(trace.read_text()) == {
        "acquired": True,
        "args": ["auth", "token", "--hostname", "github.com"],
    }
    assert len(calls) == (1 if outcome == "success" else 0)
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err


def test_token_subprocess_revalidates_registry_before_login_fetch(
    api: Any,
    monkeypatch: pytest.MonkeyPatch,
    auth_registry: tuple[Any, Path],
) -> None:
    store, trace = auth_registry
    monkeypatch.setenv("AUTH_TEST_OUTCOME", "changed")
    state = importlib.import_module("parent.state")
    gh = api.parent.GitHub("o/r", store=store)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "login fetch must not follow a registry conflict during token lookup"
        )

    monkeypatch.setattr(gh, "get", forbidden)
    try:
        with store.locked():
            with pytest.raises(state.RegistryConflict, match="registry changed"):
                gh.authenticated_login()
            # Catching the conflict must not reauthorize the stale transaction.
            with pytest.raises(state.RegistryConflict, match="registry changed"):
                store.save()
        assert json.loads(trace.read_text())["acquired"] is True
        with store.locked():
            assert store.parents["o/r#1"]["generation"] == 2
            store.save()
    finally:
        gh.close()


def test_login_cache_separates_credentials_and_api_hosts(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with server(
        api,
        monkeypatch,
        [response({"login": "first"}), response({"login": "second"})],
    ) as calls:
        for token, login in (("first-token", "first"), ("second-token", "second")):
            monkeypatch.setenv("GH_TOKEN", token)
            gh = api.parent.GitHub("o/r")
            try:
                assert gh.authenticated_login() == login
            finally:
                gh.close()
    with server(api, monkeypatch, [response({"login": "other-host"})]) as other:
        gh = api.parent.GitHub("o/r")
        try:
            assert gh.authenticated_login() == "other-host"
        finally:
            gh.close()
    assert len(calls) == 2 and len(other) == 1
    assert "first-token" not in repr(api.transport._logins)
    assert "second-token" not in repr(api.transport._logins)


def test_failed_login_is_not_cached(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with server(
        api,
        monkeypatch,
        [response({"message": "unauthorized"}, 401), response({"login": "repaired"})],
    ) as calls:
        gh = api.parent.GitHub("o/r")
        try:
            with pytest.raises(api.reader.GitHubHTTPError):
                gh.authenticated_login()
            assert gh.authenticated_login() == "repaired"
        finally:
            gh.close()
    assert len(calls) == 2


def test_write_invalidates_target_relations_without_evicting_other_issue_numbers(
    api: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = importlib.import_module("parent.state").Store(tmp_path / "registry")
    with server(
        api, monkeypatch, [response({"revision": n}) for n in range(8)]
    ) as calls:
        gh = api.parent.GitHub("o/r", store=store)
        gh.parent = 1
        try:
            with store.locked():
                for endpoint in (
                    "issues/1",
                    "issues/1/parent",
                    "issues/10",
                    "issues/10/parent",
                ):
                    gh.get(f"repos/o/r/{endpoint}")
                gh.write("PATCH", "repos/o/r/issues/1", {"body": "updated"})
                assert gh.get("repos/o/r/issues/10") == {"revision": 2}
                assert gh.get("repos/o/r/issues/10/parent") == {"revision": 3}
                assert gh.get("repos/o/r/issues/1") == {"revision": 5}
                assert gh.get("repos/o/r/issues/1/parent") == {"revision": 6}
                with gh.fresh_reads():
                    assert gh.get("repos/o/r/issues/10") == {"revision": 7}
        finally:
            gh.close()
    assert [c["path"] for c in calls[-3:]] == [
        "/repos/o/r/issues/1",
        "/repos/o/r/issues/1/parent",
        "/repos/o/r/issues/10",
    ]


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
