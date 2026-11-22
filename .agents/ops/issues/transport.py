"""Persistent GitHub HTTPS transport with explicit authentication and no replay.

CompletedProcess/TimeoutExpired are the existing reader's injection boundary;
no API request is executed in a subprocess. Only credential lookup may use gh.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast
from urllib.parse import urlsplit

import requests
from urllib3 import connection, connectionpool

_API_URL = "https://api.github.com/"
_TIMEOUT = 60
_logins: dict[tuple[str, str], str] = {}
_received_status: ContextVar[int | None] = ContextVar(
    "github_http_status", default=None
)


@dataclass(frozen=True)
class HTTPResponse:
    """Wire status and selected lowercase headers; error means incomplete I/O."""

    status: int | None
    headers: dict[str, str]
    body: str
    error: str = ""

    def http_error(self) -> str:
        try:
            data = json.loads(self.body)
            message = data.get("message", "") if isinstance(data, dict) else ""
        except ValueError:
            message = ""
        message = re.sub(r"[\x00-\x1f\x7f]", " ", str(message))
        return f"GitHub: {message} (HTTP {self.status})"


class _Response(http.client.HTTPResponse):
    def begin(self) -> None:
        # Proxy CONNECT also calls _read_status directly. Only the API response
        # is a freshness/retry boundary; a tunnel handshake is transport setup.
        self._observing = True
        try:
            super().begin()
        finally:
            self._observing = False

    def _read_status(self) -> tuple[str, int, str]:
        # requests cannot otherwise distinguish a timeout halfway through the
        # headers from one before the status line. Record the protocol event,
        # rather than infer it from exception text.
        result = cast(tuple[str, int, str], cast(Any, super())._read_status())
        if getattr(self, "_observing", False) and result[1] >= 200:
            _received_status.set(result[1])
        return result


class _Connection(connection.HTTPConnection):
    response_class = _Response


class _TLSConnection(connection.HTTPSConnection):
    response_class = _Response


class _Pool(connectionpool.HTTPConnectionPool):
    ConnectionCls = _Connection


class _TLSPool(connectionpool.HTTPSConnectionPool):
    ConnectionCls = _TLSConnection


class _Adapter(requests.adapters.HTTPAdapter):
    @staticmethod
    def _observe(manager: Any) -> None:
        manager.pool_classes_by_scheme = {
            **manager.pool_classes_by_scheme,
            "http": _Pool,
            "https": _TLSPool,
        }

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        super().init_poolmanager(*args, **kwargs)
        self._observe(self.poolmanager)

    def proxy_manager_for(self, *args: Any, **kwargs: Any) -> Any:
        manager = super().proxy_manager_for(*args, **kwargs)
        self._observe(manager)
        return manager


class _BearerAuth(requests.auth.AuthBase):
    def __init__(self, token: str) -> None:
        self.token = token

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        request.headers["Authorization"] = f"Bearer {self.token}"
        return request


class _NoAuth(requests.auth.AuthBase):
    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        # An explicit auth object prevents requests from consulting .netrc.
        request.headers.pop("Authorization", None)
        return request


@lru_cache(maxsize=8)
def _auth_token(config_dir: str | None, executable: str | None) -> str:
    # Neither gh's stderr nor its argv/result is included in any diagnostic.
    try:
        result = subprocess.run(
            [executable or "gh", "auth", "token", "--hostname", "github.com"],
            text=True,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("GitHub authentication token lookup failed") from None
    if result.returncode or not result.stdout.strip():
        raise ValueError("GitHub authentication token lookup failed")
    return result.stdout.strip()


def _token() -> str:
    if os.environ.get("GH_HOST", "github.com").casefold() != "github.com":
        raise ValueError("GitHub transport requires GH_HOST=github.com")
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or _auth_token(os.environ.get("GH_CONFIG_DIR"), shutil.which("gh"))
    )
    if not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError("invalid GitHub authentication token")
    return token


class HTTPClient:
    """One connection pool; shared by transaction and detached reader copies."""

    def __init__(self) -> None:
        self._session: requests.Session | None = None
        self._token: str | None = None

    def prepare_authentication(self) -> None:
        """Resolve credentials before entering any registry-bound callback."""
        self._token = self._token or _token()

    def authenticated_login(self, fetch: Callable[[], str]) -> str:
        # Principal identity is stable for a credential, unlike issue facts.
        # Keep only login, keyed by API host and an opaque credential digest;
        # neither tokens nor complete /user responses enter the shared cache.
        self.prepare_authentication()
        assert self._token is not None
        key = (_API_URL, hashlib.sha256(self._token.encode()).hexdigest())
        if key not in _logins:
            _logins[key] = fetch()
        return _logins[key]

    def redact(self, text: str) -> str:
        secrets = {
            self._token,
            os.environ.get("GH_TOKEN"),
            os.environ.get("GITHUB_TOKEN"),
        }
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
            text = text.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        return text

    def redact_json(self, text: str) -> str:
        """Also scrub credentials reconstructed by JSON escape decoding."""
        text = self.redact(text)
        try:
            value = json.loads(text)
        except ValueError:
            return text

        def scrub(value: Any) -> Any:
            if isinstance(value, str):
                return self.redact(value)
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, dict):
                result = {self.redact(key): scrub(item) for key, item in value.items()}
                if len(result) != len(value):
                    raise ValueError(
                        "GitHub response keys collide after credential redaction"
                    )
                return result
            return value

        cleaned = scrub(value)
        return json.dumps(cleaned, ensure_ascii=False) if cleaned != value else text

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def cache_key(self, endpoint: str) -> tuple[str, str]:
        """Scope a representation to its exact URL and selected credential."""
        url = self._endpoint_url(endpoint)
        self.prepare_authentication()
        assert self._token is not None
        return url, hashlib.sha256(self._token.encode()).hexdigest()

    def request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        include: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Legacy boundary: status line only, unchanged success/error contract."""
        response = self.request_response(method, endpoint, payload)
        status = response.status
        headers = f"HTTP/1.1 {status}\n\n" if include and status is not None else ""
        success = status is not None and 200 <= status < 300
        error = response.error or ("" if success else response.http_error())
        return subprocess.CompletedProcess(
            [], int(bool(error) or not success), headers + response.body, error
        )

    @staticmethod
    def _endpoint_url(endpoint: str) -> str:
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or endpoint.startswith("/")
            or "\\" in endpoint
            or any(ord(c) < 32 or ord(c) == 127 for c in endpoint)
            or any(p in {".", ".."} for p in parsed.path.split("/"))
        ):
            raise ValueError("GitHub API endpoint must be a relative API path")
        return _API_URL + endpoint

    def request_response(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HTTPResponse:
        """Additive API retaining wire metadata without following redirects."""
        url = self._endpoint_url(endpoint)
        if self._session is None:
            self.prepare_authentication()
            assert self._token is not None
            self._session = requests.Session()
            self._session.mount("https://", _Adapter(max_retries=0))
            self._session.mount("http://", _Adapter(max_retries=0))
            # An explicit auth object also prevents ambient .netrc credentials
            # from silently replacing the selected GitHub token.
            self._session.auth = _BearerAuth(self._token)
            self._session.headers.update(
                {
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "tennis-lab-agents",
                }
            )
        return self._exchange(self._session, method, url, payload, headers=headers)

    def download_log(self, url: str) -> HTTPResponse:
        """Fetch one signed HTTPS URL with no API auth, cookies or .netrc."""
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and parsed.port in {None, 443}
                and not parsed.fragment
                and "\\" not in url
                and not any(ord(c) < 33 or ord(c) == 127 for c in url)
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("GitHub log redirect must be an absolute HTTPS URL")
        # Never reuse the API session: even a same-host URL receives no token.
        with requests.Session() as session:
            session.auth = _NoAuth()
            session.mount("https://", _Adapter(max_retries=0))
            return self._exchange(
                session, "GET", url, headers={"Accept": "text/plain"}, plain_text=True
            )

    def _exchange(
        self,
        session: requests.Session,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        plain_text: bool = False,
    ) -> HTTPResponse:
        from parent import metrics

        if method == "GET":
            metrics.count("github_get_requests")
        # requests' default adapter has zero retries. Redirects must not move
        # credentials or change the repository/API host behind the caller.
        observation = _received_status.set(None)
        try:
            response = session.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=_TIMEOUT,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            status = _received_status.get()
            if status is not None:
                if method == "GET":
                    metrics.count(f"github_get_{status}")
                return HTTPResponse(
                    status,
                    {},
                    "",
                    f"GitHub response failed after HTTP {status} ({type(exc).__name__})",
                )
            if isinstance(exc, requests.Timeout):
                raise subprocess.TimeoutExpired(
                    "GitHub API request", _TIMEOUT
                ) from None
            # Exception messages/requests can contain credentials and URLs.
            return HTTPResponse(
                None, {}, "", f"GitHub connection failed ({type(exc).__name__})"
            )
        finally:
            _received_status.reset(observation)
        status = response.status_code
        if method == "GET":
            metrics.count(f"github_get_{status}")
        response_headers = {
            name.lower(): self.redact(value)
            for name, value in response.headers.items()
            if name.lower().startswith("x-ratelimit-")
            or name.lower()
            in {"etag", "retry-after", "x-poll-interval", "location", "link"}
        }
        try:
            # Receiving headers ends transport retries, including status=200
            # followed by a read timeout, truncated body or connection reset.
            text = response.content.decode("utf-8")
            body = self.redact(text) if plain_text else self.redact_json(text)
        except (requests.RequestException, UnicodeError) as exc:
            return HTTPResponse(
                status,
                response_headers,
                "",
                f"GitHub response body failed after HTTP {status} ({type(exc).__name__})",
            )
        finally:
            response.close()
        return HTTPResponse(status, response_headers, body)
