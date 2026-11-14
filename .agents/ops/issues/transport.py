"""Persistent GitHub HTTPS transport with explicit authentication and no replay.

CompletedProcess/TimeoutExpired are the existing reader's injection boundary;
no API request is executed in a subprocess. Only credential lookup may use gh.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, cast
from urllib.parse import urlsplit

import requests
from urllib3 import connection, connectionpool

_API_URL = "https://api.github.com/"
_TIMEOUT = 60
_received_status: ContextVar[int | None] = ContextVar(
    "github_http_status", default=None
)


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

    def request(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        *,
        include: bool = False,
    ) -> subprocess.CompletedProcess[str]:
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
        if self._session is None:
            token = self._token or _token()
            self._token = token
            self._session = requests.Session()
            self._session.mount("https://", _Adapter(max_retries=0))
            self._session.mount("http://", _Adapter(max_retries=0))
            # An explicit auth object also prevents ambient .netrc credentials
            # from silently replacing the selected GitHub token.
            self._session.auth = _BearerAuth(token)
            self._session.headers.update(
                {
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "tennis-lab-agents",
                }
            )
        # requests' default adapter has zero retries. Redirects must not move
        # credentials or change the repository/API host behind the caller.
        observation = _received_status.set(None)
        try:
            response = self._session.request(
                method,
                _API_URL + endpoint,
                json=payload,
                timeout=_TIMEOUT,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            status = _received_status.get()
            if status is not None:
                return subprocess.CompletedProcess(
                    [],
                    1,
                    f"HTTP/1.1 {status}\n\n" if include else "",
                    f"GitHub response failed after HTTP {status} ({type(exc).__name__})",
                )
            if isinstance(exc, requests.Timeout):
                raise subprocess.TimeoutExpired(
                    "GitHub API request", _TIMEOUT
                ) from None
            # Exception messages/requests can contain credentials and URLs.
            return subprocess.CompletedProcess(
                [], 1, "", f"GitHub connection failed ({type(exc).__name__})"
            )
        finally:
            _received_status.reset(observation)
        status = response.status_code
        headers = f"HTTP/1.1 {status}\n\n" if include else ""
        try:
            # Receiving headers ends transport retries, including status=200
            # followed by a read timeout, truncated body or connection reset.
            body = self.redact_json(response.content.decode("utf-8"))
        except (requests.RequestException, UnicodeError) as exc:
            return subprocess.CompletedProcess(
                [],
                1,
                headers,
                f"GitHub response body failed after HTTP {status} ({type(exc).__name__})",
            )
        finally:
            response.close()
        error = ""
        if not 200 <= status < 300:
            try:
                data = json.loads(body)
                message = data.get("message", "") if isinstance(data, dict) else ""
            except ValueError:
                message = ""
            # Keep control characters out of human-readable diagnostics.
            message = re.sub(r"[\x00-\x1f\x7f]", " ", str(message))
            error = f"GitHub: {message} (HTTP {status})"
        return subprocess.CompletedProcess(
            [], int(not 200 <= status < 300), headers + body, error
        )
