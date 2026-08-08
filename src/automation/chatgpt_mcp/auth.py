"""Single-owner OAuth 2.1 provider backed by the MCP Python SDK routes."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    TokenError,
    TokenVerifier,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore

_SCOPES = ["wsl:read", "wsl:write"]


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _append_query(url: str, values: dict[str, str | None]) -> str:
    parsed = urlsplit(url)
    query = urlencode(
        {key: value for key, value in values.items() if value is not None}
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment)
    )


def _is_allowed_redirect_uri(uri: str, *, allow_localhost: bool) -> bool:
    parsed = urlsplit(uri)
    if parsed.fragment or parsed.username is not None or parsed.password is not None:
        return False
    if (
        parsed.scheme == "https"
        and parsed.hostname == "chatgpt.com"
        and parsed.port in {None, 443}
    ):
        return parsed.path.startswith("/connector/oauth/") or (
            parsed.path == "/connector_platform_oauth_redirect"
        )
    if allow_localhost and parsed.scheme == "http":
        return parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    return False


@dataclass(frozen=True)
class PendingAuthorization:
    transaction_id: str
    client_id: str
    client_name: str
    params: AuthorizationParams


class OwnerOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken],
    TokenVerifier,
):
    """Durable OAuth provider that authorizes one local owner via a secret."""

    def __init__(
        self,
        settings: GatewaySettings,
        store: SqliteStore,
        *,
        allow_localhost_redirects: bool = False,
    ) -> None:
        self.settings = settings
        self.store = store
        self.allow_localhost_redirects = allow_localhost_redirects
        self._owner_secret = settings.read_owner_secret()

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        payload = self.store.get("clients", client_id)
        if payload is None:
            return None
        return OAuthClientInformationFull.model_validate(payload)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if not client_info.client_id:
            raise RegistrationError(
                "invalid_client_metadata", "generated client_id is missing"
            )
        if client_info.token_endpoint_auth_method not in {
            "none",
            "client_secret_post",
            "client_secret_basic",
        }:
            raise RegistrationError(
                "invalid_client_metadata",
                "unsupported token endpoint authentication method",
            )
        redirect_uris = client_info.redirect_uris or []
        if not redirect_uris or not all(
            _is_allowed_redirect_uri(
                str(uri), allow_localhost=self.allow_localhost_redirects
            )
            for uri in redirect_uris
        ):
            raise RegistrationError(
                "invalid_redirect_uri",
                "redirect URIs must be ChatGPT connector callbacks",
            )
        self.store.put(
            "clients",
            client_info.client_id,
            client_info.model_dump(mode="json"),
        )

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        if not client.client_id:
            raise TokenError("invalid_client", "OAuth client has no client_id")
        if params.resource != self.settings.resource_url:
            from mcp.server.auth.provider import AuthorizeError

            raise AuthorizeError(
                "invalid_request",
                f"resource must be {self.settings.resource_url}",
            )
        scopes = params.scopes or []
        if not set(_SCOPES).issubset(scopes):
            from mcp.server.auth.provider import AuthorizeError

            raise AuthorizeError("invalid_scope", "wsl:read and wsl:write are required")

        transaction_id = secrets.token_urlsafe(32)
        payload = {
            "transaction_id": transaction_id,
            "client_id": client.client_id,
            "client_name": client.client_name or "ChatGPT",
            "params": params.model_dump(mode="json"),
        }
        self.store.put(
            "pending_authorizations",
            transaction_id,
            payload,
            expires_at=time.time() + self.settings.authorization_ttl_seconds,
        )
        return f"{self.settings.public_base_url}/oauth/approve?{urlencode({'transaction': transaction_id})}"

    def get_pending_authorization(
        self, transaction_id: str
    ) -> PendingAuthorization | None:
        payload = self.store.get("pending_authorizations", transaction_id)
        if payload is None:
            return None
        return PendingAuthorization(
            transaction_id=str(payload["transaction_id"]),
            client_id=str(payload["client_id"]),
            client_name=str(payload["client_name"]),
            params=AuthorizationParams.model_validate(payload["params"]),
        )

    def approve_authorization(self, transaction_id: str, owner_secret: str) -> str:
        """Consume one pending request and return its validated client callback URL."""

        if not secrets.compare_digest(owner_secret, self._owner_secret):
            raise PermissionError("owner secret did not match")
        payload = self.store.pop("pending_authorizations", transaction_id)
        if payload is None:
            raise ValueError("authorization request is missing, expired, or already used")

        params = AuthorizationParams.model_validate(payload["params"])
        client_id = str(payload["client_id"])
        code_value = secrets.token_urlsafe(32)
        code = AuthorizationCode(
            code=code_value,
            scopes=params.scopes or [],
            expires_at=time.time() + self.settings.authorization_ttl_seconds,
            client_id=client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject="owner",
        )
        code_payload = code.model_dump(mode="json")
        code_payload["code"] = ""
        self.store.put(
            "authorization_codes",
            _token_key(code_value),
            code_payload,
            expires_at=code.expires_at,
        )
        return _append_query(
            str(params.redirect_uri),
            {"code": code_value, "state": params.state},
        )

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        payload = self.store.get("authorization_codes", _token_key(authorization_code))
        if payload is None:
            return None
        payload["code"] = authorization_code
        code = AuthorizationCode.model_validate(payload)
        if code.client_id != client.client_id:
            return None
        return code

    def _issue_token_pair(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str,
        subject: str,
    ) -> OAuthToken:
        now = int(time.time())
        access_value = secrets.token_urlsafe(48)
        access_expires_at = now + self.settings.access_token_ttl_seconds
        access = AccessToken(
            token=access_value,
            client_id=client_id,
            scopes=scopes,
            expires_at=access_expires_at,
            resource=resource,
            subject=subject,
        )
        access_payload = access.model_dump(mode="json")
        access_payload["token"] = ""
        self.store.put(
            "access_tokens",
            _token_key(access_value),
            access_payload,
            expires_at=access_expires_at,
        )

        refresh_value = secrets.token_urlsafe(48)
        refresh_expires_at = now + self.settings.refresh_token_ttl_seconds
        refresh_payload: dict[str, Any] = {
            "token": "",
            "client_id": client_id,
            "scopes": scopes,
            "expires_at": refresh_expires_at,
            "subject": subject,
            "resource": resource,
        }
        self.store.put(
            "refresh_tokens",
            _token_key(refresh_value),
            refresh_payload,
            expires_at=refresh_expires_at,
        )
        return OAuthToken(
            access_token=access_value,
            token_type="Bearer",
            expires_in=self.settings.access_token_ttl_seconds,
            scope=" ".join(scopes),
            refresh_token=refresh_value,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        consumed = self.store.pop(
            "authorization_codes", _token_key(authorization_code.code)
        )
        if consumed is None:
            raise TokenError("invalid_grant", "authorization code was already used")
        if authorization_code.resource != self.settings.resource_url:
            raise TokenError("invalid_grant", "authorization code resource mismatch")
        return self._issue_token_pair(
            client_id=str(client.client_id),
            scopes=authorization_code.scopes,
            resource=self.settings.resource_url,
            subject=authorization_code.subject or "owner",
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        payload = self.store.get("refresh_tokens", _token_key(refresh_token))
        if payload is None or payload.get("client_id") != client.client_id:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=str(payload["client_id"]),
            scopes=list(payload["scopes"]),
            expires_at=int(payload["expires_at"]),
            subject=str(payload.get("subject") or "owner"),
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        payload = self.store.pop("refresh_tokens", _token_key(refresh_token.token))
        if payload is None:
            raise TokenError("invalid_grant", "refresh token was already used")
        resource = str(payload.get("resource") or "")
        if resource != self.settings.resource_url:
            raise TokenError("invalid_grant", "refresh token resource mismatch")
        original_scopes = set(str(value) for value in payload.get("scopes", []))
        requested_scopes = set(scopes) if scopes else original_scopes
        if not requested_scopes.issubset(original_scopes):
            raise TokenError("invalid_scope", "refresh scope exceeds the original grant")
        return self._issue_token_pair(
            client_id=str(client.client_id),
            scopes=sorted(requested_scopes),
            resource=resource,
            subject=refresh_token.subject or "owner",
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        payload = self.store.get("access_tokens", _token_key(token))
        if payload is None:
            return None
        payload["token"] = token
        access = AccessToken.model_validate(payload)
        if access.resource != self.settings.resource_url:
            return None
        if access.expires_at is None or access.expires_at <= int(time.time()):
            return None
        if not set(_SCOPES).issubset(access.scopes):
            return None
        return access

    async def verify_token(self, token: str) -> AccessToken | None:
        return await self.load_access_token(token)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        key = _token_key(token.token)
        self.store.delete("access_tokens", key)
        self.store.delete("refresh_tokens", key)


def oauth_scopes() -> list[str]:
    """Return the stable scope list advertised by this gateway."""

    return list(_SCOPES)
