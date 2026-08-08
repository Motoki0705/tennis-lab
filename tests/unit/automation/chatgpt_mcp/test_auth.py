from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from mcp.server.auth.provider import AuthorizationParams, TokenError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from src.automation.chatgpt_mcp.auth import OwnerOAuthProvider, oauth_scopes
from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore


def _settings(tmp_path: Path) -> GatewaySettings:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    settings = GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        public_base_url="https://mcp.example.test",
    )
    settings.ensure_state()
    return settings


def _client() -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id="chatgpt-client",
        client_id_issued_at=1,
        redirect_uris=[AnyUrl("https://chatgpt.com/connector/oauth/test-callback")],
        token_endpoint_auth_method="none",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=" ".join(oauth_scopes()),
        client_name="ChatGPT",
    )


def _authorization_params(client: OAuthClientInformationFull, resource: str) -> AuthorizationParams:
    assert client.redirect_uris is not None
    return AuthorizationParams(
        state="state-value",
        scopes=oauth_scopes(),
        code_challenge="challenge-value",
        redirect_uri=client.redirect_uris[0],
        redirect_uri_provided_explicitly=True,
        resource=resource,
    )


def test_owner_oauth_round_trip_one_time_approval_and_refresh_rotation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    provider = OwnerOAuthProvider(settings, SqliteStore(settings.database_path))
    client = _client()
    asyncio.run(provider.register_client(client))
    params = _authorization_params(client, settings.resource_url)

    approval_url = asyncio.run(provider.authorize(client, params))
    transaction = parse_qs(urlsplit(approval_url).query)["transaction"][0]
    with pytest.raises(PermissionError):
        provider.approve_authorization(transaction, "wrong-secret")

    callback = provider.approve_authorization(transaction, settings.read_owner_secret())
    callback_query = parse_qs(urlsplit(callback).query)
    code_value = callback_query["code"][0]
    assert callback_query["state"] == ["state-value"]

    with pytest.raises(ValueError, match="already used"):
        provider.approve_authorization(transaction, settings.read_owner_secret())

    code = asyncio.run(provider.load_authorization_code(client, code_value))
    assert code is not None
    token_pair = asyncio.run(provider.exchange_authorization_code(client, code))
    access = asyncio.run(provider.verify_token(token_pair.access_token))
    assert access is not None
    assert access.resource == settings.resource_url
    assert access.subject == "owner"

    assert token_pair.refresh_token is not None
    refresh = asyncio.run(provider.load_refresh_token(client, token_pair.refresh_token))
    assert refresh is not None
    rotated = asyncio.run(
        provider.exchange_refresh_token(client, refresh, oauth_scopes())
    )
    assert rotated.refresh_token != token_pair.refresh_token
    assert (
        asyncio.run(provider.load_refresh_token(client, token_pair.refresh_token))
        is None
    )


def test_refresh_scope_cannot_exceed_original_grant(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = OwnerOAuthProvider(settings, SqliteStore(settings.database_path))
    client = _client()
    asyncio.run(provider.register_client(client))
    approval_url = asyncio.run(
        provider.authorize(client, _authorization_params(client, settings.resource_url))
    )
    transaction = parse_qs(urlsplit(approval_url).query)["transaction"][0]
    callback = provider.approve_authorization(transaction, settings.read_owner_secret())
    code_value = parse_qs(urlsplit(callback).query)["code"][0]
    code = asyncio.run(provider.load_authorization_code(client, code_value))
    assert code is not None
    token_pair = asyncio.run(provider.exchange_authorization_code(client, code))
    assert token_pair.refresh_token is not None
    refresh = asyncio.run(provider.load_refresh_token(client, token_pair.refresh_token))
    assert refresh is not None

    with pytest.raises(TokenError, match="exceeds the original grant"):
        asyncio.run(
            provider.exchange_refresh_token(
                client,
                refresh,
                [*oauth_scopes(), "admin"],
            )
        )


def test_registration_rejects_non_chatgpt_redirect(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = OwnerOAuthProvider(settings, SqliteStore(settings.database_path))
    client = _client().model_copy(
        update={"redirect_uris": [AnyUrl("https://attacker.example/callback")]}
    )

    with pytest.raises(Exception, match="redirect URIs"):
        asyncio.run(provider.register_client(client))


def test_registration_rejects_non_default_chatgpt_port(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = OwnerOAuthProvider(settings, SqliteStore(settings.database_path))
    client = _client().model_copy(
        update={
            "redirect_uris": [
                AnyUrl("https://chatgpt.com:444/connector/oauth/test-callback")
            ]
        }
    )

    with pytest.raises(Exception, match="redirect URIs"):
        asyncio.run(provider.register_client(client))
