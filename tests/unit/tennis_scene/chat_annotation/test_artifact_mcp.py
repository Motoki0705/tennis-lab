from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from src.tennis_scene.chat_annotation.artifacts import server


def test_real_http_mcp_discovery_and_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("clip__ball.json", '{"clip_id":"clip"}')
    monkeypatch.setattr(server, "download_zip", lambda *_: stream.getvalue())
    mcp = server.create_server(tmp_path, frozenset({"files.example.com"}))
    with TestClient(
        mcp.streamable_http_app(), base_url="http://127.0.0.1:8000"
    ) as client:

        def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
            response = client.post(
                "/mcp",
                headers={"Accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params,
                },
            )
            assert response.status_code == 200, response.text
            result: dict[str, Any] = response.json()["result"]
            return result

        assert "serverInfo" in call(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        tools = {tool["name"]: tool for tool in call("tools/list", {})["tools"]}
        assert set(tools) == {"save_artifact", "list_artifacts", "read_artifact"}
        descriptor = tools["save_artifact"]
        assert descriptor["_meta"]["openai/fileParams"] == ["file"]
        file_schema = descriptor["inputSchema"]["$defs"]["OpenAIFile"]
        assert set(file_schema["properties"]) == {
            "download_url",
            "file_id",
            "mime_type",
            "file_name",
        }
        assert set(file_schema["required"]) == {"download_url", "file_id"}
        result = call(
            "tools/call",
            {
                "name": "save_artifact",
                "arguments": {
                    "file": {
                        "download_url": "https://files.example.com/signed",
                        "file_id": "file_test",
                    },
                    "filename": "batch.zip",
                },
            },
        )
        assert not result.get("isError")
        receipt = result["structuredContent"]
        assert (tmp_path / receipt["artifact_id"]).read_bytes() == stream.getvalue()
        invalid = call(
            "tools/call",
            {
                "name": "save_artifact",
                "arguments": {
                    "file": {
                        "download_url": "https://files.example.com/signed",
                        "file_id": "file_test",
                    },
                    "filename": "../batch.zip",
                },
            },
        )
        assert invalid["isError"]


@pytest.mark.parametrize(
    "url",
    [
        "http://files.example.com/f",
        "https://evil.example/f",
        "https://files.example.com:444/f",
        "https://user:password@files.example.com/f",
        "sandbox:/mnt/data/a.zip",
    ],
)
def test_untrusted_download_url_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        server.download_zip(url, frozenset({"files.example.com"}))


def test_private_download_address_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="public"):
        server.download_zip(
            "https://files.example.com/f", frozenset({"files.example.com"})
        )


@pytest.mark.parametrize("mode", ["success", "redirect", "too_large", "timeout"])
def test_download_is_bounded_and_does_not_follow_redirects(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import httpx

    monkeypatch.setattr(
        server.socket,
        "getaddrinfo",
        lambda *_a, **_k: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    monkeypatch.setattr(server, "MAX_ZIP_BYTES", 4)

    class Response:
        status_code = 302 if mode == "redirect" else 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: Any) -> None:
            pass

        def iter_bytes(self, chunk_size: int) -> Any:
            assert chunk_size > 0
            if mode == "timeout":
                raise httpx.ReadTimeout("secret signed URL")
            yield b"zip"
            if mode == "too_large":
                yield b"extra"

    def stream(*args: Any, **kwargs: Any) -> Response:
        assert args[0] == "GET"
        assert kwargs["follow_redirects"] is False
        return Response()

    monkeypatch.setattr(server.httpx, "stream", stream)
    if mode == "success":
        assert (
            server.download_zip(
                "https://files.example.com/f", frozenset({"files.example.com"})
            )
            == b"zip"
        )
    else:
        with pytest.raises(ValueError) as result:
            server.download_zip(
                "https://files.example.com/f", frozenset({"files.example.com"})
            )
        assert "secret signed URL" not in str(result.value)
