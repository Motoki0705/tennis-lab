"""Private MCP intake for ChatGPT file inputs over Secure MCP Tunnel."""

from __future__ import annotations

import ipaddress
import logging
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from .store import MAX_ZIP_BYTES, ArtifactStore, simple_name

logger = logging.getLogger(__name__)


class OpenAIFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    download_url: str
    file_id: str
    mime_type: str = ""
    file_name: str = ""


def download_zip(url: str) -> bytes:
    scheme: str | None = None
    hostname: str | None = None
    try:
        parts = urlsplit(url)
        scheme, hostname = parts.scheme, parts.hostname
        port = parts.port
    except ValueError:
        # urllib's exceptions may echo invalid ports or authority values.
        logger.warning(
            "file download URL scheme=%r hostname=%r port=%r",
            scheme,
            hostname,
            "invalid",
        )
        raise ValueError("invalid file download URL") from None
    # Never log the URL, path, query, fragment, userinfo, file_id or headers.
    # repr escapes control characters so each diagnostic stays on one log line.
    logger.info(
        "file download URL scheme=%r hostname=%r port=%r", scheme, hostname, port
    )
    rejected = [
        condition
        for condition, failed in (
            ("scheme", scheme != "https"),
            ("hostname", not hostname),
            ("port", port not in (None, 443)),
            ("userinfo", parts.username is not None or parts.password is not None),
            ("fragment", bool(parts.fragment)),
        )
        if failed
    ]
    if rejected:
        raise ValueError(
            "file download requires a valid HTTPS URL; "
            f"rejected={','.join(rejected)}; "
            f"scheme={scheme!r} hostname={hostname!r} port={port!r}"
        )
    addresses = socket.getaddrinfo(parts.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(item[4][0]).is_global for item in addresses
    ):
        raise ValueError("file download host must resolve to public addresses")
    try:
        with httpx.stream("GET", url, follow_redirects=False, timeout=30) as response:
            if response.status_code != 200:
                raise ValueError(
                    f"file download failed (HTTP {response.status_code}); refresh the file input"
                )
            data = bytearray()
            for block in response.iter_bytes(chunk_size=64 * 1024):
                data.extend(block)
                if len(data) > MAX_ZIP_BYTES:
                    raise ValueError("download exceeds 16 MiB")
    except httpx.HTTPError:
        # Signed download URLs must not appear in tool errors or application logs.
        raise ValueError(
            "file download failed; refresh the file input and retry"
        ) from None
    return bytes(data)


def create_server(root: Path, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    store = ArtifactStore(root)
    server = FastMCP(
        "artifact-mcp-server",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )

    def save_artifact(file: OpenAIFile, filename: str) -> dict[str, Any]:
        """Submit a ZIP containing only annotation JSON files to raw storage.

        Supply the actual ChatGPT file, not a sandbox path or an invented URL.
        filename is a plain ZIP name. Returns a durable artifact_id and SHA-256.
        Identical ZIP bytes are idempotent. Saving does not mark a clip done.
        """
        simple_name(filename, ".zip")
        result: dict[str, Any] = store.save(filename, download_zip(file.download_url))
        return result

    def list_artifacts(offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """List received ZIP receipts, with pagination."""
        result: dict[str, Any] = store.list(offset, limit)
        return result

    def read_artifact(artifact_id: str) -> dict[str, Any]:
        """Verify a raw ZIP receipt and return its JSON member names (not file contents)."""
        result: dict[str, Any] = store.read(artifact_id)
        return result

    server.add_tool(
        save_artifact,
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
        meta={"openai/fileParams": ["file"]},
    )
    server.add_tool(
        list_artifacts,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    server.add_tool(
        read_artifact,
        annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    )
    return server
