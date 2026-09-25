"""Private MCP intake for ChatGPT file inputs over Secure MCP Tunnel."""

from __future__ import annotations

import argparse
import ipaddress
import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict

from .store import MAX_ZIP_BYTES, ArtifactStore, simple_name


class OpenAIFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    download_url: str
    file_id: str
    mime_type: str = ""
    file_name: str = ""


def download_zip(url: str, allowed_hosts: frozenset[str]) -> bytes:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in allowed_hosts
        or parts.port not in (None, 443)
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError("file download requires HTTPS on an explicitly allowed host")
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


def create_server(
    root: Path, allowed_hosts: frozenset[str], host: str = "127.0.0.1", port: int = 8000
) -> FastMCP:
    if not allowed_hosts:
        raise ValueError(
            "ARTIFACT_DOWNLOAD_HOSTS must explicitly list trusted file hosts"
        )
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
        return store.save(filename, download_zip(file.download_url, allowed_hosts))

    def list_artifacts(offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """List received ZIP receipts, with pagination."""
        return store.list(offset, limit)

    def read_artifact(artifact_id: str) -> dict[str, Any]:
        """Verify a raw ZIP receipt and return its JSON member names (not file contents)."""
        return store.read(artifact_id)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/chat_annotation/annotated/raw")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    hosts = frozenset(
        value.strip().lower()
        for value in os.environ.get("ARTIFACT_DOWNLOAD_HOSTS", "").split(",")
        if value.strip()
    )
    # httpx INFO messages include signed URLs; suppress those access logs.
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    create_server(args.root, hosts, args.host, args.port).run(
        transport="streamable-http"
    )


if __name__ == "__main__":
    main()
