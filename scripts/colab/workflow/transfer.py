"""Bound Contents API uploads and verify reassembly before publishing a file."""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from .common import WorkflowError

UPLOAD_CHUNK_BYTES = 8 * 1024**2


def upload_chunked(
    local_path: Path,
    remote_path: str,
    *,
    session: str,
    invoke: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> None:
    """Upload one immutable file in bounded requests, then atomically assemble it.

    Colab CLI base64-encodes Contents API payloads. Large source snapshots can
    cause a proxy connection reset; chunking also bounds the CLI's host memory.
    ``invoke`` must raise on transport errors and return captured stdout.
    """
    prefix = f"{remote_path}.upload-{uuid.uuid4().hex}"
    parts: list[str] = []
    digest = hashlib.sha256()
    total = math.ceil(local_path.stat().st_size / UPLOAD_CHUNK_BYTES)
    print(f"[tennis-colab] uploading {local_path.name} in {total} chunks", flush=True)
    with tempfile.TemporaryDirectory(prefix="tennis-colab-upload-") as directory:
        part_path = Path(directory) / "part"
        with local_path.open("rb") as source:
            while payload := source.read(UPLOAD_CHUNK_BYTES):
                digest.update(payload)
                part_path.write_bytes(payload)
                remote_part = f"{prefix}.part-{len(parts):06d}"
                invoke(["upload", "-s", session, str(part_path), remote_part])
                parts.append(remote_part)
                print(f"[tennis-colab] uploaded chunk {len(parts)}/{total}", flush=True)
        checksum = digest.hexdigest()
        marker = f"TENNIS_COLAB_UPLOAD_VERIFIED:{checksum}"
        script = Path(directory) / "assemble.py"
        script.write_text(
            "from pathlib import Path\n"
            "import hashlib, shutil\n"
            f"parts = [Path(p) for p in {parts!r}]\n"
            f"staging = Path({(prefix + '.assembling')!r})\n"
            "try:\n"
            "    with staging.open('xb') as output:\n"
            "        for part in parts:\n"
            "            with part.open('rb') as source:\n"
            "                shutil.copyfileobj(source, output, length=1024**2)\n"
            "    with staging.open('rb') as source:\n"
            "        digest = hashlib.file_digest(source, 'sha256').hexdigest()\n"
            f"    if digest != {checksum!r}:\n"
            "        raise ValueError('Chunked upload SHA-256 mismatch')\n"
            f"    staging.replace(Path({remote_path!r}))\n"
            "    for part in parts:\n"
            "        part.unlink()\n"
            f"    print({marker!r}, flush=True)\n"
            "finally:\n"
            "    staging.unlink(missing_ok=True)\n",
            encoding="utf-8",
        )
        result = invoke(
            ["exec", "-s", session, "-f", str(script), "--timeout", "300"]
        )
        # CLI 0.6.0 can return exit code 0 after a Python exception in the VM.
        if marker not in result.stdout.splitlines():
            raise WorkflowError(
                f"Chunked upload did not verify {remote_path}; the previous target was not accepted"
            )
    print(f"[tennis-colab] upload verified: {local_path.name} sha256={checksum}", flush=True)
