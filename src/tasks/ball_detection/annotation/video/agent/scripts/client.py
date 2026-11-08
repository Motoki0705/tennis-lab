"""Standalone stdlib-only worker entrypoint, copied into the private attempt input.

No project import, media path resolver, arbitrary file service or output writer is
available here. The parent validates every request, including ones bypassing this
convenience client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("video_handoff", "video_rows", "video_image", "video_edit"),
    )
    parser.add_argument("--parameters", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent.parent
    fixed = json.loads((directory / "input/tool-input.json").read_text())
    request = {
        "task_request": fixed["task_request"],
        "operation": args.operation,
        "parameters": json.loads(args.parameters.read_text()),
    }
    data = canonical(request)
    if len(data) + 1 > 32768:
        raise ValueError("request exceeds 32 KiB; use fewer rows/tiles")
    digest = hashlib.sha256(data).hexdigest()
    inbox = directory / "work/service-requests"
    inbox.mkdir(exist_ok=True)
    path = inbox / (digest + ".json")
    with tempfile.NamedTemporaryFile(dir=inbox, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    reply_path = directory / "exchange" / digest / "reply.json"
    deadline = time.monotonic() + fixed["timeout_seconds"]
    while not reply_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("parent has not replied; saved edits remain durable")
        time.sleep(0.1)
    reply = json.loads(reply_path.read_text())
    if reply["request_digest"] != digest or reply["status"] != "succeeded":
        raise RuntimeError(reply.get("reason") or "service response identity differs")
    reference = reply["result"]
    result = directory / "exchange" / digest / "result.json"
    data = result.read_bytes()
    if (
        len(data) != reference["size_bytes"]
        or hashlib.sha256(data).hexdigest() != reference["digest"]
    ):
        raise ValueError("service result bytes changed")
    print(
        json.dumps(
            {"video_service": digest, "result": reference, "view": json.loads(data)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
