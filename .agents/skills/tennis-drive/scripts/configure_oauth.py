"""Import a downloaded desktop OAuth client into a new private rclone config."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
from pathlib import Path


def prepare(client_json: Path, destination: Path, remote: str) -> dict[str, object]:
    """Create credentials without changing an existing connection or logging secrets."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", remote):
        raise ValueError("Remote name must contain letters, digits, underscore or dash")
    try:
        payload = json.loads(client_json.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Cannot read the OAuth client JSON") from error
    client = payload.get("installed") if isinstance(payload, dict) else None
    if not isinstance(client, dict):
        raise ValueError("Expected a Google OAuth Desktop app JSON ('installed')")
    client_id = client.get("client_id")
    secret = client.get("client_secret")
    if not isinstance(client_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+\.apps\.googleusercontent\.com", client_id
    ):
        raise ValueError("Invalid Google OAuth client ID")
    if not isinstance(secret, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", secret):
        raise ValueError("Invalid Google OAuth client secret")
    config = configparser.ConfigParser(interpolation=None)
    config[remote] = {
        "type": "drive",
        "client_id": client_id,
        "client_secret": secret,
        "scope": "drive",
        "use_trash": "true",
    }
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # O_EXCL also rejects a pre-existing symlink; existing connections stay intact.
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        config.write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "schema_version": 1,
        "status": "needs_browser_auth",
        "config_path": str(destination),
        "remote": remote,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-json", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    parser.add_argument("--remote", default="gdrive")
    args = parser.parse_args()
    try:
        receipt = prepare(args.client_json, args.config_output, args.remote)
    except (OSError, ValueError) as error:
        print(json.dumps({"schema_version": 1, "ok": False, "error": str(error)}))
        return 1
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
