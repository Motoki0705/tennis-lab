"""Capture one explicit Colab/Drive CLI invocation, without retrying it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    destination = Path(sys.argv[1])
    argv = sys.argv[2:]
    if destination.exists() or not argv:
        raise ValueError("Supply a new receipt and an explicit command")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    environment = dict(os.environ)
    environment["RCLONE_CONFIG"] = str(
        Path.home() / ".config/tennis-lab/rclone-drive.conf"
    )
    result = subprocess.run(argv, env=environment, capture_output=True, text=True)
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        response = result.stdout
    receipt = {
        "schema": "court_d2_cli_receipt_v1",
        "argv": argv,
        "started_at": started_at,
        "wall_seconds": time.monotonic() - started,
        "returncode": result.returncode,
        "response": response,
        "stderr": result.stderr,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
