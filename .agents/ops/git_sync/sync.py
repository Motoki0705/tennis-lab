"""File entry for systemd; all update code is imported before main can change."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.checkout_lock import run_cli  # noqa: E402

if __name__ == "__main__":
    run_cli("git_sync.cli", release_before_main=True)
