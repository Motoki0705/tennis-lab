"""File entry for systemd; all update code is imported before main can change."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from git_sync.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
