"""File entrypoint for systemd; module callers use python -m daily_ci."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from daily_ci.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
