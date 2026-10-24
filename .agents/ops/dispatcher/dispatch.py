"""File entry for systemd; module callers use python -m dispatcher."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dispatcher.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
