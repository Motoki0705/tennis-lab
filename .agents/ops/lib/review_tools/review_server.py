"""Lock the source checkout before loading the CLI implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.checkout_lock import run_cli  # noqa: E402

if __name__ == "__main__":
    run_cli(
        "lib.review_tools.review_server_impl",
    )
