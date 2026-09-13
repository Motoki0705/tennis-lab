"""Launch the NHT public batch-render acceptance probe in its isolated runtime."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nht-python", type=Path, required=True)
    parser.add_argument("--nht-repo", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run(
        [
            str(args.nht_python),
            str(args.nht_repo / "scripts/benchmark_render_batches.py"),
            "--scenes",
            str(args.scenes),
            "--output",
            str(args.output.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
