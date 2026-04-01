#!/usr/bin/env python3
"""Download the official DINO 5-scale Swin-L checkpoint from Google Drive."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import gdown


FILE_ID = "14h4UCi-HsDL01ZQRbpV47dzMST_py_vM"
DEFAULT_OUTPUT = Path("/workspace/checkpoints/DINO/checkpoint0027_5scale_swin.pth")
EXPECTED_SHA256: str | None = "17ddce1592816a0c63a2edc94d4a0877ffeb086f397a6657e151c703a4c850b5"


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the downloaded checkpoint.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download if the output file already exists.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA256 verification after download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and args.output.exists():
        print(f"Skipping download because file already exists: {args.output}")
    else:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        result = gdown.download(url=url, output=str(args.output), quiet=False, fuzzy=True)
        if result is None:
            raise RuntimeError("gdown failed to download the checkpoint")

    sha256 = compute_sha256(args.output)
    print(f"sha256: {sha256}")

    if args.no_verify or EXPECTED_SHA256 is None:
        if EXPECTED_SHA256 is None:
            print("Skipped fixed SHA256 verification because EXPECTED_SHA256 is unset")
        else:
            print("Skipped SHA256 verification")
        return

    if sha256 != EXPECTED_SHA256:
        raise ValueError(
            "Downloaded file checksum does not match the expected checkpoint: "
            f"{sha256} != {EXPECTED_SHA256}"
        )
    print("SHA256 verification passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
