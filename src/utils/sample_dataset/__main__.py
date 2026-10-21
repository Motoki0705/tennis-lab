"""Download or verify the published three-camera sample and its manifest/license."""

import argparse
from pathlib import Path

from src.utils.sample_dataset import ensure_sample_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Absolute data root (normally $PWD/data); installs below samples/",
    )
    parser.add_argument("--revision", default="main", help="Hub branch, tag or commit")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files without accessing the network",
    )
    args = parser.parse_args()
    for path in ensure_sample_dataset(
        args.data_root, revision=args.revision, verify_only=args.verify_only
    ):
        print(f"Verified {path}")
