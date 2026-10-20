"""Download or verify the published court and ball inference checkpoints."""

import argparse
from pathlib import Path

from src.utils.model_weights import ensure_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Absolute checkpoint root (normally $PWD/ckpt)",
    )
    parser.add_argument("--model", choices=("all", "court", "ball"), default="all")
    parser.add_argument("--revision", default="main", help="Hub branch, tag or commit")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files without accessing the network",
    )
    args = parser.parse_args()
    for path in ensure_weights(
        args.checkpoint_root,
        model=args.model,
        revision=args.revision,
        verify_only=args.verify_only,
    ):
        print(f"Verified {path}")
