"""Install byte-verified SMPL auxiliary files from a user-acquired GVHMR checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES, install_smpl_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gvhmr-root", type=Path, help="Official GVHMR checkout directory")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpt"))
    parser.add_argument("--list", action="store_true", help="List sources, destinations and SHA-256")
    args = parser.parse_args()
    if args.list:
        for source in SMPL_ASSET_SOURCES.values():
            print(f"{source.sha256}  {args.checkpoint_root / source.checkpoint_path}")
            print(f"  {source.source_url}")
        return
    if args.gvhmr_root is None:
        parser.error("--gvhmr-root is required unless --list is specified")
    for path in install_smpl_assets(gvhmr_root=args.gvhmr_root, checkpoint_root=args.checkpoint_root):
        print(path)


if __name__ == "__main__":
    main()
