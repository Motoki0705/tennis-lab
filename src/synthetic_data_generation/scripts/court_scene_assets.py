"""Package or verify immutable Court scene inputs without starting a GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.synthetic_data_generation.dataset.court.assets import (
    pack_scene_assets,
    verify_scene_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--source-root", type=Path, required=True)
    pack.add_argument("--destination", type=Path, required=True)
    pack.add_argument("--scene-ids", nargs="+", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        )
        if status.strip():
            raise RuntimeError("Commit the worktree before pinning an asset producer version.")

        def revision(ref: str) -> str:
            return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()

        result = pack_scene_assets(
            args.source_root,
            args.destination,
            scene_ids=args.scene_ids,
            repository_commit=revision("HEAD"),
            nht_commit=revision("HEAD:third_party/nht"),
        )
    else:
        result = verify_scene_assets(args.root, manifest_sha256=args.manifest_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
