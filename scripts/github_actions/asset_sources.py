#!/usr/bin/env python3
"""Render runner asset paths from the single storage-tier configuration."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

OPS = Path(__file__).resolve().parents[2] / ".agents/ops"
sys.path.insert(0, str(OPS))


def asset_sources(config: Any) -> dict[str, str]:
    from storage_tier.filesystem import real_directory

    values = {"MIRROR_MOUNT": str(config.mount)}
    for name in ("data", "ckpt"):
        path = config.mirror / name
        real_directory(path, config.mount)
        values[f"SOURCE_{name.upper()}"] = str(path)

    return values


def main() -> None:
    from storage_tier.engine import mounted
    from storage_tier.layout import Config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = Config.load(args.repo, args.config)
    mounted(config)
    for key, value in asset_sources(config).items():
        print(f"{key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
