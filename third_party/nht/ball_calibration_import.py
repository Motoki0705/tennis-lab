#!/usr/bin/env python3
"""Import real ball RGB/mask/camera records into an immutable NHT bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data_generation.blcs.calibration import (  # noqa: E402
    import_ball_calibration_capture,
    load_ball_calibration_import,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = import_ball_calibration_capture(
        args.capture_manifest,
        args.output_dir,
        bundle_id=args.bundle_id,
    )
    imported = load_ball_calibration_import(manifest_path)
    print(json.dumps(imported.manifest, sort_keys=True))


if __name__ == "__main__":
    main()
