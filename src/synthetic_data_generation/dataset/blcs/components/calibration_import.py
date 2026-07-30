"""Import real ball RGB, mask, and camera records into a BLCS bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]

from src.synthetic_data_generation.dataset.blcs.artifacts.calibration import (  # noqa: E402
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
