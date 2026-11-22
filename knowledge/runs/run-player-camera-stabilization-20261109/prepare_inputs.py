"""Pin C2-derived rotations and four-second portable C3 inputs for issue 1776."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tennis_scene.visualization.player_view.blender import (
    PlayerViewRenderConfig,
    prepare_blender_shot,
)
from src.tennis_scene.visualization.player_view.lenses import EquidistantLensConfig
from src.tennis_scene.visualization.player_view.stabilization import (
    StabilizationConfig,
    export_stabilized_cameras,
)
from src.utils.checksum import dual_sha256

STRENGTHS = [0.0, 0.06, 0.08, 0.12]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.path.lexists(args.output):
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent)
    )
    try:
        lens = EquidistantLensConfig(width=640, height=360)
        packages = []
        for sigma in STRENGTHS:
            name = f"sigma_{round(sigma * 1000):03d}"
            camera = staging / "cameras" / name
            export_stabilized_cameras(
                cameras_dir=args.cameras,
                output_dir=camera,
                config=StabilizationConfig(sigma),
            )
            path = prepare_blender_shot(
                bundle_dir=args.bundle,
                cameras_dir=camera,
                output_dir=staging / "packages" / name,
                config=PlayerViewRenderConfig(),
                frame_range=(0, 240),
                lens=lens,
            )
            packages.append(
                {
                    "name": name,
                    "sigma_seconds": sigma,
                    "manifest_sha256": dual_sha256(path),
                }
            )
        manifest = {
            "schema": "player_view_stabilization_comparison_inputs_v1",
            "git_ref": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_camera_manifest_sha256": dual_sha256(
                args.cameras / "manifest.json"
            ),
            "source_scene_manifest_sha256": dual_sha256(args.bundle / "manifest.json"),
            "source_frame_interval": [0, 240],
            "stabilization_context": "full C2 sequence before shot selection",
            "preview_source_frames": [0, 26, 162],
            "preview_total_images_per_player": 12,
            "player_ids": [0, 1],
            "packages": packages,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(args.output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
