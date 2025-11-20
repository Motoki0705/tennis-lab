"""CLI tool to render a synthetic tennis pose scene JSON to a video file."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from src.tennis.sim.schema import validate_scene_dict
from src.visualize.tennis_multi_cam_3d_pose import render_video


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tennis pose scene to video")
    parser.add_argument("--scene", required=True, help="Path to scene_XXXXX.json")
    parser.add_argument("--out", required=True, help="Output video path (e.g. .mp4)")
    parser.add_argument(
        "--camera_index", type=int, default=0, help="Camera index to render"
    )
    parser.add_argument(
        "--fps", type=int, default=0, help="FPS override (0 = use scene fps)"
    )
    parser.add_argument(
        "--width", type=int, default=0, help="Output width (0 = camera size)"
    )
    parser.add_argument(
        "--height", type=int, default=0, help="Output height (0 = camera size)"
    )
    return parser.parse_args(argv)


def _load_scene(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw_scene = json.load(f)
    if not isinstance(raw_scene, Mapping):
        msg = f"Scene JSON must be a mapping: {path}"
        raise ValueError(msg)
    scene = cast(Mapping[str, Any], raw_scene)
    validate_scene_dict(scene)
    return scene


def main(argv: Sequence[str] | None = None) -> int:
    """Render the provided scene JSON to a video file specified via CLI arguments."""
    args = _parse_args(argv)
    scene_path = Path(args.scene)
    if not scene_path.exists():
        sys.stderr.write(f"[render-error] scene not found: {scene_path}\n")
        return 2
    try:
        scene = _load_scene(scene_path)
        width = args.width or None
        height = args.height or None
        fps = args.fps or None
        out_path = str(Path(args.out))
        render_video(
            scene,
            out_path,
            camera_index=args.camera_index,
            width=width,
            height=height,
            fps=fps,
        )
        sys.stdout.write(f"[tennis-render] wrote video to {out_path}\n")
    except Exception as exc:  # pragma: no cover - surfacing errors with context
        sys.stderr.write(f"[render-error] {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
