"""CLI for generating synthetic tennis pose scenes (P1 minimal)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.tennis.sim.generator import GenConfig, TennisPoseSceneGenerator, write_scene_json
from src.tennis.sim.schema import validate_scene_dict


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tennis pose scenes")
    parser.add_argument("--out", required=True, help="Output directory for scenes")
    parser.add_argument("--num_scenes", type=int, default=5, help="Number of scenes")
    parser.add_argument("--num_cameras", type=int, default=4, help="Cameras per scene")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration per scene in seconds (can be fractional)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    cfg = GenConfig(fps=args.fps, duration_sec=args.duration, num_cameras=args.num_cameras)
    gen = TennisPoseSceneGenerator(cfg)
    try:
        for i in range(int(args.num_scenes)):
            scene = gen.generate_scene(scene_id=i)
            # Validate before writing
            validate_scene_dict(scene)
            path = out_dir / f"scene_{i:05d}.json"
            write_scene_json(path, scene)
        sys.stdout.write(
            f"[tennis-gen] Wrote {int(args.num_scenes)} scene(s) to {out_dir}\n"
        )
    except Exception as exc:  # pragma: no cover - surface context in CLI
        sys.stderr.write(f"[gen-error] {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

