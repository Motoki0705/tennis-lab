"""Build the fixed Meiji multiview-v3 release on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tennis_scene.ball_events.meiji_benchmark import build_meiji_benchmark
from src.utils.paths import PROJECT_ROOT


def main() -> None:
    meiji = PROJECT_ROOT / "data/tennis_multiview/processed/meiji_4cam"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=meiji / "benchmarks/multiview-v2")
    parser.add_argument("--annotations", type=Path, default=meiji / "dataset/videos/video_000/clip_000-068_ball_events")
    parser.add_argument("--output", type=Path, default=meiji / "benchmarks/multiview-v3")
    args = parser.parse_args()
    existed = args.output.exists()
    manifest = build_meiji_benchmark(args.base, args.annotations, args.output)
    print(json.dumps({
        "status": "verified" if existed else "created",
        "output": str(args.output.resolve()),
        "provenance": manifest["provenance"],
        "verification": manifest["verification"],
        "ball_events": manifest["ball_events"],
        "splits": {split: {key: value for key, value in manifest["splits"][split].items() if key != "clip_ids"}
                   for split in ("train", "val", "test")},
    }, indent=2))


if __name__ == "__main__":
    main()
