"""Render CPU RGB/pseudo-teacher contact sheets with explicit DATA/OUTPUT roots."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tennis_scene.dataset_pipeline.review import render_review
from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="DATA-relative dataset fragment")
    parser.add_argument("--video", required=True)
    parser.add_argument("--clip", required=True, help="Full canonical clip ID, e.g. video_000/clip_000")
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--output", required=True, help="OUTPUT-relative tennis_scene/visualize/experiment/run-id")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    resolver = PathResolver(RuntimePathRoots(
        project_root=root, data_root=args.data_root.resolve(), output_root=args.output_root.resolve(),
        checkpoint_root=(root / "ckpt").resolve(), artifact_root=args.output_root.resolve(),
        cache_root=(root / ".cache").resolve(), external_asset_root=(root / "third_party").resolve(),
    ))
    fragment = Path(args.output)
    if len(fragment.parts) != 4 or fragment.parts[:2] != ("tennis_scene", "visualize"):
        parser.error("--output must be tennis_scene/visualize/<experiment>/<run-id>")
    dataset = SLCSDataIndex.load(resolver.resolve(PathRole.DATA, args.dataset))
    matches = [record for record in dataset.clips if record.video_id == args.video and record.clip_id == args.clip]
    if len(matches) != 1:
        parser.error("--video/--clip must select exactly one dataset manifest record")
    image, sidecar = render_review(dataset.clip_dir(matches[0]), resolver.resolve(PathRole.OUTPUT, args.output), frames=args.frames, cameras=args.cameras)
    print(image)
    print(sidecar)


if __name__ == "__main__":
    main()
