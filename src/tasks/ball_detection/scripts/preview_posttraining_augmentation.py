"""CPU annotation preview of RGB augmentation; no model predictions are fabricated."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.tasks.ball_detection.data.coordinate_dataset import (
    CoordinateWindowDataset,
    collate_coordinate_windows,
)
from src.tasks.ball_detection.training.posttraining.augmentation import (
    AugmentationConfig,
    VideoAugmenter,
)
from src.tasks.ball_detection.training.posttraining.paths import resolver
from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathRole,
)

PATH_BOUNDARY = NonHydraPathBoundary(name="ball_detection.preview_posttraining_augmentation", fields=(
    BoundaryPathField("manifest", PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("augmentation_config", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    BoundaryPathField("output", PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY),
))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--augmentation-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--targeted-demo", action="store_true", help="Force a ball-near mask for annotation review only")
    args = p.parse_args()
    names = ("manifest", "augmentation_config", "output")
    paths = PATH_BOUNDARY.validate({k: getattr(args, k) for k in names}, resolver=resolver(args.augmentation_config.parent,
        (args.manifest,), (args.output,), args.output.parent))
    for key in names:
        setattr(args, key, paths.declared(key).path)
    torch.set_num_threads(2)
    data = CoordinateWindowDataset(args.manifest, split="train", requires_pose=False, jpeg_decoder="opencv")
    batch = collate_coordinate_windows([data[args.window_index]])
    config = AugmentationConfig.load(args.augmentation_config)
    if args.targeted_demo:
        config = replace(config, targeted_probability=1.)
    rgb, transformed, audit = VideoAugmenter(config, args.seed)(batch["rgb"], batch, epoch=0, profile="combined")
    frames = []
    for t in range(32):
        canvas = Image.new("RGB", (960, 294), "white")
        for panel, (images, labels) in enumerate(((batch["rgb"], batch), (rgb, transformed))):
            image = Image.fromarray(images[0, t].permute(1, 2, 0).numpy()).resize((480, 270))
            if bool(labels["position_valid"][0, t]):
                x, y = labels["uv"][0, t].tolist()
                ImageDraw.Draw(image).ellipse((x * 479 - 4, y * 269 - 4, x * 479 + 4, y * 269 + 4), outline="lime", width=2)
            canvas.paste(image, (panel * 480, 24))
        hidden = bool(audit["artificially_occluded"][0, t])
        ImageDraw.Draw(canvas).text((5, 5), f"Original (left) / augmented (right) | green = GT | synthetic occlusion = {hidden}", fill="black")
        frames.append(canvas)
    args.output.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output / "augmentation.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    covered = audit["artificially_occluded"][0].nonzero().flatten().tolist()
    frame_index = covered[0] if covered else 16
    frames[frame_index].save(args.output / "augmentation.png")
    (args.output / "trace.json").write_text(json.dumps(dict(clip_id=batch["clip_id"][0], decoder="opencv CPU preview only",
        seed=args.seed, preview_override_targeted=args.targeted_demo, augmentation=asdict(config),
        original_uv=batch["uv"].tolist(), transformed_uv=transformed["uv"].tolist(),
        **{key: value.tolist() for key, value in audit.items()}), indent=2))


if __name__ == "__main__":
    main()
