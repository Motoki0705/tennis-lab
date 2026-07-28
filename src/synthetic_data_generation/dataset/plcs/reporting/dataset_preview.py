"""Create an RGB and exact-instance-mask preview for a PLCS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLORS = np.asarray(
    [
        [255, 48, 48],
        [32, 128, 255],
        [255, 196, 32],
    ],
    dtype=np.float32,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=288)
    args = parser.parse_args()
    root = args.render_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite output: {output}")
    manifest = json.loads((root / "manifest.json").read_text())
    panels = []
    for frame in manifest["frames"]:
        rgb = np.asarray(Image.open(root / frame["rgb"]["relative_path"])).copy()
        masks = np.load(
            root / frame["instance_mask"]["relative_path"],
            allow_pickle=False,
        )
        labels = json.loads((root / frame["labels"]["relative_path"]).read_text())
        overlay = rgb.astype(np.float32)
        for index in range(masks.shape[-1]):
            mask = masks[..., index]
            overlay[mask] = 0.35 * overlay[mask] + 0.65 * COLORS[index]
        panel = Image.fromarray(overlay.round().astype(np.uint8))
        height = round(panel.height * args.panel_width / panel.width)
        panel = panel.resize((args.panel_width, height))
        canvas = Image.new("RGB", (args.panel_width, height + 42), "white")
        canvas.paste(panel, (0, 42))
        text = ", ".join(
            f"id={item['instance_id']} {item['pose_id']} "
            f"px={item['exact_visible_pixel_count']}"
            for item in labels["instances"]
        )
        ImageDraw.Draw(canvas).multiline_text(
            (5, 4),
            f"frame {frame['frame_index']}\n{text}",
            fill="black",
            spacing=2,
        )
        panels.append(canvas)
    result = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        result.paste(panel, (x, 0))
        x += panel.width
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(output)


if __name__ == "__main__":
    main()
