"""Save small diagnostic extremes and CPU review slideshows of all frozen views."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from analyze_probe import FROZEN, read, write
from PIL import Image, ImageDraw

from src.utils.video import VideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    args = parser.parse_args()
    analysis = read(args.analysis)
    selection = read(FROZEN / "render-selection.json")
    args.figures.mkdir(parents=True, exist_ok=True)
    cases = []
    videos = []
    for scene in ("B00", "B01", "B02", "B03"):
        canvas = Image.new("RGB", (896, 600), "black")
        draw = ImageDraw.Draw(canvas)
        for col, variant in enumerate(selection["recipes"]):
            for row, framing in enumerate(("full", "partial")):
                subset = [
                    f
                    for f in analysis["frames"]
                    if f["scene"] == scene
                    and f["variant"] == variant
                    and f["framing"] == framing
                ]
                x, y = col * 224, row * 300
                title = f"{scene} {variant}\n{framing}: "
                if not subset:
                    draw.multiline_text(
                        (x + 3, y + 4), title + "no frozen population", fill="white"
                    )
                    continue
                selected = min(
                    subset, key=lambda f: (-f["roi_alpha_lt_0_9"], f["sample_id"])
                )
                relative = f"{variant}/{scene}/{selected['sample_id']}"
                for offset, name in ((30, "rgb.jpg"), (156, "overlay.png")):
                    image = Image.open(args.review / relative / name).convert("RGB")
                    canvas.paste(
                        image.resize((224, 126), Image.Resampling.LANCZOS),
                        (x, y + offset),
                    )
                caption = (
                    title
                    + f"{'PASS' if selected['accepted'] else 'REJECT'} alpha={selected['roi_alpha_lt_0_9']:.3f}"
                )
                draw.multiline_text((x + 3, y + 3), caption, fill="white")
                draw.text(
                    (x + 3, y + 284),
                    f"hull={selected['hull_clearance_m']:.2f}m {selected['sample_id'][:10]}",
                    fill="white",
                )
                cases.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "framing": framing,
                        "criterion": "largest diagnostic ROI alpha<0.9; sample ID breaks ties",
                        "sample_id": selected["sample_id"],
                    }
                )
        canvas.quantize(colors=96).save(
            args.figures / f"{scene}-cases.png", optimize=True
        )
        video = args.review / f"{scene}-all-raw-overlay.mp4"
        count = 0
        with VideoWriter(video, fps=2, crf=18) as writer:
            for scenario in selection["scenarios"]:
                if scenario["scene"] != scene:
                    continue
                for sample in scenario["samples"]:
                    relative = f"{scenario['variant']}/{scene}/{sample['sample_id']}"
                    frame = Image.new("RGB", (1280, 400), "black")
                    frame.paste(Image.open(args.review / relative / "rgb.jpg"), (0, 40))
                    frame.paste(
                        Image.open(args.review / relative / "overlay.png"), (640, 40)
                    )
                    ImageDraw.Draw(frame).text(
                        (6, 4),
                        f"Independent frozen view: {scene} {scenario['variant']} {sample['framing']}\n{sample['sample_id']}",
                        fill="white",
                    )
                    writer.write_frame(np.asarray(frame))
                    count += 1
        assert count == 48
        videos.append(
            {
                "scene": scene,
                "file": video.name,
                "frames": count,
                "fps": 2,
                "temporal_trajectory": False,
            }
        )
    write(
        args.figures / "examples.json",
        {
            "cases": cases,
            "videos": videos,
            "png_palette_colors": 96,
            "reviewed_originals": "Native 640x360 JPEG and native overlay are in the full review; palette figures are compact diagnostic examples.",
        },
    )
    print(f"Saved {len(cases)} diagnostic cases and four 48-view CPU slideshows")


if __name__ == "__main__":
    main()
