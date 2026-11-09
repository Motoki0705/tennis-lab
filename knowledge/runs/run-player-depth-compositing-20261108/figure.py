"""Render a small color table from independent hand-calculated layer cases."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("layers.json").read_text())["cases"]
    image = Image.new("RGB", (1000, 330), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (12, 8),
        "Known layers: opaque green foreground, z=4. Numeric labels: linear RGB; swatches: sRGB display.",
        fill="black",
    )
    columns = [
        ("composite_rgb", "Required prefix result"),
        ("ed_compare_rgb", "ED compare (rejected)"),
        ("always_foreground_rgb", "Always foreground (rejected)"),
    ]
    for index, (_, label) in enumerate(columns):
        draw.text((330 + index * 220, 34), label, fill="black")
    for row, name in enumerate(
        (
            "opaque_before",
            "opaque_behind",
            "net_before_opaque_behind",
            "net_behind",
            "same_ed_a",
            "same_ed_b",
        )
    ):
        y = 60 + row * 44
        case = cases[name]
        draw.text((12, y + 4), name, fill="black")
        draw.text(
            (12, y + 20),
            f"background alpha={case['background_alpha']:g}, ED={case['expected_z']:g}",
            fill="black",
        )
        for index, (key, _) in enumerate(columns):
            color = tuple(
                round(
                    255
                    * (12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055)
                )
                for c in case[key]
            )
            x = 330 + index * 220
            draw.rectangle((x, y, x + 80, y + 32), fill=color, outline="black")
            draw.text((x + 88, y + 10), str(case[key]), fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
