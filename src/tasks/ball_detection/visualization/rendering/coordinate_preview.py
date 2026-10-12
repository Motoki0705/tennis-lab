from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from torch import Tensor


def save_query_preview(path: Path, rgb: Tensor, prediction: Tensor, batch: dict[str, Any],
                       occluded: Tensor | None = None) -> None:
    pictures = []
    images = rgb[0].detach().cpu()
    for t in range(32):
        frame = Image.fromarray(images[t].permute(1, 2, 0).numpy()).resize((640, 360))
        canvas = Image.new("RGB", (640, 384), "white")
        canvas.paste(frame, (0, 24))
        draw = ImageDraw.Draw(canvas)
        for point, valid, color in ((batch["uv"][0, t], bool(batch["position_valid"][0, t]), "lime"),
                                    (prediction[0, t], True, "orange")):
            if valid:
                x, y = float(point[0]) * 639, 24 + float(point[1]) * 359
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        hidden = bool(occluded[0, t]) if occluded is not None else False
        draw.text((4, 4), f"frame {int(batch['frame_indices'][0,t])} | GT green / prediction orange | artificial occlusion={hidden}", fill="black")
        pictures.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=100, loop=0)
