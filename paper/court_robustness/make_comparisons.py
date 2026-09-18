"""Render real CPU predictions with common geometry and honest heatmap labels."""

from __future__ import annotations

import cv2
import numpy as np
from common import ROOT, sha256, sources, write_json
from PIL import Image, ImageDraw, ImageFont, ImageOps

EDGES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)


def fit_panel(image: Image.Image, size: tuple[int, int] = (640, 390)) -> Image.Image:
    image = image.copy()
    scale = min(size[0] / image.width, size[1] / image.height)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", size, (243, 246, 248))
    panel.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return panel


def overlay(
    image: Image.Image,
    points: np.ndarray,
    color: tuple[int, int, int],
    raw: np.ndarray | None = None,
) -> Image.Image:
    a = np.asarray(image).copy()
    h, w = a.shape[:2]
    lw = max(2, round(w / 360))
    for first, second in EDGES:
        if not np.isfinite(points[[first, second]]).all():
            continue
        start, stop = points[[first, second]]
        start = tuple(np.clip(np.rint(start), -1000000, 1000000).astype(int))
        stop = tuple(np.clip(np.rint(stop), -1000000, 1000000).astype(int))
        ok, start, stop = cv2.clipLine((0, 0, w, h), start, stop)
        if ok:
            cv2.line(a, start, stop, (12, 15, 20), lw + 2, cv2.LINE_AA)
            cv2.line(a, start, stop, color, lw, cv2.LINE_AA)
    if raw is not None:
        for point in raw:
            if np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h:
                cv2.circle(
                    a,
                    tuple(np.rint(point).astype(int)),
                    lw * 2,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    a, tuple(np.rint(point).astype(int)), lw, color, -1, cv2.LINE_AA
                )
    return Image.fromarray(a)


def heatmap(probability: np.ndarray, size: tuple[int, int]) -> Image.Image:
    probability = cv2.resize(
        probability.astype(np.float32), size, interpolation=cv2.INTER_LINEAR
    )
    colored = cv2.applyColorMap(
        np.clip(probability * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    return Image.fromarray(colored[:, :, ::-1])


def external_panels(image: Image.Image, b: dict, o: dict) -> dict[str, Image.Image]:
    # Reuse saved forward outputs for an equal-cardinality sensitivity test.
    probability = b["kp_probability"]
    kp, fitted = b["argmax_kp"], b["argmax_aligned_kp"]
    panels = {
        "input": image,
        "baseline": overlay(
            image,
            b["aligned_kp"] if b["homography_found"] else np.full((14, 2), np.nan),
            (255, 98, 48),
            b["raw_kp"],
        ),
        "baseline_common": overlay(
            image,
            fitted if b["argmax_homography_found"] else np.full((14, 2), np.nan),
            (255, 98, 48),
            kp,
        ),
        "baseline_heat": heatmap(probability.max(0), image.size),
        "ours": overlay(
            image,
            o["aligned_kp"] if o["homography_found"] else np.full((14, 2), np.nan),
            (0, 225, 195),
            o["raw_kp"],
        ),
        "ours_heat": heatmap(o["line_probability"], image.size),
    }
    # This is a direct LINE output overlay, not a fitted regulation template.
    resized_line = cv2.resize(o["line_probability"], image.size)
    line_image = np.asarray(image).copy()
    active = resized_line >= 0.5
    line_image[active] = (
        line_image[active] * 0.15 + np.array([0, 240, 205]) * 0.85
    ).astype(np.uint8)
    panels["ours_line_overlay"] = Image.fromarray(line_image)
    return panels


def main() -> None:
    records = sources()
    summary = []
    overview = Image.new("RGB", (1800, len(records) * 265), "white")
    figure_manifest = {}
    for index, record in enumerate(records):
        ident = record["id"]
        image = ImageOps.exif_transpose(
            Image.open(ROOT / record["paper_path"])
        ).convert("RGB")
        b = np.load(ROOT / "evidence/predictions" / f"{ident}_baseline.npz")
        o = np.load(ROOT / "evidence/predictions" / f"{ident}_ours.npz")
        panels = external_panels(image, b, o)
        for name, panel in panels.items():
            path = ROOT / "figures" / f"{ident}_{name}.png"
            fit_panel(panel, (720, 500)).save(path)
            figure_manifest[path.name] = sha256(path)
        for j, name in enumerate(
            [
                "input",
                "baseline",
                "ours_line_overlay",
                "baseline_common",
                "ours",
                "ours_heat",
            ]
        ):
            panel = fit_panel(panels[name], (300, 230))
            overview.paste(panel, (j * 300, index * 265 + 28))
            ImageDraw.Draw(overview).text(
                (j * 300 + 6, index * 265 + 4),
                f"{ident} / {name}",
                fill="black",
                font=FONT,
            )
        summary.append(
            {
                "id": ident,
                "baseline_official_detections": int(
                    np.isfinite(b["raw_kp"]).all(1).sum()
                ),
                "baseline_official_H": bool(b["homography_found"]),
                "ours_detections": int(np.isfinite(o["raw_kp"]).all(1).sum()),
                "ours_H": bool(o["homography_found"]),
                "ours_line_max": float(o["line_probability"].max()),
                "baseline_argmax_H": bool(b["argmax_homography_found"]),
            }
        )
    overview.save(ROOT / "evidence/prediction_contact.png")
    write_json(ROOT / "evidence/detections.json", summary)
    write_json(
        ROOT / "evidence/external_figures.json",
        {
            "figures": figure_manifest,
            "line_threshold": 0.5,
            "heatmap_range": [0, 1],
            "resize": "aspect-preserving; full image, no crop",
        },
    )


if __name__ == "__main__":
    main()
