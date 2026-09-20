"""Render real CPU predictions with common geometry and honest heatmap labels."""

from __future__ import annotations

import cv2
import numpy as np
from common import ROOT, sha256, sources, write_json
from homography_evidence import EDGES, read_results
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
    inliers: np.ndarray | None = None,
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
        for index, point in enumerate(raw):
            if np.isfinite(point).all() and 0 <= point[0] < w and 0 <= point[1] < h:
                if inliers is not None and not inliers[index]:
                    cv2.drawMarker(
                        a,
                        tuple(np.rint(point).astype(int)),
                        (180, 180, 180),
                        cv2.MARKER_TILTED_CROSS,
                        lw * 5,
                        lw,
                        cv2.LINE_AA,
                    )
                    continue
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


def external_panels(
    image: Image.Image, b: dict, o: dict, homography: dict
) -> dict[str, Image.Image]:
    # Reuse saved forward outputs for an equal-cardinality sensitivity test.
    probability = b["kp_probability"]
    kp, fitted = b["argmax_kp"], b["argmax_aligned_kp"]
    panels = {
        "input": image,
        "baseline": overlay(
            image,
            b["aligned_kp"] if b["homography_found"] else np.full((14, 2), np.nan),
            (255, 98, 48),
            b["refined_kp"],
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
            np.asarray(homography["aligned_kp"])
            if homography["status"] == "ok"
            else np.full((14, 2), np.nan),
            (0, 225, 195),
            o["raw_kp"],
            np.asarray(homography["inliers"]),
        ),
        "ours_heat": heatmap(o["line_probability"], image.size),
    }
    for stage in ("kp_only", "line_selection"):
        value = homography["stages"].get(stage)
        panels[f"ours_{stage}"] = overlay(
            image,
            np.asarray(value["aligned_kp"])
            if value is not None
            else np.full((14, 2), np.nan),
            (0, 225, 195),
            o["raw_kp"],
            np.asarray(value["selected"])
            if value is not None
            else np.zeros(14, dtype=bool),
        )
    # This is a direct LINE output overlay, not a fitted regulation template.
    resized_line = cv2.resize(o["line_probability"], image.size)
    line_image = np.asarray(image).copy()
    active = resized_line >= 0.5
    line_image[active] = (
        line_image[active] * 0.15 + np.array([0, 240, 205]) * 0.85
    ).astype(np.uint8)
    panels["ours_line_overlay"] = Image.fromarray(line_image)
    panels["ours_line_mask"] = Image.fromarray(
        np.repeat((active.astype(np.uint8) * 255)[..., None], 3, axis=2)
    )
    return panels


def main() -> None:
    records = sources()
    summary = []
    overview = Image.new("RGB", (1800, len(records) * 265), "white")
    figure_manifest = {}
    homographies = read_results()
    for index, record in enumerate(records):
        ident = record["id"]
        image = ImageOps.exif_transpose(
            Image.open(ROOT / record["paper_path"])
        ).convert("RGB")
        b = np.load(ROOT / "evidence/predictions" / f"{ident}_baseline.npz")
        o = np.load(ROOT / "evidence/predictions" / f"{ident}_ours.npz")
        homography = homographies["images"][ident]
        panels = external_panels(image, b, o, homography)
        for name, panel in panels.items():
            path = ROOT / "figures" / f"{ident}_{name}.png"
            fit_panel(panel, (720, 500)).save(path)
            figure_manifest[path.name] = sha256(path)
        for j, name in enumerate(
            [
                "input",
                "baseline",
                "ours",
                "ours_line_mask",
                "ours_line_overlay",
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
                "ours_H": homography["status"] == "ok",
                "ours_inliers": homography["inlier_count"],
                "ours_inlier_rms_px": homography["inlier_rms_px"],
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
            "homography_evidence_sha256": sha256(
                ROOT / "evidence/homography/results.json"
            ),
        },
    )


if __name__ == "__main__":
    main()
