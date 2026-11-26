"""Create four-view publication panels from the reviewed frames and DINO rows."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from audit import read_json
from PIL import Image

# Main-court crops in original 1920x1080 pixels, in camera order.
COURT_CROPS = (
    (24, 310, 1560, 850),
    (85, 324, 1860, 892),
    (0, 310, 1560, 887),
    (419, 169, 1920, 740),
)
COLORS = {
    "raw": (205, 205, 205),
    "track": (246, 157, 23),
    "selected": (70, 225, 50),
    "focus": (17, 138, 255),
}
PANEL_WIDTH = 664
PANEL_HEIGHT = 250
TITLE_HEIGHT = 28


@dataclass(frozen=True)
class Figure:
    frame: int
    detail: tuple[str, tuple[int, ...]] | None = None


FIGURES = {
    "clip_002": Figure(1246),
    "clip_004": Figure(282, ("cam3", (275, 282, 290))),
    "clip_012": Figure(811),
    "clip_014": Figure(222, ("cam2", (209, 222, 236))),
    "clip_018": Figure(179),
    "clip_020": Figure(131),
    "clip_023": Figure(1046, ("cam0", (1046, 1058, 1071))),
    "clip_028": Figure(641, ("cam3", (634, 641, 648))),
    "clip_036": Figure(283, ("cam2", (241, 262, 283))),
    "clip_038": Figure(190, ("cam2", (182, 186, 190))),
    "clip_039": Figure(433),
    "clip_049": Figure(147),
    "clip_058": Figure(383),
    "clip_060": Figure(239),
}


def highlighted(
    frame: int, row: dict[str, Any], focus: list[dict[str, Any]], marked: set[int]
) -> bool:
    return row["row"] in marked or any(
        item["start"] <= frame < item["end"] and row[item["kind"]] == item["identity"]
        for item in focus
    )


def draw_labels(
    image: np.ndarray, labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]]
) -> None:
    height, width = image.shape[:2]
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.43
    occupied: list[tuple[int, int, int, int]] = []
    for label, (x, y), color in reversed(labels):
        (label_width, label_height), _ = cv2.getTextSize(label, font, scale, 1)
        x = max(1, min(x, width - label_width - 3))
        for offset in (-2, -17, 13, -32, 28):
            base = y + offset
            bounds = (x - 1, base - label_height - 2, x + label_width + 1, base + 3)
            a, b, c, d = bounds
            if (
                b < 0
                or d >= height
                or any(
                    a < right and c > left and b < bottom and d > top
                    for left, top, right, bottom in occupied
                )
            ):
                continue
            occupied.append(bounds)
            cv2.rectangle(image, (a, b), (c, d), (13, 13, 13), cv2.FILLED)
            cv2.putText(image, label, (x, base), font, scale, color, 1, cv2.LINE_AA)
            break


def section(
    clip: Path,
    frame: int,
    camera: str,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    rows: list[dict[str, Any]],
    focus: list[dict[str, Any]],
    marked: set[int],
) -> np.ndarray:
    path = clip / "native" / f"f{frame:04d}_{camera}.jpg"
    source = cv2.imread(str(path))
    if source is None or source.shape != (1080, 1920, 3):
        raise ValueError(f"Expected the native review still at {path}")
    x0, y0, x1, y1 = crop
    width, height = size
    scale = min(width / (x1 - x0), (height - TITLE_HEIGHT) / (y1 - y0))
    actual = (round((x1 - x0) * scale), round((y1 - y0) * scale))
    panel = np.full((height, width, 3), 19, np.uint8)
    image = panel[TITLE_HEIGHT : TITLE_HEIGHT + actual[1], : actual[0]]
    image[:] = cv2.resize(
        source[y0:y1, x0:x1],
        actual,
        interpolation=cv2.INTER_AREA if scale <= 1 else cv2.INTER_CUBIC,
    )
    label = f"{clip.name}  {camera}  f{frame}"
    cv2.putText(
        panel,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    def layer(row: dict[str, Any]) -> tuple[bool, bool, bool]:
        return (
            highlighted(frame, row, focus, marked),
            row["selected"],
            row["track"] is not None,
        )

    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for row in sorted(rows, key=layer):
        a, b, c, d = row["box"]
        if a >= x1 or c <= x0 or b >= y1 or d <= y0:
            continue
        priority, selected, tracked = layer(row)
        color = COLORS[
            "focus"
            if priority
            else "selected"
            if selected
            else "track"
            if tracked
            else "raw"
        ]
        x, y, u, v = (
            round((a - x0) * scale),
            round((b - y0) * scale),
            round((c - x0) * scale),
            round((d - y0) * scale),
        )
        cv2.rectangle(image, (x, y), (u, v), color, 2 if priority else 1, cv2.LINE_AA)
        if priority or selected:
            label = f"t{row['track']}" if tracked else f"r{row['row']}"
            if row["group"] is not None:
                label += f" g{row['group']}"
            elif selected:
                label += " S"
            labels.append((label, (x, y), color))
    draw_labels(image, labels)
    return panel


def render(
    clip: Path, spec: Figure, events: list[dict[str, Any]], focus: list[dict[str, Any]]
) -> np.ndarray:
    observations = read_json(clip / "observations.json")
    marked: dict[str, set[int]] = {
        camera: set() for camera in observations[str(spec.frame)]
    }
    relevant = {
        "off_target_box",
        "merged_box",
        "split_box",
        "detection_gap",
        "selection_gap",
    }
    for event in events:
        if event["kind"] in relevant:
            marked[event["camera"]].update(event.get("rows", ()))

    def panel(
        frame: int, camera: str, crop: tuple[int, int, int, int], size: tuple[int, int]
    ) -> np.ndarray:
        return section(
            clip,
            frame,
            camera,
            crop,
            size,
            observations[str(frame)][camera],
            [item for item in focus if item["camera"] == camera],
            marked[camera],
        )

    views = [
        panel(spec.frame, f"cam{i}", crop, (PANEL_WIDTH, PANEL_HEIGHT))
        for i, crop in enumerate(COURT_CROPS)
    ]
    strips = [np.hstack(views[:2]), np.hstack(views[2:])]
    if spec.detail:
        camera, frames = spec.detail
        local_focus = [item for item in focus if item["camera"] == camera]
        boxes = np.asarray(
            [
                row["box"]
                for frame in frames
                for row in observations[str(frame)][camera]
                if highlighted(frame, row, local_focus, marked[camera])
            ]
        )
        a, b = np.floor(boxes[:, :2].min(axis=0)).astype(int)
        c, d = np.ceil(boxes[:, 2:].max(axis=0)).astype(int)
        widths = (442, 442, 444)
        top, bottom = max(0, b - 32), min(1080, d + 25)
        width = max(
            c - a + 100, round((bottom - top) * widths[0] / (196 - TITLE_HEIGHT))
        )
        left = max(0, min((a + c - width) // 2, 1920 - width))
        crop = left, top, min(1920, left + width), bottom
        strips.append(
            np.hstack(
                [
                    panel(frame, camera, crop, (width, 196))
                    for frame, width in zip(frames, widths, strict=True)
                ]
            )
        )
    legend = np.full((29, 2 * PANEL_WIDTH, 3), 19, np.uint8)
    specs = (
        ("raw", "DINO row"),
        ("track", "Tracked / excluded"),
        ("selected", "Selected / g = group"),
        ("focus", "Decision or reviewed row"),
    )
    for i, (color, label) in enumerate(specs):
        start = 14 + i * 255
        cv2.rectangle(legend, (start, 8), (start + 16, 21), COLORS[color], 2)
        cv2.putText(
            legend,
            label,
            (start + 25, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (241, 241, 241),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([*strips, legend])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extract", type=Path)
    parser.add_argument("output", type=Path, help="New figure directory")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    index = {
        row["clip"]: row for row in read_json(args.extract / "index.json")["clips"]
    }
    events = read_json(Path(__file__).with_name("events.json"))
    if set(index) != set(FIGURES):
        raise ValueError("Expected extracts for all 14 #2363 stops")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "clip",
                "four_view_frame",
                "detail_camera",
                "detail_frames",
                "image",
                "sha256",
            )
        )
        for name, spec in FIGURES.items():
            image = render(
                args.extract / name,
                spec,
                [event for event in events if event["clip"] == name],
                index[name]["focus"],
            )
            path = args.output / f"{name}-f{spec.frame:04d}.pdf"
            rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rgb.save(
                path,
                resolution=144,
                quality=79,
                subsampling=0,
                optimize=True,
                title=f"{name} source frame {spec.frame}",
                creationDate=None,
                modDate=None,
            )
            camera, frames = spec.detail or ("", ())
            writer.writerow(
                (
                    name,
                    spec.frame,
                    camera,
                    " ".join(map(str, frames)),
                    path.name,
                    sha256(path.read_bytes()).hexdigest(),
                )
            )
    print(f"Saved {len(FIGURES)} publication panels")


if __name__ == "__main__":
    main()
