"""Render already-computed smoke predictions after zooming, CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.utils.video.reader import read_video_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.output / "provenance.json").read_text())
    frame = read_video_rgb(provenance["video"], max_frames=1)[0]
    assert hashlib.sha256(frame.tobytes()).hexdigest() == provenance["frame_rgb_sha256"]
    with np.load(args.output / "predictions.npz") as saved:
        poses = saved["keypoints"][:, 0]
    boxes = np.asarray(provenance["boxes_xyxy"])[:, 0]
    edges = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    tiles = []
    for i, (box, pose) in enumerate(zip(boxes, poses, strict=True)):
        x1, y1, x2, y2 = box
        margin = (y2 - y1) * 0.4
        left, top = max(0, int(x1 - margin)), max(0, int(y1 - margin))
        right, bottom = (
            min(frame.shape[1], int(x2 + margin)),
            min(frame.shape[0], int(y2 + margin)),
        )
        crop = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_RGB2BGR)
        scale = 480 / crop.shape[0]
        crop = cv2.resize(
            crop, (round(crop.shape[1] * scale), 480), interpolation=cv2.INTER_CUBIC
        )
        xy = np.rint((pose[:, :2] - [left, top]) * scale).astype(int)
        color = (40, 240, 40) if i == 0 else (40, 180, 255)
        for a, b in edges:
            if min(pose[a, 2], pose[b, 2]) >= 0.3:
                cv2.line(crop, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)
        for (x, y), score in zip(xy, pose[:, 2], strict=True):
            if score >= 0.3:
                cv2.circle(crop, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            crop,
            f"person {i}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        tiles.append(crop)
    image = np.concatenate(tiles, axis=1)
    image = cv2.copyMakeBorder(
        image, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )
    cv2.putText(
        image,
        "Predictions, not ground truth; confidence >= 0.3",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    assert cv2.imwrite(str(args.output / "overlay_zoom.png"), image)
    record = {
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": f"PYTHONPATH=. .venv/bin/python knowledge/runs/run-slcs-vitpose-redownload-smoke-v1/render_overlay.py {args.output}",
        "input_predictions_sha256": hashlib.sha256(
            (args.output / "predictions.npz").read_bytes()
        ).hexdigest(),
        "output": "overlay_zoom.png",
        "gpu_used": False,
    }
    (args.output / "visualization.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
