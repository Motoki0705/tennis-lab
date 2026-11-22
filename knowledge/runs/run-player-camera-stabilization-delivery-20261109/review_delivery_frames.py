"""Stream sample frames from delivered grids for agent inspection, without rewriting videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chosen = [0, 2, 26, 60, 162, 239]
    records = []
    for player in (0, 1):
        source = args.run / "videos" / f"comparison_player_{player}.mp4"
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open {source}")
        selected = {}
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if count in chosen:
                    if frame.shape != (720, 1280, 3):
                        raise ValueError((source, frame.shape))
                    selected[count] = frame
                count += 1
        finally:
            capture.release()
        if count != 240 or sorted(selected) != chosen:
            raise ValueError((source, count, sorted(selected)))
        contact = np.zeros((2 * 360, 3 * 640, 3), dtype=np.uint8)
        for index, number in enumerate(chosen):
            small = cv2.resize(
                selected[number], (640, 360), interpolation=cv2.INTER_AREA
            )
            top, left = (index // 3) * 360, (index % 3) * 640
            contact[top : top + 360, left : left + 640] = small
        target = args.output / f"player_{player}_delivery_contact.png"
        if target.exists():
            raise FileExistsError(target)
        if not cv2.imwrite(str(target), contact):
            raise RuntimeError(f"Cannot write {target}")
        full_size_samples = []
        for number in (26, 162):
            sample = args.output / f"player_{player}_source_{number:03d}.png"
            if sample.exists():
                raise FileExistsError(sample)
            if not cv2.imwrite(str(sample), selected[number]):
                raise RuntimeError(f"Cannot write {sample}")
            full_size_samples.append(
                {
                    "source_frame": number,
                    "path": str(sample),
                    "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                }
            )
        with source.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(
            {
                "player_id": player,
                "video": str(source),
                "video_sha256": sha,
                "source_frames": chosen,
                "decoded": count,
                "contact": str(target),
                "contact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "full_size_samples": full_size_samples,
            }
        )
    manifest = args.output / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "records": records, "human_acceptance": "pending"},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"contacts": len(records), "decoded": sum(x["decoded"] for x in records)}
        )
    )


if __name__ == "__main__":
    main()
