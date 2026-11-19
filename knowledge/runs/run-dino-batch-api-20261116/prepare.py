"""Recreate the fixed C7 inputs from the complete, read-only exported clip."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    expected = json.loads((bundle / "inputs.json").read_text())
    results = json.loads((bundle / "results.json").read_text())
    checkpoint = Path("ckpt/dino/checkpoint0029_4scale_swin.pth")
    if sha256(checkpoint) != results["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the fixed C7 comparison")
    meta = json.loads((args.clip / "clip.json").read_text())
    for key in ("clip_id", "num_frames", "width", "height", "fps", "camera_ids"):
        if meta[key] != expected["clip_metadata"][key]:
            raise ValueError(f"Clip metadata differs: {key}")
    if not meta["exported_at"]:
        raise ValueError("Clip export has not completed")
    images = {}
    cv2.setNumThreads(1)
    for camera in meta["camera_ids"]:
        wanted = [item for item in expected["frames"] if item["key"].startswith(camera + "_")]
        media = args.clip / "media" / f"{camera}.mp4"
        before = media.stat()
        if sha256(media) != wanted[0]["media_sha256"]:
            raise ValueError(f"Media differs: {camera}")
        by_frame = {item["frame"]: item for item in wanted}
        reader = cv2.VideoCapture(str(media))
        count = 0
        try:
            while True:
                valid, image = reader.read()
                if not valid:
                    break
                if count in by_frame:
                    item = by_frame[count]
                    if hashlib.sha256(image.tobytes()).hexdigest() != item["bgr_sha256"]:
                        raise ValueError(f"Decoded BGR differs: {item['key']}")
                    images[item["key"]] = image
                count += 1
        finally:
            reader.release()
        after = media.stat()
        if count != meta["num_frames"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Incomplete or changing media: {camera}")
    if list(images) != [item["key"] for item in expected["frames"]]:
        raise ValueError("Input frame order or coverage differs")
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "inputs.npz", **images)
    for name in ("inputs.json", "protocol.json", "scalar_reference.py"):
        shutil.copy2(bundle / name, args.output / name)
    print(f"Prepared {len(images)} fixed frames in {args.output}")


if __name__ == "__main__":
    main()
