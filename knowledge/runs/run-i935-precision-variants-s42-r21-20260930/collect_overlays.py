"""Verify encoded overlays and retain representative frames plus hashes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent
OUTPUT = BUNDLE.parents[2] / "outputs/campaign935-r22-video"


def main() -> None:
    cv2.setNumThreads(1)
    results: list[dict[str, Any]] = []
    for name, posters in (("clip010", (135, 146)), ("hard", (400, 402, 535))):
        directory = OUTPUT / name
        metadata = json.loads((directory / "overlay.json").read_text())
        video = Path(metadata["video"])
        assert dual_sha256(video) == metadata["sha256"]
        destination = BUNDLE / f"overlay-{name}.json"
        shutil.copy2(directory / "overlay.json", destination)
        comparisons = []
        poster_hashes = {}
        pts = []
        with av.open(str(video)) as container:
            container.streams.video[0].codec_context.thread_count = 1
            for index, decoded in enumerate(container.decode(video=0)):
                frame = metadata["start"] + index
                assert [decoded.width, decoded.height] == metadata["size_wh"]
                pts.append(decoded.pts)
                pixels = decoded.to_ndarray(format="bgr24")
                snapshot = directory / f"frame-{frame:04d}.jpg"
                if snapshot.exists():
                    original = cv2.imread(str(snapshot))
                    assert original is not None
                    before, after = original[-148:], pixels[-148:]
                    error = np.abs(before.astype(np.float64) - after.astype(np.float64))
                    bright = before.max(-1) > 120
                    lost = float((after.max(-1)[bright] < 40).mean())
                    assert error.mean() < 2 and lost < .005
                    comparisons.append({"source_frame": frame, "footer_mae": float(error.mean()), "bright_pixels_lost_fraction": lost})
                if frame in posters:
                    path = BUNDLE / f"overlay-{name}-frame-{frame:04d}.jpg"
                    assert cv2.imwrite(str(path), pixels)
                    poster_hashes[path.name] = dual_sha256(path)
        assert len(pts) == metadata["frames"]
        assert all(a < b for a, b in zip(pts, pts[1:], strict=False))
        results.append({"clip": metadata["clip"], "video": str(video), "sha256": metadata["sha256"], "frames": len(pts),
                        "start": metadata["start"], "bytes": video.stat().st_size, "posters_sha256": poster_hashes,
                        "metadata_sha256": dual_sha256(destination), "encoding_checks": comparisons})
    report = {"status": "verified", "gpu": False, "results": results,
              "render_output_bytes": sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()),
              "poster_rule": "clip010 middle + frame146 estimated occlusion/artificial gap; hard interval start, first recorded remaining-failure frame, and middle"}
    (BUNDLE / "overlay-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "verified", "videos": len(results), "frames": sum(r["frames"] for r in results),
                      "render_output_bytes": report["render_output_bytes"]}))


if __name__ == "__main__":
    main()
