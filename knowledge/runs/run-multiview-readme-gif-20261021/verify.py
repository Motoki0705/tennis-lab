"""Decode the final GIF, check source correspondence, and extract review images."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.video.reader import RandomAccessVideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    provenance = json.loads((args.frames / "provenance.json").read_text())
    sources = provenance["source_frames"]
    # Check that none of the read-only scene, videos or SMPL assets changed.
    for item in provenance["inputs"].values():
        with Path(item["path"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Input changed: {item['path']}")
    samples = [sources.index(frame) for frame in (sources[0], 180, 360, 540, 720, sources[-1])]
    args.output.mkdir(parents=True, exist_ok=False)
    max_source_error, duration_ms = 0, 0
    psnr, stills = [], []
    with ExitStack() as stack:
        readers = [stack.enter_context(RandomAccessVideoReader(Path(provenance["inputs"][camera]["path"])))
                   for camera in ("cam0", "cam1", "cam2")]
        gif = stack.enter_context(Image.open(args.gif))
        if gif.n_frames != len(sources) or gif.size != (960, 620) or gif.info["loop"] != 0:
            raise ValueError("GIF dimensions/frame count/loop differ from intended render")
        for output_index, source_index in enumerate(sources):
            gif.seek(output_index)
            duration_ms += gif.info["duration"]
            decoded = gif.convert("RGB")
            png = cv2.cvtColor(cv2.imread(str(args.frames / f"{output_index:05d}.png")), cv2.COLOR_BGR2RGB)
            mse = np.square(np.asarray(decoded).astype(float) - png.astype(float)).mean()
            psnr.append(float(10 * np.log10(255 ** 2 / mse)))
            for row, reader in enumerate(readers):
                rgb = cv2.cvtColor(reader.read(source_index), cv2.COLOR_BGR2RGB)
                expected = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
                # Exclude the camera-ID label and panel edges. Every output is checked.
                actual = png[48 + row * 180 + 32:48 + row * 180 + 176, 4:316]
                error = int(np.abs(actual.astype(int) - expected[32:176, 4:316].astype(int)).max())
                max_source_error = max(max_source_error, error)
            if output_index in samples:
                decoded.save(args.output / f"frame_{output_index:05d}_source_{source_index:05d}.png")
                stills.append(decoded.resize((416, 269), Image.Resampling.LANCZOS))
        source_duration_ms = (sources[-1] + 2 - sources[0]) / provenance["source_fps"] * 1000
    if max_source_error != 0:
        raise ValueError(f"Source frame correspondence failed: max RGB error {max_source_error}")
    if abs(duration_ms - source_duration_ms) > 20 or min(psnr) < 28:
        raise ValueError("GIF timing or palette fidelity outside expected bounds")
    if args.gif.stat().st_size >= 10_000_000:
        raise ValueError("GIF exceeds the 10 MB README budget")
    sheet = Image.new("RGB", (1248, 538))
    for index, still in enumerate(stills):
        sheet.paste(still, ((index % 3) * 416, (index // 3) * 269))
    sheet.save(args.output / "contact_sheet.png", optimize=True)
    report = {
        "gif": str(args.gif), "gif_sha256": hashlib.sha256(args.gif.read_bytes()).hexdigest(),
        "size_bytes": args.gif.stat().st_size, "width": 960, "height": 620,
        "frames": len(sources), "duration_ms": duration_ms, "loop": 0,
        "all_input_hashes_unchanged": True,
        "camera_source_comparisons": len(sources) * 3,
        "unlabelled_input_panel_max_rgb_error": max_source_error,
        "gif_to_png_psnr_min_db": min(psnr), "gif_to_png_psnr_mean_db": float(np.mean(psnr)),
        "review_output_frames": samples,
        "review_source_frames": [sources[index] for index in samples],
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
