"""Verify the saved FastHMR–GVHMR review videos against the render manifest."""

import hashlib
import json
import subprocess
from argparse import ArgumentParser
from fractions import Fraction
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path

PRODUCTS = ("before", "after", "joints", "3d")
FRAME_COUNTS = {
    "clip_003": (185, 369),
    "clip_057": (143, 285),
    "clip_004": (294, 586),
    "clip_013": (235, 469),
    "clip_040": (209, 417),
    "clip_028": (645, 1288),
}
SOURCE_FPS = Fraction(60000, 1001)


class MediaLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.targets.append(dict(attrs)["href"])


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_mp4(path, frame_count, size):
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_streams",
            "-show_entries",
            "packet=pts,duration:stream=codec_name,width,height,time_base,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    report = json.loads(probe)
    (stream,) = report["streams"]
    tick = 1 / (SOURCE_FPS * Fraction(stream["time_base"]))
    if (
        stream["codec_name"] != "h264"
        or (stream["width"], stream["height"]) != size
        or Fraction(stream["avg_frame_rate"]) != SOURCE_FPS
        or int(stream["nb_frames"]) != frame_count
        or tick.denominator != 1
    ):
        raise ValueError(f"Unexpected video format: {path}")
    timestamps = sorted(packet["pts"] for packet in report["packets"])
    if timestamps != list(range(0, frame_count * tick.numerator, tick.numerator)):
        raise ValueError(f"Video timestamps do not match the source clock: {path}")


def verify(directory, bundle):
    manifest = directory / "manifest.json"
    if manifest.read_bytes() != (bundle / "render-manifest.json").read_bytes():
        raise ValueError("Published render manifest differs from the recorded bytes")
    report = json.loads(manifest.read_text())
    clips = report["clips"]
    if report["preview"] or [clip["clip"] for clip in clips] != list(FRAME_COUNTS):
        raise ValueError("Render does not cover the fixed set of six clips")
    input_hashes = {}
    for mode in ("gvhmr-archive", "pt16"):
        reads = json.loads((bundle / f"review-{mode}.json").read_text())
        if reads["cohort_sha256"] != report["cohort_sha256"]:
            raise ValueError("Rendered cohort disagrees with the recorded input")
        for clip in reads["clips"]:
            name = clip["clip_id"].rsplit("/", 1)[-1]
            input_hashes.setdefault(name, {})[mode] = clip["sha256"]

    names = []
    for clip in clips:
        name = clip["clip"]
        frames, ticks = FRAME_COUNTS[name]
        source_frames = clip["evaluation_frames"]
        if (
            len(source_frames) != frames
            or source_frames[0] != 0
            or source_frames[-1] != ticks - 1
            or any(b - a not in (1, 2) for a, b in pairwise(source_frames))
            or clip["encoded_frames"] != ticks
            or clip["fps"] != float(SOURCE_FPS)
            or clip["input_sha256"] != input_hashes[name]
            or set(clip["files"]) != set(PRODUCTS)
        ):
            raise ValueError(f"Unexpected render lineage or timeline for {name}")
        for mode in PRODUCTS:
            item = clip["files"][mode]
            file = f"{name}-{mode}.mp4"
            path = directory / file
            if item["file"] != file or sha256(path) != item["sha256"]:
                raise ValueError(f"Unexpected file hash: {path}")
            verify_mp4(path, ticks, (1664, 732) if mode == "3d" else (1920, 1672))
            names.append(file)

    index = directory / "index.html"
    parser = MediaLinks()
    parser.feed(index.read_text())
    if parser.targets != names or set(p.name for p in directory.glob("*.mp4")) != set(
        names
    ):
        raise ValueError("Review index links differ from the expected MP4 set")
    return {
        "clips": len(clips),
        "videos": len(names),
        "evaluation_frame_rows": sum(x[0] for x in FRAME_COUNTS.values()),
        "source_ticks_per_product": sum(x[1] for x in FRAME_COUNTS.values()),
        "fps": str(SOURCE_FPS),
        "manifest_sha256": sha256(manifest),
        "index_sha256": sha256(index),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, Path(__file__).resolve().parent)))
