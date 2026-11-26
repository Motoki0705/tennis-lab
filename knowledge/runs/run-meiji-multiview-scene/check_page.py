"""Verify the offline page and write a compact receipt for its media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import av


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset(bundle: Path, relative: str) -> Path:
    path = bundle / relative
    if (
        not path.resolve().is_relative_to(bundle)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(f"Asset must be stored in the page: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.page.resolve()
    manifest = bundle / "manifest.json"
    html = bundle / "index.html"
    data = json.loads(manifest.read_text())
    body = html.read_text(encoding="utf-8")
    marker = '<script id="comparison-data" type="application/json">'
    embedded = json.loads(body.split(marker, 1)[1].split("</script>", 1)[0])
    original = json.loads((args.original / "manifest.json").read_text())
    first, second = data["stages"]
    expected = data["selection"]["included"]
    if (
        embedded != data
        or (first["id"], second["id"]) != ("stage1", "stage2")
        or len(expected) != 16
        or data["selection"] != original["selection"]
        or first != original["stages"][0]
        or [clip["id"] for clip in second["clips"]] != expected
    ):
        raise ValueError("Comparison page manifest or cohort changed")
    run_dir = Path(__file__).parent
    scenes = json.loads((run_dir / "verified-scenes.json").read_text())
    placement = {}
    for route in ("legacy", "multiview"):
        with (run_dir / f"{route}-completion.csv").open(newline="") as stream:
            placement[route] = {
                row["clip_id"]: list(
                    json.loads(row["valid_placement_frames_by_id"]).values()
                )
                for row in csv.DictReader(stream)
            }
    clips = []
    assets = {html, manifest}
    for old, current in zip(first["clips"], second["clips"], strict=True):
        clip_id = current["id"]
        provenance = current["provenance"]
        if (
            old["id"] != clip_id
            or provenance["multiview_scene_assembly"] != scenes[clip_id]["scene"]
            or provenance["placement_frames_by_route"]
            != {route: table[clip_id] for route, table in placement.items()}
        ):
            raise ValueError(f"Scene receipt differs from the 4cam readback: {clip_id}")
        entry = {"clip_id": clip_id}
        for stage, clip in (("stage1", old), ("stage2", current)):
            video = asset(bundle, clip["video"])
            poster = asset(bundle, clip["poster"])
            assets.update((video, poster))
            digest, size = sha256(video), video.stat().st_size
            if (
                digest != clip["video_sha256"]
                or size != clip["video_bytes"]
                or size > 15_000_000
            ):
                raise ValueError(f"Invalid video size or checksum: {video}")
            with av.open(str(video)) as container:
                (stream,) = container.streams.video
                if (
                    stream.codec_context.name != "h264"
                    or stream.frames != clip["video_frames"]
                ):
                    raise ValueError(f"Invalid video stream: {video}")
            if stage == "stage1" and sha256(poster) != sha256(
                args.original / clip["poster"]
            ):
                raise ValueError(f"Stage 1 poster changed: {clip_id}")
            entry[stage] = {
                "video": clip["video"],
                "video_sha256": digest,
                "video_bytes": size,
                "video_frames": clip["video_frames"],
                "poster": clip["poster"],
                "poster_sha256": sha256(poster),
            }
        clips.append(entry)
    files = {file for file in bundle.rglob("*") if file.is_file()}
    size = sum(file.stat().st_size for file in files)
    if files != assets or size > 200_000_000:
        raise ValueError(f"Unexpected files or overall bundle size: {bundle}")
    receipt = {
        "bundle_bytes": size,
        "index_sha256": sha256(html),
        "manifest_sha256": sha256(manifest),
        "codec": "h264",
        "clips": clips,
    }
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(clips)} clips, {2 * len(clips)} H.264 videos, {size:,} bytes")


if __name__ == "__main__":
    main()
