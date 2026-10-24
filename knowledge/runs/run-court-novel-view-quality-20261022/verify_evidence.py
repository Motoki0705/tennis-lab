"""Audit retained diagnostic outputs without invoking the renderer or rewriting media."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    scenes = args.scenes_root.resolve(strict=True)
    output = args.plan_root.resolve(strict=True)
    replay = args.replay_root.resolve(strict=True)
    require(not args.output.resolve().is_relative_to(scenes), "Read-only scene root")
    verified_inputs = 0
    for sid, inventory in read(bundle / "inputs.json").items():
        for name, expected in inventory["hashes"].items():
            require(
                sha256(scenes / sid / name) == expected, f"Input changed: {sid}/{name}"
            )
            verified_inputs += 1

    replay_names = ["inputs.json", "policy.json", "smoke-1.json", "smoke-4.json"]
    counts = []
    summary = {
        (row["scene"], row["condition"]): row for row in read(bundle / "summary.json")
    }
    for sid in ("B00", "B01", "B02", "B03"):
        for condition in ("recommended", "hull5", "hfov45_90", "jitter0"):
            prefix = f"{sid}/{condition}"
            replay_names.extend(
                f"{prefix}/{name}" for name in ("plan.json", "request.json")
            )
            plan = read(output / prefix / "plan.json")
            qa = read(output / prefix / "qa.json")
            request = read(output / prefix / "request.json")["cameras"]
            ids = {row["id"] for row in plan}
            require(len(plan) == 132 and len(ids) == 130, f"Plan counts: {prefix}")
            require(len(qa) == len(request) == len(ids), f"QA/request counts: {prefix}")
            require(ids == {row["id"] for row in qa}, f"QA identities: {prefix}")
            require(
                ids == {row["camera_id"] for row in request},
                f"Request identities: {prefix}",
            )
            for camera_id in ids:
                for name in ("rgb", "alpha", "depth"):
                    require(
                        (
                            output / prefix / "renders" / camera_id / f"{name}.npy"
                        ).is_file(),
                        f"Missing retained raster: {prefix}/{camera_id}/{name}",
                    )
            old = summary[sid, condition]
            for gate in (
                "existing_pre_gate",
                "existing_gate",
                "candidate_target4",
                "candidate_roi01",
            ):
                require(
                    sum(row[gate] for row in qa) == old[f"{gate}_pass"],
                    f"Gate counts: {prefix}/{gate}",
                )
            counts.append(
                {
                    "scene": sid,
                    "condition": condition,
                    "logical_records": len(plan),
                    "candidate_unique": len(ids),
                    "geometry_pass": old["existing_pre_gate_pass"],
                    "rendered_unique": len(request),
                    "existing_gate_accepted": old["existing_gate_pass"],
                    "target4_accepted_all": old["candidate_target4_pass"],
                }
            )
    for name in replay_names:
        require(
            sha256(output / name) == sha256(replay / name), f"Replay differs: {name}"
        )

    manifest = read(bundle / "media-manifest.json")
    for entry in manifest["artifacts"]:
        path = Path(entry["path"])
        require(path.stat().st_size == entry["bytes"], f"Media size: {path}")
        require(sha256(path) == entry["sha256"], f"Media hash: {path}")
        if path.suffix == ".jpg":
            with Image.open(path) as img:
                img.verify()
    videos = [
        Path(row["path"])
        for row in manifest["artifacts"]
        if row["path"].endswith(".mp4")
    ]
    sheets = [
        row for row in manifest["artifacts"] if row["path"].endswith("-contact.jpg")
    ]
    require(len(videos) == len(sheets) == 16, "Incomplete comparison bundle")
    for path in videos:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        height = 338 if "recommended-rgb-overlay" in path.name else 676
        for key, value in {
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "width": 960,
            "height": height,
            "nb_frames": "96",
        }.items():
            require(stream[key] == value, f"Video {key}: {path}")
        require(abs(float(stream["duration"]) - 8) < 0.001, f"Video duration: {path}")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "2",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            check=True,
            capture_output=True,
        )
    for entry in read(bundle / "local-evidence.json").values():
        require(
            sha256(Path(entry["path"])) == entry["sha256"],
            f"Local evidence: {entry['path']}",
        )
    pairs = read(bundle / "pair-checks.json")
    require(
        len(pairs) == 12 and all(row["passed"] for row in pairs),
        "Paired geometry checks",
    )
    for name in ("cases-clearer.png", "cases-failure.png"):
        with Image.open(bundle / name) as img:
            img.verify()
    result = {
        "scope": "retained files, saved QA counts and complete video decoding; no rendering",
        "verified_read_only_inputs": verified_inputs,
        "byte_equal_replay_files": len(replay_names),
        "media_hashes_verified": len(manifest["artifacts"]),
        "fully_decoded_videos": len(videos),
        "contact_sheets": len(sheets),
        "counts": counts,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"inputs={verified_inputs}, replay={len(replay_names)}, media={len(manifest['artifacts'])}, videos={len(videos)}, sheets={len(sheets)}: verified"
    )


if __name__ == "__main__":
    main()
