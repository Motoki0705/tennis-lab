"""Render the pinned issue 1776 preview or four-second comparison on Colab L4."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.tennis_scene.visualization.player_view.video import (
    publish_player_views,
    validate_render,
    verify_video,
)
from src.utils.checksum import dual_sha256


def contacts(directory: Path, names: list[str], player: int, count: int) -> Path:
    columns = []
    for name in names:
        cap = cv2.VideoCapture(
            str(directory / "videos" / name / f"player_{player}.mp4")
        )
        frames = []
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Missing preview frame")
            frames.append(frame)
        cap.release()
        columns.append(np.concatenate(frames, axis=0))
    image = np.concatenate(columns, axis=1)
    path = directory / f"player_{player}_contact.png"
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not write contact sheet")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preview", "full"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    inputs = json.loads((args.inputs / "manifest.json").read_text())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if "L4" not in gpu:
        raise RuntimeError(f"Expected NVIDIA L4, observed {gpu}")
    if inputs["source_frame_interval"] != [0, 240] or inputs["player_ids"] != [0, 1]:
        raise ValueError("Comparison input changed")
    preview = args.phase == "preview"
    frames = inputs["preview_source_frames"] if preview else list(range(240))
    summary = {
        "schema": "player_view_stabilization_comparison_run_v1",
        "phase": args.phase,
        "gpu": gpu,
        "source_frames": frames,
        "inputs_sha256": dual_sha256(args.inputs / "manifest.json"),
        "git_ref": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "human_acceptance": "pending",
        "variants": {},
        "comparison_videos": {},
    }
    for variant in inputs["packages"]:
        name = variant["name"]
        package = args.inputs / "packages" / name
        if dual_sha256(package / "manifest.json") != variant["manifest_sha256"]:
            raise ValueError("Package digest changed")
        info = json.loads((package / "manifest.json").read_text())
        if (
            info["camera_processing"]["config"]["sigma_seconds"]
            != variant["sigma_seconds"]
        ):
            raise ValueError("Strength differs")
        for player in [0, 1]:
            output = args.output / "renders" / name / f"player_{player}"
            command = [
                str(args.blender),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--threads",
                "4",
                "--python",
                "src/tennis_scene/visualization/blender/shots/player_view.py",
                "--",
                "--package",
                str(package),
                "--output",
                str(output),
                "--player-id",
                str(player),
                "--gpu-backend",
                "OPTIX",
            ]
            if preview:
                command += ["--source-frames", *(str(x) for x in frames)]
            output.parent.mkdir(parents=True, exist_ok=True)
            with (output.parent / f"player_{player}.log").open("w") as log:
                print(f"RENDER_START {name} ID{player}", flush=True)
                subprocess.run(
                    command, check=True, stdout=log, stderr=subprocess.STDOUT
                )
            report = validate_render(package, output, player)
            if [r["source_frame"] for r in report["frames"]] != frames:
                raise ValueError("Rendered timeline differs")
            if report["camera"]["depth_of_field"]:
                raise ValueError("Unexpected depth of field")
            print(f"RENDER_COMPLETE {name} ID{player} frames={len(frames)}", flush=True)
        videos = args.output / "videos" / name
        manifest = publish_player_views(
            package=package,
            renders=args.output / "renders" / name,
            output=videos,
            preview=preview,
        )
        summary["variants"][name] = {
            "sigma_seconds": variant["sigma_seconds"],
            "video_manifest_sha256": dual_sha256(manifest),
            "video_manifest": str(manifest.relative_to(args.output)),
            "lens": info["lens"],
        }
        # An atomic, useful checkpoint; Colab --persist saves this with completed videos.
        checkpoint = args.output / ".checkpoint.tmp"
        checkpoint.write_text(json.dumps(summary, indent=2) + "\n")
        checkpoint.replace(args.output / "checkpoint.json")
        print(f"CHECKPOINT_READY {name}", flush=True)
    names = [x["name"] for x in inputs["packages"]]
    for player in [0, 1]:
        if preview:
            contacts(args.output, names, player, len(frames))
        else:
            target = args.output / "videos" / f"comparison_player_{player}.mp4"
            command = ["ffmpeg", "-v", "error", "-n", "-filter_complex_threads", "1"]
            for name in names:
                command += [
                    "-i",
                    str(args.output / "videos" / name / f"player_{player}.mp4"),
                ]
            command += [
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2[top];[2:v][3:v]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[v]",
                "-map",
                "[v]",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ]
            subprocess.run(command, check=True)
            summary["comparison_videos"][str(player)] = verify_video(
                target, frames=240, fps=info["fps"], width=1280, height=720
            )
    summary["elapsed_seconds"] = time.monotonic() - started
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "COMPARISON_COMPLETE "
        + json.dumps(
            {"phase": args.phase, "elapsed_seconds": summary["elapsed_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
