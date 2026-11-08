"""Publish synchronized labeled videos after checking the Blender frame audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.checksum import dual_sha256

from .blender import SHOT_SCHEMA


def validate_render(package: Path, render: Path, player_id: int) -> dict[str, Any]:
    """Compare actual Blender state for every rendered frame to the pinned inputs."""
    info = json.loads((package / "manifest.json").read_text())
    report: dict[str, Any] = json.loads((render / "render.json").read_text())
    if (
        info["schema"] != SHOT_SCHEMA
        or report["schema"] != "tennis_scene_player_view_render_v1"
        or not report["complete"]
        or report["package_sha256"] != dual_sha256(package / "manifest.json")
        or report["player_id"] != player_id
    ):
        raise ValueError("Render report does not match its completed shot/player")
    if set(report["material"]) != {"OpaqueGrayPlayer", "OpaqueGreenBall"}:
        raise ValueError("Render must audit both foreground materials")
    if report["motion_blur"] or any(
        v["alpha"] != 1 or v["emission_strength"] != 0
        for v in report["material"].values()
    ):
        raise ValueError(
            "Render must use opaque, non-emissive materials without motion blur"
        )
    if not report["gpu_devices"] or report["engine"] != "CYCLES":
        raise ValueError("Expected explicit Cycles GPU rendering evidence")
    players = info["players"]
    viewer = next(p for p in players if p["player_id"] == player_id)
    sources = np.load(package / "source_frames.npy")
    mesh_valid = np.load(package / "player_smpl_valid.npy")
    ball_valid = np.load(package / "ball_3d_valid.npy")
    ball_positions = np.load(package / "ball_positions.npy")
    with np.load(package / viewer["camera"]) as arrays:
        cameras = dict(arrays)
    selected = []
    for output_index, row in enumerate(report["frames"]):
        index = row["source_frame"] - int(sources[0])
        if (
            not 0 <= index < len(sources)
            or row["cache_index"] != index
            or row["output_index"] != output_index
            or row["camera_valid"] != bool(cameras["valid"][index])
            or row["camera_reason"] != int(cameras["reason"][index])
        ):
            raise ValueError("Render source frame/camera mask correspondence differs")
        camera_valid = row["camera_valid"]
        hidden = [
            not (
                camera_valid and mesh_valid[p, index] and item["player_id"] != player_id
            )
            for p, item in enumerate(players)
        ]
        if row["player_hidden"] != hidden or row["ball_hidden"] != (
            not (camera_valid and ball_valid[index])
        ):
            raise ValueError("Render object masks/viewer visibility differ")
        if camera_valid:
            expected = np.eye(4)
            expected[:3, :3], expected[:3, 3] = (
                cameras["rotation"][index],
                cameras["location"][index],
            )
            if not np.allclose(row["camera_matrix_world"], expected, rtol=0, atol=2e-5):
                raise ValueError("Render changed the saved camera pose")
        elif row["camera_matrix_world"] is not None:
            raise ValueError("Missing camera must not acquire a replacement pose")
        if ball_valid[index] and not np.allclose(
            row["ball_position"], ball_positions[index], rtol=0, atol=1e-5
        ):
            raise ValueError("Render ball time/position differs")
        errors = np.asarray(row["pc2_max_error_m"])
        if (
            errors.shape != (len(players),)
            or not np.isfinite(errors).all()
            or (errors > 1e-5).any()
        ):
            raise ValueError("Render PC2 evaluation differs")
        selected.append(row["source_frame"])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Render frames must be nonempty and unique")
    if len(list((render / "frames").glob("*.png"))) != len(selected):
        raise ValueError("Render PNG count differs from audit")
    return report


def verify_video(
    path: Path, *, frames: int, fps: float, width: int, height: int
) -> dict[str, Any]:
    """Decode every frame and independently check the encoded FPS and duration."""
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration:frame=best_effort_timestamp_time",
                "-show_frames",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    timestamps = np.array(
        [float(row["best_effort_timestamp_time"]) for row in probe["frames"]]
    )
    if timestamps.shape != (frames,) or not np.allclose(
        timestamps, np.arange(frames) / fps, rtol=0, atol=1e-5
    ):
        raise ValueError("Encoded frame presentation times differ from the source FPS")
    probe = probe["streams"][0]
    actual_fps = float(Fraction(probe["avg_frame_rate"]))
    if (probe["width"], probe["height"]) != (width, height) or abs(
        actual_fps - fps
    ) > 1e-6:
        raise ValueError("Encoded resolution/FPS differs")
    if (
        int(probe["nb_frames"]) != frames
        or abs(float(probe["duration"]) - frames / fps) > 1 / fps
    ):
        raise ValueError("Encoded frame count/duration differs")
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError("Decoded frame shape differs")
            count += 1
    finally:
        capture.release()
    if count != frames:
        raise ValueError(f"Decoded {count} frames, expected {frames}")
    return {
        "decoded_frames": count,
        "presentation_timestamps_checked": frames,
        "fps": actual_fps,
        "duration_seconds": float(probe["duration"]),
        "width": width,
        "height": height,
        "sha256": dual_sha256(path),
    }


def publish_player_views(
    *, package: Path, renders: Path, output: Path, preview: bool = False
) -> Path:
    """Encode all selected IDs and a side-by-side comparison, with no time resampling."""
    if os.path.lexists(output):
        raise FileExistsError(f"Video output already exists: {output}")
    if any(output.resolve().is_relative_to(p.resolve()) for p in (package, renders)):
        raise ValueError("Video output must be outside read-only inputs")
    info = json.loads((package / "manifest.json").read_text())
    identities = [p["player_id"] for p in info["players"]]
    reports = [
        validate_render(package, renders / f"player_{identity}", identity)
        for identity in identities
    ]
    selected = [r["source_frame"] for r in reports[0]["frames"]]
    if any(
        selected != [r["source_frame"] for r in report["frames"]] for report in reports
    ):
        raise ValueError("All player views must render the same source frames")
    if (preview and len(selected) > 12) or (
        not preview and selected != np.load(package / "source_frames.npy").tolist()
    ):
        raise ValueError(
            "Preview is limited to 12 frames; final publication requires the entire input timeline"
        )
    width, height = (info["camera_config"][name] for name in ("width", "height"))
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p requires even image dimensions")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    processes: list[subprocess.Popen] = []
    logs = []
    filenames = [f"player_{identity}.mp4" for identity in identities] + [
        "comparison.mp4"
    ]
    try:
        rate = str(Fraction(str(info["fps"])).limit_denominator(1_000_000))
        for index, filename in enumerate(filenames):
            log = (staging / f"{filename}.log").open("wb")
            logs.append(log)
            video_width = width if index < len(identities) else width * len(identities)
            processes.append(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-n",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-s",
                        f"{video_width}x{height}",
                        "-r",
                        rate,
                        "-i",
                        "pipe:0",
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
                        str(staging / filename),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                )
            )
        for index, source_frame in enumerate(selected):
            panels = []
            for identity, report in zip(identities, reports, strict=True):
                path = renders / f"player_{identity}" / "frames" / f"{index:05d}.png"
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None or frame.shape != (height, width, 3):
                    raise ValueError(f"Missing/invalid rendered PNG: {path}")
                valid = report["frames"][index]["camera_valid"]
                if not valid and np.ptp(frame.astype(np.int16), axis=(0, 1)).max() > 2:
                    raise ValueError("Camera-missing render must be a blank frame")
                cv2.rectangle(frame, (0, 0), (width, min(50, height)), (15, 15, 15), -1)
                text = f"ID {identity} | source frame {source_frame} | camera {'VALID' if valid else 'MISSING'}"
                scale = min(0.75, width / 1100)
                cv2.putText(
                    frame,
                    text,
                    (12, min(32, height - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                if not valid:
                    cv2.putText(
                        frame,
                        "CAMERA MISSING",
                        (max(8, width // 2 - 150), height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale * 1.6,
                        (100, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )
                panels.append(frame)
            for process, frame in zip(
                processes, [*panels, np.concatenate(panels, axis=1)], strict=True
            ):
                assert process.stdin is not None
                process.stdin.write(frame.tobytes())
        for process in processes:
            assert process.stdin is not None
            process.stdin.close()
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed; inspect the encoder log")
        verification = {
            filename: verify_video(
                staging / filename,
                frames=len(selected),
                fps=info["fps"],
                width=width if i < len(identities) else width * len(identities),
                height=height,
            )
            for i, filename in enumerate(filenames)
        }
        result = {
            "schema": "tennis_scene_player_view_videos_v1",
            "player_ids": identities,
            "source_frames": selected,
            "fps": info["fps"],
            "preview": preview,
            "package_sha256": dual_sha256(package / "manifest.json"),
            "render_reports": {
                str(identity): dual_sha256(
                    renders / f"player_{identity}" / "render.json"
                )
                for identity in identities
            },
            "camera_missing_frames": {
                str(identity): [
                    r["source_frame"] for r in report["frames"] if not r["camera_valid"]
                ]
                for identity, report in zip(identities, reports, strict=True)
            },
            "settings": {
                k: info[k]
                for k in ("config", "camera_config", "policy", "inputs", "frames")
            },
            "verification": verification,
            "human_acceptance": "pending",
        }
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False)
        )
        if os.path.lexists(output):
            raise FileExistsError(f"Video output already exists: {output}")
        staging.rename(output)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        shutil.rmtree(staging)
        raise
    finally:
        for log in logs:
            log.close()
    return output / "manifest.json"
