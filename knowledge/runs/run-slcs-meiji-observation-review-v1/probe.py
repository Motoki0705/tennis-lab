"""CPU contact sheets for completed Meiji observations; no teacher acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.checksum import dual_sha256
from src.utils.io import save_json_atomic


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def inspect_camera(
    clip: dict[str, Any], camera: str, video: Path, cache: Path, output: Path
) -> dict[str, Any]:
    receipt = cache.with_suffix(".metadata.json")
    identity = json.loads(receipt.read_text())
    start_hashes = {str(p): dual_sha256(p) for p in (video, cache, receipt)}
    if start_hashes[str(video)] != identity["video_sha256"]:
        raise ValueError(f"Video identity mismatch: {video}")
    with np.load(cache, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    boxes, poses = arrays["boxes"], arrays["keypoints"]
    frames = int(clip["num_frames"])
    if boxes.shape != (2, frames, 4) or poses.shape != (2, frames, 17, 3):
        raise ValueError(f"Unexpected observation shapes: {cache}")
    if not np.isfinite(boxes).all() or not np.isfinite(poses).all():
        raise ValueError(f"Non-finite observations: {cache}")
    supported = arrays["pose_supported_mask"]
    observed = arrays["observed_masks"]
    sample_indices = arrays["detection_frame_indices"]
    centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    diagonals = np.linalg.norm(boxes[..., 2:] - boxes[..., :2], axis=-1)
    delta = np.linalg.norm(np.diff(centers, axis=1), axis=-1)
    relative_delta = delta / np.maximum(diagonals[:, :-1], 1.0)
    jumps = np.argmax(relative_delta, axis=1)
    frame_indices = sorted(
        set(np.linspace(0, frames - 1, 5, dtype=int).tolist())
        | {int(index) for jump in jumps for index in (jump, jump + 1)}
    )
    # View-local court halves, before associate_people canonicalizes scene IDs.
    # BGR colors: local near-side P0 green; local far-side P1 magenta.
    colors = ((70, 230, 70), (230, 70, 230))
    edges = (
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
    )
    tiles: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if (
                not ok
                or int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_index + 1
            ):
                raise ValueError(f"Cannot read exact frame {frame_index}: {video}")
            if frame.shape[:2] != (clip["height"], clip["width"]):
                raise ValueError(f"Video shape mismatch: {video}")
            for player, color in enumerate(colors):
                x1, y1, x2, y2 = np.rint(boxes[player, frame_index]).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    f"local P{player} support={int(supported[player, frame_index])}",
                    (max(x1, 0), max(y1 - 8, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )
                pose = poses[player, frame_index]
                for a, b in edges:
                    if min(pose[a, 2], pose[b, 2]) >= 0.3:
                        cv2.line(
                            frame,
                            tuple(np.rint(pose[a, :2]).astype(int)),
                            tuple(np.rint(pose[b, :2]).astype(int)),
                            color,
                            3,
                        )
                for x, y, confidence in pose:
                    if confidence >= 0.3:
                        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1)
            tile = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
            text = f"{clip['clip_id']} {camera} f{frame_index} / {frames}"
            cv2.putText(
                tile, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1
            )
            tiles.append(tile)
    finally:
        capture.release()
    rows = (len(tiles) + 2) // 3
    tiles.extend(np.zeros_like(tiles[0]) for _ in range(rows * 3 - len(tiles)))
    sheet = np.concatenate(
        [np.concatenate(tiles[i : i + 3], axis=1) for i in range(0, len(tiles), 3)],
        axis=0,
    )
    image_path = output / f"{camera}.jpg"
    if not cv2.imwrite(str(image_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Cannot save {image_path}")
    for path, digest in start_hashes.items():
        if dual_sha256(Path(path)) != digest:
            raise ValueError(f"Input changed during review: {path}")
    player_stats = []
    for player in range(2):
        hip = (poses[player, :, [11, 12], 2] >= 0.3).all(axis=0)
        torso = (poses[player, :, [5, 6, 11, 12], 2] >= 0.3).all(axis=0)
        player_stats.append(
            {
                "player": player,
                "detection_sample_coverage": float(
                    observed[player, sample_indices].mean()
                ),
                "pose_supported_fraction": float(supported[player].mean()),
                "hips_confident_fraction": float(hip.mean()),
                "hips_and_shoulders_confident_fraction": float(torso.mean()),
                "box_center_step_px": describe(delta[player]),
                "box_center_step_previous_diagonal": describe(relative_delta[player]),
                "max_step_segment": [int(jumps[player]), int(jumps[player] + 1)],
            }
        )
    return {
        "camera": camera,
        "frame_indices": frame_indices,
        "players": player_stats,
        "input_sha256": start_hashes,
        "producer": identity,
        "contact_sheet": str(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    report: dict[str, Any] = {
        "command": sys.argv,
        "status": "running",
        "policy": "Five evenly spaced frames plus endpoints of each player's largest adjacent box-center step. Diagnostic only; not a teacher acceptance test.",
        "player_axis": "View-local court half: P0=near, P1=far. These are not the canonical scene player IDs; associate_people aligns them before teacher inference.",
        "clips": [],
    }
    result = args.output_dir / "results.json"
    save_json_atomic(report, result)
    try:
        for clip_id in args.clip:
            if len(Path(clip_id).parts) != 2 or any(
                p in (".", "..") for p in Path(clip_id).parts
            ):
                raise ValueError(f"Invalid clip id: {clip_id}")
            video_id, name = Path(clip_id).parts
            source = (
                args.project_root
                / "data/tennis_multivew/processed/meiji_3cam/dataset/videos"
                / video_id
                / "clips"
                / name
            )
            manifest = source / "clip.json"
            clip = json.loads(manifest.read_text())
            output = args.output_dir / video_id / name
            output.mkdir(parents=True)
            cameras = [
                inspect_camera(
                    clip,
                    camera,
                    source / relative,
                    args.observation_root / clip_id / f"{camera}_people.npz",
                    output,
                )
                for camera, relative in zip(
                    clip["camera_ids"], clip["video_paths"], strict=True
                )
            ]
            report["clips"].append(
                {
                    "clip_id": clip_id,
                    "manifest_sha256": dual_sha256(manifest),
                    "cameras": cameras,
                }
            )
            save_json_atomic(report, result)
            print(json.dumps({"clip_id": clip_id, "cameras": len(cameras)}), flush=True)
        report["status"] = "done"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_json_atomic(report, result)


if __name__ == "__main__":
    main()
