"""CPU audit of the fixed three-source pilot; samples never depend on labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_refiner.data.context_cache import ContextCache
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.utils.checksum import dual_sha256


def describe(values: NDArray[Any]) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size), "min": float(values.min()),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "max": float(values.max()), "mean": float(values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "evidence", "context", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if not all(getattr(args, name).is_absolute() for name in ("store", "evidence", "context", "report")):
        parser.error("All paths must be absolute")
    args.report.mkdir(parents=True, exist_ok=False)
    cache = ContextCache(args.context, EvidenceCache(args.evidence, BallFrameStore(args.store)))
    manifest_hash = dual_sha256(args.context / "manifest.json")
    threshold = 0.15
    rows: list[dict[str, Any]] = []
    for record in cache.manifest["clips"]:
        clip_id = record["clip"]["clip_id"]
        clip = cache.evidence.store.clip_by_id(clip_id)
        generated = cache.load(clip_id)
        arrays = generated.arrays
        shard = args.store / "shards" / shard_name(clip.index)
        if dual_sha256(shard) != record["jpeg_shard_sha256"]:
            raise ValueError(f"JPEG changed: {clip_id}")
        context = arrays.model_context(clip, pose_threshold=threshold, provenance={})
        pose, court = context.require_complete()
        observed = arrays.track_observed
        selected = np.unique(np.linspace(0, clip.frame_count - 1, 5, dtype=int))
        panels = []
        frame_details = []
        for frame in selected:
            image = cache.evidence.store.read_bgr(cache.evidence.store.row_of(clip, int(frame)))
            for person in np.flatnonzero(observed[frame]):
                x, y, side = arrays.boxes_xys[frame, person]
                p0 = (round(float(x - side / 2)), round(float(y - side / 2)))
                p1 = (round(float(x + side / 2)), round(float(y + side / 2)))
                cv2.rectangle(image, p0, p1, (0, 220, 255), 1)
                cv2.putText(image, f"t{arrays.track_ids[person]}", (max(0, p0[0]), max(16, p0[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
                keypoints = arrays.keypoints[frame, person]
                for left, right in ((5, 7), (7, 9), (6, 8), (8, 10)):
                    if min(keypoints[left, 2], keypoints[right, 2]) >= threshold:
                        a = tuple(round(float(v)) for v in keypoints[left, :2])
                        b = tuple(round(float(v)) for v in keypoints[right, :2])
                        cv2.line(image, a, b, (0, 255, 0), 2, cv2.LINE_AA)
                for joint in (7, 8, 9, 10):
                    if keypoints[joint, 2] >= threshold:
                        point = tuple(round(float(v)) for v in keypoints[joint, :2])
                        cv2.circle(image, point, 3, (0, 255, 0), -1, cv2.LINE_AA)
            for joint in np.flatnonzero(arrays.court_valid):
                point = tuple(round(float(v)) for v in arrays.court_points[joint])
                cv2.circle(image, point, 5, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"k{joint}", point, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 255), 1, cv2.LINE_AA)
            # The drawing stays in stored JPEG coordinates; resize only the final panel.
            panel: NDArray[np.uint8] = np.zeros((590, 960, 3), np.uint8)
            panel[50:] = cv2.resize(image, (960, 540))
            title = f"{clip.source}: frame {frame}/{clip.frame_count - 1}; tracks {observed[frame].sum()}"
            cv2.putText(panel, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "yellow: crop/track ID; green: arms >= 0.15; magenta: static court",
                        (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
            frame_details.append({"frame_index": int(frame), "pts": int(arrays.pts[frame]),
                                  "observed_track_ids": arrays.track_ids[observed[frame]].tolist()})
        while len(panels) % 2:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.concatenate([np.concatenate(panels[start:start + 2], axis=1)
                                for start in range(0, len(panels), 2)], axis=0)
        filename = f"clip-{clip.index:05d}-context.jpg"
        if not cv2.imwrite(str(args.report / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError("Failed to write contact sheet")
        if dual_sha256(shard) != record["jpeg_shard_sha256"]:
            raise ValueError(f"JPEG changed during rendering: {clip_id}")
        outside = ((pose.uv < 0) | (pose.uv > 1)).any(axis=-1) & pose.valid
        rows.append({
            "clip_id": clip_id, "source": clip.source, "frames": clip.frame_count,
            "stored_size_wh": [clip.width, clip.height],
            "source_size_wh": [clip.source_width, clip.source_height],
            "cumulative_tracks": len(arrays.track_ids),
            "detections_per_frame": describe(arrays.detection_count),
            "observed_tracks_per_frame": describe(observed.sum(axis=1)),
            "track_observed_frame_counts": describe(observed.sum(axis=0)),
            "one_frame_tracks": int((observed.sum(axis=0) == 1).sum()),
            "frames_without_valid_arms": int((~pose.valid.any(axis=(1, 2))).sum()),
            "valid_arm_slots": int(pose.valid.sum()), "outside_image_valid_arm_slots": int(outside.sum()),
            "court_valid_points": int(court.valid.sum()),
            "court_points_stored_px": arrays.court_points[arrays.court_valid].tolist(),
            "model_context_provenance": context.provenance,
            "seconds": generated.execution["seconds"],
            "sample_selection": "five uniformly spaced source frame indices; no label access",
            "sample_frames": frame_details, "sheet": filename,
            "sheet_sha256": dual_sha256(args.report / filename),
            "context_npz_sha256": record["sha256"], "jpeg_shard_sha256": record["jpeg_shard_sha256"],
        })
    if dual_sha256(args.context / "manifest.json") != manifest_hash:
        raise ValueError("Context manifest changed during audit")
    payload = {"status": "cpu_audit_complete", "context_manifest_sha256": manifest_hash,
               "pose_threshold": threshold, "clips": rows}
    (args.report / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
