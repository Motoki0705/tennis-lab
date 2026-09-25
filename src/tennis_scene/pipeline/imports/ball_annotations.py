"""Convert reviewed video_ball_annotation.v2 points to the detector's output schema."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def convert_ball_annotation(path: Path, video: SourceVideo) -> tuple[BallDetectionOutput, dict[str, Any]]:
    annotation = json.loads(path.read_text())
    if annotation.get("schema_version") != "video_ball_annotation.v2":
        raise ValueError("External ball import requires video_ball_annotation.v2")
    source = annotation["source"]
    if (source["sha256"], source["width"], source["height"], source["frame_count"]) != (video.sha256, video.width, video.height, video.num_frames):
        raise ValueError("Ball annotation does not identify this camera's source video")
    if abs(source["fps_numerator"] / source["fps_denominator"] - video.fps) > 1e-5:
        raise ValueError("Ball annotation FPS mismatch")
    coordinates = annotation["coordinate_system"]
    if coordinates["origin"] != "top_left" or coordinates["frame_index"] != "zero_based" or coordinates["normalization"] != {"x": "x_px / width", "y": "y_px / height"}:
        raise ValueError("Unsupported external ball coordinate contract")
    rows = annotation["frames"]
    if len(rows) != video.num_frames or [r["frame_index"] for r in rows] != list(range(video.num_frames)):
        raise ValueError("Ball annotation requires every frame once, in source order")
    uv: NDArray[np.float32] = np.zeros((video.num_frames, 2), np.float32)
    confidence: NDArray[np.float32] = np.zeros(video.num_frames, np.float32)
    kinds: NDArray[np.uint8] = np.zeros(video.num_frames, np.uint8)
    kind_codes = {"unresolved": 0, "observed": 1, "interpolated": 2, "occlusion_estimated": 3}
    expected_visibility = {"unresolved": "unknown", "observed": "visible", "interpolated": "not_independently_visible", "occlusion_estimated": "occluded"}
    for frame, row in enumerate(rows):
        kind = row["status"]
        if kind not in kind_codes or row["label"] != "tennis_ball" or row["track_id"] != annotation["target"]["track_id"]:
            raise ValueError("Ball annotation contains unknown status/target")
        if row["visibility"] != expected_visibility[kind]:
            raise ValueError("Ball annotation visibility/status disagree")
        kinds[frame] = kind_codes[kind]
        point = row["center_px"]
        if kind == "unresolved":
            if point is not None:
                raise ValueError("Unresolved points cannot contain a coordinate")
            continue
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise ValueError("Each localized frame must contain exactly one ball centre")
        pixel = np.asarray([point["x"], point["y"]], np.float64)
        if not np.isfinite(pixel).all() or (pixel < 0).any() or (pixel >= [video.width, video.height]).any():
            raise ValueError("External ball coordinates are outside the source image")
        normalized = row["center_normalized"]
        if not np.allclose(pixel / [video.width, video.height], [normalized["x"], normalized["y"]], rtol=0, atol=1e-7):
            raise ValueError("Pixel/normalized annotation coordinates disagree")
        uv[frame] = pixel
        # This is a binary acceptance weight, never a fabricated detector probability.
        confidence[frame] = float(kind == "observed")
    result = BallDetectionOutput(video.camera_id, np.arange(video.num_frames, dtype=np.int64),
        uv, confidence, kinds == 1, kinds, "annotation_acceptance_not_probability")
    provenance = {"origin": "external_annotation", "source_path": str(path.resolve()), "source_sha256": dual_sha256(path),
        "annotation_schema": annotation["schema_version"], "review": annotation.get("review"),
        "counts": dict(Counter(row["status"] for row in rows)), "observation_policy": "observed_only",
        "confidence_policy": "binary_acceptance_not_probability", "frame_offset": 0,
        "estimated_coordinates_preserved": True}
    return result, provenance


def import_ball_annotations(source: ClipSource, directory: Path, store: ClipStore) -> dict[str, ArtifactRef]:
    results: dict[str, ArtifactRef] = {}
    for video in source.videos:
        value, provenance = convert_ball_annotation(directory / f"{video.camera_id}_annotations.json", video)
        name = f"ball_detection/{video.camera_id}"
        results[name] = store.publish(name, value, ArtifactCodec(BallDetectionOutput), schema=BallDetectionModule.io.output_schema,
            version=BallDetectionModule.io.version, identity={"importer": "video_ball_annotation_v2", "version": 1,
                "source_sha256": store.source_key, "annotation_sha256": provenance["source_sha256"], "observation_policy": "observed_only"},
            dependencies={}, provenance=provenance)
    return results
