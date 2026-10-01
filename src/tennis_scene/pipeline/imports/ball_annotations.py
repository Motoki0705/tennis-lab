"""Reviewed ``video_ball_annotation.v2`` points as the ball detector's output schema."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.utils.checksum import dual_sha256

IMPORTER = "video_ball_annotation_v2"
IMPORTER_VERSION = 1
POINT_KINDS = {"unresolved": 0, "observed": 1, "interpolated": 2, "occlusion_estimated": 3}
EXPECTED_VISIBILITY = {"unresolved": "unknown", "observed": "visible", "interpolated": "not_independently_visible",
                       "occlusion_estimated": "occluded"}


def convert_ball_annotation(path: Path, video: SourceVideo) -> tuple[BallDetectionOutput, dict[str, Any]]:
    """One camera's annotation; only ``observed`` points become observations.

    Interpolated and occlusion-estimated coordinates are preserved with their
    point kind, and the confidence is a binary acceptance weight, never a
    fabricated detector probability.
    """
    annotation = json.loads(path.read_text())
    if annotation.get("schema_version") != "video_ball_annotation.v2":
        raise ValueError(f"External ball import requires video_ball_annotation.v2: {path}")
    source = annotation["source"]
    if (source["sha256"], source["width"], source["height"], source["frame_count"]) != (video.sha256, video.width, video.height, video.num_frames):
        raise ValueError(f"Ball annotation does not identify {video.camera_id}'s source video: {path}")
    if abs(source["fps_numerator"] / source["fps_denominator"] - video.fps) > 1e-5:
        raise ValueError(f"Ball annotation FPS mismatch: {path}")
    coordinates = annotation["coordinate_system"]
    if (coordinates["origin"], coordinates["frame_index"], coordinates["normalization"]) != ("top_left", "zero_based", {"x": "x_px / width", "y": "y_px / height"}):
        raise ValueError(f"Unsupported external ball coordinate contract: {path}")
    rows = annotation["frames"]
    if [row["frame_index"] for row in rows] != list(range(video.num_frames)):
        raise ValueError(f"Ball annotation requires every frame once, in source order: {path}")
    uv: NDArray[np.float32] = np.zeros((video.num_frames, 2), np.float32)
    confidence: NDArray[np.float32] = np.zeros(video.num_frames, np.float32)
    kinds: NDArray[np.uint8] = np.zeros(video.num_frames, np.uint8)
    for frame, row in enumerate(rows):
        kind = row["status"]
        if kind not in POINT_KINDS or row["label"] != "tennis_ball" or row["track_id"] != annotation["target"]["track_id"]:
            raise ValueError(f"Ball annotation frame {frame} has an unknown status/target")
        if row["visibility"] != EXPECTED_VISIBILITY[kind]:
            raise ValueError(f"Ball annotation frame {frame}: visibility {row['visibility']!r} disagrees with {kind!r}")
        kinds[frame] = POINT_KINDS[kind]
        point = row["center_px"]
        if kind == "unresolved":
            if point is not None:
                raise ValueError(f"Unresolved frame {frame} cannot contain a coordinate")
            continue
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise ValueError(f"Localized frame {frame} must contain exactly one ball centre")
        pixel = np.asarray([point["x"], point["y"]], np.float64)
        if not np.isfinite(pixel).all() or (pixel < 0).any() or (pixel >= [video.width, video.height]).any():
            raise ValueError(f"Ball frame {frame} lies outside the source image")
        normalized = row["center_normalized"]
        if not np.allclose(pixel / [video.width, video.height], [normalized["x"], normalized["y"]], rtol=0, atol=1e-7):
            raise ValueError(f"Ball frame {frame}: pixel and normalized coordinates disagree")
        uv[frame] = pixel
        confidence[frame] = float(kind == "observed")
    result = BallDetectionOutput(video.camera_id, np.arange(video.num_frames, dtype=np.int64),
        uv, confidence, kinds == POINT_KINDS["observed"], kinds, "annotation_acceptance_not_probability")
    provenance = {"source_path": str(path.resolve()), "annotation_schema": annotation["schema_version"],
        "review": annotation.get("review"), "counts": dict(Counter(row["status"] for row in rows)),
        "observation_policy": "observed_only", "confidence_policy": "binary_acceptance_not_probability"}
    return result, provenance


def import_ball_annotations(nodes: Sequence[ComponentNode], store: ClipStore, source: ClipSource,
                            directory: Path) -> dict[str, ArtifactRef]:
    """Publish ``<directory>/<camera>_annotations.json`` into every ``ball_detection/<camera>`` node."""
    results: dict[str, ArtifactRef] = {}
    for video in source.videos:
        path = directory / f"{video.camera_id}_annotations.json"
        value, provenance = convert_ball_annotation(path, video)
        name = f"ball_detection/{video.camera_id}"
        results[name] = publish_import(bind_import(nodes, name, store), store, value, importer=IMPORTER,
            version=IMPORTER_VERSION, identity={"annotation_sha256": dual_sha256(path), "observation_policy": "observed_only"},
            provenance=provenance)
    return results
