"""Meiji multi-camera ``outsource/cam*_annotations.json`` -> one :class:`ClipSpec` per camera.

The external annotations (``video_ball_annotation.v2``) are ChatGPT-assisted
visual reviews, not independent human ground truth; the store keeps their
``point_kind`` so training can choose which kinds to trust. Every frame has a
record: ``observed``/``interpolated``/``occlusion_estimated`` carry a position
and ``unresolved`` does not. The format has no negative label, so an
``unresolved`` frame is "unknown", never "no ball". ``break_before`` becomes
``segment_break``; the format has no typed events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.generate_dataset.frame_store.clip import (
    BallInstance,
    ClipLabels,
    ClipSpec,
    FrameLabel,
    VideoFrames,
)
from src.tennis_scene.chat_annotation.runtime.contracts import sha256_file

SCHEMA = "video_ball_annotation.v2"
DATASET_FILE = "dataset.json"
EXPECTED_VISIBILITY = {
    "observed": "visible",
    "interpolated": "not_independently_visible",
    "occlusion_estimated": "occluded",
    "unresolved": "unknown",
}
COORDINATE_CONTRACT = ("top_left", "zero_based", {"x": "x_px / width", "y": "y_px / height"})
# Exporters round pixels to 3 decimals and normalized values to 9 decimals.
COORDINATE_AGREEMENT_PX = 1e-3


@dataclass(frozen=True, slots=True)
class MeijiSourceConfig:
    root: Path


def _frame_label(row: dict[str, Any], index: int, width: int, height: int, track: int, path: Path) -> FrameLabel:
    kind = row["status"]
    if row["frame_index"] != index or kind not in EXPECTED_VISIBILITY:
        raise ValueError(f"{path}: frame {index} has index {row['frame_index']} / status {kind!r}")
    if row["track_id"] != track or row["label"] != "tennis_ball":
        raise ValueError(f"{path}: frame {index} is not the annotated tennis ball track")
    if row["visibility"] != EXPECTED_VISIBILITY[kind]:
        raise ValueError(f"{path}: frame {index} visibility {row['visibility']!r} disagrees with {kind!r}")
    point = row["center_px"]
    if kind == "unresolved":
        if point is not None:
            raise ValueError(f"{path}: unresolved frame {index} has a position")
        xy = None
    else:
        if not isinstance(point, dict) or set(point) != {"x", "y"}:
            raise ValueError(f"{path}: located frame {index} needs exactly one centre")
        xy = (float(point["x"]), float(point["y"]))
        normalized = row["center_normalized"]
        denormalized = (float(normalized["x"]) * width, float(normalized["y"]) * height)
        if max(abs(a - b) for a, b in zip(xy, denormalized, strict=True)) > COORDINATE_AGREEMENT_PX:
            raise ValueError(f"{path}: frame {index} pixel and normalized centres disagree")
    ball = BallInstance(str(track), kind, xy, kind == "occlusion_estimated")
    return FrameLabel(int(row["pts"]), True, True, bool(row["break_before"]), "unlabeled", (ball,))


def read_camera(annotation_path: Path, video_path: Path, *, video_id: str, clip_name: str, camera_id: str) -> ClipSpec:
    """Parse one camera's annotation and bind it to the exact decoded video."""
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if annotation["schema_version"] != SCHEMA:
        raise ValueError(f"{annotation_path}: expected {SCHEMA}")
    source = annotation["source"]
    coordinates = annotation["coordinate_system"]
    if (coordinates["origin"], coordinates["frame_index"], coordinates["normalization"]) != COORDINATE_CONTRACT:
        raise ValueError(f"{annotation_path}: unsupported coordinate contract")
    if source["file_name"] != video_path.name or source["rotation"] != 0:
        raise ValueError(f"{annotation_path}: does not describe {video_path.name} without rotation")
    media_sha256 = sha256_file(video_path)
    if media_sha256 != source["sha256"]:
        raise ValueError(f"{annotation_path}: sha256 of {video_path} differs from the annotation")
    width, height = int(source["width"]), int(source["height"])
    time_base = Fraction(source["time_base"])
    fps = Fraction(int(source["fps_numerator"]), int(source["fps_denominator"]))
    rows = annotation["frames"]
    if len(rows) != int(source["frame_count"]):
        raise ValueError(f"{annotation_path}: {len(rows)} frames for frame_count {source['frame_count']}")
    track = annotation["target"]["track_id"]
    labels = [_frame_label(row, index, width, height, track, annotation_path) for index, row in enumerate(rows)]
    counts = {kind: sum(1 for row in rows if row["status"] == kind) for kind in EXPECTED_VISIBILITY}
    if counts != annotation["summary"]["counts"]:
        raise ValueError(f"{annotation_path}: summary counts {annotation['summary']['counts']} != {counts}")
    return ClipSpec(
        clip_id=f"meiji/{video_id}/{clip_name}/{camera_id}",
        source="meiji",
        group_id=video_id,
        camera_id=camera_id,
        width=width,
        height=height,
        time_base=time_base,
        fps=fps,
        has_events=False,
        annotation_path=annotation_path,
        annotation_sha256=sha256_file(annotation_path),
        media_sha256=media_sha256,
        media=VideoFrames(video_path, tuple(label.pts for label in labels), time_base),
        labels=ClipLabels.from_frames(labels, width=width, height=height),
        provenance={"video_path": str(video_path), "review": annotation["review"]},
    )


def collect_meiji(config: MeijiSourceConfig) -> list[ClipSpec]:
    """Every camera of every clip listed in ``dataset.json``; each must be annotated."""
    root = config.root
    dataset = json.loads((root / DATASET_FILE).read_text(encoding="utf-8"))
    specs: list[ClipSpec] = []
    for entry in dataset["clips"]:
        clip_dir = root / entry["path"]
        clip = json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))
        if clip["clip_id"] != entry["clip_id"] or clip["video_id"] != entry["video_id"]:
            raise ValueError(f"{clip_dir}: clip.json identity differs from {DATASET_FILE}")
        for camera in clip["cameras"]:
            camera_id = camera["camera_id"]
            specs.append(
                read_camera(
                    clip_dir / "outsource" / f"{camera_id}_annotations.json",
                    clip_dir / camera["video"],
                    video_id=entry["video_id"],
                    clip_name=entry["clip_name"],
                    camera_id=camera_id,
                )
            )
    if not specs or not any(np.any(spec.labels.instances["point_kind"] == 0) for spec in specs):
        raise ValueError(f"Meiji root {root} contains no observed ball")
    return specs
