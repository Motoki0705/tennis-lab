"""Connect the #2495 cohort to native frames and fixed top-down rows."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from pose_contract import PoseInput

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, ReferenceFrame
from src.utils.video import OpenCVVideoFrameReader


class RowKey(NamedTuple):
    frame: FrameKey
    player_id: int


@dataclass(frozen=True)
class Video:
    path: Path
    camera_id: str
    clip_id: str
    width: int
    height: int
    frames: tuple[int, ...]


def load_cohort(
    bundle_root: Path, bundle_sha256: str, media_root: Path
) -> tuple[str, dict[FrameKey, ReferenceFrame], list[Video]]:
    manifest = load_manifest(bundle_root, expected_sha256=bundle_sha256)
    source = manifest["source_benchmark"]
    source_file = pinned_file(
        media_root, {"path": "manifest.json", "sha256": source["manifest_sha256"]}
    )
    source_manifest = json.loads(source_file.read_bytes())
    if source["benchmark_id"] != source_manifest["benchmark_id"]:
        raise ValueError("Source benchmark ID differs from the fixed cohort")
    source_clips = {clip["clip_id"]: clip for clip in source_manifest["clips"]}
    videos = []
    for clip in manifest["clips"]:
        clip_id = clip["clip_id"]
        pinned_videos = clip["source_videos"]
        if (
            pinned_videos != source_clips[clip_id]["cameras"]
            or [video["camera_id"] for video in pinned_videos] != clip["camera_ids"]
        ):
            raise ValueError(f"Media axes differ from source benchmark for {clip_id}")
        for video in pinned_videos:
            if video["num_frames"] != clip["source_frames"]:
                raise ValueError(
                    f"Source frame count differs from cohort for {clip_id}"
                )
            videos.append(
                Video(
                    pinned_file(media_root, video),
                    video["camera_id"],
                    clip_id,
                    clip["width"],
                    clip["height"],
                    tuple(clip["evaluation_frames"]),
                )
            )
    references = load_reference_frames(bundle_root, expected_sha256=bundle_sha256)
    return (
        manifest["benchmark_id"],
        references,
        sorted(videos, key=lambda video: (video.clip_id, video.camera_id)),
    )


def row_keys(references: Mapping[FrameKey, ReferenceFrame]) -> tuple[RowKey, ...]:
    return tuple(
        RowKey(frame, int(player))
        for frame in sorted(references)
        for player in references[frame].player_ids
    )


def read_video(video: Video) -> Iterator[tuple[FrameKey, NDArray[np.uint8]]]:
    """Yield selected frames in source coordinates through the last request."""
    selected = iter(video.frames)
    target = next(selected)
    for packet in OpenCVVideoFrameReader(video.path, max_frames=video.frames[-1] + 1):
        if packet.original_size != (video.width, video.height):
            raise ValueError(f"Native video dimensions changed for {video.path}")
        if packet.index < target:
            continue
        if packet.index != target:
            break
        yield FrameKey(video.clip_id, video.camera_id, target), packet.frame
        following = next(selected, None)
        if following is None:
            return
        target = following
    raise ValueError(f"Missing source frame {target} in {video.path}")


def pose_inputs(
    videos: list[Video], references: Mapping[FrameKey, ReferenceFrame]
) -> Iterator[tuple[RowKey, PoseInput]]:
    for video in videos:
        for key, image in read_video(video):
            reference = references[key]
            for player, box in zip(
                reference.player_ids, reference.boxes_xyxy, strict=True
            ):
                yield RowKey(key, int(player)), PoseInput(image, box.astype(np.float32))
