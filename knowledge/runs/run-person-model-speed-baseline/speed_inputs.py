"""Freeze source pixels, detection crops and observed body windows."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.submodules.models._base.crops import iter_frame_person_crops
from src.tasks.person_tracking.features import pose_boxes
from src.tennis_scene.pipeline.storage.codec import unpack_value
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import pairwise_iou
from src.utils.inference_benchmark_inputs import PinnedBenchmarkInputs
from src.utils.io import write_json_atomic
from src.utils.video import OpenCVVideoFrameReader

BENCHMARK = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v1")
SOURCE_SHA256 = "b9a33f87ac980c55ebc285c9a5a6f5edd657687be920aea2e14857c1ee845a45"
INPUT_ID = "meiji_4cam/person-model-speed-v1"
INPUT_MANIFEST_SHA256 = (
    "64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62"
)
TIMING_PROTOCOL_SHA256 = (
    "39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2"
)
TIMING_PROTOCOL = Path(__file__).with_name("timing-protocol.json")
WORKLOAD_INDEX = {
    "dino": "frames",
    "dino-vitpose": "frames",
    "vitpose": "crops",
    "vitpose-flip": "crops",
    "body": "body_sequences",
}
CLIPS = tuple(f"video_000/clip_{number:03}" for number in (3, 57, 4, 13, 40, 28))
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
BODY_LENGTH = 16
CROP_BATCH_MULTIPLE = 8


def require_sha(path: Path, expected: str) -> Path:
    actual = dual_sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch at {path}: {actual} != {expected}")
    return path


def frame_id(clip: str, camera: str, frame: int) -> str:
    return f"{clip}/{camera}/frame_{frame:06d}"


@dataclass
class ClipRows:
    clip_id: str
    media: dict
    observed: dict
    boxes: dict
    image_frames: tuple[int, ...]
    body_windows: list[tuple[dict, dict]]


def read_component(scene_path: Path, scene: dict, node: str):
    ref = scene["artifacts"][node]
    descriptor = read_component_descriptor(
        scene_path.parent,
        ref,
        node=node,
        source_sha256=scene["source_sha256"],
    )
    payload = unpack_value(
        descriptor["payload"],
        (scene_path.parent / ref["path"]).parent,
        descriptor["arrays"],
    )
    return ref, payload


def read_source(root: Path) -> list[ClipRows]:
    root = root.resolve(strict=True)
    source = json.loads(require_sha(root / "manifest.json", SOURCE_SHA256).read_text())
    clips = {row["clip_id"]: row for row in source["clips"]}
    selected = []
    for clip in CLIPS:
        row = clips[clip]
        scene_row = row["scene"]
        scene_path = require_sha(root / scene_row["path"], scene_row["sha256"])
        scene = json.loads(scene_path.read_text())

        _, bodies = read_component(scene_path, scene, "gvhmr")
        used_by_all = set.intersection(
            *(
                {
                    int(frame)
                    for segment in body["segments"]
                    for frame in segment["source_frames"]
                }
                for body in bodies["bodies"]
            )
        )
        if len(used_by_all) < 5:
            raise ValueError(f"Fewer than five common GVHMR inference frames: {clip}")
        frames = sorted(used_by_all)
        image_frames = tuple(frames[len(frames) * fifth // 5] for fifth in range(1, 5))

        selection_ref, selection = read_component(
            scene_path, scene, "body_view_selection"
        )
        windows = []
        if len(selection["selections"]) != 4:
            raise ValueError(f"Expected four selected bodies: {clip}")
        for player in selection["selections"]:
            candidates = []
            for segment, request in enumerate(player["requests"]):
                indices = request["source_frames"]
                for start in range(len(indices) - BODY_LENGTH + 1):
                    section = indices[start : start + BODY_LENGTH]
                    if np.all(np.diff(section) == 2) and all(
                        int(i) in used_by_all for i in section
                    ):
                        center = (int(section[0]) + int(section[-1])) * 0.5
                        candidates.append(
                            (abs(center - (row["frames"] - 1) * 0.5), segment, start)
                        )
            if not candidates:
                raise ValueError(
                    f"No complete {BODY_LENGTH}-sample body window: {clip}/{player['person_id']}"
                )
            _, segment, start = min(candidates)
            request = player["requests"][segment]
            sl = slice(start, start + BODY_LENGTH)
            indices = request["source_frames"][sl]
            camera, person = player["camera_id"], player["person_id"]
            identity = {
                "id": f"{clip}/{camera}/body_{person}/samples_{int(indices[0]):06d}_{int(indices[-1]):06d}",
                "clip_id": clip,
                "person": person,
                "camera": camera,
                "segment": segment,
                "frame_ids": [frame_id(clip, camera, int(i)) for i in indices],
                "source_frames": indices.tolist(),
                "size": request["size"],
                "selection_artifact_id": selection_ref["artifact_id"],
                "selection_descriptor_sha256": selection_ref["sha256"],
                "selection_descriptor": str(
                    (scene_path.parent / selection_ref["path"]).relative_to(root)
                ),
                "items": BODY_LENGTH,
                "processed_items": BODY_LENGTH,
            }
            arrays = dict(
                boxes=request["boxes_xys"][sl],
                keypoints=request["keypoints"][sl],
                intrinsic=request["intrinsic"],
            )
            if (
                not all(np.isfinite(value).all() for value in arrays.values())
                or not (arrays["boxes"][:, 2] > 0).all()
            ):
                raise ValueError(
                    f"Invalid body crop or observation coordinates: {identity['id']}"
                )
            windows.append((identity, arrays))

        _, multiview = read_component(scene_path, scene, "multiview_scene_inputs")
        raw = multiview["people"]["raw"]
        if tuple(raw["camera_ids"]) != CAMERAS:
            raise ValueError(f"Unexpected camera set: {clip}")
        raw_boxes = dict(zip(CAMERAS, raw["boxes_xys"], strict=True))
        observed = dict(zip(CAMERAS, raw["observed"], strict=True))
        selected.append(
            ClipRows(
                clip,
                {c["camera_id"]: c for c in row["cameras"]},
                observed,
                raw_boxes,
                image_frames,
                windows,
            )
        )
    return selected


def write_frames(
    root: Path, destination: Path, clips: list[ClipRows]
) -> tuple[list[dict], list[dict], list[dict]]:
    frames, body_frames, bodies = [], [], []
    for clip in clips:
        body_indices = defaultdict(set)
        for identity, arrays in clip.body_windows:
            body_indices[identity["camera"]].update(identity["source_frames"])
            relative = Path("bodies", identity["id"] + ".npz")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
            bodies.append(
                {**identity, "path": relative.as_posix(), "sha256": dual_sha256(path)}
            )

        for camera in CAMERAS:
            media = clip.media[camera]
            wanted = set(clip.image_frames) | body_indices[camera]
            video = require_sha(root / media["path"], media["sha256"])
            with closing(
                iter(OpenCVVideoFrameReader(video, max_frames=max(wanted) + 1))
            ) as decoded:
                for packet in decoded:
                    index = packet.index
                    if index not in wanted:
                        continue
                    name = frame_id(clip.clip_id, camera, index)
                    relative = Path("frames", name + ".npy")
                    path = destination / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, packet.frame, allow_pickle=False)
                    record = {
                        "id": name,
                        "clip_id": clip.clip_id,
                        "camera": camera,
                        "frame": index,
                        "source": media["path"],
                        "source_sha256": media["sha256"],
                        "path": relative.as_posix(),
                        "sha256": dual_sha256(path),
                        "shape": list(packet.frame.shape),
                    }
                    wanted.remove(index)
                    if index in clip.image_frames:
                        frames.append(record)
                    if index in body_indices[camera]:
                        body_frames.append(record)
            if wanted:
                raise ValueError(
                    f"Missing requested video frames {clip.clip_id}/{camera}: {wanted}"
                )
        print(f"Fixed source frames and body windows: {clip.clip_id}", flush=True)
    return frames, body_frames, bodies


def square_bounds(boxes):
    center = boxes[..., :2]
    radius = boxes[..., 2:] * 0.5
    return np.concatenate((center - radius, center + radius), axis=-1)


def write_crops(
    destination: Path, frames: list[dict], clips: list[ClipRows], detector
) -> list[dict]:
    by_clip = {clip.clip_id: clip for clip in clips}
    crops = []
    for row in frames:
        image = np.load(destination / row["path"], allow_pickle=False)
        detection = detector(image)
        boxes, scores = detection.boxes_xyxy, detection.scores
        squares = pose_boxes(boxes)
        clip, frame, camera = by_clip[row["clip_id"]], row["frame"], row["camera"]
        tracked = clip.boxes[camera][frame, clip.observed[camera][frame]]
        tracked = tracked[np.isfinite(tracked).all(axis=-1) & (tracked[:, 2] > 0)]
        iou = pairwise_iou(square_bounds(squares.numpy()), square_bounds(tracked))
        best = iou.max(axis=1, initial=0)
        usable = (squares[:, 2] > 0).numpy()
        priority = sorted(
            np.flatnonzero(usable), key=lambda i: (-float(best[i]), int(i))
        )[:4]
        for rank in sorted(priority):
            pixels, _ = next(
                iter_frame_person_crops(
                    [(0, image)],
                    squares[rank : rank + 1],
                    torch.zeros(1, dtype=torch.int64),
                    batch_size=1,
                )
            )
            name = f"{row['id']}/dino_{rank:03d}"
            relative = Path("crops", name + ".npy")
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ascontiguousarray(pixels[0].numpy()), allow_pickle=False)
            crops.append(
                {
                    "id": name,
                    "frame_id": row["id"],
                    "xyxy": boxes[rank].tolist(),
                    "xys": squares[rank].tolist(),
                    "confidence": float(scores[rank]),
                    "scene_observation_iou": float(best[rank]),
                    "path": relative.as_posix(),
                    "sha256": dual_sha256(path),
                }
            )
    return crops


def publish(
    destination: Path, frames, body_frames, bodies, crops, detector_info
) -> Path:
    count = len(crops) // CROP_BATCH_MULTIPLE * CROP_BATCH_MULTIPLE
    if count < CROP_BATCH_MULTIPLE:
        raise ValueError("Too few fixed crops for the common pose sweep")
    indexes = {}
    for name, rows in (
        ("frames", frames),
        ("body_frames", body_frames),
        ("body_sequences", bodies),
        ("crops", crops),
    ):
        path = destination / f"{name}.json"
        write_json_atomic(path, rows)
        indexes[name] = {
            "path": path.name,
            "records": len(rows),
            "sha256": dual_sha256(path),
        }
    path = destination / "manifest.json"
    write_json_atomic(
        path,
        {
            "id": INPUT_ID,
            "source_benchmark_sha256": SOURCE_SHA256,
            "clip_ids": CLIPS,
            "detector": detector_info,
            "indexes": indexes,
            "pose_sweep_crop_count": count,
            "pose_sweep_tail_ids": [row["id"] for row in crops[count:]],
            "body_window_frames": BODY_LENGTH,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    )
    return path


class InputSet:
    def __init__(self, directory: Path) -> None:
        self.pins = PinnedBenchmarkInputs(
            directory,
            TIMING_PROTOCOL,
            manifest_sha256=INPUT_MANIFEST_SHA256,
            timing_sha256=TIMING_PROTOCOL_SHA256,
        )
        self.sha256 = self.pins.manifest_sha256
        self.manifest = self.pins.manifest
        if (self.manifest["id"], self.manifest["source_benchmark_sha256"]) != (
            INPUT_ID,
            SOURCE_SHA256,
        ):
            raise ValueError(f"Wrong speed input manifest: {directory}")
        self._images = {}
        self.lookups = {}
        self.lookups.update({row["id"]: row for row in self.rows("frames")})
        self.lookups.update({row["id"]: row for row in self.rows("body_frames")})

    def rows(self, name: str) -> tuple[dict, ...]:
        return self.pins.rows(name)

    def frame(self, key: str) -> np.ndarray:
        if key not in self._images:
            row = self.lookups[key]
            path = self.pins.verified_asset(row)
            image = np.load(path, allow_pickle=False)
            if image.dtype != np.uint8 or list(image.shape) != row["shape"]:
                raise ValueError(f"Wrong BGR frame shape or dtype: {path}")
            self._images[key] = image
        return self._images[key]

    def body(self, row: dict) -> dict:
        with np.load(self.pins.verified_asset(row), allow_pickle=False) as arrays:
            return {
                **row,
                **dict(arrays),
                "images": [self.frame(key) for key in row["frame_ids"]],
            }
