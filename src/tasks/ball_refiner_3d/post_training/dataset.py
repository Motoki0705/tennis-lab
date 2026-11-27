"""Verify the Meiji release and materialize its public SceneResult inputs once."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.models.components.temporal_transformer import MAX_FRAMES
from src.tasks.ball_refiner_3d.post_training.labels import Annotations, Event
from src.tasks.ball_refiner_3d.post_training.motion import (
    BodySkeleton,
    MotionFeatures,
    extract_motion,
)
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.schema import SceneResult
from src.utils.inference_benchmark_inputs import read_pinned_json
from src.utils.io import write_json_atomic

PREPARED_SCHEMA = "ball_refiner_3d.meiji_motion.v1"
SPLIT_COUNTS = {"train": 47, "val": 11, "test": 11}
INDEPENDENT_TEST = frozenset(
    f"video_000/clip_{i:03d}" for i in (27, 28, 40, 43, 56, 64)
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch at {path}: {actual} != {expected}")


def _within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes the dataset: {relative}")
    return path


@dataclass(frozen=True, slots=True)
class Clip:
    clip_id: str
    fps: float
    motion: MotionFeatures
    labels: Annotations
    reference: Annotations | None


@dataclass(frozen=True, slots=True)
class MeijiBenchmark:
    root: Path
    manifest_sha256: str
    base_sha256: str
    splits: dict[str, list[str]]
    clips: dict[str, dict[str, Any]]
    timeline: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, root: Path, manifest_sha256: str) -> MeijiBenchmark:
        manifest = read_pinned_json(root / "manifest.json", manifest_sha256)
        if (
            manifest["schema"] != "tennis_multiview_benchmark_v1"
            or manifest["benchmark_id"] != "meiji_4cam/multiview-v3"
            or manifest["scene_schema_version"] != 3
            or manifest["provenance"]["method"] != "annotation"
            or manifest["splits"]["seed"] != 42
        ):
            raise ValueError("Require the published annotated Meiji v3 benchmark")
        rows = manifest["clips"]
        clips = {row["clip_id"]: row for row in rows}
        splits: dict[str, list[str]] = {
            split: manifest["splits"][split]["clip_ids"] for split in SPLIT_COUNTS
        }
        all_ids = [clip_id for ids in splits.values() for clip_id in ids]
        if (
            len(rows) != len(clips)
            or len(all_ids) != len(clips)
            or set(all_ids) != set(clips)
            or {split: len(ids) for split, ids in splits.items()} != SPLIT_COUNTS
            or set(manifest["splits"]["pinned_test_clips"]) != INDEPENDENT_TEST
            or not INDEPENDENT_TEST.issubset(splits["test"])
        ):
            raise ValueError(
                "Manifest must retain the disjoint 47/11/11 split and six independent test clips"
            )
        timeline = _timeline(root, clips, splits)
        return cls(
            root,
            manifest_sha256,
            manifest["base"]["manifest_sha256"],
            splits,
            clips,
            timeline,
        )

    def scene(self, clip_id: str) -> SceneResult:
        row = self.clips[clip_id]
        index = row["scene"]
        verify_file(_within(self.root, index["path"]), index["sha256"])
        exports = row["exports"]
        archive = _within(self.root, exports["scene/scene"]["path"])
        sidecar = _within(self.root, exports["scene/metadata"]["path"])
        if archive.with_suffix(".metadata.json") != sidecar:
            raise ValueError(f"Archive and pinned sidecar disagree for {clip_id}")
        verify_file(archive, exports["scene/scene"]["sha256"])
        verify_file(sidecar, exports["scene/metadata"]["sha256"])
        scene = load_scene_result(archive)
        if (
            scene.schema_version != 3
            or scene.metadata["dataset_clip_id"] != clip_id
            or scene.num_frames != row["frames"]
            or scene.fps != row["fps"]
            or scene.metadata["ball_events"]["provenance"]["annotation_sha256"]
            != row["ball_events"]["annotation_sha256"]
        ):
            raise ValueError(
                f"Scene timeline/provenance differs from manifest for {clip_id}"
            )
        return scene


def _timeline(
    root: Path,
    clips: Mapping[str, dict[str, Any]],
    splits: Mapping[str, list[str]],
) -> dict[str, tuple[float, float]]:
    timeline: dict[str, tuple[float, float]] = {}
    by_video: dict[str, list[tuple[float, float, str]]] = {}
    roles = {clip_id: role for role, ids in splits.items() for clip_id in ids}
    for clip_id, row in clips.items():
        record = row["clip_json"]
        clip = read_pinned_json(_within(root, record["path"]), record["sha256"])
        start, stop = clip["global_start_sec"], clip["global_end_sec"]
        if (
            clip["clip_id"] != clip_id
            or clip["fps"] != row["fps"]
            or clip["num_frames"] != row["frames"]
            or not 1 <= row["frames"] <= MAX_FRAMES
            or not math.isfinite(clip["fps"])
            or clip["fps"] <= 0
            or not 0 <= start < stop < math.inf
            or abs((stop - start) * clip["fps"] - row["frames"]) > 1.01
        ):
            raise ValueError(f"Unexpected source timeline for {clip_id}")
        timeline[clip_id] = (start, stop)
        by_video.setdefault(clip["video_id"], []).append((start, stop, clip_id))
    for rows in by_video.values():
        previous_ends: dict[str, float] = {}
        for start, stop, clip_id in sorted(rows):
            role = roles[clip_id]
            if any(
                start < end for other, end in previous_ends.items() if role != other
            ):
                raise ValueError(f"Source intervals overlap across splits at {clip_id}")
            previous_ends[role] = max(stop, previous_ends.get(role, 0.0))
    return timeline


def prepare_clip(
    scene: SceneResult,
    clip_id: str,
    skeleton: BodySkeleton,
    reference: Annotations | None = None,
) -> Clip:
    return Clip(
        clip_id,
        scene.fps,
        extract_motion(scene, skeleton),
        Annotations.from_scene(scene),
        reference,
    )


def _independent(
    root: Path, pins: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[Annotations, str]:
    clip_id = row["clip_id"]
    digest = pins["label_sha256"][clip_id]
    doc = read_pinned_json(_within(root, f"{clip_id}/labels.json"), digest)
    expected = {
        "schema": "meiji_event_labels.v1",
        "benchmark": "multiview-v2",
        "clip_id": clip_id,
        "fps": row["fps"],
        "frames": row["frames"],
        "role": "scoring",
    }
    if any(doc.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Independent label grid/provenance differs for {clip_id}")
    return Annotations.from_document(doc), digest


def save_clip(path: Path, clip: Clip) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = clip.motion
    arrays = {
        "ball_m": m.ball_m,
        "ball_valid": m.ball_valid,
        "players": m.players,
        "root_valid": m.root_valid,
        "smpl_valid": m.smpl_valid,
        "coverage": clip.labels.coverage,
    }
    if clip.reference is not None:
        arrays["reference_coverage"] = clip.reference.coverage
    np.savez(path, **arrays)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "fps": clip.fps,
        "frames": len(m.ball_valid),
        "track_ids": m.track_ids,
        "events": [asdict(e) for e in clip.labels.events],
        "reference_events": None
        if clip.reference is None
        else [asdict(e) for e in clip.reference.events],
    }


def prepare_benchmark(
    benchmark: MeijiBenchmark,
    smplx: Path,
    independent_root: Path,
    pins_path: Path,
    pins_sha256: str,
    output: Path,
) -> Path:
    if output.resolve().is_relative_to(benchmark.root.resolve()):
        raise ValueError("Prepared inputs require their own output directory")
    pins = read_pinned_json(pins_path, pins_sha256)
    if pins[
        "manifest_sha256"
    ] != benchmark.base_sha256 or not INDEPENDENT_TEST.issubset(pins["label_sha256"]):
        raise ValueError("Independent label pins must use the benchmark's v2 base")
    references = {
        clip_id: _independent(independent_root, pins, benchmark.clips[clip_id])
        for clip_id in sorted(INDEPENDENT_TEST)
    }
    asset_sha256 = sha256_file(smplx)
    skeleton = BodySkeleton.load(smplx)
    verify_file(smplx, asset_sha256)
    output.mkdir(parents=True, exist_ok=False)
    prepared = {}
    for clip_id, row in benchmark.clips.items():
        reference, reference_sha = references.get(clip_id, (None, None))
        clip = prepare_clip(benchmark.scene(clip_id), clip_id, skeleton, reference)
        expected = tuple(
            identity
            for key in ("selected_ids_negative_y", "selected_ids_positive_y")
            for identity in _padded_ids(row["players"][key])
        )
        if clip.motion.track_ids != expected:
            raise ValueError(
                f"Stable player slots disagree with manifest for {clip_id}"
            )
        path = output / "clips" / f"{clip_id.replace('/', '--')}.npz"
        prepared[clip_id] = {
            **save_clip(path, clip),
            "source_interval_s": benchmark.timeline[clip_id],
            "scene_index_sha256": row["scene"]["sha256"],
            "scene_archive_sha256": row["exports"]["scene/scene"]["sha256"],
            "annotation_sha256": row["ball_events"]["annotation_sha256"],
            "reference_annotation_sha256": reference_sha,
        }
        print(f"Prepared {clip_id} ({row['frames']} frames)", flush=True)
    manifest_path = output / "inputs.json"
    write_json_atomic(
        manifest_path,
        {
            "schema": PREPARED_SCHEMA,
            "manifest_sha256": benchmark.manifest_sha256,
            "smplx_sha256": asset_sha256,
            "reference_pins_sha256": pins_sha256,
            "splits": benchmark.splits,
            "clips": prepared,
        },
    )
    return manifest_path


def _padded_ids(ids: list[int]) -> list[int | None]:
    return [*sorted(ids), *([None] * (2 - len(ids)))]


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    root: Path
    sha256: str
    inputs: dict[str, Any]

    @classmethod
    def load(
        cls, root: Path, prepared_sha256: str, manifest_sha256: str
    ) -> PreparedDataset:
        inputs = read_pinned_json(root / "inputs.json", prepared_sha256)
        if (
            inputs["schema"] != PREPARED_SCHEMA
            or inputs["manifest_sha256"] != manifest_sha256
        ):
            raise ValueError("Prepared inputs do not match the pinned release")
        return cls(root, prepared_sha256, inputs)

    def split(self, role: str) -> list[Clip]:
        return [self.clip(clip_id) for clip_id in self.inputs["splits"][role]]

    def clip(self, clip_id: str) -> Clip:
        row = self.inputs["clips"][clip_id]
        path = _within(self.root / "clips", row["path"])
        verify_file(path, row["sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            motion = MotionFeatures(
                arrays["ball_m"],
                arrays["ball_valid"],
                arrays["players"],
                arrays["root_valid"],
                arrays["smpl_valid"],
                tuple(row["track_ids"]),
            )
            labels = Annotations(
                tuple(Event(**e) for e in row["events"]), arrays["coverage"]
            )
            refs = row["reference_events"]
            reference = (
                None
                if refs is None
                else Annotations(
                    tuple(Event(**e) for e in refs), arrays["reference_coverage"]
                )
            )
        return Clip(clip_id, row["fps"], motion, labels, reference)
