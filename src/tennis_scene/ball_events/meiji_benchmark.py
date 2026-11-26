"""Build and verify the event-annotated Meiji multiview-v3 benchmark."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tennis_scene.ball_event_schema import EVENT_STATUS
from src.tennis_scene.ball_events.annotation import (
    ANNOTATION_SCHEMA_FILENAME,
    MEIJI_V2_MANIFEST_SHA256,
    annotation_path,
    meiji_v2_manifest_asset,
    read_meiji_v2_manifest,
)
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    EventDetectionConfig,
)
from src.tennis_scene.pipeline.imports.scene_events import (
    replay_scene_events,
    verify_replayed_scene,
)
from src.tennis_scene.pipeline.source_identity import source_commit, source_digests
from src.tennis_scene.schema import SceneResult
from src.utils.paths import PROJECT_ROOT

BASE_SUMS_SHA256 = "cbdb16b047a18747cda008a285c773cd7c3aa894fbed469b79122576daf4d295"
ANNOTATION_SCHEMA_SHA256 = "bacedb5ef378ba50529cb3607600f3a54ae0902696fcb9d402a5e5fc64180779"
CLIP_IDS = tuple(f"video_000/clip_{index:03}" for index in range(69))
SPLIT_SEED = 42
SPLIT_SIZES = {"train": 47, "val": 11, "test": 11}
NET_CLIP_COUNTS = {"train": 22, "val": 5, "test": 5}
SCORING_CLIPS = tuple(CLIP_IDS[index] for index in (27, 28, 40, 43, 56, 64))
TUNING_CLIPS = tuple(CLIP_IDS[index] for index in (1, 9))
CAM0_FRAME_CHECKS = {
    CLIP_IDS[26]: {"near": 100, "far": 172},
    CLIP_IDS[34]: {"near": 84, "far": 546},
    CLIP_IDS[36]: {"near": 217, "far": 140},
    CLIP_IDS[39]: {"near": 108, "far": 613},
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _require_digest(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _read_sums(path: Path) -> dict[Path, str]:
    files: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative in files:
            raise ValueError(f"Invalid or duplicate checksum path: {name}")
        files[relative] = digest
    return files


@dataclass(frozen=True)
class BenchmarkInputs:
    root: Path
    manifest: dict[str, Any]
    files: dict[Path, str]

    @classmethod
    def read(cls, root: Path) -> BenchmarkInputs:
        root = root.resolve()
        manifest_path = root / "manifest.json"
        sums_path = root / "SHA256SUMS"
        manifest = read_meiji_v2_manifest(meiji_v2_manifest_asset(manifest_path))
        _require_digest(sums_path, BASE_SUMS_SHA256)
        if tuple(row["clip_id"] for row in manifest["clips"]) != CLIP_IDS:
            raise ValueError("Expected the fixed 69-clip Meiji multiview-v2 manifest")
        return cls(root, manifest, _read_sums(sums_path))

    def stage(self, destination: Path, index_paths: set[Path]) -> dict[Path, str]:
        """Copy writable indexes and hardlink all immutable benchmark data."""
        shared: dict[Path, str] = {}
        for relative, digest in self.files.items():
            source, target = self.root / relative, destination / relative
            if source.resolve() != source or not source.is_file():
                raise ValueError(f"Expected a regular base benchmark file: {source}")
            _require_digest(source, digest)
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in index_paths:
                shutil.copyfile(source, target)
            elif target.name != ".scene.lock":
                target.hardlink_to(source)
                shared[relative] = digest
        return shared

    def clip_paths(self, row: dict[str, Any]) -> Path:
        """Match pinned media and the clip manifest to the selected v2 scene."""
        clip_id: str = row["clip_id"]
        video, name = clip_id.split("/")
        clip = Path("videos") / video / "clips" / name
        relative = clip / "scene/scene.json"
        if Path(row["scene"]["path"]) != relative or self.files[relative] != row["scene"]["sha256"]:
            raise ValueError(f"Unexpected v2 scene index for {row['clip_id']}")
        clip_json = clip / "clip.json"
        if self.files[clip_json] != row["clip_json"]["sha256"] or Path(row["clip_json"]["path"]) != clip_json:
            raise ValueError(f"Unexpected clip manifest for {row['clip_id']}")
        manifest = json.loads((self.root / clip_json).read_text(encoding="utf-8"))
        index = json.loads((self.root / relative).read_text(encoding="utf-8"))
        if (manifest["clip_id"], manifest["fps"], manifest["num_frames"]) != (row["clip_id"], row["fps"], row["frames"]):
            raise ValueError(f"Clip manifest timeline mismatch: {row['clip_id']}")
        expected_source = []
        for camera in row["cameras"]:
            camera_id = camera["camera_id"]
            media = clip / "media" / f"{camera_id}.mp4"
            if camera["path"] != str(media) or self.files[media] != camera["sha256"]:
                raise ValueError(f"Camera media disagrees with the pinned benchmark: {media}")
            expected_source.append((camera_id, camera["sha256"], row["fps"], row["frames"], manifest["width"], manifest["height"]))
        actual_source = [
            (v["camera_id"], v["sha256"], v["fps"], v["num_frames"], v["width"], v["height"])
            for v in index["source"]["videos"]
        ]
        if (
            index["source"]["clip_id"] != row["clip_id"]
            or index["source_sha256"] != row["scene"]["source_sha256"]
            or actual_source != expected_source
            or manifest["camera_ids"] != [camera["camera_id"] for camera in row["cameras"]]
        ):
            raise ValueError(f"Scene source disagrees with the pinned clip and media: {row['clip_id']}")
        for role, asset in index["exports"]["scene"]["files"].items():
            path = str(relative.parent / asset["path"])
            if row["exports"][f"scene/{role}"] != {**asset, "path": path}:
                raise ValueError(f"v2 export disagrees with the pinned benchmark: {row['clip_id']}")
        return relative


def _counts(result: SceneResult) -> dict[str, Any]:
    kinds, statuses = result.ball_event_kind, result.ball_event_status
    in_point, coverage = result.ball_event_in_point, result.ball_event_coverage
    assert kinds is not None and statuses is not None and in_point is not None and coverage is not None
    return {
        "events": len(kinds),
        "by_kind": {kind: int(np.count_nonzero(kinds == code)) for code, kind in enumerate(EVENT_CLASSES)},
        "by_status": {status: int(np.count_nonzero(statuses == code)) for code, status in enumerate(EVENT_STATUS)},
        "by_in_point": {"in_point": int(in_point.sum()), "outside_point": int((~in_point).sum())},
        "unlabelable_frames": int((~coverage).sum()),
    }


def make_splits(net_clips: set[str], event_counts: dict[str, int]) -> dict[str, Any]:
    """Allocate the two strata while preserving the independent scoring clips."""
    if len(net_clips) != sum(NET_CLIP_COUNTS.values()) or net_clips - set(CLIP_IDS):
        raise ValueError("Stratification expects the 32 pinned Meiji net-contact clips")
    held_out = set(SCORING_CLIPS)
    inner = set(CLIP_IDS) - held_out

    def rank(clip: str) -> bytes:
        return sha256(f"{SPLIT_SEED}:{clip}".encode()).digest()

    assignments: dict[str, list[str]] = {"test": list(SCORING_CLIPS), "val": [], "train": []}
    for stratum, quota in (
        (net_clips, NET_CLIP_COUNTS),
        (set(CLIP_IDS) - net_clips, {split: size - NET_CLIP_COUNTS[split] for split, size in SPLIT_SIZES.items()}),
    ):
        candidates = sorted(stratum & inner, key=rank)
        test_count = quota["test"] - len(stratum & held_out)
        test_clips = [clip for clip in candidates if clip not in TUNING_CLIPS][:test_count]
        if test_count < 0 or len(test_clips) != test_count:
            raise ValueError("The pinned scoring clips are incompatible with stratification")
        remaining = [clip for clip in candidates if clip not in test_clips]
        assignments["test"].extend(test_clips)
        assignments["val"].extend(remaining[:quota["val"]])
        assignments["train"].extend(remaining[quota["val"]:])
    return {
        "seed": SPLIT_SEED,
        "rule": (
            "Separate clips with and without net contact. Allocate net clips 22/5/5 "
            "by largest remainder of 32 times the requested split proportions. "
            "Pin the #2542 scoring clips to test. Fill remaining test slots in "
            "ascending SHA-256 of 'seed:clip_id', excluding #2542 tuning clips. "
            "Use the next ranked clips for val and the remaining clips for train."
        ),
        "pinned_test_clips": list(SCORING_CLIPS),
        "excluded_from_test": list(TUNING_CLIPS),
        **{
            split: {
                "clip_ids": sorted(clips),
                "clips": len(clips),
                "net_clips": len(net_clips.intersection(clips)),
                "net_events": sum(event_counts[clip] for clip in clips),
            }
            for split, clips in assignments.items()
        },
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    annotations = [row["ball_events"] for row in rows]
    totals: dict[str, Any] = {
        "events": sum(item["events"] for item in annotations),
        "unlabelable_frames": sum(item["unlabelable_frames"] for item in annotations),
    }
    for axis in ("by_kind", "by_status", "by_in_point"):
        counts: Counter[str] = Counter()
        for item in annotations:
            counts.update(item[axis])
        totals[axis] = dict(counts)
    if totals != {
        "events": 913,
        "unlabelable_frames": 1885,
        "by_kind": {"shot": 421, "bounce": 457, "net": 35},
        "by_status": {"certain": 809, "probable": 91, "uncertain": 13},
        "by_in_point": {"in_point": 640, "outside_point": 273},
    }:
        raise ValueError(f"Unexpected Meiji event annotation totals: {totals}")
    return totals


def _code_identity() -> dict[str, str]:
    changes = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
        cwd=PROJECT_ROOT, text=True,
    )
    if changes:
        raise ValueError("Commit the src changes before building the Meiji benchmark")
    return {
        "code_commit": source_commit(),
        "code_sha256": document_digest(source_digests(PROJECT_ROOT / "src")),
    }


def _readme() -> str:
    return """# Meiji 4-camera multiview-v3

The 69 fixed multiview-v2 clips contain 913 annotated ball events in SceneResult v3.
The adoption pointer for each clip is its `scene/scene.json`. The adopted
`event_detection` component contains event IDs, evidence and cameras.
The original scene arrays were compared byte for byte as NPY members. Original
immutable files are hardlinked to multiview-v2.

`manifest.json` records input and output SHA-256, the code revision, event counts,
coverage, the fixed train/val/test split and the four cam0 checks. The coverage
mask excludes each inclusive unlabelable interval while keeping recorded events.
Use `sha256sum -c SHA256SUMS` from this directory to check the release.

Rebuild from the repository with
`.venv/bin/python -m src.tennis_scene.scripts.build_meiji_event_benchmark`.
The command verifies an existing release for the same source inputs. Use a new
`--output` directory to build an independent copy.
"""


def build_meiji_benchmark(base: Path, annotations: Path, output: Path) -> dict[str, Any]:
    """Atomically publish the fixed v3 release, or verify a completed copy."""
    base, annotations, output = base.resolve(), annotations.resolve(), output.resolve()
    if (output.is_relative_to(base) or base.is_relative_to(output)
            or output.is_relative_to(annotations) or annotations.is_relative_to(output)):
        raise ValueError("Use an output directory separate from the immutable input trees")
    code = _code_identity()
    inputs = BenchmarkInputs.read(base)
    _require_digest(annotations / ANNOTATION_SCHEMA_FILENAME, ANNOTATION_SCHEMA_SHA256)
    digests = {key: _sha256(annotation_path(annotations, key)) for key in CLIP_IDS}
    rows = inputs.manifest["clips"]
    indexes = {row["clip_id"]: inputs.clip_paths(row) for row in rows}
    pins = {
        "base": {"benchmark_id": inputs.manifest["benchmark_id"], "manifest_sha256": MEIJI_V2_MANIFEST_SHA256,
                 "sha256sums_sha256": BASE_SUMS_SHA256, "shared_files": "hardlinks"},
        "provenance": {**code, "method": "annotation", "annotation_schema_sha256": ANNOTATION_SCHEMA_SHA256},
    }
    if output.exists():
        return _verify_existing(output, inputs, indexes, digests, pins)

    config = EventDetectionConfig("annotation", annotation_directory=annotations,
                                  annotation_benchmark_manifest=base / "manifest.json")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        shared = inputs.stage(staging, set(indexes.values()))
        changed: list[dict[str, Any]] = []
        for row in rows:
            clip_id = row["clip_id"]
            result, _, index = replay_scene_events(
                staging / indexes[clip_id].parent, config, code_commit=code["code_commit"], code_identity=code["code_sha256"],
            )
            info = result.metadata["ball_events"]["provenance"]
            if (info["annotation_sha256"] != digests[clip_id] or info["annotation_schema_sha256"] != ANNOTATION_SCHEMA_SHA256
                    or info["annotation_benchmark_sha256"] != MEIJI_V2_MANIFEST_SHA256):
                raise ValueError(f"Event annotation changed during generation: {clip_id}")
            relative = indexes[clip_id]
            counts = _counts(result)
            changed.append({
                **row,
                "scene": {**row["scene"], "sha256": _sha256(staging / relative), "revision": index["revision"]},
                "exports": {
                    f"scene/{role}": {**asset, "path": (relative.parent / asset["path"]).as_posix()}
                    for role, asset in index["exports"]["scene"]["files"].items()
                },
                "ball_events": {
                    **counts, "annotation_sha256": digests[clip_id],
                    "annotation_file": annotation_path(annotations, clip_id).name,
                    "component": index["artifacts"][EVENT_DETECTION],
                    "court_side": result.event_evidence["player_hint_reference"],
                },
            })
        net_counts = {row["clip_id"]: row["ball_events"]["by_kind"]["net"] for row in changed}
        splits = make_splits({clip for clip, count in net_counts.items() if count}, net_counts)
        totals = _summarize(changed)
        files = [path for path in staging.rglob("*") if path.is_file()]
        manifest = {
            **inputs.manifest, **pins, "benchmark_id": "meiji_4cam/multiview-v3",
            "created": "2026-11-26",
            "description": "Fixed multiview-v2 scenes with original-frame Meiji ball-event annotations.",
            "scene_schema_version": 3, "clips": changed, "ball_events": totals, "splits": splits,
            "cam0_visual_checks": CAM0_FRAME_CHECKS,
            "totals": {**inputs.manifest["totals"], "files": len(files), "bytes": sum(path.stat().st_size for path in files)},
            "verification": {"bit_identical_v2_clips": len(changed), "hardlinked_v2_files": len(shared)},
        }
        (staging / "README.md").write_text(_readme(), encoding="utf-8")
        write_json_atomic(staging / "manifest.json", manifest)
        files.extend((staging / "README.md", staging / "manifest.json"))
        relative_files = sorted(path.relative_to(staging) for path in files)
        (staging / "SHA256SUMS").write_text(
            "".join(f"{shared.get(path) or _sha256(staging / path)}  {path.as_posix()}\n" for path in relative_files),
            encoding="utf-8",
        )
        if _code_identity() != code:
            raise ValueError("Pipeline code changed during benchmark generation")
        BenchmarkInputs.read(base)
        os.rename(staging, output)
    return manifest


def _verify_existing(
    output: Path, base: BenchmarkInputs, indexes: dict[str, Path], digests: dict[str, str], pins: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["benchmark_id"] != "meiji_4cam/multiview-v3" or any(manifest[key] != value for key, value in pins.items()):
        raise ValueError("Existing benchmark was built from different code or inputs")
    if {row["clip_id"]: row["ball_events"]["annotation_sha256"] for row in manifest["clips"]} != digests:
        raise ValueError("Existing benchmark was built from different event annotations")
    sums = _read_sums(output / "SHA256SUMS")
    actual = {path.relative_to(output) for path in output.rglob("*") if path.is_file()} - {Path("SHA256SUMS")}
    if set(sums) != actual:
        raise ValueError("Existing benchmark SHA256SUMS does not describe every file")
    for path, digest in sums.items():
        _require_digest(output / path, digest)
    for path, digest in base.files.items():
        if path not in indexes.values() and path.name != ".scene.lock" and (
            sums.get(path) != digest or not (output / path).samefile(base.root / path)
        ):
            raise ValueError(f"Existing benchmark changed a shared v2 file: {path}")
    net_counts = {}
    for row in manifest["clips"]:
        clip_id = row["clip_id"]
        index = indexes[clip_id]
        scene = verify_replayed_scene(base.root / index, output / index)
        evidence = scene.metadata["ball_events"]
        if (evidence["source"] != "annotation" or evidence["provenance"]["annotation_sha256"] != digests[clip_id]
                or evidence["provenance"]["annotation_benchmark_sha256"] != MEIJI_V2_MANIFEST_SHA256):
            raise ValueError(f"Existing benchmark annotation provenance mismatch: {clip_id}")
        counts = _counts(scene)
        if any(row["ball_events"][key] != value for key, value in counts.items()):
            raise ValueError(f"Existing benchmark annotation counts mismatch: {clip_id}")
        net_counts[clip_id] = counts["by_kind"]["net"]
    if (manifest["ball_events"] != _summarize(manifest["clips"])
            or manifest["splits"] != make_splits({clip for clip, count in net_counts.items() if count}, net_counts)):
        raise ValueError("Existing benchmark aggregate counts or splits mismatch")
    return manifest
