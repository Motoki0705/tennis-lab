"""Read the adopted sparse event columns and the original annotation lineage."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import file_digest, sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.analytics.ingest.input_files import json_object, sha256_value
from src.tennis_scene.ball_event_labels import (
    EVENT_CLASSES_V4,
    EVENT_STATUS,
    EVENT_SUBTYPES_V4,
)
from src.tennis_scene.ball_event_schema import (
    BALL_EVENT_FIELDS,
    validate_ball_event_columns,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest, split_clip_id

SPLITS = ("train", "val", "test")
SERVE_SUBTYPES = ("serve_overhead", "serve_underhand")


@dataclass(frozen=True)
class V4Event:
    source_event_id: str
    frame: int
    kind: str
    subtype: str | None
    status: str | None
    frame_uncertainty: int | None
    court_side: int | None
    adopted_in_v4: bool
    in_point: bool
    excluded_reason: str | None
    evidence: dict[str, Any]
    outside_point_reason: str | None = None


@dataclass(frozen=True)
class V4Clip:
    clip: ClipManifest
    split: str
    global_start_sec: float
    global_end_sec: float
    events: tuple[V4Event, ...]
    coverage: NDArray[np.bool_]
    provenance: dict[str, Any]

    @property
    def clip_id(self) -> str:
        return self.clip.clip_id


@dataclass(frozen=True)
class V4Release:
    benchmark_id: str
    dataset_id: str
    video_id: str
    manifest_sha256: str
    sha256sums_sha256: str
    verified_files: int
    listed_files: int
    clips: tuple[V4Clip, ...]


def _relative(value: str) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
        or path == Path(".")
    ):
        raise ValueError(f"Invalid benchmark path: {value!r}")
    return path


class _BenchmarkFiles:
    def __init__(self, root: Path, sums: bytes) -> None:
        self.root = root.resolve()
        self.digests: dict[Path, str] = {}
        self.verified: set[Path] = set()
        for line in sums.decode("utf-8").splitlines():
            digest, name = line.split("  ", maxsplit=1)
            relative = _relative(name)
            if relative in self.digests:
                raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
            self.digests[relative] = sha256_value(digest, name)

    def expected(self, relative: Path, declared: str | None = None) -> str:
        expected = self.digests.get(relative)
        if expected is None or (declared is not None and expected != declared):
            raise ValueError(f"SHA256SUMS disagrees with the release for {relative}")
        return expected

    def path(self, relative: Path) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Benchmark path escapes its root: {relative}")
        return path

    def check(self, relative: Path, actual: str) -> None:
        if actual != self.expected(relative):
            raise ValueError(f"SHA-256 mismatch: {relative}")
        self.verified.add(relative)

    def read(self, relative: Path) -> bytes:
        raw = self.path(relative).read_bytes()
        self.check(relative, sha256(raw).hexdigest())
        return raw

    def json(self, relative: Path) -> dict[str, Any]:
        return json_object(self.read(relative), relative.as_posix())

    def reference(self, record: dict[str, Any], *, directory: Path = Path()) -> Path:
        relative = directory / _relative(record["path"])
        self.expected(relative, sha256_value(record["sha256"], str(relative)))
        return relative


def _decoded_events(
    arrays: dict[str, NDArray[Any]], labels: dict[str, Any]
) -> dict[tuple[int, str], tuple[str, str, int, int]]:
    events = {}
    for frame, kind, subtype, status, uncertainty, side in zip(
        arrays["ball_event_frame"],
        arrays["ball_event_kind"],
        arrays["ball_event_subtype"],
        arrays["ball_event_status"],
        arrays["ball_event_frame_uncertainty"],
        arrays["ball_event_side"],
        strict=True,
    ):
        events[(int(frame), labels["classes"][kind])] = (
            labels["subtypes"][subtype],
            labels["statuses"][status],
            int(uncertainty),
            int(side),
        )
    return events


def _event_rows(
    evidence: dict[str, Any],
    events: dict[tuple[int, str], tuple[str, str, int, int]],
    num_frames: int,
) -> tuple[V4Event, ...]:
    rows: list[V4Event] = []
    seen: set[str] = set()
    hint_sides = evidence["player_hint_reference"]

    def identity(item: dict[str, Any]) -> tuple[str, int]:
        source_id, frame = item["source_event_id"], item["frame"]
        if not isinstance(source_id, str) or not source_id.strip() or source_id in seen:
            raise ValueError(f"Missing or duplicate original event ID: {source_id!r}")
        if type(frame) is not int or not 0 <= frame < num_frames:
            raise ValueError(f"Invalid frame for {source_id}: {frame!r}")
        if type(item["old_in_point"]) is not bool:
            raise ValueError(f"Missing original point eligibility for {source_id}")
        seen.add(source_id)
        return source_id, frame

    for item in evidence["events"]:
        source_id, frame = identity(item)
        kind = item["kind"]
        key = (frame, kind)
        hint = item.get("player_hint") if kind == "shot" else None
        side = hint_sides[hint] if hint is not None else 0
        label = (item["subtype"], item["status"], item["frame_uncertainty"], side)
        if events.pop(key, None) != label:
            raise ValueError(f"Component evidence and NPZ disagree for {source_id}")
        rows.append(
            V4Event(
                source_id,
                frame,
                kind,
                *label,
                True,
                item["old_in_point"],
                None,
                item,
                None if item["old_in_point"] else "outside_original_point",
            )
        )
    if events:
        raise ValueError(
            "An adopted NPZ event has no original event ID in the component"
        )

    for item in evidence["excluded_events"]:
        source_id, frame = identity(item)
        reason, kind = item["reason"], item["old_kind"]
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or kind not in EVENT_CLASSES_V4
        ):
            raise ValueError(f"Invalid V4 exclusion evidence for {source_id}")
        rows.append(
            V4Event(
                source_id,
                frame,
                kind,
                None,
                None,
                None,
                None,
                False,
                False,
                reason,
                item,
                "excluded_in_v4",
            )
        )

    # An adopted bounce after a service let is still a dead-ball observation.
    result = []
    after_let = False
    for event in sorted(rows, key=lambda row: (row.frame, row.source_event_id)):
        if event.adopted_in_v4:
            if event.in_point and event.subtype in SERVE_SUBTYPES:
                after_let = False
            elif event.in_point and event.subtype == "service_let":
                after_let = True
            elif after_let and event.subtype == "bounce":
                event = replace(
                    event,
                    in_point=False,
                    outside_point_reason="post_service_let_bounce",
                )
        result.append(event)
    return tuple(result)


def _validate_coverage(
    coverage: NDArray[np.bool_], intervals: list[dict[str, Any]]
) -> None:
    expected: NDArray[np.bool_] = np.ones(len(coverage), dtype=np.bool_)
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < len(coverage)
        ):
            raise ValueError("Invalid V4 unlabelable frame interval")
        expected[start : end + 1] = False
    if not np.array_equal(coverage, expected):
        raise ValueError("Component unlabelable intervals disagree with NPZ coverage")


def _read_scene_columns(
    files: _BenchmarkFiles,
    archive_path: Path,
    metadata: dict[str, Any],
    clip: ClipManifest,
) -> dict[str, NDArray[Any]]:
    if (
        metadata.get("scene_schema_version") != 4
        or metadata.get("dataset_clip_id") != clip.clip_id
    ):
        raise ValueError("Scene metadata is not the expected clip's SceneResult v4")
    if metadata["camera_ids"] != list(clip.camera_ids):
        raise ValueError("Scene metadata cameras differ from clip.json")
    with files.path(archive_path).open("rb") as handle:
        files.check(archive_path, file_digest(handle, "sha256").hexdigest())
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            for key in ("num_frames", "fps", "width", "height"):
                item = archive[key]
                if item.ndim != 0 or item.item() != getattr(clip, key):
                    raise ValueError(f"NPZ {key} differs from clip.json")
            arrays = {
                name: archive[name]
                for name in BALL_EVENT_FIELDS
                if name in archive.files
            }
    validate_ball_event_columns(metadata, clip.num_frames, arrays, version=4)
    return arrays


def _check_component_arrays(
    files: _BenchmarkFiles,
    component: dict[str, Any],
    directory: Path,
    arrays: dict[str, NDArray[Any]],
) -> None:
    for name, exported in arrays.items():
        member = component["payload"][name]["array"]
        relative = directory / _relative(member)
        descriptor = component["arrays"][member]
        files.expected(relative, sha256_value(descriptor["sha256"], str(relative)))
        value = np.load(BytesIO(files.read(relative)), allow_pickle=False)
        if (
            not isinstance(value, np.ndarray)
            or descriptor["shape"] != list(value.shape)
            or descriptor["dtype"] != value.dtype.name
            or exported.dtype != value.dtype
            or not np.array_equal(exported, value)
        ):
            raise ValueError(f"Component array differs from the export: {name}")


def _camera_lineage(
    files: _BenchmarkFiles,
    row: dict[str, Any],
    index: dict[str, Any],
    directory: Path,
    clip: ClipManifest,
) -> None:
    cameras = row["cameras"]
    videos = index["source"]["videos"]
    for camera_id, media, manifest, source in zip(
        clip.camera_ids, clip.video_paths, cameras, videos, strict=True
    ):
        media_path = files.reference(manifest)
        if media_path != directory / media:
            raise ValueError(
                f"Manifest camera path differs from clip.json for {camera_id}"
            )
        expected = (camera_id, clip.fps, clip.num_frames)
        if (
            any(
                (entry["camera_id"], entry["fps"], entry["num_frames"]) != expected
                for entry in (manifest, source)
            )
            or manifest["sha256"] != source["sha256"]
        ):
            raise ValueError(
                f"Manifest, scene and clip camera inputs disagree for {camera_id}"
            )


def _read_clip(
    files: _BenchmarkFiles, row: dict[str, Any], split: str, dataset_id: str
) -> V4Clip:
    video_id, clip_name = split_clip_id(row["clip_id"])
    directory = Path("videos") / video_id / "clips" / clip_name
    clip_path = files.reference(row["clip_json"])
    if clip_path != directory / "clip.json":
        raise ValueError("Manifest clip.json path disagrees with clip ID")
    raw_clip = files.json(clip_path)
    clip = ClipManifest.from_dict(raw_clip, clip_dir=files.root / directory)
    if (clip.dataset_id, clip.clip_id, clip.fps, clip.num_frames) != (
        dataset_id,
        row["clip_id"],
        row["fps"],
        row["frames"],
    ):
        raise ValueError("Manifest and clip.json disagree on clip identity or timebase")
    start, end = (raw_clip[key] for key in ("global_start_sec", "global_end_sec"))
    if (
        any(
            type(value) not in (int, float) or not isfinite(value)
            for value in (start, end, clip.fps)
        )
        or not 0 <= start < end
        or not clip.cameras
    ):
        raise ValueError("Invalid clip.json source timeline or missing camera origins")
    scene_reference = row["scene"]
    index_path = files.reference(scene_reference)
    if index_path != directory / "scene/scene.json":
        raise ValueError("Manifest scene index path disagrees with clip ID")
    index = files.json(index_path)
    if (
        index["schema"] != "tennis_scene_index_v1"
        or index["source"]["clip_id"] != clip.clip_id
        or index["source_sha256"] != scene_reference["source_sha256"]
        or index["revision"] != scene_reference["revision"]
    ):
        raise ValueError("Scene index disagrees with the benchmark manifest")
    _camera_lineage(files, row, index, directory, clip)

    annotation = row["ball_events"]
    export = index["exports"]["scene"]
    component_ref = annotation["component"]
    if (
        component_ref != index["artifacts"]["event_detection"]
        or export["inputs"].get("event_detection") != component_ref
        or (component_ref["schema"], component_ref["version"]) != ("scene_result", 4)
    ):
        raise ValueError("Scene export does not use the manifest's V4 event component")
    component_path = files.reference(component_ref, directory=index_path.parent)
    component = files.json(component_path)
    if (
        component["schema"] != "component_artifact_v1"
        or component["status"] != "complete"
        or component["node"] != "event_detection"
        or component["artifact_id"] != component_ref["artifact_id"]
        or component["execution_key"] != component_ref["execution_key"]
        or component["source_sha256"] != index["source_sha256"]
        or (component["output_schema"], component["output_version"])
        != ("scene_result", 4)
    ):
        raise ValueError("V4 event component identity disagrees with the scene index")

    export_paths = {}
    for role in ("metadata", "scene"):
        manifest_path = files.reference(row["exports"][f"scene/{role}"])
        index_export = files.reference(
            export["files"][role], directory=index_path.parent
        )
        if manifest_path != index_export or not manifest_path.is_relative_to(
            index_path.parent
        ):
            raise ValueError(
                f"Manifest and index point to different scene {role} assets"
            )
        export_paths[role] = manifest_path
    archive_path, metadata_path = export_paths["scene"], export_paths["metadata"]
    if archive_path.suffix != ".npz" or metadata_path != archive_path.with_suffix(
        ".metadata.json"
    ):
        raise ValueError("Invalid scene NPZ or metadata sidecar path")
    metadata = files.json(metadata_path)
    payload = component["payload"]
    if metadata != payload["metadata"] or any(
        payload[key] != getattr(clip, key)
        for key in ("fps", "num_frames", "width", "height")
    ):
        raise ValueError("Component and scene export metadata differ")
    arrays = _read_scene_columns(files, archive_path, metadata, clip)
    _check_component_arrays(files, component, component_path.parent, arrays)
    labels = metadata["ball_events"]
    provenance = labels["provenance"]
    for key in ("annotation_sha256", "source_v1_annotation_sha256"):
        if sha256_value(annotation[key], key) != provenance[key]:
            raise ValueError(f"Manifest and scene annotation {key} disagree")
    evidence = payload["event_evidence"]
    if evidence["player_hint_reference"] != annotation["court_side"]:
        raise ValueError("Component and manifest court side references differ")
    events = _event_rows(evidence, _decoded_events(arrays, labels), clip.num_frames)
    coverage = arrays["ball_event_coverage"]
    _validate_coverage(coverage, evidence["unlabelable"])
    sources = {
        "clip_json": row["clip_json"],
        "source_media": row["cameras"],
        "scene_index": scene_reference,
        "scene_npz": row["exports"]["scene/scene"],
        "scene_metadata": row["exports"]["scene/metadata"],
        "event_detection": {**component_ref, "path": component_path.as_posix()},
        "annotation_file": annotation["annotation_file"],
        "annotation_provenance": provenance,
        "unlabelable_intervals": evidence["unlabelable"],
        "player_hint_reference": evidence["player_hint_reference"],
    }
    return V4Clip(clip, split, float(start), float(end), events, coverage, sources)


def _splits(manifest: dict[str, Any], clip_ids: set[str]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in SPLITS:
        row = manifest["splits"][split]
        members = row["clip_ids"]
        if not isinstance(members, list) or row["clips"] != len(members):
            raise ValueError(f"Invalid {split} split count")
        for clip_id in members:
            if clip_id not in clip_ids or clip_id in assignments:
                raise ValueError(f"Invalid or repeated split assignment: {clip_id}")
            assignments[clip_id] = split
    if set(assignments) != clip_ids:
        raise ValueError("Manifest splits do not cover every clip")
    return assignments


def _event_counts(clips: tuple[V4Clip, ...]) -> dict[str, Any]:
    events = [event for clip in clips for event in clip.events if event.adopted_in_v4]
    counts: dict[str, Any] = {
        "events": len(events),
        "excluded_events": sum(
            not event.adopted_in_v4 for clip in clips for event in clip.events
        ),
        "unlabelable_frames": sum(int((~clip.coverage).sum()) for clip in clips),
    }
    for axis, keys in (
        ("kind", EVENT_CLASSES_V4),
        ("subtype", EVENT_SUBTYPES_V4),
        ("status", EVENT_STATUS),
    ):
        observed = Counter(getattr(event, axis) for event in events)
        counts[f"by_{axis}"] = {key: observed[key] for key in keys}
    return counts


def read_v4(root: Path, *, manifest_sha256: str | None = None) -> V4Release:
    """Verify every S1 input against the frozen release and read one video."""
    sums = (root / "SHA256SUMS").read_bytes()
    files = _BenchmarkFiles(root, sums)
    manifest_path = Path("manifest.json")
    if manifest_sha256 is not None:
        files.expected(manifest_path, sha256_value(manifest_sha256, "manifest_sha256"))
    manifest = files.json(manifest_path)
    try:
        return _read_release(manifest, files, sha256(sums).hexdigest())
    except (KeyError, TypeError, BadZipFile, EOFError) as error:
        raise ValueError(f"Invalid V4 release {root}: {error}") from error


def _read_release(
    manifest: dict[str, Any], files: _BenchmarkFiles, sums_hash: str
) -> V4Release:
    benchmark_id = manifest["benchmark_id"]
    dataset_id, version = split_clip_id(benchmark_id)
    if (
        version != "multiview-v4"
        or manifest["schema"] != "tennis_multiview_benchmark_v1"
        or manifest["scene_schema_version"] != 4
    ):
        raise ValueError("Expected a multiview-v4 benchmark")
    rows = manifest["clips"]
    ids = [row["clip_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or len(ids) != manifest["totals"]["clips"]:
        raise ValueError("Invalid manifest clip count or duplicate clip ID")
    videos = {split_clip_id(clip_id)[0] for clip_id in ids}
    if len(videos) != 1:
        raise ValueError("Expected a single video in the frozen benchmark")
    assignments = _splits(manifest, set(ids))
    clips = []
    for row in sorted(rows, key=lambda item: item["clip_id"]):
        clip_id = row["clip_id"]
        try:
            clip = _read_clip(files, row, assignments[clip_id], dataset_id)
            counts = _event_counts((clip,))
            if any(row["ball_events"][key] != value for key, value in counts.items()):
                raise ValueError(
                    "Manifest per-clip event counts disagree with the scene"
                )
        except (KeyError, TypeError, ValueError, BadZipFile, EOFError) as error:
            raise ValueError(f"{clip_id}: {error}") from error
        clips.append(clip)
    result = tuple(clips)
    if manifest["ball_events"] != _event_counts(result):
        raise ValueError("Manifest aggregate event counts disagree with the scenes")
    return V4Release(
        benchmark_id,
        dataset_id,
        next(iter(videos)),
        files.expected(Path("manifest.json")),
        sums_hash,
        len(files.verified),
        len(files.digests),
        result,
    )
