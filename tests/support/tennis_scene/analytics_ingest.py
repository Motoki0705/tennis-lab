"""Small on-disk V4 inputs for the audit and PostgreSQL ingest tests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.configuration import PathResolver, RuntimePathRoots

CLASSES = ("shot", "bounce", "net", "unknown")
STATUSES = ("certain", "probable", "uncertain")
SUBTYPES = (
    "serve_overhead",
    "serve_underhand",
    "groundstroke_forehand",
    "groundstroke_backhand",
    "volley_forehand",
    "volley_backhand",
    "volley_center",
    "smash",
    "service_let",
    "net_in",
    "net_fault",
    "bounce",
    "unknown",
)


@dataclass(frozen=True)
class IngestFixture:
    root: Path
    projects: Path
    data_root: Path

    def cli_args(self) -> list[str]:
        return [
            "--v4-root",
            str(self.root),
            "--projects",
            str(self.projects),
            "--data-root",
            str(self.data_root),
        ]

    def resolver(self) -> PathResolver:
        return PathResolver(RuntimePathRoots(*(self.data_root,) * 7))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_sums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
    )


def _ref(path: Path, root: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def _event(
    source: str,
    frame: int,
    subtype: str,
    *,
    hint: str | None = None,
    status: str = "certain",
    in_point: bool = True,
) -> dict[str, Any]:
    kind = (
        "shot"
        if subtype in SUBTYPES[:8]
        else "bounce"
        if subtype == "bounce"
        else "net"
    )
    return {
        "id": f"edited-{source}",
        "source_event_id": source,
        "kind": kind,
        "subtype": subtype,
        "status": status,
        "frame": frame,
        "frame_uncertainty": 2,
        "old_kind": kind,
        "old_subtype": "serve" if "serve_" in subtype else None,
        "old_in_point": in_point,
        "player_hint": hint,
    }


def _event_counts(
    events: list[dict[str, Any]], excluded: int, gaps: int
) -> dict[str, Any]:
    counts: dict[str, Any] = {
        "events": len(events),
        "excluded_events": excluded,
        "unlabelable_frames": gaps,
    }
    for axis, keys in (("kind", CLASSES), ("status", STATUSES), ("subtype", SUBTYPES)):
        counter = Counter(event[axis] for event in events)
        counts[f"by_{axis}"] = {key: counter[key] for key in keys}
    return counts


def _create_clip(
    root: Path,
    name: str,
    start: float,
    events: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    *,
    end: float | None = None,
) -> dict[str, Any]:
    clip_id = f"video_000/{name}"
    directory = root / "videos/video_000/clips" / name
    scene_dir = directory / "scene"
    export_dir = scene_dir / "exports/adopted"
    component_path = (
        scene_dir / "components/event_detection/adopted/event_detection.json"
    )
    media = directory / "media/cam0.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture clip bytes")
    timebase = {"fps": 30.0, "num_frames": 36, "width": 30, "height": 20}
    if end is None:
        end = start + 1.2
    camera = {"camera_id": "cam0", "num_frames": 36, "fps": 30.0, **_ref(media, root)}
    clip_path = directory / "clip.json"
    write_json(
        clip_path,
        {
            **timebase,
            "version": 2,
            "dataset_id": "meiji_4cam",
            "clip_id": clip_id,
            "video_id": "video_000",
            "clip_name": name,
            "global_start_sec": start,
            "global_end_sec": end,
            "camera_ids": ["cam0"],
            "video_paths": ["media/cam0.mp4"],
            "cameras": [
                {
                    "camera_id": "cam0",
                    "video": "media/cam0.mp4",
                    "source_path": "/capture/tennis_multiview/raw/meiji_4cam/video_000/cam0.mp4",
                    "offset_sec": 1.25,
                    "source_fps": 30.0,
                    "source_frame_start": 0,
                    "source_frame_end": 35,
                    "letterbox": None,
                }
            ],
            "sync_source": "clip_studio",
            "exported_at": "2026-01-01T00:00:00+00:00",
        },
    )
    provenance = {
        "annotation_sha256": "a" * 64,
        "source_v1_annotation_sha256": "b" * 64,
        "code_commit": "0" * 40,
        "labeler": "reviewer",
        "labeler_original": "Reviewer",
    }
    court_side = {"near": -1, "far": 1, "camera_id": "cam0", "camera_center_y_m": -18}
    metadata = {
        "dataset_clip_id": clip_id,
        "camera_ids": ["cam0"],
        "scene_schema_version": 4,
        "ball_events": {
            "version": 2,
            "source": "annotation",
            "classes": list(CLASSES),
            "subtypes": list(SUBTYPES),
            "statuses": list(STATUSES),
            "provenance": provenance,
        },
    }
    metadata_path = export_dir / "scene.metadata.json"
    write_json(metadata_path, metadata)
    npz_path = export_dir / "scene.npz"
    coverage: NDArray[np.bool_] = np.ones(36, dtype=np.bool_)
    for interval in gaps:
        coverage[interval["start"] : interval["end"] + 1] = False
    arrays = {
        "ball_event_frame": np.array(
            [event["frame"] for event in events], dtype=np.int32
        ),
        "ball_event_kind": np.array(
            [CLASSES.index(event["kind"]) for event in events], dtype=np.uint8
        ),
        "ball_event_subtype": np.array(
            [SUBTYPES.index(event["subtype"]) for event in events], dtype=np.uint8
        ),
        "ball_event_status": np.array(
            [STATUSES.index(event["status"]) for event in events], dtype=np.uint8
        ),
        "ball_event_frame_uncertainty": np.array(
            [event["frame_uncertainty"] for event in events], dtype=np.uint8
        ),
        "ball_event_side": np.array(
            [court_side.get(event["player_hint"], 0) for event in events], dtype=np.int8
        ),
        "ball_event_coverage": coverage,
    }
    np.savez(
        npz_path,
        fps=timebase["fps"],
        num_frames=timebase["num_frames"],
        width=timebase["width"],
        height=timebase["height"],
        **arrays,
    )
    component_path.parent.mkdir(parents=True)
    descriptors, references = {}, {}
    for index, (key, array) in enumerate(arrays.items()):
        member = component_path.parent / f"array_{index:04d}.npy"
        np.save(member, array, allow_pickle=False)
        descriptors[member.name] = {
            "sha256": digest(member),
            "shape": list(array.shape),
            "dtype": array.dtype.name,
        }
        references[key] = {"array": member.name}
    source_sha = "c" * 64
    write_json(
        component_path,
        {
            "schema": "component_artifact_v1",
            "node": "event_detection",
            "status": "complete",
            "output_schema": "scene_result",
            "output_version": 4,
            "artifact_id": "d" * 64,
            "execution_key": "e" * 64,
            "source_sha256": source_sha,
            "arrays": descriptors,
            "payload": {
                **timebase,
                **references,
                "metadata": metadata,
                "event_evidence": {
                    "events": list(reversed(events)),
                    "excluded_events": excluded,
                    "unlabelable": gaps,
                    "player_hint_reference": court_side,
                },
            },
        },
    )
    component_ref = {
        **_ref(component_path, scene_dir),
        "artifact_id": "d" * 64,
        "execution_key": "e" * 64,
        "schema": "scene_result",
        "version": 4,
    }
    index_path = scene_dir / "scene.json"
    write_json(
        index_path,
        {
            "schema": "tennis_scene_index_v1",
            "revision": 4,
            "source_sha256": source_sha,
            "source": {"clip_id": clip_id, "videos": [camera]},
            "exports": {
                "scene": {
                    "files": {
                        "metadata": _ref(metadata_path, scene_dir),
                        "scene": _ref(npz_path, scene_dir),
                    },
                    "inputs": {"event_detection": component_ref},
                }
            },
            "artifacts": {"event_detection": component_ref},
        },
    )
    return {
        "clip_id": clip_id,
        "fps": 30.0,
        "frames": 36,
        "clip_json": _ref(clip_path, root),
        "scene": {**_ref(index_path, root), "source_sha256": source_sha, "revision": 4},
        "cameras": [camera],
        "exports": {
            "scene/metadata": _ref(metadata_path, root),
            "scene/scene": _ref(npz_path, root),
        },
        "ball_events": {
            **_event_counts(events, len(excluded), int((~coverage).sum())),
            **provenance,
            "component": component_ref,
            "annotation_file": f"{name}_ball_events.json",
            "court_side": court_side,
        },
    }


def make_ingest_fixture(data_root: Path, *, overlapping: bool = False) -> IngestFixture:
    root = data_root / "frozen"
    times = [10.0, 20.0, 30.0]
    names = ["clip_009", "clip_012", "clip_030"]
    point_events = [
        [
            _event("e001", 4, "serve_overhead", hint="near"),
            _event("e002", 6, "service_let"),
            _event("e003", 7, "bounce", in_point=False),
            _event("e004", 15, "serve_overhead", hint="near"),
            _event("e005", 18, "bounce"),
            _event("e006", 21, "groundstroke_forehand", hint="far"),
            _event("e007", 26, "net_fault"),
        ],
        [
            _event("e090", 4, "serve_underhand", hint="far"),
            _event("e091", 10, "bounce"),
            _event("e095", 18, "groundstroke_backhand", hint="near"),
            _event("e096", 23, "bounce", status="probable"),
        ],
        [
            _event("e201", 3, "serve_underhand", hint="near"),
            _event("e202", 5, "service_let"),
            _event("e203", 8, "bounce"),
            _event("e204", 11, "serve_underhand", hint="near"),
            _event("e205", 13, "net_fault"),
            _event("e206", 20, "serve_underhand", hint="near"),
            _event("e207", 23, "net_fault"),
        ],
    ]
    excluded = [
        [
            {
                "id": "e008",
                "source_event_id": "e008",
                "frame": 28,
                "old_kind": "bounce",
                "old_subtype": None,
                "old_in_point": True,
                "reason": "bounce after terminal net",
            }
        ],
        [],
        [],
    ]
    unlabelable = [
        [{"start": 33, "end": 34, "reason": "occlusion after terminal contact"}],
        [],
        [],
    ]
    before_points, winners = [[0, 0], [1, 0], [1, 1]], ["A", "B", "B"]
    if overlapping:
        names.append("clip_010")
        times.append(10.0 + 1 / 30)
        point_events.append(
            [{**row, "frame": row["frame"] - 1} for row in point_events[0]]
        )
        excluded.append([])
        unlabelable.append([])
        before_points.append([0, 0])
        winners.append("A")
    clips, project_clips = [], []
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for i, (name, start, events, omitted, gaps) in enumerate(
        zip(names, times, point_events, excluded, unlabelable, strict=True)
    ):
        clips.append(_create_clip(root, name, start, events, omitted, gaps))
        splits[("train", "val", "test")[i % 3]].append(f"video_000/{name}")
        project_clips.append(
            {
                "name": name,
                "start_sec": start,
                "end_sec": start + 1.2,
                "score": {
                    "status": "confirmed",
                    "before_source": "anchor" if i == 0 else "chain",
                    "server": "A",
                    "server_source": "auto",
                    "winner": winners[i],
                    "before": {"games": [0, 0], "points": before_points[i]},
                    "note": "free text about an out ball" if i == 1 else "",
                },
            }
        )
    write_json(
        root / "manifest.json",
        {
            "schema": "tennis_multiview_benchmark_v1",
            "benchmark_id": "meiji_4cam/multiview-v4",
            "scene_schema_version": 4,
            "clips": clips,
            "totals": {"clips": len(clips)},
            "splits": {
                key: {"clip_ids": value, "clips": len(value)}
                for key, value in splits.items()
            },
            "ball_events": _event_counts(
                [event for events in point_events for event in events],
                1,
                2,
            ),
        },
    )
    write_sums(root)
    projects = data_root / "projects.json"
    write_json(
        projects,
        {
            "version": 2,
            "dataset_id": "meiji_4cam",
            "projects": {
                "video_000": {
                    "match": {
                        "sides": {"A": "Side A", "B": "Side B"},
                        "first_server": "A",
                        "rules": {"type": "soft_tennis", "match_games": 3},
                    },
                    "sources": [
                        {
                            "path": "tennis_multiview/raw/meiji_4cam/video_000/cam0.mp4",
                            "camera_id": "cam0",
                            "offset_sec": 1.25,
                        }
                    ],
                    "clips": list(reversed(project_clips)),
                }
            },
        },
    )
    return IngestFixture(root, projects, data_root)
