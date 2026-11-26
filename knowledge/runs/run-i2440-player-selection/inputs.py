"""Shared readers and fixed conditions for the player-selection comparison."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, unpack_value
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_SINGLES_WIDTH

ValueT = TypeVar("ValueT")
STORE = "annotations/tennis_scene/scene.json"
DEV_CLIPS = (
    "video_000/clip_000",
    "video_000/clip_007",
    "video_001/clip_001",
    "video_002/clip_013",
)
SELECTION = {
    "baseline": LinkingConfig(core_half_width_m=HALF_SINGLES_WIDTH, min_presence_fraction=0.25),
    "wide": LinkingConfig(core_half_width_m=HALF_DOUBLES_WIDTH, min_presence_fraction=0.25),
    "short": LinkingConfig(core_half_width_m=HALF_SINGLES_WIDTH, min_presence_fraction=0.125),
    "wide_short": LinkingConfig(core_half_width_m=HALF_DOUBLES_WIDTH, min_presence_fraction=0.125),
}


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_bytes()))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    index: Path
    status: str
    reason: str | None
    error: str | None


class SavedRun:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.input_path = self.root / "input.json"
        (self.progress_path,) = (self.root / "all/progress").glob("*/progress.json")
        self.outcome_path = self.root / "all/outcome.json"
        records = read_json(self.input_path)["clips"]
        self.progress = read_json(self.progress_path)
        outcomes = read_json(self.outcome_path)["statuses"]
        clips = self.progress["clips"]
        if (
            self.progress["dataset_id"] != "meiji_4cam"
            or self.progress["video_id"] != "video_000"
            or len(records) != 69
            or {record["clip_id"] for record in records} != set(clips)
            or {key: value["status"] for key, value in clips.items()} != outcomes
        ):
            raise ValueError("Input, progress and outcome must describe the same 69 clips")
        self.records = [
            ClipRecord(
                record["clip_id"],
                self.root / "inputs/dataset" / record["path"] / STORE,
                outcomes[record["clip_id"]],
                clips[record["clip_id"]].get("reason"),
                clips[record["clip_id"]].get("error"),
            )
            for record in records
        ]

    def receipt(self) -> dict[str, str]:
        paths = (self.input_path, self.progress_path, self.outcome_path)
        return {path.relative_to(self.root).as_posix(): dual_sha256(path) for path in paths}

    def select(self, clip_ids: list[str] | None) -> Iterator[ClipRecord]:
        if clip_ids and set(clip_ids) - {record.clip_id for record in self.records}:
            raise ValueError("Replay requested unknown clip IDs")
        for record in self.records:
            if not clip_ids or record.clip_id in clip_ids:
                yield record


def read_store(record: ClipRecord) -> StoreReader:
    store = StoreReader(record.index)
    if store.source["clip_id"] != record.clip_id or len(store.camera_ids) != 4:
        raise ValueError(f"{record.clip_id}: unexpected scene index")
    return store


def source_from_index(store: StoreReader) -> ClipSource:
    return ClipSource(
        store.source["clip_id"],
        tuple(
            SourceVideo(**{**video, "path": Path(video["path"])}) for video in store.source["videos"]
        ),
    )


def read_artifact(
    store: StoreReader,
    node: str,
    schema: str,
    version: int,
    output: type[ValueT],
) -> ValueT:
    ref = store.references[node]
    if (ref["schema"], ref["version"]) != (schema, version):
        raise ValueError(f"{store.index_path}: wrong {node} schema/version")
    desc = store.descriptor(node)
    return cast(ValueT, ArtifactCodec(output).load(
        desc["payload"], (store.root / ref["path"]).parent, desc["arrays"]
    ))


def read_fields(
    store: StoreReader, node: str, schema: str, version: int, *fields: str
) -> dict[str, Any]:
    ref = store.references[node]
    if (ref["schema"], ref["version"]) != (schema, version):
        raise ValueError(f"{store.index_path}: wrong {node} schema/version")
    desc = store.descriptor(node)
    directory = (store.root / ref["path"]).parent
    result = {}
    for field in fields:
        value = desc["payload"]
        for key in field.split("."):
            value = value[key]
        result[field] = unpack_value(value, directory, desc["arrays"])
    return result
