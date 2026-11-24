"""Read one immutable input revision and publish its derived tracking artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from src.tennis_scene.multiview_pose import schema as frame_schema
from src.tennis_scene.multiview_tracking.schema import (
    NODE,
    SCHEMA,
    VERSION,
    PoseTracks,
    TrackingConfig,
)
from src.tennis_scene.multiview_tracking.selection import (
    COURT_HALF_X_M,
    COURT_HALF_Y_M,
    MIN_RESIDENCE_FRACTION,
)
from src.tennis_scene.multiview_tracking.tracking import track
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.source_identity import source_digests
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor

_Output = TypeVar("_Output", frame_schema.FramePoseCandidates, PoseTracks)


def _read(
    directory: Path,
    node: str,
    schema: str,
    version: int,
    output_type: type[_Output],
    *,
    expected_artifact_id: str | None = None,
) -> tuple[ClipSource, ArtifactRef, _Output]:
    root = directory.resolve()
    index = json.loads((root / "scene.json").read_text())
    if (
        index.get("schema") != "tennis_scene_index_v1"
        or document_digest(index["source"]) != index["source_sha256"]
    ):
        raise ValueError("Invalid clip store source")
    record = index["artifacts"].get(node)
    if record is None or (record["schema"], record["version"]) != (schema, version):
        raise ValueError(f"Missing {schema} v{version} in {root}")
    ref = ArtifactRef(**record)
    if expected_artifact_id is not None and ref.artifact_id != expected_artifact_id:
        raise ValueError(f"The active {node} differs from the pinned artifact ID")
    descriptor = read_component_descriptor(
        root, record, node=node, source_sha256=index["source_sha256"]
    )
    if (
        descriptor.get("status") != "complete"
        or descriptor.get("execution_key") != ref.execution_key
    ):
        raise ValueError(f"Incomplete or inconsistent artifact: {node}")
    value = ArtifactCodec(output_type).load(
        descriptor["payload"], (root / ref.path).parent, descriptor["arrays"]
    )
    source: ClipSource = restore_type(index["source"], ClipSource)
    if (
        value.camera_ids != source.camera_ids
        or len(value.frame_offsets) != source.num_frames + 1
    ):
        raise ValueError(f"{node} and its source have different camera/frame axes")
    return source, ref, value


def read_candidate_store(
    directory: Path, *, expected_artifact_id: str | None = None
) -> tuple[ClipSource, ArtifactRef, frame_schema.FramePoseCandidates]:
    return _read(
        directory,
        frame_schema.NODE,
        frame_schema.SCHEMA,
        frame_schema.VERSION,
        frame_schema.FramePoseCandidates,
        expected_artifact_id=expected_artifact_id,
    )


def read_tracks(
    directory: Path, *, expected_artifact_id: str | None = None
) -> PoseTracks:
    _, _, value = _read(
        directory,
        NODE,
        SCHEMA,
        VERSION,
        PoseTracks,
        expected_artifact_id=expected_artifact_id,
    )
    return value


def track_store(
    candidates: Path,
    output: Path,
    config: TrackingConfig,
    *,
    expected_artifact_id: str | None = None,
) -> ArtifactRef:
    origin, destination = candidates.resolve(), output.resolve()
    if origin.is_relative_to(destination) or destination.is_relative_to(origin):
        raise ValueError("Tracking requires a separate output clip store")
    source, ref, frame_poses = read_candidate_store(
        origin, expected_artifact_id=expected_artifact_id
    )
    tracks = track(frame_poses, fps=source.fps, config=config)
    store = ClipStore(destination, json_value(source))
    if store.references().keys() - {NODE}:
        raise ValueError(
            "Tracking output store already contains another recipe's artifacts"
        )
    # The frame pose producer is an independent clip store, so pin it in provenance.
    provenance = {
        "frame_pose_candidates": json_value(ref),
        "config": json_value(config),
        "selection_region": {
            "half_x_m": COURT_HALF_X_M,
            "half_y_m": COURT_HALF_Y_M,
            "min_residence_fraction": MIN_RESIDENCE_FRACTION,
        },
    }
    identity = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_sha256": store.source_key,
        "implementation": document_digest(source_digests(Path(__file__).parent)),
        **provenance,
    }
    return store.publish(
        NODE,
        tracks,
        ArtifactCodec(PoseTracks),
        schema=SCHEMA,
        version=VERSION,
        identity=identity,
        dependencies={},
        provenance=provenance,
    )
