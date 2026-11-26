"""Publish and read versioned frame candidates in an independent clip store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.tennis_scene.multiview_pose.inputs import SavedPoseInputs
from src.tennis_scene.multiview_pose.schema import (
    NODE,
    SCHEMA,
    VERSION,
    FramePoseCandidates,
    PoseConfig,
)
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.source_identity import source_digests
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor


def publish_candidates(
    directory: Path,
    upstream: Path,
    inputs: SavedPoseInputs,
    candidates: FramePoseCandidates,
    config: PoseConfig,
) -> ArtifactRef:
    destination, origin = directory.resolve(), upstream.resolve()
    if destination.is_relative_to(origin) or origin.is_relative_to(destination):
        raise ValueError(
            "Pose output requires a separate store from the upstream observations"
        )
    source = inputs.observations.source
    if (
        candidates.camera_ids != source.camera_ids
        or len(candidates.frame_offsets) != source.num_frames + 1
        or any(
            not np.array_equal(a, d.frame_offsets)
            for a, d in zip(
                candidates.detection_offsets,
                inputs.observations.detections,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "Pose output and raw observations have different camera/row timelines"
        )
    store = ClipStore(destination, json_value(source))
    if store.references().keys() - {NODE}:
        raise ValueError(
            "Pose output store already contains another recipe's artifacts"
        )
    # External revision pins belong in provenance, since these producers live in a separate store.
    evidence = {
        "observations": json_value(inputs.references),
        "court_side": inputs.court_side,
        "config": json_value(config),
    }
    identity = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_sha256": store.source_key,
        "implementation": document_digest(source_digests(Path(__file__).parent)),
        **evidence,
    }
    ref: ArtifactRef = store.publish(
        NODE,
        candidates,
        ArtifactCodec(FramePoseCandidates),
        schema=SCHEMA,
        version=VERSION,
        identity=identity,
        dependencies={},
        provenance=evidence,
    )
    return ref


def read_candidates(directory: Path) -> FramePoseCandidates:
    root = directory.resolve()
    index = json.loads((root / "scene.json").read_text())
    if (
        index.get("schema") != "tennis_scene_index_v1"
        or document_digest(index["source"]) != index["source_sha256"]
    ):
        raise ValueError("Invalid frame pose store source")
    ref = index["artifacts"].get(NODE)
    if ref is None or (ref["schema"], ref["version"]) != (SCHEMA, VERSION):
        raise ValueError("Missing frame pose candidates v1")
    doc = read_component_descriptor(
        root, ref, node=NODE, source_sha256=index["source_sha256"]
    )
    result: FramePoseCandidates = ArtifactCodec(FramePoseCandidates).load(
        doc["payload"], (root / ref["path"]).parent, doc["arrays"]
    )
    source = restore_type(index["source"], ClipSource)
    if (
        result.camera_ids != source.camera_ids
        or len(result.frame_offsets) != source.num_frames + 1
    ):
        raise ValueError("Frame pose output and stored source timelines disagree")
    return result
