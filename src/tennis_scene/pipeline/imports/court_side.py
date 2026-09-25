"""Publish explicitly confirmed sides, with no synthetic model probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_association import CourtSideOutput
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


def import_confirmed_side(node: ComponentNode, store: ClipStore, half_turns: tuple[bool, ...], *, confirmation: Mapping[str, Any]) -> ArtifactRef:
    if node.io.output_schema != "court_side" or node.source != "load":
        raise ValueError("Confirmed sides require an explicitly load-only court_side node")
    dependencies: dict[str, ArtifactRef] = {}
    for port, producer in node.bindings.items():
        ref = store.active(producer)
        if ref is None:
            raise ValueError(f"Cannot bind confirmed sides before {producer} exists")
        dependencies[port] = ref
    calibration = store.load(dependencies["calibration"], ArtifactCodec(CourtCalibrationOutput))
    value = CourtSideOutput(calibration.calibration.camera_ids, calibration.reference_camera, half_turns, None)
    return store.publish(node.name, value, ArtifactCodec(CourtSideOutput), schema=node.io.output_schema,
        version=node.io.version, identity={"importer": "confirmed_court_side", "version": 1, "source_sha256": store.source_key,
            "half_turns": list(half_turns), "confirmation": dict(confirmation)}, dependencies=dependencies,
        provenance={"origin": "confirmed_side", "confirmation": dict(confirmation), "model_inference": False})
