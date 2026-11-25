"""Verify the ECPose producer and its per-camera views for dataset publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.ecpose import (
    ECPoseDetectionsModule,
    ECPoseFeaturesModule,
    ECPoseModule,
)
from src.tennis_scene.pipeline.recipes import ECPOSE_RECIPE
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor


def require_current_ecpose_artifacts(
    root: Path,
    document: Mapping[str, Any],
    camera_ids: Sequence[str],
    publication: Mapping[str, object],
) -> dict[str, Any]:
    """Return the validated references that must appear in the scene's lineage."""
    models, overrides = (
        publication.get("models"),
        publication.get("component_overrides"),
    )
    if (
        not isinstance(models, Mapping)
        or not models.get(ECPOSE_RECIPE)
        or not isinstance(overrides, Mapping)
    ):
        raise ValueError("ECPose publication requires declared model settings")
    required: dict[str, Any] = {}

    def read(
        component: type[Any],
        camera: str,
        settings: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> dict[str, Any]:
        io = component.io
        name = f"{io.name}/{camera}"
        reference = document["artifacts"].get(name)
        if not isinstance(reference, dict) or (
            reference.get("schema"),
            reference.get("version"),
        ) != (io.output_schema, io.version):
            raise ValueError(
                f"ECPose publication requires {name}: {io.output_schema} v{io.version}"
            )
        descriptor = read_component_descriptor(
            root, reference, node=name, source_sha256=document["source_sha256"]
        )
        if descriptor["provenance"].get("origin") != "component":
            raise ValueError(f"Published {name} requires component origin")
        override = overrides.get(name)
        expected_class = (
            f"{component.__module__}.{component.__qualname__}"
            if override is None
            else override["override"]
        )
        identity = descriptor["identity"]
        if (
            identity.get("settings") != (settings if override is None else override)
            or identity.get("implementation") != publication["code_sha256"]
            or identity.get("component_class") != expected_class
            or identity.get("scope") != camera
            or reference["execution_key"] != document_digest(identity)
        ):
            raise ValueError(
                f"Published {name} differs from the declared ECPose execution settings"
            )
        if descriptor["dependencies"] != dependencies:
            raise ValueError(f"Published {name} has different ECPose dependencies")
        required[name] = reference
        return reference

    projection = {"recipe": ECPOSE_RECIPE}
    for camera in camera_ids:
        joint = read(
            ECPoseModule, camera, {**projection, "models": models[ECPOSE_RECIPE]}, {}
        )
        detections = read(ECPoseDetectionsModule, camera, projection, {"joint": joint})
        read(
            ECPoseFeaturesModule,
            camera,
            projection,
            {"joint": joint, "detections": detections},
        )
    return required
