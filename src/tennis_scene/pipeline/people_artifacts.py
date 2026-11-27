"""Verify the declared full-frame detector and top-down pose observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.submodules.models.rtmpose.pose2d import SCORE_KIND
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesModule,
)
from src.tennis_scene.pipeline.model_assets import (
    DetectorName,
    PoseName,
    people_model_names,
)
from src.tennis_scene.pipeline.storage.scene_index import read_component_descriptor


def _declared_model_settings(
    observation: Mapping[str, Any], role: str, architecture: str, *, legacy: bool, camera: str,
) -> Mapping[str, Any]:
    models = observation.get("models", {})
    if models == {} and legacy:
        # Observations saved before model declarations still have a legacy selector contract.
        return {}
    model = models[role] if isinstance(models, Mapping) and set(models) == {role} else None
    if not isinstance(model, Mapping) or model.get("architecture") != architecture:
        raise ValueError(f"Saved {role} model declaration must match {architecture}: {camera}")
    settings = model.get("settings", {})
    if not isinstance(settings, Mapping):
        raise ValueError(f"Invalid {role} model declaration settings: {camera}")
    return settings


def stored_people_models(detection: Any, pose: Any, *, camera: str,
                         require_unmerged: bool) -> tuple[DetectorName, PoseName]:
    if not isinstance(detection, Mapping) or detection.get("enabled") is not True:
        raise ValueError(f"Saved DINO or RF-DETR-L observations require enabled=true: {camera}")
    if not isinstance(pose, Mapping) or pose.get("enabled") is not True:
        raise ValueError(f"Saved ViTPose or RTMPose-X observations require enabled=true: {camera}")
    if (detection.get("scope") != "full_frame" or type(detection.get("merge_duplicates")) is not bool
        or require_unmerged and detection["merge_duplicates"]):
        raise ValueError(f"Person observations require unmerged full-frame detections: {camera}" if require_unmerged
                         else f"Person observations require full-frame detections: {camera}")
    # Legacy observations may omit selectors, but any available model declaration must agree.
    values = {}
    if "detector" in detection:
        values["detector"] = detection["detector"]
    if "pose" in pose:
        values["pose"] = pose["pose"]
    detector, pose_model = people_model_names(values)
    legacy_detector, legacy_pose = detector == "dino", pose_model == "vitpose-h"
    _declared_model_settings(detection, "detector", "dino_4scale_swin_l" if legacy_detector else detector,
                             legacy=legacy_detector, camera=camera)
    pose_settings = _declared_model_settings(pose, "pose", "vitpose_huge_coco17" if legacy_pose else pose_model,
                                             legacy=legacy_pose, camera=camera)
    confidence = pose.get("joint_confidence"), pose_settings.get("joint_confidence")
    if pose_model == "rtmpose-x":
        if confidence != (SCORE_KIND, SCORE_KIND):
            raise ValueError(f"RTMPose-X observation requires declared raw SimCC confidence: {camera}")
    elif SCORE_KIND in confidence:
        raise ValueError(f"ViTPose-H observation cannot declare raw SimCC confidence: {camera}")
    return detector, pose_model


def require_current_people_artifacts(
    root: Path, document: Mapping[str, Any], camera_ids: Sequence[str], publication: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the selected model settings in the exported scene's lineage."""
    choices, models = publication["people_models"], publication["models"]
    if not isinstance(choices, Mapping) or set(choices) != {"detector", "pose"} or not isinstance(models, Mapping):
        raise ValueError("Publication requires the declared detector and pose models")
    selected = people_model_names(choices)
    component_settings = publication.get("people_component_settings")
    if not isinstance(component_settings, Mapping):
        raise ValueError("Publication requires the declared people component settings")
    required: dict[str, Any] = {}
    for camera in camera_ids:
        descriptors = []
        dependencies: dict[str, Any] = {}
        for component in (PersonDetectionModule, PersonPoseFeaturesModule):
            io = component.io
            node = f"{io.name}/{camera}"
            reference = document["artifacts"].get(node)
            if not isinstance(reference, dict) or (reference.get("schema"), reference.get("version")) != (
                io.output_schema, io.version,
            ):
                raise ValueError(f"Publication requires {node}: {io.output_schema} v{io.version}")
            descriptor = read_component_descriptor(root, reference, node=node, source_sha256=document["source_sha256"])
            identity = descriptor["identity"]
            settings = identity.get("settings")
            if (descriptor["provenance"].get("origin") != "component"
                or identity.get("implementation") != publication["code_sha256"]
                or identity.get("scope") != camera
                or identity.get("component_class") != f"{component.__module__}.{component.__qualname__}"
                or reference["execution_key"] != document_digest(identity)
                or not isinstance(settings, Mapping)
                or descriptor["dependencies"] != dependencies):
                raise ValueError(f"Published {node} differs from its declared people model or dependencies")
            expected, declared_models = component_settings.get(io.name), models.get(io.name)
            if (not isinstance(expected, Mapping) or not isinstance(declared_models, Mapping)
                or settings != {**expected, "models": declared_models}):
                raise ValueError(f"Published {node} differs from its declared people settings")
            descriptors.append(settings)
            required[node] = reference
            dependencies = {"detections": reference}
        if stored_people_models(*descriptors, camera=camera, require_unmerged=False) != selected:
            raise ValueError(f"Published people models differ from the configured selection: {camera}")
    return required
