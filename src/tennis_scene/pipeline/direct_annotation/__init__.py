"""Pinned direct-annotation settings and independent scene/publication receipts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.body_artifacts import require_current_body_artifacts
from src.tennis_scene.pipeline.components.event_detection import EVENT_DETECTION
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports import (
    BallAnnotationImporter,
    FixedGeometryImporter,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256

if TYPE_CHECKING:
    from src.submodules.configuration import InferencePrecision, SubmoduleRuntimeConfig
    from src.tasks.annotation.scene_workflow import SceneAnnotationWorkflow
    from src.tasks.base.annotation.paths import AnnotationLayout


@dataclass(frozen=True)
class DirectAnnotationConfig:
    request: Path
    request_sha256: str
    bundle: Path
    bundle_sha256: str
    court_detection: Literal["stored_artifact", "verified_calibration_frame0"]
    publish_dataset: bool
    resume: Literal["workflow", "publication"]

    def workflow(
        self, source: ClipSource, layout: AnnotationLayout
    ) -> SceneAnnotationWorkflow:
        from src.tasks.annotation.scene_workflow import (
            SceneAnnotationRequest,
            SceneAnnotationWorkflow,
        )

        if dual_sha256(self.request) != self.request_sha256:
            raise ValueError("Scene annotation request checksum mismatch")
        request = SceneAnnotationRequest.model_validate_json(self.request.read_bytes())
        if request.configuration.publish_dataset != self.publish_dataset:
            raise ValueError(
                "Dataset publication selection differs from the pinned request"
            )
        videos = request.configuration.videos
        by_camera = {v.annotation.source.camera_id: v for v in videos}
        if len(by_camera) != len(videos) or set(by_camera) != set(source.camera_ids):
            raise ValueError(
                "Scene annotation request must contain exactly one input per source camera"
            )
        expected = {c: v.annotation for c, v in by_camera.items()}
        # Validate the caller's pinned camera/frame identities before any worker starts.
        BallAnnotationImporter(
            {c: Path(v.path) for c, v in by_camera.items()}, expected
        ).validate_source(source)
        if any(
            Path(by_camera[v.camera_id].path).resolve() != v.path for v in source.videos
        ):
            raise ValueError("Scene and annotation original video paths differ")
        return SceneAnnotationWorkflow(layout, request)

    def geometry(self) -> FixedGeometryImporter:
        return FixedGeometryImporter(
            self.bundle, self.bundle_sha256, self.court_detection
        )


def _person_precision(runtime: SubmoduleRuntimeConfig) -> dict[str, InferencePrecision]:
    return {
        "dino_detector": runtime.dino_detector.precision,
        "vitpose": runtime.vitpose.precision,
    }


def scene_receipt(
    store: ClipStore,
    config: DirectAnnotationConfig,
    request_digest: str,
    archive: Path,
    *,
    people_runtime: SubmoduleRuntimeConfig,
    model_identity: str | None = None,
    event_settings: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    node = "scene_assembly" if event_settings is None else EVENT_DETECTION
    result: dict[str, object] = {
        "status": "complete",
        "source_sha256": store.source_key,
        "request_sha256": request_digest,
        "bundle_sha256": config.bundle_sha256,
        "court_detection": config.court_detection,
        "person_precision": _person_precision(people_runtime),
        **({"model_identity": model_identity} if model_identity is not None else {}),
        "scene_artifact": json_value(store.references()[node]),
        **({EVENT_DETECTION: event_settings} if event_settings is not None else {}),
        "archive": str(archive),
        "archive_sha256": dual_sha256(archive),
        "metadata_sha256": dual_sha256(archive.with_suffix(".metadata.json")),
    }
    write_json_atomic(store.root / "scene-completion.json", result)
    return result


def saved_scene(
    store: ClipStore,
    config: DirectAnnotationConfig,
    request_digest: str,
    *,
    people_runtime: SubmoduleRuntimeConfig,
    model_identity: str | None = None,
    event_settings: Mapping[str, Any] | None = None,
) -> tuple[SceneResult, dict[str, object]]:
    receipt = json.loads((store.root / "scene-completion.json").read_text())
    references = store.references()
    node = "scene_assembly" if event_settings is None else EVENT_DETECTION
    if (
        receipt["status"] != "complete"
        or receipt["source_sha256"] != store.source_key
        or receipt["request_sha256"] != request_digest
        or receipt["bundle_sha256"] != config.bundle_sha256
        or receipt["court_detection"] != config.court_detection
        or receipt.get(EVENT_DETECTION) != event_settings
        or document_digest(receipt["scene_artifact"])
        != document_digest(json_value(references.get(node)))
    ):
        raise ValueError("Saved scene completion differs from the selected workflow")
    if "person_precision" in receipt:
        saved_precision = receipt["person_precision"]
    else:
        # Receipts written before precision was configurable used FP32 for both models.
        saved_precision = {"dino_detector": "fp32", "vitpose": "fp32"}
    requested_precision = _person_precision(people_runtime)
    if saved_precision != requested_precision:
        raise ValueError(
            "Saved scene person precision differs from the selected workflow "
            f"(saved {saved_precision!r}, requested {requested_precision!r})"
        )
    if receipt.get("model_identity") != model_identity:
        raise ValueError("Saved scene model identity differs from the selected workflow")
    archive = Path(receipt["archive"])
    required = require_current_body_artifacts(store.root, json_value(references["scene_assembly"]))
    required[node] = json_value(references[node])
    if indexed_scene_path(store.root / 'scene.json', required_components=required) != archive.resolve():
        raise ValueError('Saved scene archive differs from its current component export')
    if (
        dual_sha256(archive) != receipt["archive_sha256"]
        or dual_sha256(archive.with_suffix(".metadata.json"))
        != receipt["metadata_sha256"]
    ):
        raise ValueError("Saved scene export changed")
    return load_scene_result(archive), receipt
