"""Pinned direct-annotation settings and independent scene/publication receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports import (
    BallAnnotationImporter,
    FixedGeometryImporter,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256

if TYPE_CHECKING:
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


def scene_receipt(
    store: ClipStore, config: DirectAnnotationConfig, request_digest: str, archive: Path
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "complete",
        "source_sha256": store.source_key,
        "request_sha256": request_digest,
        "bundle_sha256": config.bundle_sha256,
        "court_detection": config.court_detection,
        "scene_artifact": json_value(store.references()["scene_assembly"]),
        "archive": str(archive),
        "archive_sha256": dual_sha256(archive),
        "metadata_sha256": dual_sha256(archive.with_suffix(".metadata.json")),
    }
    write_json_atomic(store.root / "scene-completion.json", result)
    return result


def saved_scene(
    store: ClipStore, config: DirectAnnotationConfig, request_digest: str
) -> tuple[SceneResult, dict[str, object]]:
    receipt = json.loads((store.root / "scene-completion.json").read_text())
    if (
        receipt["status"] != "complete"
        or receipt["source_sha256"] != store.source_key
        or receipt["request_sha256"] != request_digest
        or receipt["bundle_sha256"] != config.bundle_sha256
        or receipt["court_detection"] != config.court_detection
        or document_digest(receipt["scene_artifact"])
        != document_digest(json_value(store.references()["scene_assembly"]))
    ):
        raise ValueError("Saved scene completion differs from the selected workflow")
    archive = Path(receipt["archive"])
    if (
        dual_sha256(archive) != receipt["archive_sha256"]
        or dual_sha256(archive.with_suffix(".metadata.json"))
        != receipt["metadata_sha256"]
    ):
        raise ValueError("Saved scene export changed")
    return load_scene_result(archive), receipt
