"""Application boundary for the declared component runner and clip store."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tennis_scene.pipeline.artifacts import (
    document_digest,
    json_value,
    write_json_atomic,
)
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.direct_annotation import saved_scene, scene_receipt
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.imports import BallAnnotationImporter
from src.tennis_scene.pipeline.inference_runtime import ModelEventHook
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.video_court.diagnostics import (
    clip_report,
    drift_identity,
    load_video_drift,
)
from src.tennis_scene.pipeline.video_court.policy import apply_drift_policy
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.annotation.local_controller import LocalAnnotationController
    from src.tasks.base.annotation.paths import AnnotationLayout
    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact


class TennisSceneOrchestrator:
    def __init__(self, config: PipelineRuntimeConfig, *, components: dict[str, Any] | None = None,
                 annotation_layout: AnnotationLayout | None = None,
                 annotation_controller_factory: Callable[..., LocalAnnotationController] | None = None,
                 annotation_wait: Callable[[LocalAnnotationController], None] | None = None,
                 model_event: ModelEventHook | None = None) -> None:
        self.config, self.components = config, components
        self.annotation_layout = annotation_layout
        self.annotation_controller_factory, self.annotation_wait = annotation_controller_factory, annotation_wait
        self.model_event = model_event
        self.last_receipt: dict[str, Any] = {}
        self.last_runner: ComponentRunner | None = None
        self.last_store: ClipStore | None = None
        self.video_court: VideoCourtArtifact | None = None
        self._video_court_key: tuple[object, object, str] | None = None
        source = PROJECT_ROOT / "src"
        self.code_identity = document_digest({str(path.relative_to(source)): dual_sha256(path) for path in sorted(source.rglob("*.py"))})

    @classmethod
    def from_runtime_config(cls, cfg: PipelineRuntimeConfig, *, model_event: ModelEventHook | None = None) -> TennisSceneOrchestrator:
        return cls(cfg, model_event=model_event)

    @classmethod
    def from_config(cls, cfg: DictConfig, *, model_event: ModelEventHook | None = None) -> TennisSceneOrchestrator:
        from src.tennis_scene.configuration import PipelineRuntimeConfig
        return cls.from_runtime_config(PipelineRuntimeConfig.from_config(cfg), model_event=model_event)

    def publication_identity(self) -> dict[str, Any]:
        cfg = self.config
        shared = self.prepare_video_court()
        return {"schema": "declared_component_pipeline_v1", "code_sha256": self.code_identity,
            "settings": json_value(cfg.processing_settings), "execution": dict(cfg.component_sources),
            "checkpoints": {key: cfg.court_kp.asset_catalog.file_identity(path) for key, path in enabled_model_assets(cfg).items()},
            **({"court_model_assets": CourtKPModule(cfg.court_kp).asset_identities()}
               if cfg.direct_annotation is None and cfg.video_court is None else {}),
            **({"court_model": CourtKPModule(cfg.court_kp).model_identity()}
               if cfg.direct_annotation is None and cfg.video_court is None else {}),
            **({'video_court_cameras': shared.manifest['cameras'],
                'video_court_unobservable_policy': cfg.video_court_unobservable_policy,
                'video_court_drift_cameras': drift_identity(shared).get('camera_digests', {})} if shared is not None else {})}

    def prepare_video_court(self) -> VideoCourtArtifact | None:
        from src.tennis_scene.pipeline.video_court import prepare_video_court
        if self.config.video_court is None:
            return None
        key = (self.config.video_court, self.config.video_court_drift, self.code_identity)
        if self.video_court is None or self._video_court_key != key:
            self.video_court = prepare_video_court(self.config.video_court, self.config.resolver, code_identity=self.code_identity)
            if self.config.video_court_drift is not None:
                self.video_court = load_video_drift(self.video_court, self.config.video_court_drift)
            self._video_court_key = key
        self.video_court.verify()
        return self.video_court

    def run(self, video_paths: Sequence[Path], *, video_role: PathRole, camera_ids: Sequence[str],
            store_root: Path | None, max_frames: int | None = None, clip_id: str | None = None) -> SceneResult:
        """Run the standard recipe into an explicitly chosen component store.

        ``store_root`` is the clip-owned store (structured datasets keep it at
        ``<clip>/annotations/tennis_scene``); ``None`` selects
        ``cache.directory/<source digest>``. The store is never inferred from
        the shape of the video paths.
        """
        paths = tuple(self.config.resolver.validate(video_role, Path(p)) for p in video_paths)
        inputs = prepare_clip_source(paths, camera_ids, max_frames=max_frames, clip_id=clip_id or self.config.clip_id)
        source = inputs.source
        if self.config.camera_geometry.reference_camera is not None and self.config.camera_geometry.reference_camera not in source.camera_ids:
            raise ValueError("Reference camera is not a source camera")
        source_document = json_value(source)
        root = self.config.cache_directory / document_digest(source_document)[:20] if store_root is None else store_root
        store = ClipStore(root, source_document)
        runner = None
        self.last_runner, self.last_store = None, store
        self.last_receipt = {"schema": "tennis_scene_run_v3", "status": "running", "clip_id": source.clip_id,
            "source": source_document, "scene_index": str(store.index_path)}
        try:
            shared = self.prepare_video_court()
            if shared is not None:
                shared.bind(source)
                self.last_receipt['video_court'] = {'directory': str(shared.root), 'artifact_digest': shared.digest,
                                                   'camera_digests': shared.manifest['cameras'],
                                                   'drift_artifact': drift_identity(shared),
                                                   'unobservable_policy': self.config.video_court_unobservable_policy,
                                                   'clip_diagnostics': json_value({camera: apply_drift_policy(
                                                       clip_report(shared, camera, source.clip_id), self.config.video_court_unobservable_policy,
                                                       common_status=shared.cameras[camera]['status'])
                                                                                  for camera in source.camera_ids})}
            direct = self.config.direct_annotation
            workflow = None
            if direct is not None:
                from src.tasks.base.annotation.paths import AnnotationLayout
                workflow = direct.workflow(source, self.annotation_layout or AnnotationLayout.production())
                self.last_receipt.update(
                    annotation={"status": "pending", "request": workflow.reference.model_dump(mode="json")},
                    scene={"status": "pending"},
                    dataset={"status": "pending" if direct.publish_dataset else "not_requested"},
                    learning={"status": "not_requested"},
                )
                if direct.resume == "publication":
                    completed = workflow.saved()
                else:
                    nodes = standard_definition(self.config, source, code_identity=self.code_identity, overrides=self.components)
                    fixed = direct.geometry()
                    nodes = fixed.bind_nodes(nodes)
                    fixed.publish(nodes, store, source)
                    options = {} if self.annotation_controller_factory is None else {"controller_factory": self.annotation_controller_factory}
                    completed = workflow.annotate(wait=self.annotation_wait, **options)
                self.last_receipt["annotation"] = {"status": "complete", "receipt": str(workflow.directory / "annotations.json"),
                    "completion": completed.model_dump(mode="json")}
                from src.tasks.ball_detection.annotation.video import read_annotation
                from src.tennis_scene.evaluation.annotation_acceptance import (
                    summarize_video_annotations,
                )
                annotation_paths, expected_inputs = workflow.inputs(completed)
                counts = summarize_video_annotations(
                    [read_annotation(p, expected_input=expected_inputs[c], require_complete=True) for c, p in annotation_paths.items()],
                    expected_inputs=tuple(expected_inputs.values()),
                )
                write_json_atomic(store.root / "annotation-summary.json", counts)
                self.last_receipt["annotation"]["summary"] = str(store.root / "annotation-summary.json")
                write_json_atomic(store.root / "run.json", self.last_receipt)
                if direct.resume == "publication":
                    scene, saved = saved_scene(store, direct, workflow.reference.digest)
                    self.last_receipt["scene"] = saved
                else:
                    annotations, expected = workflow.inputs(completed)
                    importer = BallAnnotationImporter(annotations, expected)
                    nodes = importer.bind_nodes(nodes)
                    importer.publish(nodes, store, source)
            else:
                nodes = standard_definition(self.config, source, code_identity=self.code_identity, overrides=self.components, video_court=shared)
            if direct is None or direct.resume != "publication":
                runner = ComponentRunner(nodes, store, overwrite=self.config.cache_overwrite, model_event=self.model_event)
                self.last_runner = runner
                runner.run()
                inputs.snapshot.assert_unchanged()
                if shared is not None:
                    shared.verify()
                scene, archive = runner.finalize_clip()
                if direct is not None and workflow is not None:
                    self.last_receipt["scene"] = scene_receipt(store, direct, workflow.reference.digest, archive)
            if workflow is not None:
                write_json_atomic(store.root / "run.json", self.last_receipt)
                try:
                    published = workflow.publish()
                    self.last_receipt["dataset"] = {"status": "not_requested" if published is None else "complete",
                        "receipt": None if published is None else published.model_dump(mode="json")}
                except Exception as exc:
                    self.last_receipt["dataset"] = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
                    raise
            self.last_receipt.update(status=scene.metadata["status"], validity=scene.metadata["validity_statistics"])
            return scene
        except Exception as exc:
            self.last_receipt.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            for stage in ("annotation", "scene"):
                if self.last_receipt.get(stage, {}).get("status") == "pending":
                    self.last_receipt[stage].update(status="incomplete", reason=str(exc))
            if isinstance(exc, ReconstructionUnavailable):
                self.last_receipt.update(error_reason=exc.reason, error_diagnostics=json_value(exc.diagnostics))
            raise
        finally:
            if runner is not None:
                self.last_receipt.update(active_stage=runner.active_node, stage_status=runner.statuses,
                    stage_seconds=runner.seconds, artifacts=json_value(runner.references))
            write_json_atomic(store.root / "run.json", self.last_receipt)
