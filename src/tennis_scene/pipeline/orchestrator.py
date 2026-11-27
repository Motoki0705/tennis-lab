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
from src.tennis_scene.pipeline.clip_inputs import InputSnapshot, prepare_clip_source
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    EventDetectionModule,
)
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesModule,
    appearance_feature_settings,
)
from src.tennis_scene.pipeline.definition import (
    binds_appearance_media,
    check_shared_side_execution,
    component_override_identity,
    enabled_model_assets,
    person_detection_settings,
    person_pose_feature_settings,
    publication_model_identities,
    standard_definition,
    validate_override_names,
)
from src.tennis_scene.pipeline.direct_annotation import saved_scene, scene_receipt
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.imports import BallAnnotationImporter
from src.tennis_scene.pipeline.inference_runtime import ModelEventHook
from src.tennis_scene.pipeline.model_assets import ModelAssetSnapshot
from src.tennis_scene.pipeline.recipes import STANDARD_RECIPE, recipe_directory
from src.tennis_scene.pipeline.runner import (
    ComponentNode,
    ComponentRunner,
    PlanExecutor,
)
from src.tennis_scene.pipeline.source_identity import (
    compatible_source_digests,
    source_digests,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.video_court.diagnostics import (
    clip_report,
    drift_identity,
    load_video_drift,
)
from src.tennis_scene.pipeline.video_court.policy import apply_drift_policy
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.annotation.local_controller import LocalAnnotationController
    from src.tasks.base.annotation.paths import AnnotationLayout
    from src.tennis_scene.configuration import PipelineRuntimeConfig
    from src.tennis_scene.pipeline.batching import SourceMemory
    from src.tennis_scene.pipeline.video_court.store import VideoCourtArtifact
    from src.tennis_scene.pipeline.video_court_side import VideoCourtSideArtifact


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
        self._scene_export: Path | None = None
        self._publication_assets: ModelAssetSnapshot | None = None
        self.video_court: VideoCourtArtifact | None = None
        self._video_court_key: tuple[object, object, str] | None = None
        self.video_court_side: VideoCourtSideArtifact | None = None
        self._video_court_side_key: tuple[object, ...] | None = None
        self.code_identity = document_digest(compatible_source_digests(
            source_digests(PROJECT_ROOT / "src"), ball_court_side=config.court_side_evidence == "ball",
            tensorrt=config.people.uses_tensorrt or (config.ecpose is not None and config.ecpose.engine is not None),
            standard_recipe=config.recipe == STANDARD_RECIPE, standard_people=config.ecpose is None,
            alternative_people=config.people.uses_alternative_models))

    @classmethod
    def from_runtime_config(cls, cfg: PipelineRuntimeConfig, *, model_event: ModelEventHook | None = None) -> TennisSceneOrchestrator:
        return cls(cfg, model_event=model_event)

    @classmethod
    def from_config(cls, cfg: DictConfig, *, model_event: ModelEventHook | None = None) -> TennisSceneOrchestrator:
        from src.tennis_scene.configuration import PipelineRuntimeConfig
        return cls.from_runtime_config(PipelineRuntimeConfig.from_config(cfg), model_event=model_event)

    def publication_identity(self) -> dict[str, Any]:
        cfg = self.config
        event_config = cfg.require_event_detection()
        check_shared_side_execution(cfg, self.components)
        validate_override_names(self.components or {}, cfg.recipe)
        overrides = {name: component_override_identity(cfg, component) for name, component in (self.components or {}).items()}
        shared = self.prepare_video_court()
        side = self.prepare_video_court_side(shared)
        settings = json_value(cfg.processing_settings)
        people = settings.get("people_models", {})
        people_runtime = people.get("runtime", {})
        # Omit selectors with legacy values and the dormant model settings.
        # Older saved configurations lack these fields entirely.
        if cfg.people.detector == "dino":
            people.pop("detector", None)
            people.pop("rfdetr_checkpoint", None)
            people_runtime.pop("rfdetr_l", None)
        else:
            people.pop("dino_checkpoint", None)
            people.pop("dino_repository", None)
            people_runtime.pop("dino_detector", None)
        if cfg.people.pose == "vitpose-h":
            people.pop("pose", None)
            people.pop("rtmpose_checkpoint", None)
            people_runtime.pop("rtmpose_x", None)
        else:
            people.pop("vitpose_checkpoint", None)
            people_runtime.pop("vitpose", None)
        for name in ("dino_detector", "vitpose", "hmr2", "clip_reid", "gvhmr"):
            options = people_runtime.get(name, {})
            if options.get("backend") == "pytorch":
                del options["backend"]
            if options.get("engines") == []:
                del options["engines"]
            if options.get("precision") == "fp32":
                del options["precision"]
        if people_runtime.get("dino_detector", {}).get("batch_size") == 1:
            del people_runtime["dino_detector"]["batch_size"]
        for name, default in (("clip_reid", 64), ("gvhmr", 1)):
            if people_runtime.get(name) == {"batch_size": default}:
                del people_runtime[name]
        event = event_config.settings()
        if event_config.annotation_directory is not None:
            event["annotation_directory"] = str(event_config.annotation_directory)
            event["annotation_benchmark_manifest"] = str(event_config.annotation_benchmark_manifest)
        identity = {"schema": "declared_component_pipeline_v1", "code_sha256": self.code_identity,
            "settings": settings, "execution": dict(cfg.component_sources),
            EVENT_DETECTION: event,
            "checkpoints": {key: cfg.court_kp.asset_catalog.file_identity(path) for key, path in enabled_model_assets(cfg).items()},
            "models": publication_model_identities(cfg),
            "component_overrides": overrides,
            **({"people_models": {"detector": cfg.people.detector, "pose": cfg.people.pose}}
               if cfg.people.uses_alternative_models else {}),
            **({"recipe": cfg.recipe} if cfg.recipe != STANDARD_RECIPE else {}),
            **({"appearance_preprocessing": appearance_feature_settings(cfg.tracking.features)}
               if cfg.enabled["person_observations"] else {}),
            **({"bind_appearance_media": True} if binds_appearance_media(cfg, self.components or {}) else {}),
            **({"court_model_assets": CourtKPModule(cfg.court_kp).asset_identities()}
               if cfg.direct_annotation is None and cfg.video_court is None else {}),
            **({"court_model": CourtKPModule(cfg.court_kp).model_identity()}
               if cfg.direct_annotation is None and cfg.video_court is None else {}),
            **({"video_court_side_digest": side.digest, "video_court_side_unobservable_policy": cfg.court_side_unobservable_policy,
                "video_court_side_clips": side.publication_verdicts(shared, cfg.court_side_unobservable_policy, cfg.video_court_unobservable_policy)}
               if side is not None and shared is not None else {}),
            **({'video_court_cameras': shared.manifest['cameras'],
                'video_court_unobservable_policy': cfg.video_court_unobservable_policy,
                'video_court_drift_cameras': drift_identity(shared).get('camera_digests', {})} if shared is not None else {})}
        if cfg.people.uses_alternative_models and cfg.enabled["person_observations"]:
            pose = PersonPoseFeaturesModule(cfg.people, cfg.tracking.features)
            identity["people_component_settings"] = json_value({
                "person_detection": person_detection_settings(cfg),
                "person_pose_features": person_pose_feature_settings(cfg, pose),
            })
        self._publication_assets = ModelAssetSnapshot.capture(identity)
        self._publication_assets.assert_unchanged()
        return identity

    def validate_plan_assets(self, nodes: Sequence[ComponentNode]) -> None:
        """The plan must bind the asset versions already promised by its marker."""
        if self._publication_assets is not None:
            for node in nodes:
                self._publication_assets.require_settings(node.settings)
            self._publication_assets.assert_unchanged()

    def finalize_clip(self, executor: PlanExecutor, clip_id: str, snapshot: InputSnapshot, *,
                      video_court: VideoCourtArtifact | None, video_court_side: VideoCourtSideArtifact | None,
                      memory: SourceMemory | None = None) -> tuple[SceneResult, Path]:
        """Verify the exact inputs/shared products of this plan immediately before export."""
        snapshot.assert_unchanged()
        if video_court is not None:
            video_court.verify()
        if video_court_side is not None:
            video_court_side.verify()
        if self._publication_assets is not None:
            self._publication_assets.assert_unchanged()
        scene, archive = executor.finalize_clip(clip_id, memory=memory)
        return scene, archive

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

    def completed_scene_export(self) -> Path:
        """The exact export produced by the last run, before another writer acts."""
        if self._scene_export is None:
            raise RuntimeError("No completed scene export for the last run")
        return self._scene_export

    def prepare_video_court_side(self, court: VideoCourtArtifact | None) -> VideoCourtSideArtifact | None:
        from src.tennis_scene.pipeline.video_court_side.prepare import (
            prepare_video_court_side,
            verify_prepared_court_side,
        )

        cfg = self.config
        if cfg.video_court_side is None:
            return None
        check_shared_side_execution(cfg, self.components)
        if court is None:
            raise ValueError("Shared court side requires shared calibration")
        # The court policy governs adoption on each call. Measurements may be reused across policies.
        key = (cfg.video_court_side, court.digest, drift_identity(court).get("artifact_digest"),
               cfg.court_side, cfg.sampling_max_frames, self.code_identity)
        if self.video_court_side is None or self._video_court_side_key != key:
            self.video_court_side = prepare_video_court_side(cfg.video_court_side, cfg.resolver, court=court,
                thresholds=cfg.court_side, max_frames=cfg.sampling_max_frames, code_identity=self.code_identity,
                court_policy=cfg.video_court_unobservable_policy)
            self._video_court_side_key = key
        else:
            verify_prepared_court_side(self.video_court_side, cfg.video_court_side, cfg.resolver,
                                       code_identity=self.code_identity)
            self.video_court_side.check_settings(court, cfg.court_side, cfg.sampling_max_frames, court_policy=cfg.video_court_unobservable_policy)
        return self.video_court_side

    def run(self, video_paths: Sequence[Path], *, video_role: PathRole, camera_ids: Sequence[str],
            store_root: Path | None, max_frames: int | None = None, clip_id: str | None = None) -> SceneResult:
        """Run the selected recipe into an explicitly chosen component store.

        ``store_root`` is the clip-owned store (structured datasets keep it at
        ``<clip>/annotations/tennis_scene``); ``None`` selects
        ``cache.directory/<source digest>``. The store is never inferred from
        the shape of the video paths. The recipe and people models determine
        the store within that root.
        """
        event_config = self.config.require_event_detection()
        self._scene_export = None
        paths = tuple(self.config.resolver.validate(video_role, Path(p)) for p in video_paths)
        inputs = prepare_clip_source(paths, camera_ids, max_frames=max_frames, clip_id=clip_id or self.config.clip_id)
        source = inputs.source
        if self.config.camera_geometry.reference_camera is not None and self.config.camera_geometry.reference_camera not in source.camera_ids:
            raise ValueError("Reference camera is not a source camera")
        source_document = json_value(source)
        root = self.config.cache_directory / document_digest(source_document)[:20] if store_root is None else store_root
        store = ClipStore(recipe_directory(root, self.config.recipe, people_models=self.config.people.selection), source_document)
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
            side = self.prepare_video_court_side(shared)
            if side is not None:
                self.last_receipt["video_court_side"] = side.clip_report(source.clip_id, self.config.court_side_unobservable_policy)
            direct = self.config.direct_annotation
            workflow = None
            direct_model_identity = None
            direct_event_settings = None
            if direct is not None:
                from src.tasks.base.annotation.paths import AnnotationLayout
                if self.config.people.uses_tensorrt or self.config.people.uses_alternative_models:
                    direct_model_identity = document_digest(publication_model_identities(self.config))
                direct_event_settings = EventDetectionModule(event_config, source.clip_id).cache_identity()
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
                    scene, saved = saved_scene(store, direct, workflow.reference.digest,
                        people_runtime=self.config.people.runtime, model_identity=direct_model_identity,
                        event_settings=direct_event_settings, code_identity=self.code_identity)
                    archive = Path(str(saved["archive"]))
                    self.last_receipt["scene"] = saved
                else:
                    annotations, expected = workflow.inputs(completed)
                    importer = BallAnnotationImporter(annotations, expected)
                    nodes = importer.bind_nodes(nodes)
                    importer.publish(nodes, store, source)
            else:
                nodes = standard_definition(self.config, source, code_identity=self.code_identity, overrides=self.components,
                                            video_court=shared, video_court_side=side)
            if direct is None or direct.resume != "publication":
                self.validate_plan_assets(nodes)
                runner = ComponentRunner(nodes, store, overwrite=self.config.cache_overwrite, model_event=self.model_event)
                self.last_runner = runner
                runner.run()
                scene, archive = self.finalize_clip(runner.executor, runner.clip_id, inputs.snapshot,
                                                   video_court=shared, video_court_side=side)
                if direct is not None and workflow is not None:
                    self.last_receipt["scene"] = scene_receipt(store, direct, workflow.reference.digest, archive,
                        people_runtime=self.config.people.runtime, model_identity=direct_model_identity,
                        event_settings=direct_event_settings)
            self._scene_export = archive
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
