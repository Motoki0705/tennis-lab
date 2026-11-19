"""Execute a frozen video's clips in one dependency plan and model runtime."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.batching import BatchConfig, SourceMemory
from src.tennis_scene.pipeline.clip_inputs import (
    PreparedDatasetClip,
    prepare_video_inputs,
)
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    completed_annotation,
    publish_annotation,
    record_annotation_failure,
)
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable, is_clip_failure
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ExecutionPlan,
    NodeKey,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore, ClipStoreCache
from src.utils.configuration import PathRole
from src.utils.inference_recording import InferenceRecorder, recording_context, span
from src.utils.io import utc_now_iso


@dataclass(frozen=True)
class VideoClipOutcome:
    clip_id: str
    status: Literal["generated", "skipped", "failed", "blocked"]
    annotation_path: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class VideoBatchResult:
    outcomes: tuple[VideoClipOutcome, ...]
    progress_path: Path
    plan: ExecutionPlan

    @property
    def exit_code(self) -> int:
        return int(any(outcome.status in {"failed", "blocked"} for outcome in self.outcomes))


class VideoBatchProgress:
    """An atomic progress receipt, with the C1 recording kept beside it."""

    def __init__(self, directory: Path, dataset_id: str, video_id: str,
                 prepared: Sequence[PreparedDatasetClip], pipeline_yaml: str, identity: Mapping[str, Any],
                 *, overwrite: bool, continue_on_error: bool, recording_devices: Sequence[str],
                 batching: BatchConfig) -> None:
        self.path = directory / uuid4().hex / "progress.json"
        cross_source = batching.enabled and any(batching.sizes.for_model(name) > 1 for name in batching.sizes.__slots__)
        self.recorder = InferenceRecorder(self.path.with_name("recording.json"), devices=recording_devices,
            metadata={"dataset_id": dataset_id, "video_id": video_id, "batching": json_value(batching),
                      "condition": "model_reuse_and_batching" if cross_source else "model_reuse_only"})
        self.plan: ExecutionPlan | None = None
        self.outcomes: dict[str, VideoClipOutcome] = {}
        self.cursor = 0
        self.document: dict[str, Any] = {
            "schema": "tennis_scene_video_batch_v1", "dataset_id": dataset_id, "video_id": video_id,
            "status": "running", "started_at": utc_now_iso(), "active_node": None,
            "pipeline_config_yaml": pipeline_yaml, "publication_identity": json_value(identity),
            "overwrite": overwrite, "continue_on_error": continue_on_error,
            "batching": json_value(batching), "passes": [],
            "targets": [{"record": clip.inputs.record.to_dict(), "manifest_sha256": clip.inputs.manifest_sha256,
                         "media_sha256": dict(clip.inputs.media_sha256)} for clip in prepared],
            "clips": {clip.inputs.record.clip_id: {"status": "pending"} for clip in prepared},
            "nodes": {}, "models": [], "batch_shapes": {}, "recording": "recording.json",
        }
        self.flush()

    def flush(self) -> None:
        # C1 records the actual model input, including tails and flip-test crops.
        events = self.recorder.events[self.cursor:]
        self.cursor += len(events)
        for event in events:
            attributes = event["attributes"]
            if event["stage"] in {"model_load", "model_unload"}:
                self.document["models"].append(event)
            if event["stage"] == "inference" and "batch_shape" in attributes:
                name = str(attributes.get("model", attributes.get("node", "unknown")))
                shape = json.dumps(attributes["batch_shape"])
                counts = self.document["batch_shapes"].setdefault(name, {})
                count = counts.setdefault(shape, {"succeeded": 0, "failed": 0, "items": attributes["batch_items"]})
                count["succeeded" if event["status"] == "ok" else "failed"] += 1
        if self.plan is not None:
            nodes: dict[str, dict[str, Any]] = {clip_id: {} for clip_id in self.plan.clips}
            for clip_id, name in self.plan.order:
                key = clip_id, name
                nodes[clip_id][name] = {"status": self.plan.statuses.get(key, "pending"), "seconds": self.plan.seconds.get(key)}
            self.document["nodes"] = nodes
        write_json_atomic(self.path, self.document)

    @contextmanager
    def node_event(self, key: NodeKey) -> Iterator[None]:
        self.document["active_node"] = list(key)
        self.flush()
        try:
            yield
        finally:
            self.document["active_node"] = None
            self.flush()

    def outcome(self, outcome: VideoClipOutcome) -> None:
        self.outcomes[outcome.clip_id] = outcome
        self.document["clips"][outcome.clip_id] = json_value(outcome)
        self.flush()


def run_video_batch(dataset_directory: Path, video_id: str, orchestrator: TennisSceneOrchestrator, *,
                    pipeline_config_yaml: str, clip_ids: Sequence[str] | None = None,
                    overwrite: bool = False, continue_on_error: bool = True,
                    progress_directory: Path | None = None,
                    recording_devices: Sequence[str] = (), batching: BatchConfig | None = None) -> VideoBatchResult:
    """Preflight the fixed inputs, then drain the plan using explicit batch limits."""
    cfg = orchestrator.config
    batching = BatchConfig() if batching is None else batching
    store_cache = ClipStoreCache(batching.store_cache_bytes)
    if cfg.max_frames is not None or cfg.direct_annotation is not None:
        raise ValueError("Video batches require complete clips and the model pipeline")
    cfg.resolver.validate(PathRole.DATA, dataset_directory)
    dataset_id, prepared = prepare_video_inputs(dataset_directory, video_id, clip_ids)
    for clip in prepared:
        for video in clip.media.source.videos:
            cfg.resolver.validate(PathRole.DATA, video.path)
    directory = (dataset_directory / "videos" / video_id / "annotations" / "tennis_scene" / "batch"
                 if progress_directory is None else progress_directory)
    progress = VideoBatchProgress(directory, dataset_id, video_id, prepared, pipeline_config_yaml, {},
        overwrite=overwrite, continue_on_error=continue_on_error, recording_devices=recording_devices, batching=batching)
    bound = {clip.inputs.record.clip_id: clip for clip in prepared}
    clips = []

    def failed(clip_id: str, error: Exception) -> None:
        clip = bound[clip_id]
        message = record_annotation_failure(clip.inputs.clip_dir, clip_id, error)
        progress.outcome(VideoClipOutcome(clip_id, "failed", error=message))
        if isinstance(error, ReconstructionUnavailable):
            progress.document["clips"][clip_id].update(reason=error.reason, diagnostics=json_value(error.diagnostics))
            progress.flush()

    try:
        with progress.recorder:
            progress.document["active_stage"] = "prepare_video_court"
            with span("preprocess", operation="prepare_video_court"):
                if cfg.video_court is not None and (cfg.video_court.dataset_id, cfg.video_court.video_id) != (dataset_id, video_id):
                    raise ValueError("Shared court configuration belongs to another dataset/video")
                video_court = orchestrator.prepare_video_court()
                if video_court is not None:
                    for clip in prepared:
                        video_court.bind(clip.media.source)
                    progress.document["video_court"] = {
                        "artifact_digest": video_court.digest, "cameras": video_court.manifest["cameras"],
                        "unobservable_policy": cfg.video_court_unobservable_policy,
                        "drift_digest": None if cfg.video_court_drift is None else cfg.video_court_drift.digest,
                    }
            progress.document["active_stage"] = "prepare_video_court_side"
            progress.flush()
            with span("preprocess", operation="prepare_video_court_side"):
                if cfg.video_court_side is not None and (cfg.video_court_side.dataset_id, cfg.video_court_side.video_id) != (dataset_id, video_id):
                    raise ValueError("Shared court side configuration belongs to another dataset/video")
                video_court_side = orchestrator.prepare_video_court_side(video_court)
                if video_court_side is not None:
                    progress.document["video_court_side"] = {
                        "evidence": video_court_side.decision.evidence,
                        "artifact_digest": video_court_side.digest,
                        "diagnostics_digest": video_court_side.decision.diagnostics_digest,
                        "unobservable_policy": cfg.court_side_unobservable_policy,
                        "clips": {clip.inputs.record.clip_id: video_court_side.clip_report(
                            clip.inputs.record.clip_id, cfg.court_side_unobservable_policy) for clip in prepared},
                    }
            with span("measurement", operation="publication_identity"):
                identity = orchestrator.publication_identity()
            progress.document.update(publication_identity=json_value(identity), active_stage=None)
            progress.flush()
            for clip in prepared:
                inputs, source = clip.inputs, clip.media.source
                key = inputs.record.clip_id
                try:
                    marker = completed_annotation(inputs, identity, overwrite=overwrite)
                    if marker is not None:
                        progress.outcome(VideoClipOutcome(key, "skipped", marker))
                        continue
                    reference = cfg.camera_geometry.reference_camera
                    if reference is not None and reference not in source.camera_ids:
                        raise ValueError("Reference camera is not a source camera")
                    nodes = standard_definition(cfg, source, code_identity=orchestrator.code_identity,
                                                overrides=orchestrator.components, video_court=video_court,
                                                video_court_side=video_court_side)
                    orchestrator.validate_plan_assets(nodes)
                    store = ClipStore(inputs.clip_dir / ANNOTATION_RELATIVE_DIR, json_value(source),
                                      memory_entries=0, shared_cache=store_cache)
                    clips.append(ClipExecution(key, nodes, store, cfg.cache_overwrite))
                except Exception as error:
                    failed(key, error)
                    if not continue_on_error or not is_clip_failure(error):
                        raise
            plan = ExecutionPlan(clips)
            progress.plan = plan
            executor = PlanExecutor(plan, model_event=orchestrator.model_event, node_event=progress.node_event, batching=batching)

            def publish(clip_id: str) -> None:
                clip = bound[clip_id]
                clip.inputs.snapshot.assert_unchanged()
                memory = SourceMemory(executor.working_memory) if batching.enabled else None
                try:
                    scene, archive = orchestrator.finalize_clip(executor, clip_id, clip.media.snapshot,
                        video_court=video_court, video_court_side=video_court_side, memory=memory)
                    marker = publish_annotation(clip.inputs, scene, pipeline_config_yaml=pipeline_config_yaml,
                                                publication_identity=identity, expected_scene=archive)
                finally:
                    if memory is not None:
                        memory.close()
                progress.outcome(VideoClipOutcome(clip_id, "generated", marker))

            pass_size = batching.clips_per_pass or max(1, len(clips))
            progress.recorder.metadata["passes"] = progress.document["passes"]
            try:
                for offset in range(0, len(clips), pass_size):
                    clip_group = {spec.clip_id for spec in clips[offset:offset + pass_size]}
                    pass_index = len(progress.document["passes"])
                    progress.document["passes"].append({"index": pass_index, "clips": [key for key in bound if key in clip_group]})
                    targets = [key for key in plan.order if key in plan.required and key[0] in clip_group]
                    with recording_context(video_pass=pass_index), span("video_pass", clip_count=len(clip_group)):
                        executor.run(targets=targets, on_clip_complete=publish,
                                     continue_on_error=continue_on_error, on_clip_error=failed)
            finally:
                budget = {"working_peak_bytes": executor.working_memory.peak, "store_cache_peak_bytes": store_cache.peak_bytes}
                progress.recorder.metadata["budget"] = progress.document["budget"] = budget
            progress.recorder.metadata["outcomes"] = json_value(progress.outcomes)
    except BaseException as error:
        progress.document.update(status="interrupted" if not isinstance(error, Exception) else "failed",
                                 error=str(error), error_type=type(error).__name__)
        if progress.plan is not None:
            for clip_id in progress.plan.clips:
                if clip_id not in progress.outcomes:
                    progress.plan.block_clip(clip_id)
        for clip_id in bound:
            if clip_id not in progress.outcomes:
                progress.outcome(VideoClipOutcome(clip_id, "blocked", error=str(error)))
        raise
    else:
        progress.document["status"] = "failed" if any(o.status == "failed" for o in progress.outcomes.values()) else "complete"
    finally:
        progress.document["finished_at"] = utc_now_iso()
        progress.flush()
    return VideoBatchResult(tuple(progress.outcomes[key] for key in bound), progress.path, plan)
