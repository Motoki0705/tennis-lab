"""Enrich the assembled scene with sparse ball events on its frame grid."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.tennis_scene.ball_event_schema import (
    EVENT_SOURCES,
    EventSource,
    ball_event_metadata,
    validate_scene_ball_events,
)
from src.tennis_scene.ball_events import analyze_scene_events
from src.tennis_scene.ball_events.annotation import (
    ANNOTATION_SCHEMA_FILENAME,
    annotation_path,
    meiji_v2_manifest_asset,
    read_ball_annotations,
)
from src.tennis_scene.ball_events.rules import DEFAULT_THRESHOLDS, RuleThresholds
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
)
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.schema import SceneResult

EVENT_DETECTION = "event_detection"
REFINER_SEED = 42


@dataclass(frozen=True)
class EventDetectionConfig:
    method: EventSource
    annotation_directory: Path | None = None
    checkpoint: Path | None = None
    checkpoint_sha256: str | None = None
    fps_tolerance: float | None = None
    device: str = "cpu"
    thresholds: RuleThresholds = DEFAULT_THRESHOLDS
    annotation_benchmark_manifest: Path | None = None

    def __post_init__(self) -> None:
        if self.method not in EVENT_SOURCES:
            raise ValueError("event_detection.method must explicitly select rule, refiner or annotation")
        if self.method == "annotation":
            if self.annotation_directory is None or not self.annotation_directory.is_absolute():
                raise ValueError("annotation method requires a resolved annotation_directory")
            if self.annotation_benchmark_manifest is None or not self.annotation_benchmark_manifest.is_absolute():
                raise ValueError("annotation method requires a resolved annotation_benchmark_manifest")
        elif self.annotation_directory is not None or self.annotation_benchmark_manifest is not None:
            raise ValueError("Annotation inputs require event_detection.method=annotation")
        refiner_fields = (self.checkpoint, self.checkpoint_sha256, self.fps_tolerance)
        if self.method == "refiner":
            if self.checkpoint is None or not self.checkpoint.is_absolute():
                raise ValueError("refiner method requires a resolved checkpoint")
            if self.checkpoint_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256) is None:
                raise ValueError("refiner method requires checkpoint_sha256")
            if self.fps_tolerance is None or not math.isfinite(self.fps_tolerance) or self.fps_tolerance < 0:
                raise ValueError("refiner method requires a finite nonnegative fps_tolerance")
        elif any(value is not None for value in refiner_fields):
            raise ValueError("Refiner options require event_detection.method=refiner")

    def model_specs(self) -> dict[str, ModelSpec[Any]]:
        if self.checkpoint is None:
            return {}
        from src.tasks.ball_refiner_3d.model_io.checkpoint import CHECKPOINT_SCHEMA

        self.require_checkpoint()
        return {"refiner": ModelSpec(
            "ball_refiner_3d", {"checkpoint": self.checkpoint}, CHECKPOINT_SCHEMA,
            self.device, "float32", {}, self._load_predictor,
        )}

    def require_checkpoint(self) -> None:
        if self.checkpoint is not None and file_identity(self.checkpoint)["sha256"] != self.checkpoint_sha256:
            raise ValueError(f"Refiner checkpoint SHA-256 mismatch: {self.checkpoint}")

    def _load_predictor(self) -> Any:
        from src.tasks.ball_refiner_3d.inference.predictor import RefinerPredictor

        assert self.checkpoint is not None
        self.require_checkpoint()
        return RefinerPredictor.from_checkpoint(self.checkpoint, device=self.device)

    def settings(self, code_commit: str) -> dict[str, Any]:
        settings: dict[str, Any] = {"method": self.method, "code_commit": code_commit}
        if self.method == "rule":
            settings["thresholds"] = asdict(self.thresholds)
        elif self.method == "refiner":
            from src.tasks.ball_refiner_3d.inference.segmentation import DEFAULT_PICKING

            settings.update(
                checkpoint_sha256=self.checkpoint_sha256, fps_tolerance=self.fps_tolerance,
                seed=REFINER_SEED, device=self.device, picking=asdict(DEFAULT_PICKING),
            )
        return settings


def annotation_assets(directory: Path, clip_id: str, *, benchmark_manifest: Path) -> dict[str, dict[str, str]]:
    manifest = meiji_v2_manifest_asset(benchmark_manifest)
    if file_identity(benchmark_manifest) != manifest:
        raise ValueError(f"Event annotation requires the fixed Meiji multiview-v2 manifest SHA-256: {benchmark_manifest}")
    return {
        "annotation": file_identity(annotation_path(directory, clip_id)),
        "schema": file_identity(directory / ANNOTATION_SCHEMA_FILENAME),
        "benchmark_manifest": manifest,
    }


def annotation_digests(assets: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    return {
        "annotation_sha256": assets["annotation"]["sha256"],
        "annotation_schema_sha256": assets["schema"]["sha256"],
        "annotation_benchmark_sha256": assets["benchmark_manifest"]["sha256"],
    }


@dataclass(frozen=True)
class EventDetectionInput:
    source: ClipSource
    scene: SceneResult


@dataclass(kw_only=True)
class EventDetectionResult(SceneResult):
    """The v3 scene and evidence persisted only in the component descriptor."""

    event_evidence: dict[str, Any] = field(default_factory=dict)


class EventDetectionInputAssembler:
    version = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> EventDetectionInput:
        return EventDetectionInput(context.source, artifacts["scene"])


class EventDetectionModule(InferenceComponent[EventDetectionInput, EventDetectionResult]):
    io = ComponentIO(
        EVENT_DETECTION, EventDetectionInput, EventDetectionResult,
        {"scene": InputPort("scene_result", 2)}, "scene_result", 3,
    )

    def __init__(self, config: EventDetectionConfig, clip_id: str, *, code_commit: str) -> None:
        self.config, self.clip_id, self.code_commit = config, clip_id, code_commit
        self._specs = config.model_specs()
        self._assets: dict[str, dict[str, str]] = {}
        if config.annotation_directory is not None:
            assert config.annotation_benchmark_manifest is not None
            self._assets = annotation_assets(
                config.annotation_directory, clip_id, benchmark_manifest=config.annotation_benchmark_manifest,
            )

    def model_specs(self) -> Mapping[str, ModelSpec[Any]]:
        return self._specs

    def cache_identity(self) -> dict[str, Any]:
        settings = self.config.settings(self.code_commit)
        if self.config.method == "annotation":
            settings.update(annotation_digests(self._assets))
        return settings

    def require_inputs(self) -> None:
        for asset in self._assets.values():
            if file_identity(Path(asset["path"])) != asset:
                raise ValueError(f"Changed event annotation asset: {asset['path']}")
        self.config.require_checkpoint()

    def validate_output(self, result: EventDetectionResult, settings: Mapping[str, Any]) -> None:
        if result.schema_version != 3:
            raise ValueError("event_detection must produce SceneResult v3")
        validate_scene_ball_events(result, version=3)
        events = result.metadata["ball_events"]
        provenance = events["provenance"]
        if events["source"] != settings["method"] or any(
            provenance.get(key) != value
            for key, value in settings.items() if key not in {"method", "models"}
        ):
            raise ValueError("Scene ball-event provenance disagrees with event_detection settings")

    def process_with_models(self, inputs: EventDetectionInput, models: ModelBorrower) -> EventDetectionResult:
        scene, source = inputs.scene, inputs.source
        if scene.schema_version != 2:
            raise ValueError("event_detection requires the SceneResult v2 assembly")
        if (scene.metadata.get("dataset_clip_id"), scene.num_frames, scene.fps, (scene.width, scene.height)) != (
            self.clip_id, source.num_frames, source.fps, source.size,
        ) or source.clip_id != self.clip_id:
            raise ValueError("event_detection source clip or timeline disagrees with the scene assembly")

        config = self.config
        provenance = {key: value for key, value in self.cache_identity().items() if key != "method"}
        if config.method == "annotation":
            labels = read_ball_annotations(scene, source, **self._assets)
            events, coverage, columns, evidence = labels.events, labels.coverage, labels.columns, labels.evidence
            provenance.update(labels.provenance)
        elif config.method == "rule":
            candidates, coverage = analyze_scene_events(scene, thresholds=config.thresholds)
            events = [asdict(event) for event in candidates]
            columns = {"ball_event_score": np.asarray([event["score"] for event in events], dtype=np.float32)}
            evidence = {"events": events}
        else:
            from src.tasks.ball_refiner_3d.inference.scene import (
                predict_scene_result_events,
            )
            assert config.fps_tolerance is not None
            _, events = predict_scene_result_events(
                models.borrow("refiner"), scene, seed=REFINER_SEED, fps_tolerance=config.fps_tolerance,
            )
            coverage = np.ones(scene.num_frames, dtype=np.bool_)
            columns = {"ball_event_score": np.asarray([event["score"] for event in events], dtype=np.float32)}
            evidence = {"events": events}
        values = {item.name: getattr(scene, item.name) for item in fields(SceneResult)}
        values.update(
            columns,
            metadata={**scene.metadata, "scene_schema_version": 3, "ball_events": ball_event_metadata(config.method, provenance)},
            ball_event_frame=np.asarray([row["frame"] for row in events], dtype=np.int32),
            ball_event_kind=np.asarray([EVENT_CLASSES.index(row["kind"]) for row in events], dtype=np.uint8),
            ball_event_coverage=coverage,
        )
        result = EventDetectionResult(**values, event_evidence=evidence)
        self.validate_output(result, self.cache_identity())
        return result
