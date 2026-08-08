"""Composition root for preflight-first canonical scene execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    PublicationMode,
    ScenePipelineRequest,
    StageExecutionContext,
    StageHandler,
    StageName,
    StageSpec,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.publication import StagePublisher
from src.synthetic_data_generation.pipeline.registry import StageRegistry
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace


@dataclass(frozen=True, slots=True)
class _Context(StageExecutionContext):
    request: ScenePipelineRequest
    stage: StageSpec
    owner_path: Path
    staging_path: Path


class ScenePipelineRunner:
    """Execute typed handlers without stage-name conditionals or silent fallback."""

    def __init__(
        self,
        *,
        workspace: SceneWorkspace,
        registry: StageRegistry,
        handlers: Mapping[str, StageHandler],
        resolved_config_yaml: str,
    ) -> None:
        if not resolved_config_yaml.strip():
            raise ValueError("resolved_config_yaml must not be empty.")
        _configuration_authority(resolved_config_yaml)
        self.workspace = workspace
        self.registry = registry
        self.handlers = dict(handlers)
        self.resolved_config_yaml = resolved_config_yaml

    def run(self, request: ScenePipelineRequest) -> MutableRunManifest:
        """Run one request, leaving no partial/stale stage marked completed."""
        if request.scene_id != self.workspace.scene_id:
            raise ValueError("Request scene_id disagrees with the resolved workspace.")
        selected = self.registry.selected_for_request(request)
        order = self.registry.ordered(set(StageName))
        start_index = order.index(request.from_stage)
        selected_execution = tuple(
            stage for stage in selected if order.index(stage) >= start_index
        )
        if not selected_execution:
            raise ValueError("from_stage is after every selected stage.")

        self._require_handlers(selected)
        manifest = self._load_or_create_manifest(request)
        manifest.assert_request_compatible(request)
        self._preflight_before_invalidation(request, manifest, selected, start_index)

        self.workspace.root.mkdir(parents=True, exist_ok=True)
        config_path = self.workspace.resolved_config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_staging = config_path.with_suffix(config_path.suffix + ".tmp")
        config_staging.write_text(self.resolved_config_yaml, encoding="utf-8")
        config_staging.replace(config_path)
        manifest.targets = sorted(target.value for target in request.targets)
        invalidated = self.registry.descendants(request.from_stage, include_self=True)
        for stage in reversed(invalidated):
            self.workspace.invalidate_outputs(self.registry.spec(stage))
            manifest.invalidate(stage)
        for target in DatasetTarget:
            if target not in request.targets and target.stage in invalidated:
                manifest.skip(target.stage)
        manifest.save(self.workspace.run_manifest_path)

        for stage in selected_execution:
            self._execute_stage(request, stage, manifest)
        return manifest

    def _load_or_create_manifest(
        self,
        request: ScenePipelineRequest,
    ) -> MutableRunManifest:
        path = self.workspace.run_manifest_path
        if not path.exists():
            return MutableRunManifest.create(request)
        manifest = MutableRunManifest.load(path)
        recovered = False
        for stage, record in manifest.stages.items():
            if record.status is StageStatus.RUNNING:
                spec = self.registry.spec(stage)
                if spec.publication_mode is PublicationMode.ATOMIC_OUTPUTS:
                    StagePublisher(
                        self.workspace,
                        spec,
                    ).recover_interrupted_publication()
                record.status = StageStatus.FAILED
                record.summary = {}
                record.error = "InterruptedError: prior process ended while stage was running"
                recovered = True
        if recovered:
            manifest.save(path)
        return manifest

    def _require_handlers(self, selected: tuple[StageName, ...]) -> None:
        required = {self.registry.spec(stage).handler_key for stage in selected}
        missing = sorted(required - set(self.handlers))
        unexpected = sorted(set(self.handlers) - {spec.handler_key for spec in self.registry.specs.values()})
        if missing or unexpected:
            raise ValueError(f"Handler registry mismatch; missing={missing}, unexpected={unexpected}.")

    def _preflight_before_invalidation(
        self,
        request: ScenePipelineRequest,
        manifest: MutableRunManifest,
        selected: tuple[StageName, ...],
        start_index: int,
    ) -> None:
        """Validate request, retained upstream, config and handler preflight first."""
        order = self.registry.ordered(set(StageName))
        if request.from_stage is not StageName.INGEST:
            if not self.workspace.resolved_config_path.is_file():
                raise FileNotFoundError("Existing scene is missing resolved-config.yaml.")
            existing_config = self.workspace.resolved_config_path.read_text(encoding="utf-8")
            if _configuration_authority(existing_config) != _configuration_authority(
                self.resolved_config_yaml
            ):
                raise ValueError(
                    "Resolved configuration changed; rerun from ingest rather than invalidating downstream."
                )
        for stage in order[:start_index]:
            if stage not in selected:
                continue
            if manifest.stages[stage].status is not StageStatus.COMPLETED:
                raise ValueError(f"Retained upstream stage is not completed: {stage.value}.")
            self.workspace.validate_required_outputs(self.registry.spec(stage))
        start_spec = self.registry.spec(request.from_stage)
        context = _Context(
            request=request,
            stage=start_spec,
            owner_path=self.workspace.owner_path(start_spec),
            staging_path=self.workspace.staging_path(start_spec),
        )
        self.handlers[start_spec.handler_key].preflight(context)

    def _execute_stage(
        self,
        request: ScenePipelineRequest,
        stage: StageName,
        manifest: MutableRunManifest,
    ) -> None:
        spec = self.registry.spec(stage)
        handler = self.handlers[spec.handler_key]
        publisher = StagePublisher(self.workspace, spec)
        staging_path = (
            publisher.prepare()
            if spec.publication_mode is PublicationMode.ATOMIC_OUTPUTS
            else self.workspace.staging_path(spec)
        )
        context = _Context(
            request=request,
            stage=spec,
            owner_path=self.workspace.owner_path(spec),
            staging_path=staging_path,
        )
        manifest.begin(stage)
        manifest.save(self.workspace.run_manifest_path)
        try:
            handler.preflight(context)
            summary = handler.execute(context)
            handler.validate(context)
            if spec.publication_mode is PublicationMode.ATOMIC_OUTPUTS:
                publisher.publish()
            self.workspace.validate_required_outputs(spec)
            manifest.complete(stage, summary.values)
            manifest.save(self.workspace.run_manifest_path)
        except BaseException as error:
            publisher.abandon()
            self.workspace.invalidate_outputs(spec)
            manifest.fail(stage, error)
            manifest.save(self.workspace.run_manifest_path)
            raise


def _configuration_authority(resolved_yaml: str) -> Mapping[str, object]:
    """Exclude only the per-invocation rerun cursor from config comparison."""
    loaded: object = yaml.safe_load(resolved_yaml)
    if not isinstance(loaded, Mapping) or any(
        not isinstance(key, str) for key in loaded
    ):
        raise ValueError("resolved-config.yaml must contain a string-keyed mapping.")
    authority: dict[str, object] = dict(loaded)
    if "request" not in authority:
        raise ValueError("resolved-config.yaml must contain request.from_stage.")
    request = authority["request"]
    if not isinstance(request, Mapping) or any(
        not isinstance(key, str) for key in request
    ):
        raise ValueError("resolved-config.yaml request must be a string-keyed mapping.")
    stable_request: dict[str, object] = dict(request)
    if "from_stage" not in stable_request:
        raise ValueError("resolved-config.yaml must contain request.from_stage.")
    del stable_request["from_stage"]
    authority["request"] = stable_request
    return authority


__all__ = ["ScenePipelineRunner"]
