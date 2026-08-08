"""Definition-driven, preflight-first canonical scene execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionContext,
    StageExecutionSummary,
    StageInputKind,
    StageName,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.registry import StageRegistry
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace


@dataclass(frozen=True, slots=True)
class _Context(StageExecutionContext):
    request: ScenePipelineRequest
    stage: StageDefinition[StageExecutionSummary]
    owner_path: Path
    staging_path: Path


class ScenePipelineRunner:
    """Execute complete typed definitions without handler or publisher switches."""

    def __init__(
        self,
        *,
        workspace: SceneWorkspace,
        registry: StageRegistry,
        resolved_config_yaml: str,
    ) -> None:
        if not resolved_config_yaml.strip():
            raise ValueError("resolved_config_yaml must not be empty.")
        _configuration_authority(resolved_config_yaml)
        self.workspace = workspace
        self.registry = registry
        self.resolved_config_yaml = resolved_config_yaml

    def run(self, request: ScenePipelineRequest) -> MutableRunManifest:
        """Run one request, leaving no partial/stale stage marked completed."""
        if request.scene_id != self.workspace.scene_id:
            raise ValueError("Request scene_id disagrees with the resolved workspace.")
        selected = self.registry.selected_for_request(request)
        order = self.registry.ordered(set(StageName))
        positions = {definition.name: index for index, definition in enumerate(order)}
        start_index = positions[request.from_stage]
        selected_execution = tuple(
            definition
            for definition in selected
            if positions[definition.name] >= start_index
        )
        if not selected_execution:
            raise ValueError("from_stage is after every selected stage.")

        manifest = self._load_or_create_manifest(request)
        manifest.assert_request_compatible(request)
        self._preflight_before_invalidation(
            request,
            manifest,
            selected,
            selected_execution,
            start_index,
        )

        self.workspace.root.mkdir(parents=True, exist_ok=True)
        config_path = self.workspace.resolved_config_path
        config_staging = config_path.with_suffix(config_path.suffix + ".tmp")
        config_staging.write_text(self.resolved_config_yaml, encoding="utf-8")
        config_staging.replace(config_path)
        manifest.targets = sorted(target.value for target in request.targets)
        invalidated = self.registry.descendants(
            request.from_stage,
            include_self=True,
        )
        invalidated_names = {definition.name for definition in invalidated}
        for definition in reversed(invalidated):
            # The rerun cursor's prior complete owner stays visible until the new
            # validated snapshot atomically replaces it. Descendants are stale and
            # are physically unpublished before any execution starts.
            if definition.name is not request.from_stage:
                definition.invalidate_publication(self.workspace)
            manifest.invalidate(definition.name)
        for target in DatasetTarget:
            if target not in request.targets and target.stage in invalidated_names:
                manifest.skip(target.stage)
        manifest.save(self.workspace.run_manifest_path)

        for definition in selected_execution:
            self._execute_stage(request, definition, manifest)
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
                self.registry.definition(stage).recover_publication(self.workspace)
                record.status = StageStatus.FAILED
                record.summary = {}
                record.error = (
                    "InterruptedError: prior process ended while stage was running"
                )
                recovered = True
        if recovered:
            manifest.save(path)
        return manifest

    def _preflight_before_invalidation(
        self,
        request: ScenePipelineRequest,
        manifest: MutableRunManifest,
        selected: tuple[StageDefinition[StageExecutionSummary], ...],
        selected_execution: tuple[StageDefinition[StageExecutionSummary], ...],
        start_index: int,
    ) -> None:
        """Validate request, retained upstream, config, handler and publication first."""
        order = self.registry.ordered(set(StageName))
        selected_names = {definition.name for definition in selected}
        if request.from_stage is not StageName.INGEST:
            if not self.workspace.resolved_config_path.is_file():
                raise FileNotFoundError("Existing scene is missing resolved-config.yaml.")
            existing_config = self.workspace.resolved_config_path.read_text(encoding="utf-8")
            if _configuration_authority(existing_config) != _configuration_authority(
                self.resolved_config_yaml
            ):
                raise ValueError(
                    "Resolved configuration changed; rerun from ingest rather than "
                    "invalidating downstream."
                )
        for definition in order[:start_index]:
            if definition.name not in selected_names:
                continue
            if manifest.stages[definition.name].status is not StageStatus.COMPLETED:
                raise ValueError(
                    "Retained upstream stage is not completed: "
                    f"{definition.name.value}."
                )
            self.workspace.validate_required_outputs(definition)
        for definition in selected_execution:
            definition.preflight_publication(self.workspace)
        start_definition = self.registry.definition(request.from_stage)
        self._validate_definition_inputs(request, start_definition)
        context = _Context(
            request=request,
            stage=start_definition,
            owner_path=self.workspace.owner_path(start_definition),
            staging_path=self.workspace.staging_path(start_definition),
        )
        start_definition.preflight(context)

    def _validate_definition_inputs(
        self,
        request: ScenePipelineRequest,
        definition: StageDefinition[StageExecutionSummary],
    ) -> None:
        for stage_input in definition.required_inputs:
            if not stage_input.applies_to(request):
                continue
            if stage_input.kind is StageInputKind.SOURCE_VIDEO:
                if not request.source_video.is_file():
                    raise FileNotFoundError(
                        f"Request source video disappeared: {request.source_video}."
                    )
            elif stage_input.kind is StageInputKind.RESOLVED_CONFIGURATION:
                _configuration_authority(self.resolved_config_yaml)
            elif stage_input.kind is StageInputKind.STAGE_OUTPUT:
                if stage_input.producer is None or stage_input.relative_path is None:
                    raise RuntimeError("Invalid stage input escaped registry validation.")
                self.workspace.validate_stage_input(
                    self.registry.definition(stage_input.producer),
                    stage_input.relative_path,
                )
            else:  # pragma: no cover - exhaustive StrEnum guard
                raise ValueError(f"Unknown stage input kind: {stage_input.kind!r}.")

    def _execute_stage(
        self,
        request: ScenePipelineRequest,
        definition: StageDefinition[StageExecutionSummary],
        manifest: MutableRunManifest,
    ) -> None:
        staging_path = definition.prepare_publication(self.workspace)
        context = _Context(
            request=request,
            stage=definition,
            owner_path=self.workspace.owner_path(definition),
            staging_path=staging_path,
        )
        manifest.begin(definition.name)
        manifest.save(self.workspace.run_manifest_path)
        try:
            self._validate_definition_inputs(request, definition)
            definition.preflight(context)
            summary = definition.execute(context)
            definition.validate(context)
            publication = definition.publish(self.workspace)
            if publication.owner_path != context.owner_path:
                raise ValueError("Publication returned a non-canonical owner path.")
            self.workspace.validate_required_outputs(definition)
            manifest.complete(definition.name, summary.values)
            manifest.save(self.workspace.run_manifest_path)
        except BaseException as error:
            definition.abandon_publication(self.workspace)
            manifest.fail(definition.name, error)
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
