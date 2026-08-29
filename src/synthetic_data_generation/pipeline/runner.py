"""Definition-driven, preflight-first canonical scene execution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionContext,
    StageExecutionPlan,
    StageExecutionSummary,
    StageInputKind,
    StageName,
    StageStatus,
)
from src.synthetic_data_generation.pipeline.registry import StageRegistry
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace

_LEGACY_COURT_REUSE_ADDITIVE_NHT_KEYS = frozenset(
    {"training_python_path", "trainer_path"}
)


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

        manifest = self._load_or_create_manifest(request)
        manifest.assert_request_compatible(request)
        reusable_stages = self._reusable_stages(request, manifest)
        plan = self.registry.execution_for_request(
            request,
            reusable_stages=reusable_stages,
        )
        self._preflight_before_invalidation(request, manifest, plan)
        self._recover_interrupted_stages(manifest)

        self.workspace.root.mkdir(parents=True, exist_ok=True)
        config_path = self.workspace.resolved_config_path
        config_staging = config_path.with_suffix(config_path.suffix + ".tmp")
        config_staging.write_text(self.resolved_config_yaml, encoding="utf-8")
        config_staging.replace(config_path)
        manifest.targets = sorted(target.value for target in request.targets)
        invalidated_names = {definition.name for definition in plan.invalidated}
        execution_names = {definition.name for definition in plan.execution}
        replacement_roots = {
            definition.name
            for definition in plan.execution
            if not (set(definition.dependencies) & execution_names)
        }
        replacement_roots.add(plan.cursor.name)
        for definition in reversed(plan.invalidated):
            # Each independently scheduled replacement root keeps an old complete
            # owner until atomic publication. Graph descendants are stale and are
            # physically unpublished before any execution starts.
            if definition.name not in replacement_roots:
                definition.invalidate_publication(self.workspace)
            manifest.invalidate(definition.name)
        for target in DatasetTarget:
            if target not in request.targets and target.stage in invalidated_names:
                manifest.skip(target.stage)
        manifest.save(self.workspace.run_manifest_path)

        for definition in plan.execution:
            self._execute_stage(request, definition, manifest)
        return manifest

    def _load_or_create_manifest(
        self,
        request: ScenePipelineRequest,
    ) -> MutableRunManifest:
        path = self.workspace.run_manifest_path
        if not path.exists():
            return MutableRunManifest.create(request)
        return MutableRunManifest.load(path)

    def _recover_interrupted_stages(self, manifest: MutableRunManifest) -> None:
        """Recover interrupted publication only after request preflight succeeds."""
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
            manifest.save(self.workspace.run_manifest_path)

    def _reusable_stages(
        self,
        request: ScenePipelineRequest,
        manifest: MutableRunManifest,
    ) -> set[StageName]:
        """Return completed publications whose selected dependency closure is valid."""
        selected = self.registry.selected_for_request(request)
        selected_names = {definition.name for definition in selected}
        demanded_outputs: dict[StageName, set[Path]] = {
            stage: set() for stage in selected_names
        }
        for definition in selected:
            for stage_input in definition.required_inputs:
                if (
                    stage_input.kind is StageInputKind.STAGE_OUTPUT
                    and stage_input.applies_to(request)
                ):
                    producer = stage_input.producer
                    relative_path = stage_input.relative_path
                    if producer is None or relative_path is None:
                        raise RuntimeError(
                            "Invalid stage input escaped registry validation."
                        )
                    demanded_outputs[producer].add(relative_path)

        reusable: set[StageName] = set()
        for definition in selected:
            record = manifest.stages[definition.name]
            selected_dependencies = set(definition.dependencies) & selected_names
            if (
                record.status is not StageStatus.COMPLETED
                or not selected_dependencies <= reusable
            ):
                continue
            try:
                self.workspace.validate_required_outputs(definition)
                for relative_path in demanded_outputs[definition.name]:
                    self.workspace.validate_stage_input(definition, relative_path)
                definition.validate_reusable_publication(
                    self.workspace.owner_path(definition)
                )
            except (OSError, TypeError, ValueError):
                continue
            reusable.add(definition.name)
        return reusable

    def _preflight_before_invalidation(
        self,
        request: ScenePipelineRequest,
        manifest: MutableRunManifest,
        plan: StageExecutionPlan,
    ) -> None:
        """Validate request, retained upstream, config, handler and publication first."""
        if plan.cursor.name is not StageName.INGEST:
            if not self.workspace.resolved_config_path.is_file():
                raise FileNotFoundError(
                    "Existing scene is missing resolved-config.yaml."
                )
            existing_config = self.workspace.resolved_config_path.read_text(
                encoding="utf-8"
            )
            if not _resolved_configuration_is_reusable(
                existing_config,
                self.resolved_config_yaml,
                request=request,
                plan=plan,
            ):
                raise ValueError(
                    "Resolved configuration changed; rerun from ingest rather than "
                    "invalidating downstream."
                )
        for definition in plan.retained_ancestors:
            if manifest.stages[definition.name].status is not StageStatus.COMPLETED:
                raise ValueError(
                    "Retained prerequisite stage is not completed: "
                    f"{definition.name.value}."
                )
            self.workspace.validate_required_outputs(definition)
            definition.validate_reusable_publication(
                self.workspace.owner_path(definition)
            )
        first_execution = plan.execution[0]
        self._validate_definition_inputs(request, first_execution)
        context = _Context(
            request=request,
            stage=first_execution,
            owner_path=self.workspace.owner_path(first_execution),
            staging_path=self.workspace.staging_path(first_execution),
        )
        first_execution.preflight(context)
        for definition in plan.invalidated:
            definition.preflight_publication(self.workspace)

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
                    raise RuntimeError(
                        "Invalid stage input escaped registry validation."
                    )
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
    """Exclude per-invocation start and terminal cursors from config comparison."""
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
    if "through_stage" not in stable_request:
        raise ValueError("resolved-config.yaml must contain request.through_stage.")
    del stable_request["through_stage"]
    authority["request"] = stable_request
    return authority


def _resolved_configuration_is_reusable(
    existing_yaml: str,
    requested_yaml: str,
    *,
    request: ScenePipelineRequest,
    plan: StageExecutionPlan,
) -> bool:
    """Compare exact config authority with one narrow Court/report exception."""
    existing = _configuration_authority(existing_yaml)
    requested = _configuration_authority(requested_yaml)
    if plan.cursor.name is not StageName.COURT_DATASET or request.targets != frozenset(
        {DatasetTarget.COURT}
    ):
        return existing == requested
    scoped_existing = _court_report_scoped_authority(
        existing, require_court_target=False
    )
    scoped_requested = _court_report_scoped_authority(
        requested, require_court_target=True
    )
    if scoped_existing is None or scoped_requested is None:
        return existing == requested
    return _court_report_scoped_authorities_match(
        scoped_existing,
        scoped_requested,
    )


def _court_report_scoped_authorities_match(
    existing: Mapping[str, object],
    requested: Mapping[str, object],
) -> bool:
    """Allow only typed-required NHT path additions to a legacy authority."""
    if existing == requested:
        return True
    existing_nht = existing.get("nht")
    requested_nht = requested.get("nht")
    if (
        not isinstance(existing_nht, Mapping)
        or any(not isinstance(key, str) for key in existing_nht)
        or not isinstance(requested_nht, Mapping)
        or any(not isinstance(key, str) for key in requested_nht)
    ):
        return False
    existing_keys = set(existing_nht)
    requested_keys = set(requested_nht)
    requested_only_keys = requested_keys - existing_keys
    if (
        not requested_only_keys
        or not requested_only_keys <= _LEGACY_COURT_REUSE_ADDITIVE_NHT_KEYS
        or existing_keys - requested_keys
    ):
        return False
    if any(
        not isinstance(requested_nht[key], str) or not requested_nht[key].strip()
        for key in requested_only_keys
    ):
        return False
    if any(requested_nht[key] != value for key, value in existing_nht.items()):
        return False
    normalized_requested = dict(requested)
    normalized_requested["nht"] = dict(existing_nht)
    return existing == normalized_requested


def _court_report_scoped_authority(
    authority: Mapping[str, object],
    *,
    require_court_target: bool,
) -> Mapping[str, object] | None:
    """Remove only fields owned by a Court-only cursor and its report."""
    scoped = deepcopy(dict(authority))
    dataset = scoped.get("dataset")
    request = scoped.get("request")
    if not isinstance(dataset, dict) or "court" not in dataset:
        return None
    if not isinstance(request, dict):
        return None
    targets = request.get("targets")
    if require_court_target and targets != [DatasetTarget.COURT.value]:
        return None
    dataset = dict(dataset)
    del dataset["court"]
    scoped["dataset"] = dataset
    request = dict(request)
    request.pop("targets", None)
    scoped["request"] = request
    return scoped


__all__ = ["ScenePipelineRunner"]
