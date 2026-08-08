"""Definition-driven lifecycle, invalidation, target, and failure tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest

from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    DatasetTarget,
    ScenePipelineRequest,
    ScenePipelineRunner,
    SceneWorkspace,
    StageExecutionSummary,
    StageName,
    StageStatus,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.contracts import StageExecutionContext
from src.synthetic_data_generation.pipeline.publication import (
    AtomicDirectoryPublication,
    AtomicPublicationUnavailableError,
)
from src.synthetic_data_generation.pipeline.registry import StageRegistry
from src.utils.configuration import PathResolver, RuntimePathRoots


@dataclass
class _FakeHandler:
    payload: str
    fail_preflight: bool = False
    fail_execute: bool = False
    invalid_summary: bool = False

    def preflight(self, context: StageExecutionContext) -> None:
        if self.fail_preflight:
            raise ValueError(f"preflight failed for {context.stage.name.value}")

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        destination = context.staging_path
        for relative in context.stage.required_outputs:
            path = destination / relative
            if relative.name in {
                "export",
                "diagnostics",
                "samples",
                "backgrounds",
                "scenes",
            }:
                path.mkdir(parents=True, exist_ok=True)
                (path / "manifest.json").write_text(self.payload, encoding="utf-8")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.payload, encoding="utf-8")
        if context.stage.name is StageName.RECONSTRUCTION:
            export = destination / "export"
            for name in ("scene.json", "cameras.json", "points_scene.npy"):
                (export / name).write_text(self.payload, encoding="utf-8")
            for name in ("images", "model"):
                (export / name).mkdir(exist_ok=True)
        if self.fail_execute:
            (destination / "partial.tmp").write_text("partial", encoding="utf-8")
            raise RuntimeError(f"execute failed for {context.stage.name.value}")
        if self.invalid_summary:
            return cast(StageExecutionSummary, {"payload": self.payload})
        return StageExecutionSummary({"payload": self.payload})

    def validate(self, context: StageExecutionContext) -> None:
        for relative in context.stage.required_outputs:
            if not (context.staging_path / relative).exists():
                raise ValueError(f"missing {relative}")


def _workspace(tmp_path: Path) -> SceneWorkspace:
    roots = RuntimePathRoots(
        project_root=tmp_path.resolve(),
        data_root=(tmp_path / "data").resolve(),
        checkpoint_root=(tmp_path / "ckpt").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        cache_root=(tmp_path / "cache").resolve(),
        external_asset_root=(tmp_path / "external").resolve(),
    )
    return SceneWorkspace.resolve(PathResolver(roots), "scene-a")


def _request(
    tmp_path: Path,
    *,
    from_stage: StageName = StageName.INGEST,
) -> ScenePipelineRequest:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source.resolve(),
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=from_stage,
        config_schema="scene_pipeline_v1",
    )


def _registry(
    *,
    payload: str,
) -> tuple[StageRegistry, dict[StageName, _FakeHandler]]:
    by_stage = {stage: _FakeHandler(payload) for stage in StageName}
    handlers = CanonicalStageHandlers(
        ingest=by_stage[StageName.INGEST],
        reconstruction=by_stage[StageName.RECONSTRUCTION],
        alignment=by_stage[StageName.ALIGNMENT],
        court_dataset=by_stage[StageName.COURT_DATASET],
        blcs_dataset=by_stage[StageName.BLCS_DATASET],
        plcs_dataset=by_stage[StageName.PLCS_DATASET],
        report=by_stage[StageName.REPORT],
    )
    return canonical_registry(handlers), by_stage


def _runner(
    tmp_path: Path,
    registry: StageRegistry,
    *,
    resolved_config_yaml: str = (
        "schema: scene_pipeline_v1\nrequest:\n  from_stage: ingest\n"
    ),
) -> ScenePipelineRunner:
    return ScenePipelineRunner(
        workspace=_workspace(tmp_path),
        registry=registry,
        resolved_config_yaml=resolved_config_yaml,
    )


def test_runner_generates_only_explicit_target_and_report(tmp_path: Path) -> None:
    request = _request(tmp_path)
    registry, _ = _registry(payload="first")
    runner = _runner(tmp_path, registry)

    manifest = runner.run(request)

    assert manifest.stages[StageName.COURT_DATASET].status is StageStatus.COMPLETED
    assert manifest.stages[StageName.BLCS_DATASET].status is StageStatus.SKIPPED
    assert manifest.stages[StageName.PLCS_DATASET].status is StageStatus.SKIPPED
    assert not (runner.workspace.root / "datasets/blcs/dataset.json").exists()
    assert not (runner.workspace.root / "datasets/plcs/dataset.json").exists()
    persisted = json.loads(
        runner.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    assert persisted["stages"]["report"]["status"] == "completed"
    assert not runner.workspace.transaction_root.exists()


def test_preflight_failure_does_not_invalidate_completed_outputs(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    court_dataset = first.workspace.root / "datasets/court/dataset.json"
    before = court_dataset.read_text(encoding="utf-8")
    second_registry, handlers = _registry(payload="second")
    handlers[StageName.ALIGNMENT].fail_preflight = True
    second = _runner(tmp_path, second_registry)

    with pytest.raises(ValueError, match="preflight failed"):
        second.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert court_dataset.read_text(encoding="utf-8") == before
    assert (second.workspace.root / "reconstruction/export/scene.json").exists()
    assert not second.workspace.transaction_root.exists()


def test_atomic_capability_failure_precedes_destructive_invalidation(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    court_dataset = first.workspace.root / "datasets/court/dataset.json"
    report = first.workspace.root / "report/report.json"

    def unavailable(source: Path, destination: Path, flags: int) -> None:
        raise AtomicPublicationUnavailableError("exchange unavailable")

    second_registry, _ = _registry(payload="second")
    definitions = {
        name: replace(definition)
        for name, definition in second_registry.definitions.items()
    }
    court = definitions[StageName.COURT_DATASET]
    definitions[StageName.COURT_DATASET] = replace(
        court,
        publication=AtomicDirectoryPublication(rename_operation=unavailable),
    )
    runner = _runner(tmp_path, StageRegistry(definitions))

    with pytest.raises(AtomicPublicationUnavailableError, match="unavailable"):
        runner.run(_request(tmp_path, from_stage=StageName.COURT_DATASET))

    assert court_dataset.read_text(encoding="utf-8") == "first"
    assert report.read_text(encoding="utf-8") == "first"


def test_alignment_rerun_preserves_reconstruction_and_atomically_replaces_cursor(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    reconstruction = first.workspace.root / "reconstruction/export/scene.json"
    before = reconstruction.read_text(encoding="utf-8")
    second_registry, _ = _registry(payload="second")
    second = _runner(tmp_path, second_registry)

    second.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert reconstruction.read_text(encoding="utf-8") == before
    assert (
        second.workspace.root / "alignment/alignment.json"
    ).read_text(encoding="utf-8") == "second"
    assert (
        second.workspace.root / "datasets/court/dataset.json"
    ).read_text(encoding="utf-8") == "second"
    assert not second.workspace.transaction_root.exists()


def test_rerun_cursor_can_change_but_production_configuration_cannot(
    tmp_path: Path,
) -> None:
    ingest_config = "request:\n  from_stage: ingest\nsettings:\n  seed: 695\n"
    alignment_config = "request:\n  from_stage: alignment\nsettings:\n  seed: 695\n"
    changed_config = "request:\n  from_stage: alignment\nsettings:\n  seed: 696\n"
    first_registry, _ = _registry(payload="first")
    first = _runner(
        tmp_path,
        first_registry,
        resolved_config_yaml=ingest_config,
    )
    first.run(_request(tmp_path))
    reconstruction = first.workspace.root / "reconstruction/export/scene.json"
    retained = reconstruction.read_text(encoding="utf-8")

    rerun_registry, _ = _registry(payload="second")
    rerun = _runner(
        tmp_path,
        rerun_registry,
        resolved_config_yaml=alignment_config,
    )
    rerun.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert reconstruction.read_text(encoding="utf-8") == retained
    changed_registry, _ = _registry(payload="forbidden")
    changed = _runner(
        tmp_path,
        changed_registry,
        resolved_config_yaml=changed_config,
    )
    with pytest.raises(ValueError, match="Resolved configuration changed"):
        changed.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))
    assert reconstruction.read_text(encoding="utf-8") == retained
    assert (
        changed.workspace.root / "datasets/court/dataset.json"
    ).read_text(encoding="utf-8") == "second"


def test_domain_failure_keeps_no_partial_or_completed_output(tmp_path: Path) -> None:
    registry, handlers = _registry(payload="first")
    handlers[StageName.COURT_DATASET].fail_execute = True
    runner = _runner(tmp_path, registry)

    with pytest.raises(RuntimeError, match="execute failed"):
        runner.run(_request(tmp_path))

    manifest = json.loads(
        runner.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["stages"]["court_dataset"]["status"] == "failed"
    assert manifest["stages"]["report"]["status"] == "invalidated"
    assert not (runner.workspace.root / "datasets/court").exists()
    assert not runner.workspace.transaction_root.exists()


def test_failed_fixed_path_rerun_retains_old_complete_owner_until_retry(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    court_dataset = first.workspace.root / "datasets/court/dataset.json"

    failed_registry, failed_handlers = _registry(payload="partial")
    failed_handlers[StageName.COURT_DATASET].fail_execute = True
    failed = _runner(tmp_path, failed_registry)
    with pytest.raises(RuntimeError, match="execute failed"):
        failed.run(_request(tmp_path, from_stage=StageName.COURT_DATASET))

    persisted = json.loads(
        failed.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    assert persisted["stages"]["court_dataset"]["status"] == "failed"
    assert court_dataset.read_text(encoding="utf-8") == "first"
    assert not failed.workspace.transaction_root.exists()

    retry_registry, _ = _registry(payload="retry")
    retry = _runner(tmp_path, retry_registry)
    retry.run(_request(tmp_path, from_stage=StageName.COURT_DATASET))

    assert court_dataset.read_text(encoding="utf-8") == "retry"
    assert not (court_dataset.parent / "partial.tmp").exists()


def test_definition_rejects_handler_summary_type_at_execution(tmp_path: Path) -> None:
    registry, handlers = _registry(payload="bad")
    handlers[StageName.INGEST].invalid_summary = True
    runner = _runner(tmp_path, registry)

    with pytest.raises(TypeError, match="expected StageExecutionSummary"):
        runner.run(_request(tmp_path))

    persisted = json.loads(
        runner.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    assert persisted["stages"]["ingest"]["status"] == "failed"
