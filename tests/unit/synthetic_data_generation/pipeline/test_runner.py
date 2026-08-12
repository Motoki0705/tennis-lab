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
from src.synthetic_data_generation.pipeline.run_manifest import MutableRunManifest
from src.utils.configuration import PathResolver, RuntimePathRoots


@dataclass
class _FakeHandler:
    payload: str
    execution_order: list[StageName] | None = None
    fail_preflight: bool = False
    fail_execute: bool = False
    invalid_summary: bool = False
    preflight_calls: int = 0
    execute_calls: int = 0

    def preflight(self, context: StageExecutionContext) -> None:
        self.preflight_calls += 1
        if self.fail_preflight:
            raise ValueError(f"preflight failed for {context.stage.name.value}")

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        self.execute_calls += 1
        if self.execution_order is not None:
            self.execution_order.append(context.stage.name)
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
    targets: frozenset[DatasetTarget] = frozenset({DatasetTarget.COURT}),
) -> ScenePipelineRequest:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source.resolve(),
        targets=targets,
        from_stage=from_stage,
        config_schema="scene_pipeline_v1",
    )


def _registry(
    *,
    payload: str,
    execution_order: list[StageName] | None = None,
) -> tuple[StageRegistry, dict[StageName, _FakeHandler]]:
    by_stage = {
        stage: _FakeHandler(payload, execution_order=execution_order)
        for stage in StageName
    }
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


def test_incompatible_cursor_is_rejected_without_workspace_mutation(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    before = {
        path.relative_to(first.workspace.root): path.read_bytes()
        for path in first.workspace.root.rglob("*")
        if path.is_file()
    }
    run_json_before = first.workspace.run_manifest_path.read_bytes()
    court_before = (first.workspace.root / "datasets/court/dataset.json").read_bytes()
    report_before = (first.workspace.root / "report/report.json").read_bytes()
    second_registry, handlers = _registry(payload="forbidden")
    second = _runner(tmp_path, second_registry)

    with pytest.raises(ValueError, match="not selected by request targets"):
        second.run(_request(tmp_path, from_stage=StageName.PLCS_DATASET))

    after = {
        path.relative_to(second.workspace.root): path.read_bytes()
        for path in second.workspace.root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert second.workspace.run_manifest_path.read_bytes() == run_json_before
    assert (second.workspace.root / "datasets/court/dataset.json").read_bytes() == court_before
    assert (second.workspace.root / "report/report.json").read_bytes() == report_before
    assert all(handler.preflight_calls == 0 for handler in handlers.values())
    assert all(handler.execute_calls == 0 for handler in handlers.values())


def test_dataset_cursor_does_not_rerun_selected_sibling_datasets(
    tmp_path: Path,
) -> None:
    all_targets = frozenset(DatasetTarget)
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=all_targets))
    blcs = first.workspace.root / "datasets/blcs/dataset.json"
    plcs = first.workspace.root / "datasets/plcs/dataset.json"
    second_registry, handlers = _registry(payload="second")
    second = _runner(tmp_path, second_registry)

    manifest = second.run(
        _request(
            tmp_path,
            from_stage=StageName.COURT_DATASET,
            targets=all_targets,
        )
    )

    assert manifest.stages[StageName.COURT_DATASET].attempt == 2
    assert manifest.stages[StageName.REPORT].attempt == 2
    for sibling in (StageName.BLCS_DATASET, StageName.PLCS_DATASET):
        assert manifest.stages[sibling].status is StageStatus.COMPLETED
        assert manifest.stages[sibling].attempt == 1
        assert handlers[sibling].preflight_calls == 0
        assert handlers[sibling].execute_calls == 0
    assert blcs.read_text(encoding="utf-8") == "first"
    assert plcs.read_text(encoding="utf-8") == "first"


def test_blcs_cursor_repairs_invalidated_plcs_before_report(tmp_path: Path) -> None:
    all_targets = frozenset(DatasetTarget)
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=all_targets))
    manifest = MutableRunManifest.load(first.workspace.run_manifest_path)
    manifest.stages[StageName.BLCS_DATASET].attempt = 2
    manifest.invalidate(StageName.PLCS_DATASET)
    manifest.stages[StageName.PLCS_DATASET].attempt = 0
    manifest.invalidate(StageName.REPORT)
    manifest.save(first.workspace.run_manifest_path)
    first.workspace.invalidate_outputs(
        first_registry.definition(StageName.PLCS_DATASET)
    )
    first.workspace.invalidate_outputs(first_registry.definition(StageName.REPORT))
    execution_order: list[StageName] = []
    repair_registry, handlers = _registry(
        payload="repair",
        execution_order=execution_order,
    )
    repair = _runner(tmp_path, repair_registry)

    repaired = repair.run(
        _request(
            tmp_path,
            from_stage=StageName.BLCS_DATASET,
            targets=all_targets,
        )
    )

    assert execution_order == [
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    ]
    assert repaired.stages[StageName.BLCS_DATASET].attempt == 3
    assert repaired.stages[StageName.PLCS_DATASET].attempt == 1
    assert repaired.stages[StageName.REPORT].status is StageStatus.COMPLETED
    assert handlers[StageName.COURT_DATASET].execute_calls == 0
    assert (
        repair.workspace.root / "datasets/plcs/dataset.json"
    ).read_text(encoding="utf-8") == "repair"


def test_missing_completed_sibling_publication_is_rebuilt(tmp_path: Path) -> None:
    targets = frozenset({DatasetTarget.BLCS, DatasetTarget.PLCS})
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=targets))
    missing = first.workspace.root / "datasets/plcs/dataset.json"
    missing.unlink()
    execution_order: list[StageName] = []
    repair_registry, _ = _registry(
        payload="repair",
        execution_order=execution_order,
    )

    repaired = _runner(tmp_path, repair_registry).run(
        _request(
            tmp_path,
            from_stage=StageName.BLCS_DATASET,
            targets=targets,
        )
    )

    assert execution_order == [
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    ]
    assert repaired.stages[StageName.PLCS_DATASET].attempt == 2
    assert missing.read_text(encoding="utf-8") == "repair"


def test_stale_completed_descendants_are_rebuilt_from_invalid_prerequisite(
    tmp_path: Path,
) -> None:
    all_targets = frozenset(DatasetTarget)
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=all_targets))
    manifest = MutableRunManifest.load(first.workspace.run_manifest_path)
    manifest.invalidate(StageName.ALIGNMENT)
    manifest.save(first.workspace.run_manifest_path)
    execution_order: list[StageName] = []
    repair_registry, _ = _registry(
        payload="repair",
        execution_order=execution_order,
    )

    repaired = _runner(tmp_path, repair_registry).run(
        _request(
            tmp_path,
            from_stage=StageName.REPORT,
            targets=all_targets,
        )
    )

    assert execution_order == [
        StageName.ALIGNMENT,
        StageName.COURT_DATASET,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    ]
    assert repaired.stages[StageName.INGEST].attempt == 1
    assert repaired.stages[StageName.RECONSTRUCTION].attempt == 1
    assert repaired.stages[StageName.ALIGNMENT].attempt == 2
    assert all(repaired.stages[target.stage].attempt == 2 for target in DatasetTarget)


def test_repaired_sibling_failure_keeps_report_invalidated_and_no_partial_output(
    tmp_path: Path,
) -> None:
    targets = frozenset({DatasetTarget.BLCS, DatasetTarget.PLCS})
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=targets))
    manifest = MutableRunManifest.load(first.workspace.run_manifest_path)
    plcs_record = manifest.stages[StageName.PLCS_DATASET]
    plcs_record.status = StageStatus.FAILED
    plcs_record.summary = {}
    plcs_record.error = "RuntimeError: prior failed PLCS replacement"
    manifest.invalidate(StageName.REPORT)
    manifest.save(first.workspace.run_manifest_path)
    first.workspace.invalidate_outputs(first_registry.definition(StageName.REPORT))
    repair_registry, handlers = _registry(payload="partial")
    handlers[StageName.PLCS_DATASET].fail_execute = True
    repair = _runner(tmp_path, repair_registry)

    with pytest.raises(RuntimeError, match="execute failed for plcs_dataset"):
        repair.run(
            _request(
                tmp_path,
                from_stage=StageName.BLCS_DATASET,
                targets=targets,
            )
        )

    failed = MutableRunManifest.load(repair.workspace.run_manifest_path)
    assert failed.stages[StageName.BLCS_DATASET].status is StageStatus.COMPLETED
    assert failed.stages[StageName.PLCS_DATASET].status is StageStatus.FAILED
    assert failed.stages[StageName.REPORT].status is StageStatus.INVALIDATED
    assert (
        repair.workspace.root / "datasets/plcs/dataset.json"
    ).read_text(encoding="utf-8") == "first"
    assert not repair.workspace.transaction_root.exists()


def test_alignment_subset_rerun_cleans_unselected_stale_descendants(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=frozenset(DatasetTarget)))
    second_registry, handlers = _registry(payload="second")
    second = _runner(tmp_path, second_registry)

    manifest = second.run(
        _request(
            tmp_path,
            from_stage=StageName.ALIGNMENT,
            targets=frozenset({DatasetTarget.COURT}),
        )
    )

    for target in (DatasetTarget.BLCS, DatasetTarget.PLCS):
        record = manifest.stages[target.stage]
        assert record.status is StageStatus.SKIPPED
        assert record.attempt == 1
        assert handlers[target.stage].execute_calls == 0
        assert not (second.workspace.root / "datasets" / target.value).exists()
    assert manifest.stages[StageName.ALIGNMENT].attempt == 2
    assert manifest.stages[StageName.COURT_DATASET].attempt == 2
    assert manifest.stages[StageName.REPORT].attempt == 2


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


def test_interrupted_cursor_is_recovered_and_retried_after_preflight(
    tmp_path: Path,
) -> None:
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path))
    persisted = json.loads(
        first.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    persisted["stages"]["court_dataset"]["status"] = "running"
    first.workspace.run_manifest_path.write_text(
        json.dumps(persisted), encoding="utf-8"
    )
    second_registry, _ = _registry(payload="second")
    second = _runner(tmp_path, second_registry)

    manifest = second.run(
        _request(tmp_path, from_stage=StageName.COURT_DATASET)
    )

    court = manifest.stages[StageName.COURT_DATASET]
    assert court.status is StageStatus.COMPLETED
    assert court.attempt == 2
    assert (
        second.workspace.root / "datasets/court/dataset.json"
    ).read_text(encoding="utf-8") == "second"


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
