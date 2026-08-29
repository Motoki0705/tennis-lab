"""Definition-driven lifecycle, invalidation, target, and failure tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCS_COORDINATE_CONTRACT,
    PLCSSourceSupportPlane,
)
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
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


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
        if context.stage.name is StageName.PLCS_DATASET:
            (destination / "dataset.json").write_text(
                json.dumps(_plcs_publication(self.payload)),
                encoding="utf-8",
            )
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


def _plcs_publication(payload: str) -> dict[str, object]:
    return {
        "schema": PLCS_DATASET_SCHEMA,
        "domain": "plcs",
        "fixture_payload": payload,
        "metadata": {
            "coordinate_contract": PLCS_COORDINATE_CONTRACT.to_dict(),
            "court_coordinate_normalization": (
                court_coordinate_normalization_metadata()
            ),
            "logical_scenes": [
                {
                    "tracks": [
                        {
                            "support_plane": (
                                PLCSSourceSupportPlane.from_surface_minimum(
                                    initial_root_translation_z_m=0.0,
                                    support_local_z_m=0.0,
                                ).to_dict()
                            )
                        }
                    ]
                }
            ],
        },
    }


def _request(
    tmp_path: Path,
    *,
    from_stage: StageName = StageName.INGEST,
    targets: frozenset[DatasetTarget] = frozenset({DatasetTarget.COURT}),
    source_video: Path | None = None,
) -> ScenePipelineRequest:
    source = source_video or tmp_path / "source.mp4"
    if source_video is None:
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


def _court_reuse_config_yaml(
    *,
    from_stage: StageName,
    targets: frozenset[DatasetTarget],
    court_schema_version: str,
    nht: dict[str, object],
    pipeline_seed: int = 695,
    source_video: Path | None = None,
) -> str:
    """Build the minimal resolved authority used by Court-only reuse tests."""
    request: dict[str, object] = {
        "from_stage": from_stage.value,
        "targets": sorted(target.value for target in targets),
    }
    if source_video is not None:
        request["source_video"] = str(source_video.resolve())
    return yaml.safe_dump(
        {
            "request": request,
            "dataset": {
                "court": {
                    "schema_version": court_schema_version,
                    "view": (
                        "four-targets"
                        if court_schema_version == "v1"
                        else "court-center-only"
                    ),
                },
                "blcs": {"schema": "stable-blcs"},
                "plcs": {"schema": "stable-plcs"},
            },
            "nht": nht,
            "pipeline": {"seed": pipeline_seed},
        },
        sort_keys=False,
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
    assert (
        second.workspace.root / "datasets/court/dataset.json"
    ).read_bytes() == court_before
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
    plcs_manifest = _plcs_publication("first")
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
    assert json.loads(plcs.read_text(encoding="utf-8")) == plcs_manifest


@pytest.mark.parametrize(
    "mutation",
    ("v4-schema", "missing-coordinate-contract", "invalid-support-schema"),
)
def test_completed_plcs_with_stale_coordinate_contract_is_atomically_rebuilt(
    tmp_path: Path,
    mutation: str,
) -> None:
    targets = frozenset({DatasetTarget.BLCS, DatasetTarget.PLCS})
    first_registry, _ = _registry(payload="first")
    first = _runner(tmp_path, first_registry)
    first.run(_request(tmp_path, targets=targets))
    owner = first.workspace.root / "datasets" / "plcs"
    manifest = _plcs_publication("first")
    stale = deepcopy(manifest)
    if mutation == "v4-schema":
        stale["schema"] = "tennis_plcs_compact_dataset_v4"
    elif mutation == "missing-coordinate-contract":
        metadata = cast(dict[str, object], stale["metadata"])
        del metadata["coordinate_contract"]
    else:
        metadata = cast(dict[str, object], stale["metadata"])
        logical_scenes = cast(list[object], metadata["logical_scenes"])
        logical_scene = cast(dict[str, object], logical_scenes[0])
        tracks = cast(list[object], logical_scene["tracks"])
        track = cast(dict[str, object], tracks[0])
        support = cast(dict[str, object], track["support_plane"])
        support["schema"] = "plcs_initial_foot_joint_support_v1"
    (owner / "dataset.json").write_text(json.dumps(stale), encoding="utf-8")
    execution_order: list[StageName] = []
    repair_registry, handlers = _registry(
        payload="replacement",
        execution_order=execution_order,
    )

    repaired = _runner(tmp_path, repair_registry).run(
        _request(
            tmp_path,
            from_stage=StageName.BLCS_DATASET,
            targets=targets,
        )
    )

    assert repaired.stages[StageName.BLCS_DATASET].attempt == 2
    assert repaired.stages[StageName.PLCS_DATASET].attempt == 2
    assert execution_order == [
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
        StageName.REPORT,
    ]
    assert handlers[StageName.PLCS_DATASET].execute_calls == 1
    assert json.loads((owner / "dataset.json").read_text(encoding="utf-8")) == (
        _plcs_publication("replacement")
    )
    assert not first.workspace.transaction_root.exists()


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
    assert json.loads(
        (repair.workspace.root / "datasets/plcs/dataset.json").read_text(
            encoding="utf-8"
        )
    ) == _plcs_publication("repair")


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
    assert json.loads(missing.read_text(encoding="utf-8")) == _plcs_publication(
        "repair"
    )


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
    assert json.loads(
        (repair.workspace.root / "datasets/plcs/dataset.json").read_text(
            encoding="utf-8"
        )
    ) == _plcs_publication("first")
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
    assert (second.workspace.root / "alignment/alignment.json").read_text(
        encoding="utf-8"
    ) == "second"
    assert (second.workspace.root / "datasets/court/dataset.json").read_text(
        encoding="utf-8"
    ) == "second"
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
    assert (changed.workspace.root / "datasets/court/dataset.json").read_text(
        encoding="utf-8"
    ) == "second"


def test_relocated_identical_source_is_reusable_outside_court_exception(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original" / "source.mp4"
    original.parent.mkdir()
    original.write_bytes(b"portable-video")
    relocated = tmp_path / "relocated" / "source.mp4"
    relocated.parent.mkdir()
    relocated.write_bytes(original.read_bytes())
    first_config = yaml.safe_dump(
        {
            "request": {
                "from_stage": StageName.INGEST.value,
                "source_video": str(original.resolve()),
            },
            "settings": {"seed": 695},
        },
        sort_keys=False,
    )
    rerun_config = yaml.safe_dump(
        {
            "request": {
                "from_stage": StageName.ALIGNMENT.value,
                "source_video": str(relocated.resolve()),
            },
            "settings": {"seed": 695},
        },
        sort_keys=False,
    )
    first_registry, _ = _registry(payload="first")
    first = _runner(
        tmp_path,
        first_registry,
        resolved_config_yaml=first_config,
    )
    first.run(_request(tmp_path, source_video=original))
    canonical_source = first.workspace.root / "source/video.mp4"
    canonical_source.write_bytes(original.read_bytes())
    reconstruction = first.workspace.root / "reconstruction/export/scene.json"
    retained = reconstruction.read_bytes()

    rerun_registry, handlers = _registry(payload="relocated")
    rerun = _runner(
        tmp_path,
        rerun_registry,
        resolved_config_yaml=rerun_config,
    )
    rerun.run(
        _request(
            tmp_path,
            from_stage=StageName.ALIGNMENT,
            source_video=relocated,
        )
    )

    assert reconstruction.read_bytes() == retained
    assert handlers[StageName.INGEST].execute_calls == 0
    assert handlers[StageName.RECONSTRUCTION].execute_calls == 0
    assert handlers[StageName.ALIGNMENT].execute_calls == 1


def test_relocated_source_with_different_bytes_fails_before_mutation(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original" / "source.mp4"
    original.parent.mkdir()
    original.write_bytes(b"canonical-video")
    first_config = yaml.safe_dump(
        {
            "request": {
                "from_stage": StageName.INGEST.value,
                "source_video": str(original.resolve()),
            },
            "settings": {"seed": 695},
        },
        sort_keys=False,
    )
    first_registry, _ = _registry(payload="first")
    first = _runner(
        tmp_path,
        first_registry,
        resolved_config_yaml=first_config,
    )
    first.run(_request(tmp_path, source_video=original))
    canonical_source = first.workspace.root / "source/video.mp4"
    canonical_source.write_bytes(original.read_bytes())
    manifest_before = first.workspace.run_manifest_path.read_bytes()
    config_before = first.workspace.resolved_config_path.read_bytes()

    relocated = tmp_path / "relocated" / "source.mp4"
    relocated.parent.mkdir()
    relocated.write_bytes(b"different-video")
    rerun_config = yaml.safe_dump(
        {
            "request": {
                "from_stage": StageName.ALIGNMENT.value,
                "source_video": str(relocated.resolve()),
            },
            "settings": {"seed": 695},
        },
        sort_keys=False,
    )
    rerun_registry, handlers = _registry(payload="forbidden")
    rerun = _runner(
        tmp_path,
        rerun_registry,
        resolved_config_yaml=rerun_config,
    )

    with pytest.raises(ValueError, match="source video disagrees"):
        rerun.run(
            _request(
                tmp_path,
                from_stage=StageName.ALIGNMENT,
                source_video=relocated,
            )
        )

    assert all(handler.execute_calls == 0 for handler in handlers.values())
    assert rerun.workspace.run_manifest_path.read_bytes() == manifest_before
    assert rerun.workspace.resolved_config_path.read_bytes() == config_before


def test_court_only_cursor_allows_only_court_config_change_and_reuses_upstream(
    tmp_path: Path,
) -> None:
    all_targets = frozenset(DatasetTarget)
    original_source = tmp_path / "original" / "source.mp4"
    original_source.parent.mkdir()
    original_source.write_bytes(b"portable-court-video")
    relocated_source = tmp_path / "relocated" / "source.mp4"
    relocated_source.parent.mkdir()
    relocated_source.write_bytes(original_source.read_bytes())
    first_yaml = _court_reuse_config_yaml(
        from_stage=StageName.INGEST,
        targets=all_targets,
        court_schema_version="v1",
        nht={"backend": "public-cli"},
        source_video=original_source,
    )
    court_v2_yaml = _court_reuse_config_yaml(
        from_stage=StageName.COURT_DATASET,
        targets=frozenset({DatasetTarget.COURT}),
        court_schema_version="v2",
        nht={
            "backend": "public-cli",
            "training_python_path": "/runtime/python",
            "trainer_path": "/runtime/nht/train.py",
        },
        source_video=relocated_source,
    )
    first_registry, _ = _registry(payload="retained")
    first = _runner(tmp_path, first_registry, resolved_config_yaml=first_yaml)
    first.run(
        _request(
            tmp_path,
            targets=all_targets,
            source_video=original_source,
        )
    )
    (first.workspace.root / "source/video.mp4").write_bytes(
        original_source.read_bytes()
    )
    retained_paths = {
        StageName.RECONSTRUCTION: first.workspace.root
        / "reconstruction/export/scene.json",
        StageName.ALIGNMENT: first.workspace.root / "alignment/alignment.json",
        StageName.BLCS_DATASET: first.workspace.root / "datasets/blcs/dataset.json",
        StageName.PLCS_DATASET: first.workspace.root / "datasets/plcs/dataset.json",
    }
    before = {stage: path.read_bytes() for stage, path in retained_paths.items()}

    second_registry, handlers = _registry(payload="court-v2")
    second = _runner(tmp_path, second_registry, resolved_config_yaml=court_v2_yaml)
    second.run(
        _request(
            tmp_path,
            from_stage=StageName.COURT_DATASET,
            targets=frozenset({DatasetTarget.COURT}),
            source_video=relocated_source,
        )
    )

    assert handlers[StageName.COURT_DATASET].execute_calls == 1
    assert handlers[StageName.REPORT].execute_calls == 1
    for retained_stage in (
        StageName.INGEST,
        StageName.RECONSTRUCTION,
        StageName.ALIGNMENT,
        StageName.BLCS_DATASET,
        StageName.PLCS_DATASET,
    ):
        assert handlers[retained_stage].execute_calls == 0
    assert {
        stage: path.read_bytes() for stage, path in retained_paths.items()
    } == before
    assert (second.workspace.root / "datasets/court/dataset.json").read_text(
        encoding="utf-8"
    ) == "court-v2"

    forbidden_yaml = _court_reuse_config_yaml(
        from_stage=StageName.COURT_DATASET,
        targets=frozenset({DatasetTarget.COURT}),
        court_schema_version="v2",
        nht={
            "backend": "public-cli",
            "training_python_path": "/runtime/python",
            "trainer_path": "/runtime/nht/train.py",
        },
        pipeline_seed=696,
        source_video=relocated_source,
    )
    forbidden_registry, forbidden_handlers = _registry(payload="forbidden")
    forbidden = _runner(
        tmp_path,
        forbidden_registry,
        resolved_config_yaml=forbidden_yaml,
    )
    with pytest.raises(ValueError, match="Resolved configuration changed"):
        forbidden.run(
            _request(
                tmp_path,
                from_stage=StageName.COURT_DATASET,
                targets=frozenset({DatasetTarget.COURT}),
                source_video=relocated_source,
            )
        )
    assert all(handler.execute_calls == 0 for handler in forbidden_handlers.values())
    assert {
        stage: path.read_bytes() for stage, path in retained_paths.items()
    } == before


@pytest.mark.parametrize(
    ("nht", "from_stage", "targets"),
    (
        (
            {
                "backend": "mutated-cli",
                "training_python_path": "/runtime/python",
                "trainer_path": "/runtime/nht/train.py",
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "training_python_path": "/runtime/python",
                "trainer_path": "/runtime/nht/train.py",
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "backend": "public-cli",
                "training_python_path": "/runtime/python",
                "trainer_path": "/runtime/nht/train.py",
                "workspace_path": "/unowned/path",
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "backend": "public-cli",
                "training_python_path": " ",
                "trainer_path": "/runtime/nht/train.py",
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "backend": "public-cli",
                "training_python_path": "/runtime/python",
                "trainer_path": 7,
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "backend": "public-cli",
                "training_python_path": "/runtime/python",
                "trainer_path": "/runtime/nht/train.py",
            },
            StageName.ALIGNMENT,
            frozenset({DatasetTarget.COURT}),
        ),
        (
            {
                "backend": "public-cli",
                "training_python_path": "/runtime/python",
                "trainer_path": "/runtime/nht/train.py",
            },
            StageName.COURT_DATASET,
            frozenset({DatasetTarget.COURT, DatasetTarget.BLCS}),
        ),
    ),
    ids=(
        "existing-nht-value-mutated",
        "existing-nht-key-removed",
        "unrelated-nht-key-added",
        "added-path-is-blank",
        "added-path-is-not-a-string",
        "wrong-cursor",
        "wrong-target-set",
    ),
)
def test_legacy_nht_path_additions_are_rejected_outside_exact_court_authority(
    tmp_path: Path,
    nht: dict[str, object],
    from_stage: StageName,
    targets: frozenset[DatasetTarget],
) -> None:
    all_targets = frozenset(DatasetTarget)
    legacy_yaml = _court_reuse_config_yaml(
        from_stage=StageName.INGEST,
        targets=all_targets,
        court_schema_version="v1",
        nht={"backend": "public-cli"},
    )
    first_registry, _ = _registry(payload="retained")
    first = _runner(tmp_path, first_registry, resolved_config_yaml=legacy_yaml)
    first.run(_request(tmp_path, targets=all_targets))
    resolved_config_before = first.workspace.resolved_config_path.read_bytes()
    requested_yaml = _court_reuse_config_yaml(
        from_stage=from_stage,
        targets=targets,
        court_schema_version="v2",
        nht=nht,
    )
    second_registry, handlers = _registry(payload="forbidden")
    second = _runner(tmp_path, second_registry, resolved_config_yaml=requested_yaml)

    with pytest.raises(ValueError, match="Resolved configuration changed"):
        second.run(_request(tmp_path, from_stage=from_stage, targets=targets))

    assert second.workspace.resolved_config_path.read_bytes() == resolved_config_before
    assert all(handler.execute_calls == 0 for handler in handlers.values())


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

    manifest = second.run(_request(tmp_path, from_stage=StageName.COURT_DATASET))

    court = manifest.stages[StageName.COURT_DATASET]
    assert court.status is StageStatus.COMPLETED
    assert court.attempt == 2
    assert (second.workspace.root / "datasets/court/dataset.json").read_text(
        encoding="utf-8"
    ) == "second"


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
