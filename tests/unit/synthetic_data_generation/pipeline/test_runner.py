"""State, invalidation, target, and failure tests for the composition root."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.synthetic_data_generation.pipeline import (
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
from src.utils.configuration import PathResolver, RuntimePathRoots


@dataclass
class _FakeHandler:
    payload: str
    fail_preflight: bool = False
    fail_execute: bool = False

    def preflight(self, context: StageExecutionContext) -> None:
        if self.fail_preflight:
            raise ValueError(f"preflight failed for {context.stage.name.value}")

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        destination = (
            context.owner_path
            if context.stage.name is StageName.RECONSTRUCTION
            else context.staging_path
        )
        for relative in context.stage.required_outputs:
            path = destination / relative
            if relative.name not in {"diagnostics", "samples"}:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.payload, encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "manifest.json").write_text(self.payload, encoding="utf-8")
        if self.fail_execute:
            (destination / "partial.tmp").write_text("partial", encoding="utf-8")
            raise RuntimeError(f"execute failed for {context.stage.name.value}")
        return StageExecutionSummary({"payload": self.payload})

    def validate(self, context: StageExecutionContext) -> None:
        root = (
            context.owner_path
            if context.stage.name is StageName.RECONSTRUCTION
            else context.staging_path
        )
        for relative in context.stage.required_outputs:
            if not (root / relative).exists():
                raise ValueError(f"missing {relative}")


def _workspace(tmp_path) -> SceneWorkspace:
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


def _request(tmp_path, *, from_stage: StageName = StageName.INGEST) -> ScenePipelineRequest:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="scene-a",
        source_video=source.resolve(),
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=from_stage,
        config_schema="scene_pipeline_v1",
    )


def _handlers(*, payload: str) -> dict[str, _FakeHandler]:
    return {
        spec.handler_key: _FakeHandler(payload)
        for spec in canonical_registry().specs.values()
    }


def _runner(
    tmp_path,
    handlers,
    *,
    resolved_config_yaml: str = (
        "schema: scene_pipeline_v1\nrequest:\n  from_stage: ingest\n"
    ),
) -> ScenePipelineRunner:
    return ScenePipelineRunner(
        workspace=_workspace(tmp_path),
        registry=canonical_registry(),
        handlers=handlers,
        resolved_config_yaml=resolved_config_yaml,
    )


def test_runner_generates_only_explicit_target_and_report(tmp_path) -> None:
    request = _request(tmp_path)
    runner = _runner(tmp_path, _handlers(payload="first"))

    manifest = runner.run(request)

    assert manifest.stages[StageName.COURT_DATASET].status is StageStatus.COMPLETED
    assert manifest.stages[StageName.BLCS_DATASET].status is StageStatus.SKIPPED
    assert manifest.stages[StageName.PLCS_DATASET].status is StageStatus.SKIPPED
    assert not (runner.workspace.root / "datasets/blcs/dataset.json").exists()
    assert not (runner.workspace.root / "datasets/plcs/dataset.json").exists()
    persisted = json.loads(runner.workspace.run_manifest_path.read_text(encoding="utf-8"))
    assert persisted["stages"]["report"]["status"] == "completed"


def test_preflight_failure_does_not_invalidate_completed_outputs(tmp_path) -> None:
    first = _runner(tmp_path, _handlers(payload="first"))
    first.run(_request(tmp_path))
    court_dataset = first.workspace.root / "datasets/court/dataset.json"
    before = court_dataset.read_text(encoding="utf-8")
    handlers = _handlers(payload="second")
    handlers["alignment"].fail_preflight = True
    second = _runner(tmp_path, handlers)

    with pytest.raises(ValueError, match="preflight failed"):
        second.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert court_dataset.read_text(encoding="utf-8") == before
    assert (second.workspace.root / "reconstruction/export/scene.json").exists()


def test_alignment_rerun_preserves_reconstruction_and_replaces_descendants(tmp_path) -> None:
    first = _runner(tmp_path, _handlers(payload="first"))
    first.run(_request(tmp_path))
    reconstruction = first.workspace.root / "reconstruction/export/scene.json"
    before = reconstruction.read_text(encoding="utf-8")
    second = _runner(tmp_path, _handlers(payload="second"))

    second.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert reconstruction.read_text(encoding="utf-8") == before
    assert (second.workspace.root / "alignment/alignment.json").read_text(encoding="utf-8") == "second"
    assert (second.workspace.root / "datasets/court/dataset.json").read_text(encoding="utf-8") == "second"


def test_rerun_cursor_can_change_but_production_configuration_cannot(tmp_path) -> None:
    ingest_config = "request:\n  from_stage: ingest\nsettings:\n  seed: 695\n"
    alignment_config = "request:\n  from_stage: alignment\nsettings:\n  seed: 695\n"
    changed_config = "request:\n  from_stage: alignment\nsettings:\n  seed: 696\n"
    first = _runner(
        tmp_path,
        _handlers(payload="first"),
        resolved_config_yaml=ingest_config,
    )
    first.run(_request(tmp_path))
    reconstruction = first.workspace.root / "reconstruction/export/scene.json"
    retained = reconstruction.read_text(encoding="utf-8")

    rerun = _runner(
        tmp_path,
        _handlers(payload="second"),
        resolved_config_yaml=alignment_config,
    )
    rerun.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))

    assert reconstruction.read_text(encoding="utf-8") == retained
    changed = _runner(
        tmp_path,
        _handlers(payload="forbidden"),
        resolved_config_yaml=changed_config,
    )
    with pytest.raises(ValueError, match="Resolved configuration changed"):
        changed.run(_request(tmp_path, from_stage=StageName.ALIGNMENT))
    assert reconstruction.read_text(encoding="utf-8") == retained
    assert (
        changed.workspace.root / "datasets/court/dataset.json"
    ).read_text(encoding="utf-8") == "second"


def test_domain_failure_cannot_leave_partial_or_completed_output(tmp_path) -> None:
    handlers = _handlers(payload="first")
    handlers["court_dataset"].fail_execute = True
    runner = _runner(tmp_path, handlers)

    with pytest.raises(RuntimeError, match="execute failed"):
        runner.run(_request(tmp_path))

    manifest = json.loads(runner.workspace.run_manifest_path.read_text(encoding="utf-8"))
    assert manifest["stages"]["court_dataset"]["status"] == "failed"
    assert manifest["stages"]["report"]["status"] == "invalidated"
    assert not (runner.workspace.root / "datasets/court/dataset.json").exists()
    assert not (runner.workspace.root / "datasets/court/staging").exists()
