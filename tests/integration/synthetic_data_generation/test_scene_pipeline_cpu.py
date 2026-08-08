"""CPU integration of the canonical workspace, DAG, and failure recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
class _FileBoundaryHandler:
    """Fake external/domain boundary with the same publication semantics."""

    payload: str
    fail_after_partial: bool = False

    def preflight(self, context: StageExecutionContext) -> None:
        if context.request.scene_id != context.owner_path.parent.name and (
            context.stage.name is StageName.RECONSTRUCTION
        ):
            raise ValueError("reconstruction owner disagrees with scene")

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        root = (
            context.owner_path
            if context.stage.name is StageName.RECONSTRUCTION
            else context.staging_path
        )
        for relative in context.stage.required_outputs:
            path = root / relative
            if relative.name in {"diagnostics", "samples"}:
                path.mkdir(parents=True, exist_ok=True)
                (path / "inventory.json").write_text(
                    json.dumps({"stage": context.stage.name.value}),
                    encoding="utf-8",
                )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.payload, encoding="utf-8")
        if self.fail_after_partial:
            (root / "partial.tmp").write_text("partial", encoding="utf-8")
            raise RuntimeError(f"injected {context.stage.name.value} failure")
        return StageExecutionSummary(
            {"stage": context.stage.name.value, "payload": self.payload}
        )

    def validate(self, context: StageExecutionContext) -> None:
        root = (
            context.owner_path
            if context.stage.name is StageName.RECONSTRUCTION
            else context.staging_path
        )
        missing = [
            str(relative)
            for relative in context.stage.required_outputs
            if not (root / relative).exists()
        ]
        if missing:
            raise ValueError(f"missing staged outputs: {missing}")


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
    return SceneWorkspace.resolve(PathResolver(roots), "B00")


def _request(tmp_path: Path, *, from_stage: StageName) -> ScenePipelineRequest:
    video = tmp_path / "B00.mp4"
    video.write_bytes(b"video")
    return ScenePipelineRequest(
        scene_id="B00",
        source_video=video.resolve(),
        targets=frozenset(DatasetTarget),
        from_stage=from_stage,
        config_schema="canonical_scene_pipeline_v1",
    )


def _handlers(payload: str) -> dict[str, _FileBoundaryHandler]:
    return {
        spec.handler_key: _FileBoundaryHandler(payload)
        for spec in canonical_registry().specs.values()
    }


def _runner(
    tmp_path: Path,
    handlers: dict[str, _FileBoundaryHandler],
) -> ScenePipelineRunner:
    return ScenePipelineRunner(
        workspace=_workspace(tmp_path),
        registry=canonical_registry(),
        handlers=handlers,
        resolved_config_yaml=(
            "schema: canonical_scene_pipeline_v1\n"
            "request:\n"
            "  from_stage: ingest\n"
        ),
    )


def test_all_domains_publish_once_and_reconstruction_rerun_replaces_everything(
    tmp_path: Path,
) -> None:
    first = _runner(tmp_path, _handlers("first"))
    first_manifest = first.run(_request(tmp_path, from_stage=StageName.INGEST))

    assert all(
        record.status is StageStatus.COMPLETED
        for record in first_manifest.stages.values()
    )
    assert [path.name for path in first.workspace.root.glob("run.json")] == [
        "run.json"
    ]
    for domain in DatasetTarget:
        assert (
            first.workspace.root / "datasets" / domain.value / "dataset.json"
        ).read_text(encoding="utf-8") == "first"

    second = _runner(tmp_path, _handlers("second"))
    second.run(_request(tmp_path, from_stage=StageName.RECONSTRUCTION))

    assert (
        second.workspace.root / "reconstruction/export/scene.json"
    ).read_text(encoding="utf-8") == "second"
    for domain in DatasetTarget:
        root = second.workspace.root / "datasets" / domain.value
        assert (root / "dataset.json").read_text(encoding="utf-8") == "second"
        assert not (root / "staging").exists()


def test_failed_domain_is_unpublished_and_clean_retry_cannot_reuse_partial_data(
    tmp_path: Path,
) -> None:
    handlers = _handlers("failed-attempt")
    handlers["plcs_dataset"].fail_after_partial = True
    failing = _runner(tmp_path, handlers)

    with pytest.raises(RuntimeError, match="injected plcs_dataset failure"):
        failing.run(_request(tmp_path, from_stage=StageName.INGEST))

    persisted = json.loads(
        failing.workspace.run_manifest_path.read_text(encoding="utf-8")
    )
    assert persisted["stages"]["plcs_dataset"]["status"] == "failed"
    assert persisted["stages"]["report"]["status"] == "invalidated"
    failed_root = failing.workspace.root / "datasets/plcs"
    assert not (failed_root / "dataset.json").exists()
    assert not (failed_root / "staging").exists()

    retry = _runner(tmp_path, _handlers("retry"))
    result = retry.run(_request(tmp_path, from_stage=StageName.PLCS_DATASET))

    assert result.stages[StageName.PLCS_DATASET].status is StageStatus.COMPLETED
    assert (failed_root / "dataset.json").read_text(encoding="utf-8") == "retry"
    assert not tuple(failed_root.rglob("partial.tmp"))
