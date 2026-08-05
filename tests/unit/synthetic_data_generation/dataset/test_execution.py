"""Tests for path-driven execution and non-gating quality metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.synthetic_data_generation.dataset.execution import (
    build_dataset_plan,
    execute_pipeline,
    render_dataset,
)
from src.synthetic_data_generation.dataset.pipeline import PathPipelineManifest
from src.utils.configuration import PathContractError, RuntimePathRoots


def _manifest(tmp_path: Path) -> PathPipelineManifest:
    source = tmp_path / "third_party/nht/data"
    artifacts = tmp_path / "third_party/nht/artifacts/synthetic-data"
    outputs = tmp_path / "outputs/synthetic_data_generation"
    dataset = tmp_path / "data/synthetic_data_generation"
    return PathPipelineManifest(
        runtime_roots=RuntimePathRoots(
            project_root=tmp_path,
            data_root=tmp_path / "data",
            checkpoint_root=tmp_path / "ckpt",
            artifact_root=tmp_path / "third_party/nht/artifacts",
            output_root=tmp_path / "outputs",
            cache_root=tmp_path / ".cache",
            external_asset_root=tmp_path / "third_party",
        ),
        source_root=source,
        artifact_root=artifacts,
        execution_root=outputs,
        dataset_root=dataset,
        alignment_observations=source / "alignment-observations.json",
        render_jobs=source / "render-jobs.json",
        pipeline_manifest=outputs / "path-manifest.json",
        alignment_metrics=artifacts / "alignment-metrics.json",
        dataset_plan=artifacts / "dataset-plan.json",
        render_manifest=artifacts / "render-manifest.json",
        quality_metrics=artifacts / "quality-metrics.json",
        visualization=outputs / "pipeline-summary.html",
    )


def _write_inputs(manifest: PathPipelineManifest) -> None:
    manifest.source_root.mkdir(parents=True)
    manifest.alignment_observations.write_text(
        json.dumps({"residuals": [1_000_000.0, -1_000_000.0]}),
        encoding="utf-8",
    )
    (manifest.source_root / "prepared-render.bin").write_bytes(b"\x00" * 8)
    (manifest.source_root / "reference.bin").write_bytes(b"\xff" * 8)
    manifest.render_jobs.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "sample",
                        "input": "prepared-render.bin",
                        "output": "renders/sample.bin",
                        "reference": "reference.bin",
                        "arguments": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_poor_metrics_are_written_and_never_stop_downstream_stages(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)
    manifest.write()

    visualization = execute_pipeline(
        manifest,
        renderer_mode="prepared_outputs",
        renderer_command=(),
        working_directory=manifest.source_root,
    )

    alignment = json.loads(manifest.alignment_metrics.read_text())
    quality = json.loads(manifest.quality_metrics.read_text())
    assert alignment["root_mean_square_error"] == 1_000_000.0
    assert quality["mean_absolute_byte_error"] == 1.0
    assert manifest.dataset_plan.is_file()
    assert manifest.render_manifest.is_file()
    assert (manifest.dataset_root / "renders/sample.bin").is_file()
    assert visualization.is_file()
    assert "Synthetic-data pipeline summary" in visualization.read_text()


def test_missing_input_and_malformed_jobs_remain_errors(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.source_root.mkdir(parents=True)
    manifest.render_jobs.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed JSON"):
        build_dataset_plan(manifest, renderer_command=())

    manifest.render_jobs.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "missing",
                        "input": "missing.bin",
                        "output": "render.bin",
                        "reference": None,
                        "arguments": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Render input"):
        build_dataset_plan(manifest, renderer_command=())


def test_actual_renderer_failure_stops_execution(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)
    manifest.write()
    build_dataset_plan(
        manifest,
        renderer_command=(sys.executable, "-c", "raise SystemExit(7)"),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        render_dataset(
            manifest,
            renderer_mode="execute",
            working_directory=manifest.source_root,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input", "/tmp/outside-input.bin"),
        ("output", "/tmp/outside-output.bin"),
        ("reference", "/tmp/outside-reference.bin"),
        ("input", "../outside-input.bin"),
        ("output", "../outside-output.bin"),
        ("reference", "../outside-reference.bin"),
    ],
)
def test_render_jobs_reject_absolute_and_escaping_paths_before_output(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)
    payload = json.loads(manifest.render_jobs.read_text(encoding="utf-8"))
    payload["jobs"][0][field] = value
    manifest.render_jobs.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PathContractError):
        build_dataset_plan(manifest, renderer_command=())

    assert not manifest.dataset_plan.exists()


def test_tampered_dataset_plan_paths_are_rejected_before_side_effects(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)
    build_dataset_plan(manifest, renderer_command=())
    plan = json.loads(manifest.dataset_plan.read_text(encoding="utf-8"))
    plan["jobs"][0]["output"] = str(tmp_path / "outside.bin")
    manifest.dataset_plan.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(PathContractError):
        render_dataset(
            manifest,
            renderer_mode="prepared_outputs",
            working_directory=manifest.source_root,
        )

    assert not (tmp_path / "outside.bin").exists()
    assert not (manifest.execution_root / "renderer-logs").exists()


def test_render_jobs_require_explicit_closed_schema(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)
    payload = json.loads(manifest.render_jobs.read_text(encoding="utf-8"))
    del payload["jobs"][0]["arguments"]
    manifest.render_jobs.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*arguments"):
        build_dataset_plan(manifest, renderer_command=())


def test_renderer_placeholder_typos_fail_before_plan_publication(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    _write_inputs(manifest)

    with pytest.raises(ValueError, match="Unknown renderer path placeholder"):
        build_dataset_plan(manifest, renderer_command=("renderer", "{ouptut}"))

    assert not manifest.dataset_plan.exists()
