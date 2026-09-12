"""Opt-in real-CUDA validation for rendered alignment line inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw, ImageFont

from src.synthetic_data_generation.alignment.evidence_source import (
    ProductionCourtLineDetector,
    _alignment_line_heatmaps,
    _classify_observable_cameras,
    _estimate_ground_plane,
    _fixed_camera_selection,
    _partition_cameras_with_holdout_tail,
    _project_probability_to_ground,
    _require_observable_camera_minima,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    WEIGHTED_PROJECTION_HEATMAP_FILE,
    aggregate_line_heatmaps,
    validate_line_heatmaps,
    write_line_heatmaps,
)
from src.synthetic_data_generation.alignment.line_inference_cache import (
    load_or_predict_line_probabilities,
)
from src.synthetic_data_generation.alignment.line_inputs import (
    AlignmentLineInputBatch,
    CourtLineInputSource,
    NHTRenderedAlignmentLineInputSource,
)
from src.synthetic_data_generation.alignment.validation import (
    load_alignment_evidence,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.reconstruction import (
    validate_standard_scene_export,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderClient

_ENABLE_ENVIRONMENT_VARIABLE = "TENNIS_RUN_B00_RENDERED_ALIGNMENT_GPU_TEST"
_FIXTURE_ROOT_ENVIRONMENT_VARIABLE = "TENNIS_B00_FIXTURE_REPOSITORY"


@pytest.mark.skipif(
    os.environ.get(_ENABLE_ENVIRONMENT_VARIABLE) != "1",
    reason=f"set {_ENABLE_ENVIRONMENT_VARIABLE}=1 inside the shared GPU queue",
)
def test_real_b00_rendered_line_detection_and_projection() -> None:
    """Render B00 views, detect their lines, and write the projected heatmap."""
    code_repository = Path(__file__).resolve().parents[3]
    fixture_repository = Path(
        os.environ.get(_FIXTURE_ROOT_ENVIRONMENT_VARIABLE, code_repository)
    ).resolve(strict=True)
    repro_value = os.environ.get("TENNIS_REPRO_DIR")
    if repro_value is None:
        pytest.fail("The real B00 CUDA validation must run through training-queue.")
    repro_root = Path(cast(str, repro_value)).resolve() / "b00-rendered-alignment"
    repro_root.mkdir(parents=True, exist_ok=False)

    runtime = _runtime_configuration(
        code_repository=code_repository,
        fixture_repository=fixture_repository,
    )
    scene_root = runtime.workspace.root
    scene = validate_standard_scene_export(
        scene_root / "reconstruction/export/scene.json"
    )
    baseline_alignment_root = scene_root / "alignment"
    baseline_projection_path = (
        baseline_alignment_root / "line-heatmaps" / WEIGHTED_PROJECTION_HEATMAP_FILE
    )
    for required in (
        baseline_alignment_root / "ground-line-map.npz",
        baseline_projection_path,
    ):
        if not required.is_file():
            pytest.fail(f"Required B00 alignment fixture is unavailable: {required}")
    baseline_evidence = load_alignment_evidence(
        baseline_alignment_root / "ground-line-map.npz"
    )

    settings = runtime.alignment.evidence
    selected = _fixed_camera_selection(
        scene.cameras,
        settings=settings,
    ).ordered_cameras
    fit_assigned, holdout_assigned = _partition_cameras_with_holdout_tail(
        selected,
        settings=settings,
    )
    assert tuple(camera.camera_id for camera in fit_assigned) == (
        baseline_evidence.partitions.fit_camera_ids
    )
    assert tuple(camera.camera_id for camera in holdout_assigned) == (
        baseline_evidence.partitions.holdout_camera_ids
    )

    input_source = NHTRenderedAlignmentLineInputSource(
        client=NHTRenderClient(),
        executable=runtime.nht.render_executable,
        environment=runtime.nht.environment,
        timeout_seconds=runtime.nht.render_timeout_seconds,
    )
    detector = ProductionCourtLineDetector(
        settings.line_model,
        runtime.resolver,
        seed=settings.seed,
    )
    input_source.preflight(scene, selected)
    detector.preflight()
    inputs = input_source.load(scene, selected)
    assert inputs.source is CourtLineInputSource.NHT_RENDERED_RGB
    assert inputs.camera_ids == tuple(camera.camera_id for camera in selected)

    probabilities = load_or_predict_line_probabilities(
        scene=scene,
        inputs=inputs,
        detector_identity=detector.inference_cache_identity(),
        predict_probability=detector.predict_probability,
    )
    assert tuple(probabilities) == inputs.camera_ids
    plane = _estimate_ground_plane(
        np.asarray(scene.points_scene[:, :3], dtype=np.float64),
        fit_assigned,
        seed=settings.seed,
        settings=settings.ground_plane,
    )
    projected_by_camera = {
        camera.camera_id: _project_probability_to_ground(
            probabilities[camera.camera_id],
            camera=camera,
            plane=plane,
            model_settings=settings.line_model,
            projection_settings=settings.projection,
        )
        for camera in selected
    }
    observable = _classify_observable_cameras(
        fit_assigned,
        holdout_assigned,
        projected_by_camera=projected_by_camera,
        settings=settings,
    )
    _require_observable_camera_minima(observable, settings=settings)

    heatmaps = _alignment_line_heatmaps(
        camera_prefix=selected,
        probabilities=probabilities,
        line_inputs=inputs,
        projected_by_camera=projected_by_camera,
        metric_adapter=baseline_evidence.metric_adapter,
        ground_plane_frame=baseline_evidence.ground_plane_frame,
        settings=settings.projection,
        aggregate_camera_ids=tuple(camera.camera_id for camera in observable.fit),
    )
    output = repro_root / "rendered-line-heatmaps"
    write_line_heatmaps(output, heatmaps=heatmaps)
    validated = validate_line_heatmaps(output)
    assert validated.input_source == CourtLineInputSource.NHT_RENDERED_RGB.value
    assert validated.camera_ids == inputs.camera_ids

    rendered_differences = _captured_rgb_differences(inputs=inputs)
    assert all(value > 0.0 for value in rendered_differences)
    rasters = aggregate_line_heatmaps(heatmaps)
    nonzero_cells = int(np.count_nonzero(rasters.evidence_sum))
    assert nonzero_cells > 0
    rendered_projection_path = output / WEIGHTED_PROJECTION_HEATMAP_FILE
    baseline_copy = repro_root / "legacy-captured-weighted-projection.png"
    with Image.open(baseline_projection_path) as image:
        image.convert("RGB").save(baseline_copy)
    comparison_path = repro_root / "weighted-projection-comparison.png"
    _write_comparison(
        baseline_copy,
        rendered_projection_path,
        output_path=comparison_path,
    )

    selected_counts = {
        camera_id: projected.selected_line_pixel_count
        for camera_id, projected in projected_by_camera.items()
    }
    projected_counts = {
        camera_id: len(projected.points_nht_scene)
        for camera_id, projected in projected_by_camera.items()
    }
    metrics = {
        "schema": "b00_rendered_alignment_line_validation_v1",
        "scene_id": scene.scene_id,
        "input_source": heatmaps.input_source,
        "selected_camera_count": len(selected),
        "fit_camera_count": len(observable.fit),
        "holdout_camera_count": len(observable.holdout),
        "excluded_camera_ids": [item.camera_id for item in observable.exclusions],
        "rendered_inputs_different_from_captured_count": sum(
            value > 0.0 for value in rendered_differences
        ),
        "rendered_vs_captured_mean_absolute_rgb_difference": {
            "minimum": min(rendered_differences),
            "median": float(np.median(rendered_differences)),
            "maximum": max(rendered_differences),
        },
        "selected_line_pixel_count": {
            "minimum": min(selected_counts.values()),
            "median": float(np.median(tuple(selected_counts.values()))),
            "maximum": max(selected_counts.values()),
            "total": sum(selected_counts.values()),
        },
        "projected_line_point_count": {
            "minimum": min(projected_counts.values()),
            "median": float(np.median(tuple(projected_counts.values()))),
            "maximum": max(projected_counts.values()),
            "total": sum(projected_counts.values()),
        },
        "projection_raster_shape": list(heatmaps.raster_shape),
        "projection_nonzero_cell_count": nonzero_cells,
        "projection_evidence_sum": float(np.sum(rasters.evidence_sum)),
        "detector": detector.inference_cache_identity(),
        "input": inputs.cache_identity(),
        "artifacts": {
            "baseline_projection": str(baseline_copy),
            "rendered_projection": str(rendered_projection_path),
            "comparison": str(comparison_path),
            "line_heatmaps": str(output),
        },
    }
    (repro_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


def _runtime_configuration(
    *,
    code_repository: Path,
    fixture_repository: Path,
) -> ScenePipelineConfiguration:
    config_root = code_repository / "src/synthetic_data_generation/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[f"roots.project_root={fixture_repository}"],
        )
    return ScenePipelineConfiguration.from_config(config)


def _captured_rgb_differences(*, inputs: AlignmentLineInputBatch) -> list[float]:
    views = inputs.views
    differences: list[float] = []
    for view in views:
        with Image.open(view.camera.image_path) as image:
            captured = np.asarray(image.convert("RGB"), dtype=np.uint8)
        assert captured.shape == view.image_rgb.shape
        difference = np.abs(captured.astype(np.int16) - view.image_rgb.astype(np.int16))
        differences.append(float(np.mean(difference)))
    return differences


def _write_comparison(
    legacy_path: Path,
    rendered_path: Path,
    *,
    output_path: Path,
) -> None:
    with (
        Image.open(legacy_path) as legacy_image,
        Image.open(rendered_path) as rendered_image,
    ):
        legacy = legacy_image.convert("RGB")
        rendered = rendered_image.convert("RGB")
    target_height = max(legacy.height, rendered.height)
    legacy = _resize_to_height(legacy, target_height)
    rendered = _resize_to_height(rendered, target_height)
    gap = 24
    header_height = 48
    canvas = Image.new(
        "RGB",
        (legacy.width + gap + rendered.width, header_height + target_height),
        "white",
    )
    canvas.paste(legacy, (0, header_height))
    canvas.paste(rendered, (legacy.width + gap, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    draw.text((12, 10), "Legacy: captured RGB", fill="black", font=font)
    draw.text(
        (legacy.width + gap + 12, 10),
        "New: NHT-rendered RGB",
        fill="black",
        font=font,
    )
    canvas.save(output_path)


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = int(round(image.width * height / image.height))
    return image.resize((width, height), resample=Image.Resampling.NEAREST)
