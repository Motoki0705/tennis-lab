"""Focused Court array-scan and measured-performance contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.dataset.court import assembler
from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    CourtProjection,
    MultiCourtProjection,
    SemanticClass,
    SemanticPoint,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    DatasetSplit,
    PlannedCourtSample,
)
from src.synthetic_data_generation.dataset.court.performance import (
    CourtPerformanceEvidence,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_PERFORMANCE_SCHEMA_V2,
    COURT_PERFORMANCE_SCHEMA_V3,
    COURT_SEMANTIC_CLASS_NAMES_V2,
    COURT_SEMANTIC_CLASS_NAMES_V3,
)
from src.synthetic_data_generation.dataset.court.shards import (
    CourtRenderedSample,
    CourtRenderResult,
    CourtShardTiming,
)
from src.synthetic_data_generation.dataset.runtime import (
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
    PerformanceTimer,
    directory_size_bytes,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def test_performance_evidence_round_trips_measured_court_budget() -> None:
    evidence = _post_render_rejection_evidence(fresh_rendered_sample_count=3)

    reopened = CourtPerformanceEvidence.from_dict(evidence.to_dict())

    assert reopened == evidence
    assert reopened.metrics.nht_invocations <= reopened.resolved_shard_count
    assert reopened.metrics.published_bytes == 900
    assert reopened.post_render_rejected_sample_count == 1
    assert reopened.accepted_staged_complete_array_scans == 2
    assert reopened.post_render_rejected_staged_complete_array_scans == 1
    assert reopened.budget.maximum_complete_array_scans_per_sample == 2


@pytest.mark.parametrize(
    ("schema", "class_names", "schema_value"),
    [
        (
            COURT_PERFORMANCE_SCHEMA_V2,
            COURT_SEMANTIC_CLASS_NAMES_V2,
            "court_dataset_performance_v3",
        ),
        (
            COURT_PERFORMANCE_SCHEMA_V3,
            COURT_SEMANTIC_CLASS_NAMES_V3,
            "court_dataset_performance_v4",
        ),
    ],
)
def test_singleton_performance_evidence_requires_exact_versioned_schema(
    schema: str,
    class_names: tuple[str, ...],
    schema_value: str,
) -> None:
    v1 = _post_render_rejection_evidence(fresh_rendered_sample_count=3)
    singleton = replace(
        v1,
        schema=schema,
        visible_points_by_class={name: 1 for name in class_names},
    )

    assert CourtPerformanceEvidence.from_dict(singleton.to_dict()) == singleton
    assert singleton.to_dict()["schema"] == schema_value

    mixed = copy.deepcopy(singleton.to_dict())
    semantic = mixed["semantic"]
    assert isinstance(semantic, dict)
    semantic["visible_points_by_class"] = {name: 1 for name in SEMANTIC_CLASS_NAMES}
    with pytest.raises(ValueError, match="semantic classes"):
        CourtPerformanceEvidence.from_dict(mixed)

    unknown = copy.deepcopy(singleton.to_dict())
    unknown["schema"] = "court_dataset_performance_v5"
    with pytest.raises(ValueError, match="Unknown Court performance schema"):
        CourtPerformanceEvidence.from_dict(unknown)


def test_array_scan_budget_is_equivalent_for_fresh_and_reused_shards() -> None:
    fresh = _post_render_rejection_evidence(fresh_rendered_sample_count=3)
    reused = _post_render_rejection_evidence(fresh_rendered_sample_count=0)

    assert fresh.metrics.complete_array_scans == 6
    assert reused.metrics.complete_array_scans == 3
    assert fresh.fresh_run_complete_array_scan_requirement == 6
    assert reused.fresh_run_complete_array_scan_requirement == 6
    assert fresh.complete_array_scan_budget_capacity == 6
    assert reused.complete_array_scan_budget_capacity == 6
    assert fresh.retained_nht_array_bytes == reused.retained_nht_array_bytes == 0

    for evidence in (fresh, reused):
        with pytest.raises(ValueError, match="cannot cover"):
            replace(
                evidence,
                budget=_budget(maximum_complete_array_scans_per_sample=1),
                complete_array_scan_budget_capacity=3,
            )


def test_performance_evidence_rejects_genuine_extra_staged_array_scan() -> None:
    evidence = _post_render_rejection_evidence(fresh_rendered_sample_count=3)
    excessive_metrics = replace(evidence.metrics, complete_array_scans=7)

    with pytest.raises(ValueError, match="Every accepted Court proposal"):
        replace(
            evidence,
            metrics=excessive_metrics,
            accepted_staged_complete_array_scans=3,
            staged_complete_array_scans=4,
        )


def test_performance_evidence_rejects_retained_nht_arrays() -> None:
    evidence = _post_render_rejection_evidence(fresh_rendered_sample_count=3)

    with pytest.raises(ValueError, match="cannot retain dense NHT arrays"):
        replace(evidence, retained_nht_array_bytes=1)


def _post_render_rejection_evidence(
    *,
    fresh_rendered_sample_count: int,
) -> CourtPerformanceEvidence:
    renderable_sample_count = 3
    reused_rendered_sample_count = renderable_sample_count - fresh_rendered_sample_count
    nht_invocations = int(fresh_rendered_sample_count > 0)
    metrics = DatasetPerformanceMetrics(
        domain="court",
        wall_seconds=12.0,
        cpu_seconds=4.0,
        peak_rss_bytes=1024,
        execution_device="cuda:0",
        cuda_peak_bytes=0,
        nht_invocations=nht_invocations,
        background_cache_misses=0,
        complete_array_scans=fresh_rendered_sample_count + renderable_sample_count,
        generated_bytes=1000,
        published_bytes=900,
        dense_reference_bytes=900,
        frame_count=2,
        camera_count=2,
        sample_count=2,
    )
    return CourtPerformanceEvidence(
        budget=_budget(),
        metrics=metrics,
        resolved_shard_count=8,
        maximum_shard_sample_count=3,
        request_path_count=nht_invocations,
        proposal_count=3,
        accepted_frame_count=2,
        rejected_frame_count=1,
        pre_render_checked_sample_count=3,
        pre_render_rejected_sample_count=0,
        renderable_sample_count=renderable_sample_count,
        post_render_rejected_sample_count=1,
        depth_conversion_count=2,
        fresh_rendered_sample_count=fresh_rendered_sample_count,
        reused_rendered_sample_count=reused_rendered_sample_count,
        nht_boundary_complete_array_scans=fresh_rendered_sample_count,
        accepted_staged_complete_array_scans=2,
        post_render_rejected_staged_complete_array_scans=1,
        staged_complete_array_scans=renderable_sample_count,
        fresh_run_complete_array_scan_requirement=6,
        complete_array_scan_budget_capacity=6,
        scene_validation_count=nht_invocations,
        preview_validation_count=2 * fresh_rendered_sample_count,
        loaded_array_bytes=300 if nht_invocations else 0,
        maximum_nht_live_array_bytes=100 if nht_invocations else 0,
        retained_nht_array_bytes=0,
        external_nht_boundary_wall_seconds=3.0 if nht_invocations else 0.0,
        shard_wall_seconds={"shard-000": 3.0} if nht_invocations else {},
        visible_points_by_class={name: 1 for name in SEMANTIC_CLASS_NAMES},
    )


def test_performance_evidence_counts_pre_render_rejection_without_array_scan() -> None:
    evidence = CourtPerformanceEvidence(
        budget=_budget(maximum_nht_invocations=1, maximum_batch_frames=1),
        metrics=DatasetPerformanceMetrics(
            domain="court",
            wall_seconds=1.0,
            cpu_seconds=0.5,
            peak_rss_bytes=1024,
            execution_device="cuda:0",
            cuda_peak_bytes=0,
            nht_invocations=1,
            background_cache_misses=0,
            complete_array_scans=2,
            generated_bytes=100,
            published_bytes=100,
            dense_reference_bytes=100,
            frame_count=1,
            camera_count=1,
            sample_count=1,
        ),
        resolved_shard_count=1,
        maximum_shard_sample_count=1,
        request_path_count=1,
        proposal_count=2,
        accepted_frame_count=1,
        rejected_frame_count=1,
        pre_render_checked_sample_count=2,
        pre_render_rejected_sample_count=1,
        renderable_sample_count=1,
        post_render_rejected_sample_count=0,
        depth_conversion_count=1,
        fresh_rendered_sample_count=1,
        reused_rendered_sample_count=0,
        nht_boundary_complete_array_scans=1,
        accepted_staged_complete_array_scans=1,
        post_render_rejected_staged_complete_array_scans=0,
        staged_complete_array_scans=1,
        fresh_run_complete_array_scan_requirement=2,
        complete_array_scan_budget_capacity=2,
        scene_validation_count=1,
        preview_validation_count=2,
        loaded_array_bytes=100,
        maximum_nht_live_array_bytes=100,
        retained_nht_array_bytes=0,
        external_nht_boundary_wall_seconds=0.25,
        shard_wall_seconds={"shard-000": 0.25},
        visible_points_by_class={name: 1 for name in SEMANTIC_CLASS_NAMES},
    )

    assert evidence.pre_render_rejected_sample_count == 1
    assert evidence.metrics.complete_array_scans == 2


def test_staged_evaluation_preserves_rgb_alpha_and_converts_depth_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = _rendered(tmp_path)
    rgb_before = np.load(rendered.rgb_path, allow_pickle=False)
    alpha_before = np.load(rendered.alpha_path, allow_pickle=False)
    np.save(
        rendered.depth_path,
        np.full((3, 4, 1), 4.0, dtype=np.float32),
        allow_pickle=False,
    )
    calls = 0
    actual_conversion = NHTRenderArrays.metric_depth

    def counted_conversion(
        arrays: NHTRenderArrays,
        *,
        nht_scene_units_per_metre: float,
    ) -> NDArray[np.float32]:
        nonlocal calls
        calls += 1
        converted: NDArray[np.float32] = actual_conversion(
            arrays,
            nht_scene_units_per_metre=nht_scene_units_per_metre,
        )
        return converted

    monkeypatch.setattr(NHTRenderArrays, "metric_depth", counted_conversion)
    result = assembler._evaluate_staged_sample(
        rendered,
        projection=_projection(rendered),
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.diag((2.0, 2.0, 2.0, 1.0)).astype(np.float64)
        ),
    )

    assert result.accepted
    assert calls == 1
    assert result.complete_array_scan_count == 1
    np.testing.assert_array_equal(
        np.load(rendered.rgb_path, allow_pickle=False),
        rgb_before,
    )
    np.testing.assert_array_equal(
        np.load(rendered.alpha_path, allow_pickle=False),
        alpha_before,
    )
    np.testing.assert_allclose(
        np.load(rendered.depth_path, allow_pickle=False),
        2.0,
    )


def test_performance_writer_persists_exact_published_bytes(tmp_path: Path) -> None:
    root = tmp_path / "court"
    (root / "diagnostics").mkdir(parents=True)
    rendered = _rendered(tmp_path / "nht")
    render_result = CourtRenderResult(
        samples=(rendered,),
        pre_render_projections=(_projection(rendered),),
        pre_render_rejected_sample_ids=(),
        resolved_shard_count=1,
        nht_invocations=1,
        request_path_count=1,
        maximum_shard_sample_count=1,
        generated_bytes=10_000,
        nht_complete_array_scans=1,
        scene_validation_count=1,
        preview_validation_count=2,
        loaded_array_bytes=192,
        maximum_nht_live_array_bytes=192,
        retained_nht_array_bytes=0,
        shard_timings=(
            CourtShardTiming(
                shard_id="shard-000",
                camera_count=1,
                wall_seconds=0.0,
            ),
        ),
    )

    evidence = assembler._write_performance_evidence(
        root,
        timer=PerformanceTimer(),
        render_result=render_result,
        proposal_count=1,
        accepted_frame_count=1,
        rejected_frame_count=0,
        accepted_staged_complete_array_scans=1,
        post_render_rejected_staged_complete_array_scans=0,
        budget=_budget(maximum_nht_invocations=1, maximum_batch_frames=1),
        visible_by_class={name: 1 for name in SEMANTIC_CLASS_NAMES},
    )

    assert evidence.metrics.published_bytes == directory_size_bytes(root)
    assert (root / "diagnostics" / "performance.json").is_file()


def _projection(rendered: CourtRenderedSample) -> MultiCourtProjection:
    point = SemanticPoint(
        physical_index=0,
        uv=(1.0, 1.0),
        camera_depth_m=1.0,
        scene_xyz_m=(0.0, 0.0, 0.0),
        in_front=True,
        in_frame=True,
        renderer_visible=None,
    )
    classes = tuple(
        SemanticClass(
            class_id=index,
            class_name=name,
            points=(point, point),
        )
        for index, name in enumerate(SEMANTIC_CLASS_NAMES)
    )
    return MultiCourtProjection(
        camera_id=rendered.sample.sample_id,
        width=rendered.sample.camera.width,
        height=rendered.sample.camera.height,
        courts=(CourtProjection(court_instance_id="court-a", classes=classes),),
    )


def _budget(
    *,
    maximum_nht_invocations: int = 8,
    maximum_batch_frames: int = 600,
    maximum_complete_array_scans_per_sample: int = 2,
) -> DatasetPerformanceBudget:
    return DatasetPerformanceBudget(
        maximum_wall_seconds=1800.0,
        maximum_published_bytes=35 * 1024**3,
        maximum_published_fraction_of_dense_reference=1.0,
        maximum_nht_invocations=maximum_nht_invocations,
        maximum_background_cache_misses=1,
        maximum_complete_array_scans_per_sample=(
            maximum_complete_array_scans_per_sample
        ),
        maximum_batch_frames=maximum_batch_frames,
        execution_device="cuda:0",
        require_cuda=True,
    )


def _rendered(root: Path) -> CourtRenderedSample:
    camera = SceneCamera(
        camera_id="court-sample-000000",
        source_frame_index=0,
        width=4,
        height=3,
        intrinsics=(4.0, 0.0, 1.5, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="generated/court-sample-000000.png",
    )
    sample = PlannedCourtSample(
        sample_index=0,
        sample_id=camera.camera_id,
        trajectory_group_id="group-a",
        trajectory_id="trajectory-a",
        view_id="view-a",
        trajectory_frame_index=0,
        split=DatasetSplit.TRAIN,
        shard_id="shard-000",
        camera_center_scene_m=(0.0, 0.0, 0.0),
        camera=camera,
    )
    camera_root = root / sample.sample_id
    camera_root.mkdir(parents=True)
    np.save(camera_root / "rgb.npy", np.zeros((3, 4, 3), dtype=np.float32))
    np.save(camera_root / "alpha.npy", np.ones((3, 4, 1), dtype=np.float32))
    np.save(camera_root / "depth.npy", np.ones((3, 4, 1), dtype=np.float32))
    Image.new("RGB", (4, 3)).save(camera_root / "rgb.png")
    Image.new("L", (4, 3)).save(camera_root / "alpha.png")
    return CourtRenderedSample(
        sample=sample,
        rgb_path=camera_root / "rgb.npy",
        rgb_preview_path=camera_root / "rgb.png",
        alpha_path=camera_root / "alpha.npy",
        alpha_preview_path=camera_root / "alpha.png",
        depth_path=camera_root / "depth.npy",
    )
