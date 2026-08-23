"""Real B00 CUDA acceptance over the canonical video-to-report workspace."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.dataset.blcs.assembler import (
    validate_blcs_dataset,
)
from src.synthetic_data_generation.dataset.court import (
    COURT_SCHEMA_V2,
)
from src.synthetic_data_generation.dataset.court.assembler import (
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.plcs.validation import (
    validate_plcs_dataset,
)
from src.synthetic_data_generation.dataset.runtime import DatasetPerformanceMetrics
from src.synthetic_data_generation.reconstruction.scene_export import (
    validate_standard_scene_export,
)
from src.utils.paths import PROJECT_ROOT

pytestmark = [pytest.mark.cuda, pytest.mark.local_data, pytest.mark.e2e]


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a JSON array.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, name=f"JSON at {path}")


def _b00_root() -> Path:
    return Path(PROJECT_ROOT) / "data/synthetic_data_generation/scenes/B00"


def test_b00_video_to_report_meets_all_quantitative_and_motion_gates() -> None:
    assert torch.cuda.is_available(), "B00 acceptance requires a real CUDA device."
    root = _b00_root()
    run = _json(root / "run.json")
    stages = _mapping(run["stages"], name="run.stages")
    stage_statuses = {
        name: _mapping(record, name=f"run.stages.{name}")["status"]
        for name, record in stages.items()
    }
    assert stage_statuses == {
        "ingest": "completed",
        "reconstruction": "completed",
        "alignment": "completed",
        "court_dataset": "completed",
        "blcs_dataset": "completed",
        "plcs_dataset": "completed",
        "report": "completed",
    }
    assert (root / "report/index.html").is_file()
    assert (root / "report/report.json").is_file()

    scene = validate_standard_scene_export(root / "reconstruction/export/scene.json")
    alignment = validate_alignment_outputs(root / "alignment")
    court = validate_court_dataset(root / "datasets/court")
    blcs = validate_blcs_dataset(root / "datasets/blcs")
    plcs = validate_plcs_dataset(root / "datasets/plcs")
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(
            PROJECT_ROOT / "src/synthetic_data_generation/configs"
        ),
    ):
        runtime = ScenePipelineConfiguration.from_config(
            compose(config_name="run_scene_pipeline")
        )

    assert court.performance.budget == runtime.court.performance
    court.performance.metrics.validate_budget(runtime.court.performance)
    blcs.performance.validate_budget(runtime.blcs.performance)
    plcs_performance = DatasetPerformanceMetrics.from_dict(
        _json(root / "datasets/plcs/diagnostics/performance.json")
    )
    plcs_performance.validate_budget(runtime.plcs.performance)
    assert blcs.performance.nht_invocations == 3
    assert blcs.performance.background_cache_misses == 18
    assert blcs.performance.cuda_peak_bytes > 0
    assert plcs_performance.nht_invocations == 1
    assert plcs_performance.background_cache_misses == 12
    assert plcs_performance.cuda_peak_bytes > 0

    assert scene.scene_id == "B00"
    assert len(scene.cameras) >= 12
    assert len(alignment.layout.courts) >= 2
    assert court.trajectory_group_count >= 24
    assert court.accepted_frame_count >= 2000
    assert court.proposal_count <= 5000
    assert court.maximum_adjacent_step_m <= 1.05
    assert court.accepted_fraction >= 0.9

    court_manifest = _json(root / "datasets/court/dataset.json")
    assert court_manifest["schema"] == COURT_SCHEMA_V2.dataset_schema
    court_metrics = _mapping(court_manifest["metrics"], name="court.metrics")
    assert court_metrics["split_leakage_count"] == 0
    coverage = _mapping(court_metrics["coverage_counts"], name="court.coverage_counts")
    assert all(
        _integer(coverage[name], name=f"court.coverage_counts.{name}") > 0
        for name in ("full", "near_full", "partial")
    )
    visible = _mapping(
        court_metrics["renderer_visible_points_by_class"],
        name="court.renderer_visible_points_by_class",
    )
    semantic_class_names = COURT_SCHEMA_V2.semantic_class_names
    assert set(visible) == set(semantic_class_names)
    assert all(
        _integer(visible[name], name=f"court.visible.{name}") > 0
        for name in semantic_class_names
    )

    blcs_inventory = blcs.manifest.frame_inventory
    assert blcs_inventory.source_count == len(blcs_inventory.planned_indices)
    assert blcs_inventory.planned_indices == blcs_inventory.rendered_indices
    assert blcs_inventory.rendered_indices == blcs_inventory.labelled_indices
    blcs_diagnostics = _json(root / "datasets/blcs/diagnostics/metrics.json")
    assert blcs_diagnostics["camera_profiles"] == ["default"]
    camera_counts = _mapping(
        blcs_diagnostics["camera_counts_per_trajectory"],
        name="blcs.camera_counts_per_trajectory",
    )
    assert all(
        _integer(value, name=f"blcs.camera_counts_per_trajectory.{name}") == 6
        for name, value in camera_counts.items()
    )
    assert (
        _integer(
            blcs_diagnostics["court_count_difference"],
            name="blcs.court_count_difference",
        )
        <= 1
    )

    assert plcs["scene_id"] == "B00"
    plcs_manifest = _json(root / "datasets/plcs/dataset.json")
    plcs_inventory = _mapping(
        plcs_manifest["frame_inventory"], name="plcs.frame_inventory"
    )
    counts = [
        _integer(plcs_inventory[name], name=f"plcs.frame_inventory.{name}")
        for name in ("source", "planned", "rendered", "labelled")
    ]
    assert len(set(counts)) == 1
    assert counts[0] == 2234
    plcs_metadata = _mapping(plcs_manifest["metadata"], name="plcs.metadata")
    assert _integer(
        plcs_metadata["logical_scene_count"], name="plcs.logical_scene_count"
    ) == len(alignment.layout.courts)
    logical_scene_manifests = tuple(
        _mapping(scene_manifest, name=f"plcs.metadata.logical_scenes[{index}]")
        for index, scene_manifest in enumerate(
            _sequence(
                plcs_metadata["logical_scenes"],
                name="plcs.metadata.logical_scenes",
            )
        )
    )
    assert {manifest["scene_id"] for manifest in logical_scene_manifests} == {
        "B00",
        "B00-plcs-002",
    }
    assert all(
        _mapping(manifest["frame_inventory"], name="logical_scene.frame_inventory")
        == {
            "source": 1117,
            "planned": 1117,
            "rendered": 1117,
            "labelled": 1117,
            "first_frame": 0,
            "last_frame": 1116,
        }
        for manifest in logical_scene_manifests
    )
    plcs_diagnostics = _json(
        root / "datasets/plcs/diagnostics/motion-camera-court.json"
    )
    assert plcs_diagnostics["amass_compatible"] is True
    assert _integer(
        plcs_diagnostics["logical_scene_count"],
        name="plcs.diagnostics.logical_scene_count",
    ) == 2
    logical_scene_diagnostics = tuple(
        _mapping(scene_diagnostics, name=f"plcs.logical_scenes[{index}]")
        for index, scene_diagnostics in enumerate(
            _sequence(
                plcs_diagnostics["logical_scenes"],
                name="plcs.logical_scenes",
            )
        )
    )
    assert all(
        _integer(scene_diagnostics["global_frame_count"], name="global_frame_count")
        == 1117
        for scene_diagnostics in logical_scene_diagnostics
    )
    for scene_diagnostics in logical_scene_diagnostics:
        camera_distribution = _mapping(
            scene_diagnostics["camera_distribution"],
            name="plcs.logical_scene.camera_distribution",
        )
        assert camera_distribution["profile"] == "default"
        assert (
            _integer(camera_distribution["camera_count"], name="plcs.camera_count")
            == 6
        )
    court_balance = _mapping(
        plcs_diagnostics["court_balance"], name="plcs.court_balance"
    )
    assert set(
        _mapping(court_balance["counts"], name="plcs.court_balance.counts")
    ) == {court.court_instance_id for court in alignment.layout.courts}
    assert all(
        _integer(count, name=f"plcs.court_balance.counts.{court_id}") == 1
        for court_id, count in _mapping(
            court_balance["counts"], name="plcs.court_balance.counts"
        ).items()
    )
    assert (
        _integer(
            court_balance["maximum_count_difference"],
            name="plcs.maximum_count_difference",
        )
        <= 1
    )
    motion = _sequence(plcs_diagnostics["motion"], name="plcs.motion")
    motion_records = tuple(
        _mapping(record, name=f"plcs.motion[{index}]")
        for index, record in enumerate(motion)
    )
    assert {
        _string(record["category"], name="plcs.motion.category")
        for record in motion_records
    } == {
        "running",
        "walking",
        "general",
    }
    for index, record in enumerate(motion_records):
        articulation = _mapping(
            record["articulation"], name=f"plcs.motion[{index}].articulation"
        )
        assert _integer(record["frame_count"], name="motion.frame_count") == _integer(
            articulation["frame_count"], name="articulation.frame_count"
        )
        assert (
            _number(
                articulation["gaussian_nonrigid_residual_m"],
                name="articulation.gaussian_nonrigid_residual_m",
            )
            > 1.0e-4
        )
        deformed = _sequence(
            articulation["deformed_frame_indices"],
            name="articulation.deformed_frame_indices",
        )
        assert len(deformed) > 0
        category = _string(record["category"], name="motion.category")
        if category in {"running", "walking"}:
            regions = _mapping(
                articulation["region_displacement_m"],
                name="articulation.region_displacement_m",
            )
            assert all(
                _number(regions[name], name=f"articulation.region.{name}") > 1.0e-4
                for name in ("legs", "arms", "torso")
            )
