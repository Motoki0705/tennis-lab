from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.scale import publication
from src.synthetic_data_generation.dataset.court.scale.core import QualityPolicy, digest
from src.synthetic_data_generation.dataset.court.scale.curves import ArcMap
from src.synthetic_data_generation.dataset.court.scale.saved_train import (
    SOURCE_SCHEMA,
    _definition,
    iter_saved_train_groups,
    saved_train_header,
    saved_train_recipe,
)
from src.synthetic_data_generation.dataset.court.scale.validation import (
    validate_plan_geometry,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays


def source():
    policy = {
        "mode": "uniform_arc_length",
        "max_arc_step_m": 1.05,
        "minimum_sample_count": 24,
        "sample_count_multiple": 8,
        "seed": 695,
        "stable_field_order": [
            "shape",
            "center_kind",
            "radius_scale",
            "axis_ratio",
            "orientation_degrees",
            "base_height_m",
            "vertical_modulation_m",
            "curve_mode",
        ],
        "coverage_objective": [
            "coverage_mode",
            "semantic_visibility",
            "trajectory_group",
        ],
        "proposal_budget": 4800,
        "minimum_trajectory_groups": 24,
        "minimum_accepted_frames": 2000,
        "minimum_accepted_fraction": 0.9,
        "split_fractions": [0.8, 0.1, 0.1],
        "shard_count": 8,
    }
    groups = []
    for i, split in enumerate(("train", "validation", "train", "test")):
        groups.append(
            {
                "trajectory": {
                    "trajectory_id": f"trajectory-{i}",
                    "trajectory_group_id": f"group-{i}",
                    "shape": "circle",
                    "center_kind": "complex",
                    "center_court_instance_id": None,
                    "base_radius_m": 12.0 + i,
                    "radius_scale": 0.8,
                    "axis_ratio": 1.0,
                    "orientation_radians": 0.7,
                    "base_height_m": 2.25,
                    "vertical_amplitude_m": 0.25,
                    "vertical_cycles": 1,
                    "vertical_phase_radians": 1.1,
                    "curve_mode": "sinusoidal_height",
                },
                "center": {
                    "center_kind": "complex",
                    "court_instance_id": None,
                    "reference_court_instance_id": "court-0",
                    "scene_from_center": np.eye(4).ravel().tolist(),
                    "center_scene_m": [0.0, 0.0, 0.0],
                    "base_radius_m": 12.0,
                    "captured_offset_median_m": 10.0,
                    "captured_offset_q90_m": 12.0,
                    "captured_camera_count": 12,
                },
                "views": [
                    {
                        "view_id": f"view-{i}",
                        "target_kind": "court",
                        "target_mode": "court_center",
                        "coverage_mode": "partial",
                        "look_at_height_m": 1.5,
                        "hfov_degrees": 45.0,
                        "look_at_jitter_radius_m": 1.0,
                    }
                ],
                "split": split,
                "shard_id": "shard-0",
                "target_court_policy": {
                    "mode": "nearest_camera",
                    "centre_court_instance_id": None,
                },
                "sample_count": 0,
            }
        )
    result = {
        "schema": SOURCE_SCHEMA,
        "scene_id": "fixture",
        "dataset_sha256": "a" * 64,
        "index_sha256": "b" * 64,
        "source_seed": 695,
        "sampling_policy": policy,
        "width": 80,
        "height": 48,
        "trajectory_groups": groups,
    }
    recipe = saved_train_recipe(result)
    for group in groups:
        curve, _ = _definition(group, recipe)
        group["sample_count"] = ArcMap.freeze(curve, recipe).base_count
    return result


def candidate(captured_cameras, multi_court_layout, k=0):
    header = saved_train_header(
        source=source(),
        seed=42,
        density_level=k,
        cameras=captured_cameras,
        layout=multi_court_layout,
    )
    return {
        **header,
        "trajectory_storage": "inline_v1",
        "trajectory_groups": list(iter_saved_train_groups(header, multi_court_layout)),
    }


def test_importer_preserves_nonplanar_train_curves_views_and_nested_teachers(
    captured_cameras, multi_court_layout
):
    coarse = candidate(captured_cameras, multi_court_layout)
    dense = candidate(tuple(reversed(captured_cameras)), multi_court_layout, 1)
    a = validate_plan_geometry(coarse, multi_court_layout)
    b = validate_plan_geometry(dense, multi_court_layout)
    assert len(b) == 2 * len(a)
    assert a.keys() < b.keys()
    assert all(record == b[key] for key, record in a.items())
    assert {
        g["provenance"]["saved_group"]["trajectory"]["trajectory_group_id"]
        for g in dense["trajectory_groups"]
    } == {"group-0", "group-2"}
    for group in dense["trajectory_groups"]:
        assert group["view"]["hfov_degrees"] == 45.0
        assert group["curve"]["parameters"]["orientation"] == 0.7
        assert group["curve"]["parameters"]["phase"] == 1.1
    assert all(r["split"] == "train" for r in b.values())


@pytest.mark.parametrize("mutation", ["split", "view", "phase", "recipe", "id"])
def test_importer_rejects_changed_source_group_contract(
    mutation, captured_cameras, multi_court_layout
):
    value = candidate(captured_cameras, multi_court_layout)
    group = value["trajectory_groups"][0]
    if mutation == "split":
        group["provenance"]["saved_group"]["split"] = "test"
    elif mutation == "view":
        group["view"]["hfov_degrees"] = 75
    elif mutation == "phase":
        group["curve"]["parameters"]["phase"] = 0.0
    elif mutation == "recipe":
        value["recipe"]["width"] = 81
        value["recipe_sha256"] = digest(value["recipe"])
    else:
        group["trajectory_id"] = "0" * 64
    with pytest.raises(ValueError):
        validate_plan_geometry(value, multi_court_layout)


def test_saved_train_publication_strict_roundtrip(
    tmp_path, monkeypatch, captured_cameras, multi_court_layout
):
    value = candidate(captured_cameras, multi_court_layout)
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda _: SimpleNamespace(layout=multi_court_layout),
    )
    request = publication.freeze_request(
        tmp_path,
        plans=[value],
        assets=[{"scene_id": "fixture", "alignment": {}}],
        quality=QualityPolicy(),
        renderer={"schema": "test"},
    )

    def render(_, records):
        for r in records:
            h, w = r["height"], r["width"]
            yield (
                r["sample_id"],
                NHTRenderArrays(
                    rgb=np.ones((h, w, 3), dtype=np.float32),
                    alpha=np.ones((h, w, 1), dtype=np.float32),
                    depth=np.ones((h, w, 1), dtype=np.float32),
                ),
            )

    path = publication.execute_request(tmp_path, request, render)
    version = publication.load_dataset_version(path, verify_images=True)
    assert version["method"] == "saved_train_trajectories"
    assert version["scenes"][0]["counts"]["qa_accepted"] > 0


def test_nested_selection_is_order_independent_proportional_and_spatially_spread():
    helper = run_path(
        str(Path("knowledge/runs/run-court-scene-diversity-6000-v1/generate.py"))
    )
    order = helper["nested_order"]
    records = [
        {
            "trajectory_id": gid,
            "sample_id": f"{gid}-{j}",
            "scale": {"arc_fraction": [j, n]},
        }
        for gid, n in (("a", 32), ("b", 64))
        for j in range(n)
    ]
    first, second = order(records), order(list(reversed(records)))
    assert first == second
    assert len(first) == len(set(first)) == 96
    for size in (12, 24, 48):
        assert sum(s.startswith("a-") for s in first[:size]) == size // 3
    a = [int(s.split("-")[1]) for s in first if s.startswith("a-")]
    assert (a[1] - a[0]) % 32 == 16
