"""CPU checks for pairing, near-plane clipping and hole/visibility accounting."""

# mypy: disallow_untyped_decorators=False
from __future__ import annotations

import importlib
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_probe as analyze
import plan_probe as plan

from src.synthetic_data_generation.rendering.nht import NHTRenderArrays

previous_tests = importlib.import_module(
    "knowledge.runs.run-court-nested-plan-cpu-20261022.test_plan"
)


@pytest.fixture
def scene() -> Any:
    return previous_tests.scene.__wrapped__()


def test_pairs_change_only_the_requested_variable(scene: Any) -> None:
    track = plan.paired_track(scene, 0, 0)
    base = plan.make_record(
        track, "recommended", Fraction(7, 96), 1, "continuous", 7, 0
    )
    narrow = plan.make_record(
        track, "hfov45_90", Fraction(7, 96), 1, "continuous", 7, 0
    )
    no_jitter = plan.make_record(
        track, "jitter0", Fraction(7, 96), 1, "continuous", 7, 0
    )
    assert base["pair_id"] == narrow["pair_id"] == no_jitter["pair_id"]
    assert base["position"] == narrow["position"] == no_jitter["position"]
    assert base["pose"] == narrow["pose"]
    assert base["K"] != narrow["K"]
    assert base["K"] == no_jitter["K"]
    assert base["pose"] != no_jitter["pose"]
    assert (
        plan.make_record(track, "recommended", Fraction(14, 192), 1, "sparse", 20, 0)[
            "pose"
        ]
        == base["pose"]
    )


def test_hull_alternative_recomputes_radius_from_unbounded_support(scene: Any) -> None:
    base = plan.paired_track(scene, 0, 0)
    expanded = plan.paired_track(scene, 0, 5)
    assert expanded.spec.base_radius_m > base.spec.base_radius_m
    assert expanded.family == base.family
    assert expanded.center == base.center
    assert expanded.spec.radius_scale == base.spec.radius_scale


def test_near_plane_clipping_preserves_intersections() -> None:
    polygon = np.array(
        [[-1.0, -1.0, -0.5], [1.0, -1.0, 0.5], [1.0, 1.0, 0.5], [-1.0, 1.0, -0.5]]
    )
    result = analyze.near_clip(polygon)
    assert result.shape == (4, 3)
    assert result[:, 2].min() >= 0.01 - 1e-12
    assert np.count_nonzero(np.isclose(result[:, 2], 0.01)) == 2
    assert analyze.near_clip(polygon - [0, 0, 2]).shape == (0, 3)


def test_roi_holes_and_existing_visibility_are_separate(scene: Any) -> None:
    track = plan.paired_track(scene, 0, 0)
    record = plan.make_record(track, "recommended", Fraction(7, 96), 2, "sparse", 0, 0)
    height, width = record["height"], record["width"]
    roi = analyze.court_roi(record)
    assert roi.any()
    rgb = np.full((height, width, 3), 0.3, np.float32)
    depth = np.ones((height, width, 1), np.float32)
    alpha = np.ones_like(depth)
    perfect = analyze.diagnostics(record, NHTRenderArrays(rgb, alpha, depth), scene)
    assert perfect["roi_joint_hole"] == 0
    assert perfect["renderer_target_kp"] == perfect["geometry_target_kp"]
    assert perfect["all_positive_depth"] == 1
    alpha = alpha.copy()
    alpha[..., 0][roi] = 0
    holes = analyze.diagnostics(record, NHTRenderArrays(rgb, alpha, depth), scene)
    assert holes["roi_alpha_hole"] == 1
    assert holes["roi_depth_hole"] == 0
    assert holes["roi_positive_depth"] == 1
    assert not holes["candidate_roi10"]
    empty = analyze.diagnostics(record, NHTRenderArrays(rgb, alpha, depth * 0), scene)
    assert empty["renderer_target_kp"] == 0
    assert not empty["existing_post_gate"]
    assert empty["roi_depth_hole"] == 1
