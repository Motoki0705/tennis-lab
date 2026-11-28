from __future__ import annotations

import pytest

from src.tennis_scene.analytics.profiles import resolve_profile


def test_meiji_and_itf_use_distinct_tape_geometry_and_ball_radii() -> None:
    soft = resolve_profile(dataset_id="meiji_4cam")
    hard = resolve_profile(dataset_id="meiji_4cam", profile_key="tennis_itf_standard")
    assert soft.profile_key == "soft_tennis_standard"
    assert soft.source_kind == hard.source_kind == "rule_default"
    assert (soft.court_length_m, soft.doubles_width_m, soft.service_line_m) == (
        23.77,
        10.97,
        6.40,
    )
    assert (soft.ball_radius_m, hard.ball_radius_m) == (0.033, 0.0335)
    assert [soft.net_height_at_x(x) for x in (-4.9, 0, 2.5)] == [1.07] * 3
    assert hard.net_height_at_x(0.0) == 0.914
    assert hard.net_height_at_x(-3.1995) == pytest.approx(0.953)
    assert hard.net_height_at_x(6.399) == pytest.approx(1.07)
    assert hard.crossing_in_net_span(-6.399)
    assert not hard.crossing_in_net_span(6.400)


def test_measured_net_interpolates_asymmetric_knots_and_retains_provenance() -> None:
    profile = resolve_profile(
        dataset_id="meiji_4cam",
        overrides={
            "profile_key": "venue-height-2026",
            "source_kind": "measured",
            "source_reference": "tape survey 2026",
            "net_profile_kind": "measured_curve",
            "net_height_center_m": 1.04,
            "net_height_post_m": 1.08,
            "net_samples_x_height_m": [
                [-6.399, 1.08],
                [-2.0, 1.02],
                [0.0, 1.04],
                [6.399, 1.08],
            ],
        },
    )
    assert profile.net_height_at_x(-1.0) == pytest.approx(1.03)
    assert profile.net_height_at_x(8.0) == pytest.approx(1.08)
    assert (
        profile.to_record()["parameters_json"]["source_reference"] == "tape survey 2026"
    )
    assert resolve_profile(dataset_id="meiji_4cam").net_height_at_x(-1.0) == 1.07


def test_custom_flat_profile_requires_consistent_heights_and_new_key() -> None:
    overrides = {
        "profile_key": "venue-flat",
        "source_kind": "custom",
        "net_height_center_m": 1.06,
        "net_height_post_m": 1.06,
        "ball_radius_m": 0.034,
    }
    profile = resolve_profile(dataset_id="meiji_4cam", overrides=overrides)
    assert profile.net_height_at_x(5.0) == 1.06
    assert profile.ball_radius_m == 0.034
    with pytest.raises(ValueError, match="new profile key"):
        resolve_profile(
            dataset_id="meiji_4cam",
            overrides={**overrides, "profile_key": "soft_tennis_standard"},
        )
    with pytest.raises(ValueError, match="equal center and post"):
        resolve_profile(
            dataset_id="meiji_4cam", overrides={**overrides, "net_height_post_m": 1.05}
        )
    with pytest.raises(ValueError, match="Select a standard"):
        resolve_profile(dataset_id="other")
