from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCurveMode,
    OrbitShape,
    OrbitTargetKind,
    OrbitTrajectorySpec,
    OrbitViewSpec,
)


def _trajectory() -> OrbitTrajectorySpec:
    return OrbitTrajectorySpec(
        trajectory_id="trajectory-a",
        trajectory_group_id="group-a",
        shape=OrbitShape.CIRCLE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=20.0,
        radius_scale=1.0,
        axis_ratio=1.0,
        orientation_radians=0.0,
        base_height_m=6.0,
        vertical_amplitude_m=0.0,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=OrbitCurveMode.PLANAR,
    )


def test_typed_contracts_reject_unknown_keys_and_modes() -> None:
    trajectory = _trajectory().to_dict()
    trajectory["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        OrbitTrajectorySpec.from_mapping(trajectory)

    view = OrbitViewSpec(
        view_id="view-a",
        target_kind=OrbitTargetKind.COMPLEX,
        target_court_instance_id=None,
        target_mode="center",
        coverage_mode=OrbitCoverageMode.FULL,
        look_at_height_m=0.0,
        hfov_degrees=60.0,
    ).to_dict()
    view["coverage_mode"] = "smoke"
    with pytest.raises(ValueError):
        OrbitViewSpec.from_mapping(view)


def test_shape_and_target_semantics_fail_closed() -> None:
    values = _trajectory().to_dict()
    values["axis_ratio"] = 0.8
    with pytest.raises(ValueError, match="circle"):
        OrbitTrajectorySpec.from_mapping(values)
    with pytest.raises(ValueError, match="exactly for court targets"):
        OrbitViewSpec(
            view_id="view-a",
            target_kind=OrbitTargetKind.COURT,
            target_court_instance_id=None,
            target_mode="center",
            coverage_mode=OrbitCoverageMode.FULL,
            look_at_height_m=0.0,
            hfov_degrees=60.0,
        )
