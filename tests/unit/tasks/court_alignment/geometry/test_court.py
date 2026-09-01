"""Tests for the explicit ground-court KP14 geometry contract."""

import math

import torch

from src.tasks.court_alignment.geometry.court import (
    GROUND_COURT_DOUBLES_FOOTPRINT_INDEX,
    GROUND_COURT_HALF_TURN_INDEX,
    GROUND_COURT_KP14_COUNT,
    GROUND_COURT_KP14_NAMES,
    GROUND_COURT_KP14_SCHEMA,
    GROUND_COURT_LINE_EDGES,
    GroundCourtInstance,
    canonical_court_keypoints,
    canonical_court_line_segments,
    court_doubles_footprint_for_instance,
    court_keypoints_for_instance,
    doubles_footprints_overlap,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    COURT_KP_NAMES,
    COURT_SKELETON,
)


def test_ground_contract_is_explicit_and_itf_sized() -> None:
    points = canonical_court_keypoints()
    assert GROUND_COURT_KP14_SCHEMA == "ground_court_kp14_v1"
    assert len(GROUND_COURT_KP14_NAMES) == GROUND_COURT_KP14_COUNT == 14
    assert points.shape == (14, 2)
    torch.testing.assert_close(points[0], torch.tensor((-5.485, 11.885)))
    torch.testing.assert_close(points[8], torch.tensor((-4.115, 6.4)))
    assert COURT_KP_NAMES[:14] == GROUND_COURT_KP14_NAMES


def test_ground_contract_reuses_authoritative_skeleton_and_half_turn() -> None:
    expected_edges = tuple(
        edge for edge in COURT_SKELETON if edge[0] < 14 and edge[1] < 14
    )
    assert expected_edges == GROUND_COURT_LINE_EDGES
    assert GROUND_COURT_HALF_TURN_INDEX == CAMERA_VIEW_HALF_TURN_INDEX


def test_canonical_lines_include_center_marks() -> None:
    segments = canonical_court_line_segments()
    assert segments.shape == (15, 2, 2)
    torch.testing.assert_close(segments[-2, 0], torch.tensor((0.0, 11.885)))
    torch.testing.assert_close(segments[-2, 1], torch.tensor((0.0, 11.785)))


def test_similarity_transform_maps_net_center_to_instance_center() -> None:
    instance = GroundCourtInstance(
        instance_id=3,
        center_xy_px=(20.0, 30.0),
        rotation_rad=math.pi / 2.0,
        scale_px_per_metre=2.0,
    )
    points = court_keypoints_for_instance(instance)
    torch.testing.assert_close(points[12], torch.tensor((7.2, 30.0)))
    torch.testing.assert_close(points[13], torch.tensor((32.8, 30.0)))


def test_doubles_footprint_uses_cyclic_corners_and_similarity_transform() -> None:
    instance = GroundCourtInstance(
        instance_id=3,
        center_xy_px=(20.0, 30.0),
        rotation_rad=math.pi / 2.0,
        scale_px_per_metre=2.0,
    )
    footprint = court_doubles_footprint_for_instance(instance)
    assert GROUND_COURT_DOUBLES_FOOTPRINT_INDEX == (0, 1, 3, 2)
    assert footprint.shape == (4, 2)
    torch.testing.assert_close(footprint[0], torch.tensor((-3.77, 19.03)))
    torch.testing.assert_close(footprint[1], torch.tensor((-3.77, 40.97)))


def test_doubles_footprint_sat_allows_contact_and_configured_tolerance() -> None:
    first = court_doubles_footprint_for_instance(
        GroundCourtInstance(0, (0.0, 0.0), 0.0, 1.0)
    )
    touching = court_doubles_footprint_for_instance(
        GroundCourtInstance(1, (10.97, 0.0), 0.0, 1.0)
    )
    slight_overlap = court_doubles_footprint_for_instance(
        GroundCourtInstance(1, (10.5, 0.0), 0.0, 1.0)
    )
    assert not doubles_footprints_overlap(first, touching)
    assert doubles_footprints_overlap(first, slight_overlap)
    assert not doubles_footprints_overlap(
        first, slight_overlap, tolerance_px=0.5
    )


def test_doubles_footprint_sat_handles_rotation_and_scale() -> None:
    first = court_doubles_footprint_for_instance(
        GroundCourtInstance(0, (0.0, 0.0), 0.0, 1.0)
    )
    rotated = court_doubles_footprint_for_instance(
        GroundCourtInstance(1, (0.0, 0.0), math.pi / 4.0, 0.5)
    )
    distant = court_doubles_footprint_for_instance(
        GroundCourtInstance(1, (40.0, 40.0), math.pi / 4.0, 0.5)
    )
    assert doubles_footprints_overlap(first, rotated)
    assert not doubles_footprints_overlap(first, distant)
