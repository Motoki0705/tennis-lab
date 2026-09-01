"""Tests for full-resolution line, KP, and centre-vote targets."""

import torch

from src.tasks.court_alignment.geometry.court import GroundCourtInstance
from src.tasks.court_alignment.geometry.rasterization import (
    render_center_vote_targets,
    render_court_line_mask,
    render_keypoint_heatmaps,
)
from src.utils.data.heatmaps import heatmaps_to_peaks, refine_peaks_log_parabolic


def _instance(instance_id: int, center: tuple[float, float]) -> GroundCourtInstance:
    return GroundCourtInstance(
        instance_id=instance_id,
        center_xy_px=center,
        rotation_rad=0.0,
        scale_px_per_metre=1.0,
    )


def test_binary_line_mask_has_full_resolution_shape_and_range() -> None:
    mask = render_court_line_mask((48, 64), (_instance(0, (32.0, 24.0)),))
    assert mask.shape == (1, 48, 64)
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert int(mask.sum()) > 0


def test_multiple_courts_create_multiple_peaks_per_semantic_channel() -> None:
    points = torch.tensor(
        [
            [[10.0, 10.0]] * 14,
            [[30.0, 30.0]] * 14,
        ],
        dtype=torch.float32,
    )
    visibility = torch.ones((2, 14), dtype=torch.bool)
    heatmaps = render_keypoint_heatmaps(
        (48, 48), points, visibility, sigma_px=0.75
    )
    assert heatmaps.shape == (14, 48, 48)
    assert torch.count_nonzero(heatmaps[0] == 1.0) == 2
    assert float(heatmaps.min()) >= 0.0
    assert float(heatmaps.max()) <= 1.0


def test_smaller_sigma_is_more_local() -> None:
    points = torch.tensor([[[20.0, 20.0]] * 14])
    visibility = torch.ones((1, 14), dtype=torch.bool)
    narrow = render_keypoint_heatmaps(
        (48, 48), points, visibility, sigma_px=0.5
    )[0]
    broad = render_keypoint_heatmaps(
        (48, 48), points, visibility, sigma_px=2.0
    )[0]
    assert float(narrow[20, 21]) < float(broad[20, 21])


def test_lattice_normalization_preserves_subpixel_log_parabolic_shape() -> None:
    points = torch.tensor([[[10.4, 8.3]] * 14])
    visibility = torch.ones((1, 14), dtype=torch.bool)
    heatmaps = render_keypoint_heatmaps(
        (24, 24), points, visibility, sigma_px=0.75
    ).unsqueeze(0)
    lattice, _, _ = heatmaps_to_peaks(
        heatmaps, threshold=0.25, nms_kernel=3, max_peaks=1
    )
    refined = refine_peaks_log_parabolic(heatmaps, lattice)
    torch.testing.assert_close(
        refined[0, 0, 0] * 23.0,
        torch.tensor((10.4, 8.3)),
        atol=1.0e-4,
        rtol=0.0,
    )


def test_center_votes_point_to_the_correct_court() -> None:
    points = torch.tensor([[[8.0, 8.0]] * 14, [[24.0, 24.0]] * 14])
    centers = torch.tensor([[4.0, 4.0], [28.0, 28.0]])
    visibility = torch.ones((2, 14), dtype=torch.bool)
    votes, mask = render_center_vote_targets(
        (40, 40), points, centers, visibility, sigma_px=1.0
    )
    assert votes.shape == (2, 40, 40)
    assert mask.shape == (1, 40, 40)
    torch.testing.assert_close(votes[:, 8, 8], torch.tensor((-4.0, -4.0)))
    torch.testing.assert_close(votes[:, 24, 24], torch.tensor((4.0, 4.0)))
    assert mask.dtype == torch.bool


def test_center_vote_mask_is_independent_of_gaussian_sigma() -> None:
    points = torch.tensor([[[10.25, 10.25]] * 14, [[30.25, 30.25]] * 14])
    centers = torch.tensor([[4.0, 4.0], [36.0, 36.0]])
    visibility = torch.ones((2, 14), dtype=torch.bool)
    _, narrow_mask = render_center_vote_targets(
        (48, 48), points, centers, visibility, sigma_px=0.5, vote_radius_px=4.0
    )
    _, broad_mask = render_center_vote_targets(
        (48, 48), points, centers, visibility, sigma_px=2.0, vote_radius_px=4.0
    )
    assert torch.equal(narrow_mask, broad_mask)
    _, underflow_mask = render_center_vote_targets(
        (48, 48), points, centers, visibility, sigma_px=0.01, vote_radius_px=4.0
    )
    assert torch.equal(narrow_mask, underflow_mask)


def test_center_vote_owner_is_independent_of_gaussian_sigma() -> None:
    # Both candidates support the queried pixel, but sigma=0.01 underflows
    # the farther Gaussian.  Ownership must still follow geometric distance.
    points = torch.tensor([[[8.0, 10.0]] * 14, [[9.0, 10.0]] * 14])
    centers = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    visibility = torch.ones((2, 14), dtype=torch.bool)
    narrow_votes, narrow_mask = render_center_vote_targets(
        (24, 24), points, centers, visibility, sigma_px=0.01, vote_radius_px=3.0
    )
    broad_votes, broad_mask = render_center_vote_targets(
        (24, 24), points, centers, visibility, sigma_px=2.0, vote_radius_px=3.0
    )
    assert torch.equal(narrow_mask, broad_mask)
    torch.testing.assert_close(narrow_votes[:, 10, 10], torch.tensor((90.0, 90.0)))
    torch.testing.assert_close(narrow_votes, broad_votes)
