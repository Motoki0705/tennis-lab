"""Candidate evidence keeps weak hypotheses and native spatial uncertainty."""

from __future__ import annotations

import pytest
import torch

from src.tasks.ball_detection.model_io.candidates import decode_candidates
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallModelIOError,
)


def test_weak_peaks_survive_and_unused_slots_are_explicit() -> None:
    maps = torch.zeros(2, 3, 9, 11)
    maps[..., 2, 3], maps[..., 7, 9] = .8, .00001
    output = decode_candidates(maps, config=BallCandidateConfig(4, 3, 5), subpixel_refine=False)
    assert output.coords.shape == (2, 3, 4, 2)
    assert output.valid[0, 0].tolist() == [True, True, False, False]
    torch.testing.assert_close(output.scores[0, 0], torch.tensor([.8, .00001, 0., 0.]))
    torch.testing.assert_close(output.cells[0, 0], torch.tensor([[3, 2], [9, 7], [0, 0], [0, 0]]))
    torch.testing.assert_close(output.patches[0, 0, 0], maps[0, 0, :5, 1:6])
    assert not output.coords[..., 2:, :].any()
    assert not output.patch_valid[..., 2:, :, :].any()


def test_patch_border_mask_distinguishes_padding_from_true_zero() -> None:
    maps = torch.zeros(1, 1, 4, 5)
    maps[..., 0, 0] = .7
    output = decode_candidates(maps, config=BallCandidateConfig(2, 3, 3), subpixel_refine=False)
    assert output.cells[0, 0, 0].tolist() == [0, 0]
    assert output.patch_valid[0, 0, 0].tolist() == [
        [False, False, False], [False, True, True], [False, True, True],
    ]
    assert output.patches[0, 0, 0, 1, 2] == 0  # valid background pixel
    torch.testing.assert_close(output.patches[0, 0, 0, 1, 1], torch.tensor(.7))


def test_uniform_map_has_no_contrastive_peaks_but_small_grids_reserve_k() -> None:
    maps = torch.full((1, 1, 2, 2), .9)
    output = decode_candidates(maps, config=BallCandidateConfig(8, 3, 5), subpixel_refine=True)
    assert output.scores.shape == (1, 1, 8)
    assert not output.valid.any() and not output.patches.any()
    assert not output.coords.any() and not output.cells.any()
    torch.testing.assert_close(maps, torch.full_like(maps, .9))  # no mutation
    single = decode_candidates(maps[..., :1, :1], config=BallCandidateConfig(), subpixel_refine=True)
    assert single.valid.sum() == 1 and single.patch_valid.sum() == 1


def test_subpixel_coordinates_do_not_shift_native_patches_or_scores() -> None:
    y, x = torch.meshgrid(torch.arange(9.), torch.arange(11.), indexing="ij")
    maps = torch.exp(-((x - 4.3) ** 2 + (y - 3.7) ** 2) / 2)[None, None]
    output = decode_candidates(maps, config=BallCandidateConfig(1, 5, 3), subpixel_refine=True)
    torch.testing.assert_close(output.coords[0, 0, 0], torch.tensor([4.3 / 10, 3.7 / 8]))
    assert output.cells[0, 0, 0].tolist() == [4, 4]
    torch.testing.assert_close(output.patches[0, 0, 0], maps[0, 0, 3:6, 3:6])
    torch.testing.assert_close(output.scores[0, 0, 0], maps[0, 0, 4, 4])
    assert output.patches.dtype == torch.float32 and output.cells.dtype == torch.int64
    assert output.patches.device.type == "cpu"


@pytest.mark.parametrize("name,value", [
    ("max_candidates", 0), ("max_candidates", True),
    ("nms_kernel", 2), ("patch_size", -1), ("patch_size", 2),
])
def test_invalid_config_is_rejected(name: str, value: int) -> None:
    with pytest.raises(ValueError):
        BallCandidateConfig(**{name: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_heatmaps_are_rejected(value: float) -> None:
    with pytest.raises(BallModelIOError, match="finite probabilities"):
        decode_candidates(torch.full((1, 1, 3, 3), value), config=BallCandidateConfig(), subpixel_refine=False)
