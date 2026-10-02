from pathlib import Path

import numpy as np
import pytest

from src.tasks.player_association.appearance.kpr import load_checkpoint, prompt_heatmaps


def test_kpr_prompt_groups_confidence_and_outside_crop_visibility() -> None:
    points: np.ndarray = np.zeros((1, 17, 3), np.float64)
    points[0, 9] = [.5, .5, 1.4]  # left wrist; raw ViTPose peak >1 is allowed
    points[0, 10] = [1.2, .5, 1.]  # right wrist outside crop is invisible
    points[0, 0] = [.2, .2, .3]  # strict >.3 cutoff
    maps = prompt_heatmaps(points)
    assert maps.shape == (1, 8, 384, 128)
    assert maps[0, 3, 192, 64] == 1  # bg, negative, head, left_arm
    assert maps[0, 0, 192, 64] == 0
    assert maps[0, 0, 0, 0] == 1
    assert maps[0, 1:3].sum() == 0
    assert maps[0, 4:].sum() == 0


def test_kpr_negative_prompts_have_separate_channel_and_empty_pose_is_background() -> None:
    points: np.ndarray = np.zeros((1, 17, 3), np.float32)
    negative: np.ndarray = np.zeros((1, 2, 17, 3), np.float32)
    negative[0, 0, 5] = [.25, .25, .9]
    negative[0, 1, 5] = [.75, .75, .9]
    maps = prompt_heatmaps(points, negative)
    assert maps[0, 1, 96, 32] == maps[0, 1, 288, 96] == 1
    assert maps[0, 2:].sum() == 0
    empty = prompt_heatmaps(points)
    assert (empty[:, 0] == 1).all() and empty[:, 1:].sum() == 0
    points[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        prompt_heatmaps(points)


def test_kpr_legacy_pickle_is_never_opened_for_an_unrecorded_weight(tmp_path: Path) -> None:
    path = tmp_path / 'other.pth'
    path.write_bytes(b'not the official checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        load_checkpoint(path)
