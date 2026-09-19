import pytest
import torch

from src.tasks.slcs.evaluation.conditions import condition_inputs


def test_conditions_preserve_ground_truth_and_only_mask_declared_inputs():
    batch = {
        "player_kp": torch.ones(1, 2, 9, 17, 2),
        "player_kp_vis": torch.ones(1, 2, 9, 17),
        "player_valid": torch.ones(1, 2, 9, dtype=torch.bool),
        "ball_uv": torch.ones(1, 9, 2),
        "ball_vis": torch.ones(1, 9, dtype=torch.bool),
        "court_kp": torch.ones(1, 9, 14, 2),
        "court_vis": torch.ones(1, 9, 14),
        "dino_tokens": torch.ones(1, 3, 4, 8),
        "dino_padding_mask": torch.zeros(1, 3, dtype=torch.bool),
        "padding_mask": torch.zeros(1, 9, dtype=torch.bool),
        "target_ball_position": torch.randn(1, 9, 3),
    }
    for mode in ("full", "no_rgb", "detector_gap", "rgb_only"):
        result = condition_inputs(batch, mode)
        torch.testing.assert_close(
            result["target_ball_position"], batch["target_ball_position"]
        )
    missing = condition_inputs(batch, "detector_gap")
    assert not missing["ball_vis"][0, 3:6].any()
    assert missing["ball_vis"][0, :3].all() and missing["ball_vis"][0, 6:].all()
    assert missing["court_vis"].all() and missing["dino_tokens"].all()
    assert condition_inputs(batch, "no_rgb")["dino_padding_mask"].all()
    assert not condition_inputs(batch, "rgb_only")["court_vis"].any()
    assert batch["player_kp"].all() and batch["ball_vis"].all()
    with pytest.raises(ValueError, match="Unknown"):
        condition_inputs(batch, "unknown")


@pytest.mark.parametrize("length", [1, 2, 3, 4, 8, 9])
def test_gap_without_rgb_matches_composition_and_preserves_padding(length: int) -> None:
    batch = {
        "player_kp": torch.ones(1, 2, 9, 17, 2),
        "player_kp_vis": torch.ones(1, 2, 9, 17),
        "player_valid": torch.ones(1, 2, 9, dtype=torch.bool),
        "ball_uv": torch.ones(1, 9, 2),
        "ball_vis": torch.ones(1, 9, dtype=torch.bool),
        "court_kp": torch.ones(1, 9, 14, 2),
        "court_vis": torch.ones(1, 9, 14),
        "dino_tokens": torch.randn(1, 3, 4, 8),
        "dino_padding_mask": torch.zeros(1, 3, dtype=torch.bool),
        "padding_mask": (torch.arange(9) >= length)[None],
        "target_ball_position": torch.randn(1, 9, 3),
        "target_player_position": torch.randn(1, 2, 9, 3),
        "target_player_rotation": torch.randn(1, 2, 9, 2),
        "player_mask": torch.rand(1, 2, 9) > .5,
        "ball_mask": torch.rand(1, 9) > .5,
        "player_weight": torch.rand(1, 2, 9),
        "ball_weight": torch.rand(1, 9),
    }
    before = {key: value.clone() for key, value in batch.items()}
    gap = condition_inputs(batch, "detector_gap")
    expected = condition_inputs(gap, "no_rgb")
    actual = condition_inputs(batch, "detector_gap_no_rgb")
    for key in batch:
        assert torch.equal(actual[key], expected[key]), key
        assert torch.equal(batch[key], before[key]), key
        if key.startswith("target_") or key in {"padding_mask", "player_mask", "ball_mask", "player_weight", "ball_weight", "court_kp", "court_vis"}:
            assert torch.equal(actual[key], batch[key]), key
    start, end = length // 3, max(length // 3 + 1, 2 * length // 3)
    assert not actual["ball_vis"][0, start:end].any()
    assert actual["ball_vis"][0, end:].all()
    assert not actual["dino_tokens"].any()
    assert actual["dino_padding_mask"].all()
