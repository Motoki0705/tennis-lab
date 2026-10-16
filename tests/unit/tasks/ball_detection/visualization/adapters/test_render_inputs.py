"""Training visualization must preserve the declared RGB/MDD preprocessing."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.model_io.contracts import BallModelIOError
from src.tasks.ball_detection.model_io.normalization import BallImageNormalization
from src.tasks.ball_detection.visualization.adapters.render_inputs import (
    build_render_animation_inputs,
)
from tests.unit.tasks.ball_detection.model_io.test_adapters import _mdd_adapter


@pytest.mark.parametrize("normalization", [
    BallImageNormalization(),
    BallImageNormalization(True, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    BallImageNormalization(True, (0.1, 0.2, 0.3), (0.2, 0.4, 0.8)),
])
def test_render_restores_rgb_and_matches_model_mdd_for_selected_sample(
    normalization: BallImageNormalization,
) -> None:
    adapter = _mdd_adapter()
    adapter.spec = replace(adapter.spec, input_mode="mdd", input_layout="bcthw", in_channels=2)
    raw = torch.linspace(0, 1, 2 * 3 * 3 * 2 * 2).reshape(2, 3, 3, 2, 2)
    raw[:, 1] = 1 - raw[:, 1]  # Include both brighten and darken transitions.
    prepared = normalization.apply(raw)
    original = prepared.clone()
    rendered = build_render_animation_inputs(
        images_btchw=prepared,
        pred_heatmaps_bthw=torch.zeros(2, 3, 2, 2),
        peak_threshold=0.5,
        image_normalization=normalization,
        model_io=adapter,
        sample_idx=1,
    )
    # RGB must show raw colors, while MDD must match the actual model's input.
    expected_rgb = raw[1].permute(0, 2, 3, 1).numpy() * 255
    rgb = np.stack(rendered["frames_rgb"])
    np.testing.assert_allclose(rgb, expected_rgb, rtol=0, atol=1)
    features = adapter.prepare_images(raw, image_normalization=normalization).model_input
    expected_mdd = np.stack([
        features[1, 1].numpy(), features[1, 0].numpy(), np.zeros((3, 2, 2)),
    ], axis=-1)
    np.testing.assert_array_equal(
        np.stack(rendered["mdd_frames_rgb"]), (expected_mdd * 255).astype(np.uint8),
    )
    assert rgb.dtype == np.uint8
    torch.testing.assert_close(prepared, original, rtol=0, atol=0)


def test_render_does_not_clip_invalid_preprocessed_input_into_valid_rgb() -> None:
    with pytest.raises(BallModelIOError, match="normalization bounds"):
        build_render_animation_inputs(
            images_btchw=torch.full((1, 2, 3, 2, 2), 8.0),
            pred_heatmaps_bthw=torch.zeros(1, 2, 2, 2),
            peak_threshold=0.5,
            image_normalization=BallImageNormalization(True),
            model_io=_mdd_adapter(),
        )
