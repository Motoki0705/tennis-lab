from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.model_io.predict import (
    predict_coordinates,
    predict_dpt_heatmaps,
)
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector, MDDDPTDetector
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


@pytest.mark.parametrize("channels", [2, 5])
@pytest.mark.parametrize("kind", ["dpt", "query"])
@pytest.mark.parametrize("shape,dtype", [
    ((1, 31, 3, 8, 8), torch.uint8), ((1, 32, 3, 8, 8), torch.float32),
    ((1, 32, 5, 8, 8), torch.uint8), ((1, 5, 32, 8, 8), torch.float32),
    ((1, 32, 3, 7, 8), torch.uint8), ((0, 32, 3, 8, 8), torch.uint8),
])
def test_rejects_wrong_external_input_before_forward(
    channels: int, kind: str, shape: tuple[int, ...], dtype: torch.dtype,
) -> None:
    config = replace(small_config(), input_channels=channels)
    model = MDDDPTDetector(config) if kind == "dpt" else DeepMDDQueryDetector(config)
    rgb = torch.zeros(shape, dtype=dtype)
    with patch.object(model, "forward") as forward:
        with pytest.raises(ValueError, match="RGB"):
            if isinstance(model, MDDDPTDetector):
                predict_dpt_heatmaps(model, rgb)
            else:
                predict_coordinates(model, dict(rgb=rgb, timestamps=torch.arange(32)[None].float()), torch.device("cpu"))
        forward.assert_not_called()
