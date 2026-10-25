"""Connect each concrete predictor/configuration to the shared RGB contract."""

from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.model_io import mdd_coordinates, predict
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDPretrainConfig,
)


def _prediction_config(channels: int) -> MDDPretrainConfig:
    return MDDPretrainConfig(
        stem_channels=(4, 8, 8, 8),
        mixed_channels=(8, 16),
        residual_blocks=(0, 1, 1, 1, 1, 1),
        decoder_channels=8,
        dim=16,
        heads=2,
        layers=1,
        ffn_dim=64,
        dropout=0.0,
        rope_base=10_000.0,
        activation_checkpointing=False,
        encoder_variant="residual",
        temporal_mixing="dense3d",
        input_channels=channels,
    )


@pytest.mark.parametrize("channels", [2, 5])
@pytest.mark.parametrize(
    "model_type", [MDDDPTDetector, DeepMDDQueryDetector], ids=["dpt", "query"]
)
def test_rejects_wrong_external_input_before_forward(
    channels: int,
    model_type: type[MDDDPTDetector] | type[DeepMDDQueryDetector],
) -> None:
    model = model_type(_prediction_config(channels))
    assert model.config.input_channels == channels
    rgb = torch.zeros(1, 32, 3, 8, 8, dtype=torch.float32)
    # DPT imports the validator; the query timestamp validator looks it up locally.
    validation_module = (
        predict if isinstance(model, MDDDPTDetector) else mdd_coordinates
    )
    with (
        patch.object(model, "forward") as forward,
        patch.object(
            validation_module,
            "validate_rgb_input",
            wraps=validation_module.validate_rgb_input,
        ) as validate,
    ):
        with pytest.raises(ValueError, match="RGB"):
            if isinstance(model, MDDDPTDetector):
                predict.predict_dpt_heatmaps(model, rgb)
            else:
                predict.predict_coordinates(
                    model,
                    {"rgb": rgb, "timestamps": torch.arange(32)[None].float()},
                    torch.device("cpu"),
                )
        validate.assert_called_once_with(rgb, frames=model.config.frames)
        forward.assert_not_called()
