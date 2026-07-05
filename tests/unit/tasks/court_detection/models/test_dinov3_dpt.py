from __future__ import annotations

import torch
from torch import nn

from src.tasks.court_detection.models.decoder import CourtDPTDecoder
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.utils.models.loading import DINOv3BackboneAdapter


class FakeDINOv3(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(12))
        self.requested_layers: tuple[int, ...] | None = None
        self.seen_input_shape: tuple[int, ...] | None = None

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_input_shape = tuple(inputs.shape)
        batch_size = inputs.shape[0]
        num_tokens = (inputs.shape[-2] // self.patch_size) * (
            inputs.shape[-1] // self.patch_size
        )
        return {
            "x_norm_patchtokens": inputs.new_zeros(
                batch_size,
                num_tokens,
                self.embed_dim,
            )
        }

    def get_intermediate_layers(
        self,
        inputs: torch.Tensor,
        *,
        n: tuple[int, ...],
        reshape: bool,
        return_class_token: bool,
        norm: bool,
    ) -> tuple[torch.Tensor, ...]:
        if reshape or return_class_token or not norm:
            raise AssertionError("CourtDINOv3Encoder must request normalized tokens.")
        self.requested_layers = n
        return tuple(self.forward_features(inputs)["x_norm_patchtokens"] + idx for idx in n)


def test_dinov3_encoder_reassembles_four_intermediate_token_maps() -> None:
    fake = FakeDINOv3()
    encoder = CourtDINOv3Encoder(
        backbone=DINOv3BackboneAdapter(fake),
        out_indices=(2, 5, 8, 11),
    )

    features = encoder(torch.randn(2, 3, 16, 20))

    assert fake.requested_layers == (2, 5, 8, 11)
    assert len(features) == 4
    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 4, 5),
        (2, 8, 4, 5),
        (2, 8, 4, 5),
        (2, 8, 4, 5),
    ]


def test_dinov3_encoder_pads_inputs_to_patch_grid() -> None:
    fake = FakeDINOv3()
    encoder = CourtDINOv3Encoder(
        backbone=DINOv3BackboneAdapter(fake),
        out_indices=(2, 5, 8, 11),
    )

    features = encoder(torch.randn(2, 3, 17, 19))

    assert fake.seen_input_shape == (2, 3, 20, 20)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
    ]


def test_dpt_decoder_progressively_fuses_reassembled_features() -> None:
    decoder = CourtDPTDecoder(
        encoder_channels=(8, 8, 8, 8),
        decoder_channels=16,
        reassemble_factors=(4.0, 2.0, 1.0, 0.5),
    )
    features = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))

    output = decoder(features)

    assert output.shape == (2, 16, 16, 16)
