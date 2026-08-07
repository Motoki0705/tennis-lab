from __future__ import annotations

import pytest
import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtKeypointModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
    CourtModelSpec,
)
from src.tasks.court_detection.models.decoder import CourtDPTDecoder
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.utils.models.loading import DINOv3BackboneAdapter
from src.utils.models.lora import LoRAConfig


class FakeDINOv3(nn.Module):
    embed_dim = 8
    patch_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(12))
        self.requested_layers: tuple[int, ...] | None = None
        self.seen_input_shape: tuple[int, ...] | None = None
        self.grad_enabled: bool | None = None
        self.invalid_response = False

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
    ) -> object:
        self.grad_enabled = torch.is_grad_enabled()
        if reshape or return_class_token or not norm:
            raise AssertionError("CourtDINOv3Encoder must request normalized tokens.")
        self.requested_layers = n
        if self.invalid_response:
            return (torch.zeros(1),)
        return tuple(self.forward_features(inputs)["x_norm_patchtokens"] + idx for idx in n)


def _encoder(fake: FakeDINOv3) -> CourtDINOv3Encoder:
    return CourtDINOv3Encoder(
        backbone=DINOv3BackboneAdapter(fake),
        out_indices=(2, 5, 8, 11),
        in_channels=3,
        repository_path=None,
        checkpoint_path=None,
        backbone_name=None,
        strict=None,
        train_mode="frozen",
        last_n_blocks=0,
        lora=LoRAConfig(
            enabled=False,
            rank=8,
            alpha=16.0,
            dropout=0.0,
            target_modules=("qkv", "proj", "fc1", "fc2"),
        ),
        layer_mode="uniform",
    )


class _CountingCourtDINOModel(CourtHierarchicalModel):
    def __init__(self, encoder: CourtDINOv3Encoder) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.num_classes = 14
        self.encoder = encoder
        self.calls = 0

    def forward(
        self,
        x: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        del feature_1, feature_2, feature_3, feature_4
        return x.new_zeros(x.shape[0], 14, x.shape[-2], x.shape[-1])


def _adapter(model: _CountingCourtDINOModel) -> CourtKeypointModelIO:
    adapter = CourtKeypointModelIO(
        CourtModelSpec(
            task="kp",
            in_channels=3,
            output_channels=14,
            short_side=32,
            encoder_kind="dinov3",
        ),
        focal_gamma=2.0,
        execution_boundary=CourtDINOv3ExecutionBoundary(frozen_backbone=True),
    )
    adapter.validate_model_pair(model)
    return adapter


def test_dinov3_boundary_reassembles_four_intermediate_token_maps() -> None:
    fake = FakeDINOv3()
    model = _CountingCourtDINOModel(_encoder(fake))
    adapter = _adapter(model)

    call = adapter.prepare_images(torch.zeros(2, 3, 16, 20))
    features = call.model_args[1:]

    assert fake.requested_layers == (2, 5, 8, 11)
    assert fake.grad_enabled is False
    assert len(features) == 4
    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 4, 5),
        (2, 8, 4, 5),
        (2, 8, 4, 5),
        (2, 8, 4, 5),
    ]


def test_dinov3_boundary_pads_inputs_to_patch_grid() -> None:
    fake = FakeDINOv3()
    model = _CountingCourtDINOModel(_encoder(fake))
    adapter = _adapter(model)

    call = adapter.prepare_images(torch.zeros(2, 3, 17, 19))
    features = call.model_args[1:]

    assert fake.seen_input_shape == (2, 3, 20, 20)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
    ]


def test_invalid_dinov3_response_fails_before_court_model_forward() -> None:
    fake = FakeDINOv3()
    fake.invalid_response = True
    model = _CountingCourtDINOModel(_encoder(fake))
    adapter = _adapter(model)
    pair = bind_model_io(model, adapter)
    batch = {
        "image": torch.zeros(1, 3, 16, 20),
        "heatmap": torch.zeros(1, 14, 16, 20),
        "keypoints": torch.zeros(1, 14, 2),
        "kp_visible": torch.ones(1, 14, dtype=torch.bool),
    }

    with pytest.raises(CourtModelIOError, match="return four tensors"):
        pair.run(batch)

    assert model.calls == 0


def test_dpt_decoder_progressively_fuses_reassembled_features() -> None:
    decoder = CourtDPTDecoder(
        encoder_channels=(8, 8, 8, 8),
        decoder_channels=16,
        reassemble_factors=(4.0, 2.0, 1.0, 0.5),
    )
    features = tuple(torch.randn(2, 8, 4, 4) for _ in range(4))

    output = decoder(features)

    assert isinstance(decoder.reassembly[0], nn.Upsample)
    assert decoder.reassembly[0].scale_factor == 4.0
    assert isinstance(decoder.reassembly[1], nn.Upsample)
    assert decoder.reassembly[1].scale_factor == 2.0
    assert isinstance(decoder.reassembly[2], nn.Identity)
    assert isinstance(decoder.reassembly[3], nn.Upsample)
    assert decoder.reassembly[3].scale_factor == 0.5
    assert output.shape == (2, 16, 16, 16)
