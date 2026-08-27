"""Tests for the DINOv3 execution boundary and DPT decoder."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.tasks.court_detection.configuration import (
    CourtDecoderConfig,
    CourtEncoderConfig,
    CourtLossConfig,
    CourtModelConfig,
    CourtTransformerEncoderConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetKind,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
    CourtModelSpec,
)
from src.tasks.court_detection.models import hierarchical_model as model_module
from src.tasks.court_detection.models.decoder import CourtDPTDecoder
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.pose_head import CourtModelOutput
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
        self.seen_input: torch.Tensor | None = None
        self.grad_enabled: bool | None = None
        self.invalid_response = False

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        self.seen_input_shape = tuple(inputs.shape)
        self.seen_input = inputs.detach()
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
            raise AssertionError("DINOv3 must return normalized patch tokens.")
        self.requested_layers = n
        if self.invalid_response:
            return (torch.zeros(1),)
        tokens = self.forward_features(inputs)["x_norm_patchtokens"]
        return tuple(tokens + index for index in n)


def _bundle() -> CourtTargetBundleSpec:
    return CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=7,
                channel_names=tuple(f"kp_{index}" for index in range(7)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )


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
    def __init__(
        self,
        encoder: CourtDINOv3Encoder,
        bundle: CourtTargetBundleSpec,
    ) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.target_bundle_spec = bundle
        self.encoder = encoder
        self.calls = 0

    def forward(
        self,
        x: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
        patch_valid_mask: torch.Tensor | None = None,
    ) -> dict[CourtTargetKind, torch.Tensor]:
        self.calls += 1
        assert patch_valid_mask is None
        assert all(
            value is not None
            for value in (feature_1, feature_2, feature_3, feature_4)
        )
        return {"kp": x.new_zeros(x.shape[0], 7, x.shape[-2], x.shape[-1])}


def _loss_config(*, pose: bool = False) -> CourtLossConfig:
    return CourtLossConfig.from_mapping(
        {
            "seg": {"ce_weight": 1.0, "dice_weight": 1.0, "weight": 1.0},
            "kp": {"focal_gamma": 2.0, "weight": 1.0},
            "line": {
                "bce_weight": 1.0,
                "dice_weight": 1.0,
                "pos_weight": 1.0,
                "weight": 1.0,
            },
            "pose": {
                "enabled": pose,
                "translation_weight": 1.0 if pose else 0.0,
                "rotation_weight": 1.0 if pose else 0.0,
                "focal_weight": 1.0 if pose else 0.0,
            },
            "consistency": {
                "enabled": False,
                "weight": 0.0,
                "temperature": 1.0,
                "huber_delta": 0.01,
                "min_depth_m": 0.1,
                "depth_scale_m": 1.0,
                "cheirality_weight": 0.0,
                "warmup_fraction": 0.0,
                "gradient_flow": "both",
            },
        }
    )


def _adapter(model: CourtHierarchicalModel) -> CourtModelIOAdapter:
    bundle = model.target_bundle_spec
    adapter = CourtModelIOAdapter(
        CourtModelSpec(
            target_bundle=bundle,
            in_channels=3,
            short_side=32,
            encoder_kind="dinov3",
        ),
        loss_config=_loss_config(),
        execution_boundary=CourtDINOv3ExecutionBoundary(frozen_backbone=True),
    )
    adapter.validate_model_pair(model)
    return adapter


def _pose_adapter(model: CourtHierarchicalModel) -> CourtPoseModelIOAdapter:
    adapter = CourtPoseModelIOAdapter(
        CourtModelSpec(
            target_bundle=model.target_bundle_spec,
            in_channels=3,
            short_side=32,
            encoder_kind="dinov3",
        ),
        loss_config=_loss_config(pose=True),
        execution_boundary=CourtDINOv3ExecutionBoundary(frozen_backbone=True),
    )
    adapter.validate_model_pair(model)
    return adapter


def test_dinov3_boundary_reassembles_four_intermediate_maps() -> None:
    fake = FakeDINOv3()
    model = _CountingCourtDINOModel(_encoder(fake), _bundle())
    adapter = _adapter(model)

    call = adapter.prepare_images(torch.zeros(2, 3, 17, 19))
    features = call.model_args[1:]

    assert fake.requested_layers == (2, 5, 8, 11)
    assert fake.grad_enabled is False
    assert fake.seen_input_shape == (2, 3, 20, 20)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
        (2, 8, 5, 5),
    ]


@pytest.mark.parametrize(
    ("height", "width", "expected_height", "expected_width"),
    [(8, 12, 8, 12), (7, 10, 8, 12)],
)
def test_dinov3_boundary_only_adds_right_bottom_replicate_padding(
    height: int,
    width: int,
    expected_height: int,
    expected_width: int,
) -> None:
    fake = FakeDINOv3()
    model = _CountingCourtDINOModel(_encoder(fake), _bundle())
    adapter = _adapter(model)
    images = torch.linspace(
        -1.0,
        1.0,
        2 * 3 * height * width,
        dtype=torch.float32,
    ).reshape(2, 3, height, width)

    call = adapter.prepare_images(images)

    assert call.model_args[0] is images
    assert fake.seen_input is not None
    padded = fake.seen_input
    assert padded.shape == (2, 3, expected_height, expected_width)
    torch.testing.assert_close(padded[:, :, :height, :width], images)
    if expected_width > width:
        torch.testing.assert_close(
            padded[:, :, :height, width:],
            images[:, :, :, -1:].expand(-1, -1, -1, expected_width - width),
        )
    if expected_height > height:
        torch.testing.assert_close(
            padded[:, :, height:, :],
            padded[:, :, height - 1 : height, :].expand(
                -1,
                -1,
                expected_height - height,
                -1,
            ),
        )


def test_invalid_dinov3_response_fails_before_model_forward() -> None:
    fake = FakeDINOv3()
    fake.invalid_response = True
    model = _CountingCourtDINOModel(_encoder(fake), _bundle())
    adapter = _adapter(model)

    with pytest.raises(CourtModelIOError, match="return four tensors"):
        adapter.prepare_images(torch.zeros(1, 3, 16, 20))

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


def _enabled_model_config() -> CourtModelConfig:
    return CourtModelConfig(
        name="court_hierarchical",
        in_channels=3,
        encoder=CourtEncoderConfig(
            name="dinov3",
            repository_path=None,
            checkpoint_path=None,
            backbone_name=None,
            strict=None,
            train_mode=None,
            last_n_blocks=None,
            out_indices=None,
            layer_mode=None,
            lora=None,
        ),
        decoder=CourtDecoderConfig(
            name="dpt",
            size="tiny",
            channels=64,
            reassemble_factors=(4.0, 2.0, 1.0, 0.5),
        ),
        transformer_encoder=CourtTransformerEncoderConfig(
            name="transformer",
            enabled=True,
            dim=8,
            depth=1,
            num_heads=2,
            head_dim=4,
            ffn_dim=16,
            rope_dim=4,
            rope_theta=10000.0,
            dropout=0.0,
            attention_type="mha",
            n_kv_heads=None,
            ffn_type="swiglu",
        ),
    )


def test_prepared_dinov3_features_flow_through_transformer_dpt_and_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDINOv3()
    encoder = _encoder(fake)
    monkeypatch.setattr(
        model_module,
        "build_court_encoder",
        lambda **kwargs: encoder,
    )
    model = CourtHierarchicalModel(_enabled_model_config(), _bundle())
    adapter = _adapter(model)

    call = adapter.prepare_images(torch.zeros(2, 3, 17, 19))
    boundary = adapter.execution_boundary
    assert isinstance(boundary, CourtDINOv3ExecutionBoundary)
    call = boundary.attach_patch_valid_mask(
        call,
        torch.tensor([[17, 19], [9, 12]], dtype=torch.long),
    )
    patch_valid_mask = call.model_args[-1]
    output = model(*call.model_args)

    assert patch_valid_mask.shape == (2, 5, 5)
    assert torch.count_nonzero(patch_valid_mask[0]).item() == 25
    assert torch.count_nonzero(patch_valid_mask[1]).item() == 9
    assert isinstance(output, CourtModelOutput)
    assert output.pose is not None
    assert output.pose.values.shape == (2, 10)
    assert output.dense_logits["kp"].shape == (2, 7, 17, 19)
    assert fake.requested_layers == (2, 5, 8, 11)
    assert fake.grad_enabled is False

    loss = output.dense_logits["kp"].square().mean() + output.pose.values.square().mean()
    loss.backward()

    selected_gradients = (
        model.transformer_encoder.pose_query.grad,
        next(model.pose_head.parameters()).grad,
        next(model.heads["kp"].parameters()).grad,
    )
    assert all(gradient is not None for gradient in selected_gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        and bool(torch.count_nonzero(gradient))
        for gradient in selected_gradients
        if gradient is not None
    )


def test_pose_training_propagates_content_size_as_dino_patch_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeDINOv3()
    encoder = _encoder(fake)
    monkeypatch.setattr(
        model_module,
        "build_court_encoder",
        lambda **kwargs: encoder,
    )
    model = CourtHierarchicalModel(_enabled_model_config(), _bundle())
    adapter = _pose_adapter(model)
    batch_size, height, width = 2, 20, 20
    raw_pose = torch.tensor(
        [0.0, -20.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0]
    ).repeat(batch_size, 1)
    batch = {
        "image": torch.zeros(batch_size, 3, height, width),
        "targets": {
            "kp": {
                "heatmap": torch.zeros(batch_size, 7, height, width),
                "points_xy": torch.zeros(batch_size, 7, 1, 2),
                "point_visible": torch.ones(
                    batch_size, 7, 1, dtype=torch.bool
                ),
                "physical_indices": torch.arange(7, dtype=torch.long)
                .view(1, 7, 1)
                .expand(batch_size, -1, -1),
            }
        },
        "pose_target": {
            "translation_m": raw_pose[:, :3],
            "rotation": torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
            "log_focal": raw_pose[:, -1],
            "intrinsics": torch.tensor(
                [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
            )
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            "semantic_to_physical": torch.arange(14, dtype=torch.long)
            .view(1, 14)
            .expand(batch_size, -1),
            "raw_pose10d": raw_pose,
        },
        "image_size": torch.tensor([[20, 20], [20, 20]], dtype=torch.long),
        "content_size_hw": torch.tensor(
            [[20, 20], [9, 12]], dtype=torch.long
        ),
    }

    call = adapter.prepare_training_batch(batch)
    patch_valid_mask = call.model_call.model_args[-1]
    inference_call = adapter.build_call(batch)
    inference_patch_valid_mask = inference_call.args[-1]
    output = model(*call.model_call.model_args)

    torch.testing.assert_close(inference_patch_valid_mask, patch_valid_mask)
    assert patch_valid_mask.dtype is torch.bool
    assert patch_valid_mask.shape == (2, 5, 5)
    assert torch.count_nonzero(patch_valid_mask[0]).item() == 25
    assert torch.count_nonzero(patch_valid_mask[1]).item() == 9
    assert isinstance(output, CourtModelOutput)
