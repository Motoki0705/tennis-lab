"""Tests for Court model composition and bundle-derived heads."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from src.tasks.court_detection.configuration import (
    CourtDecoderConfig,
    CourtEncoderConfig,
    CourtModelConfig,
)
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.models import hierarchical_model as model_module
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


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
            ),
            "seg": CourtTargetSpec(
                kind="seg",
                schema="test_seg",
                output_channels=7,
                channel_names=tuple(f"class_{index}" for index in range(7)),
                target_dtype=torch.long,
                precomputed=True,
            ),
            "line": CourtTargetSpec(
                kind="line",
                schema="test_line",
                output_channels=1,
                channel_names=("line",),
                target_dtype=torch.float32,
                precomputed=True,
            ),
        }
    )


def test_dinov3_dpt_factory_binds_exact_bundle(monkeypatch) -> None:
    bundle = _bundle()
    expected = object.__new__(CourtHierarchicalModel)
    nn.Module.__init__(expected)
    expected.in_channels = 3
    expected.target_bundle_spec = bundle
    encoder = object.__new__(CourtDINOv3Encoder)
    nn.Module.__init__(encoder)
    expected.encoder = encoder

    def fake_from_config(
        config: object,
        selected_bundle: CourtTargetBundleSpec,
    ) -> CourtHierarchicalModel:
        _ = config
        assert selected_bundle == bundle
        return expected

    monkeypatch.setattr(
        CourtHierarchicalModel,
        "from_config",
        staticmethod(fake_from_config),
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[
                "data/processing=all",
                "model/encoder=dinov3",
                "model/decoder=dpt",
            ],
        )

    pair = build_court_detection_pair(config, target_bundle=bundle)

    assert pair.model is expected
    assert isinstance(pair.adapter, CourtModelIOAdapter)
    assert pair.adapter.spec.target_bundle == bundle


class _TinyEncoder(nn.Module):
    feature_channels = (4, 4, 4, 4)
    requires_prepared_features = False

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor):
        feature = self.projection(images)
        return (feature, feature, feature, feature)


class _TinyDecoder(nn.Module):
    output_channels = 4

    def forward(self, features):
        assert all(feature is not None for feature in features)
        return features[0]


def test_shared_decoder_multi_head_forward_backward(monkeypatch) -> None:
    encoder = _TinyEncoder()
    decoder = _TinyDecoder()
    monkeypatch.setattr(
        model_module,
        "build_court_encoder",
        lambda **kwargs: encoder,
    )
    monkeypatch.setattr(
        model_module,
        "build_court_decoder",
        lambda **kwargs: decoder,
    )
    config = CourtModelConfig(
        name="court_hierarchical",
        in_channels=3,
        encoder=CourtEncoderConfig(
            name="default",
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
            name="fpn",
            channels=(4, 4, 4, 4),
            reassemble_factors=None,
        ),
    )
    bundle = _bundle()
    model = CourtHierarchicalModel(config, bundle)
    images = torch.randn(2, 3, 8, 8)

    outputs = model(images)
    loss = sum(value.square().mean() for value in outputs.values())
    loss.backward()

    assert {kind: value.shape for kind, value in outputs.items()} == {
        "kp": (2, 7, 8, 8),
        "seg": (2, 7, 8, 8),
        "line": (2, 1, 8, 8),
    }
    assert encoder.projection.weight.grad is not None
