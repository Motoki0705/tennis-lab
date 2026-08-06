"""Tests for the typed SLCS DINO frame-token adapter and factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src.tasks.base.model_io import (
    ModelAdapterMismatchError,
    ModelInputContractError,
    ModelOutputContractError,
)
from src.tasks.slcs.configuration import SLCSPrecomputeConfig
from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
from src.tasks.slcs.model_io.factory import create_slcs_frame_token_encoder
from src.tasks.slcs.model_io.frame_tokens import (
    BoundSLCSFrameTokenEncoder,
    SLCSFrameTokenIOAdapter,
)
from src.utils.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.utils.device import DeviceSelectionError


class _FakeBackbone(nn.Module):
    def __init__(
        self,
        spec: DinoTokenSpec,
        *,
        output: str = "valid",
        embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = spec.embed_dim if embed_dim is None else embed_dim
        self.patch_size = spec.patch_size
        self.num_tokens = spec.num_tokens
        self.output = output
        self.inputs: list[Tensor] = []

    def forward_features(self, inputs: Tensor) -> Mapping[str, object]:
        self.inputs.append(inputs.detach().clone())
        if self.output == "missing":
            return {}
        shape = (inputs.shape[0], self.num_tokens, self.embed_dim)
        tokens = torch.ones(shape, dtype=torch.float32, device=inputs.device)
        if self.output == "wrong_shape":
            tokens = tokens[:, :-1]
        elif self.output == "nonfinite":
            tokens[0, 0, 0] = torch.nan
        elif self.output == "integer":
            tokens = tokens.to(torch.int64)
        return {"x_norm_patchtokens": tokens}


@pytest.fixture
def frame_spec() -> DinoTokenSpec:
    return DinoTokenSpec(
        backbone="fake",
        patch_size=2,
        image_height=4,
        image_width=6,
        embed_dim=5,
        frame_stride=2,
    )


def _encoder(
    spec: DinoTokenSpec,
    *,
    output: str = "valid",
) -> tuple[BoundSLCSFrameTokenEncoder, _FakeBackbone]:
    model = _FakeBackbone(spec, output=output)
    adapter = SLCSFrameTokenIOAdapter(spec, torch.device("cpu"))
    adapter.validate_model(model)
    return BoundSLCSFrameTokenEncoder(model=model, adapter=adapter), model


def test_valid_frames_are_normalized_and_decoded_by_adapter(
    frame_spec: DinoTokenSpec,
) -> None:
    encoder, model = _encoder(frame_spec)
    frames: NDArray[np.uint8] = np.zeros((2, 4, 6, 3), dtype=np.uint8)
    frames[1] = 255

    tokens = encoder(frames)

    assert tokens.shape == (2, frame_spec.num_tokens, frame_spec.embed_dim)
    assert tokens.dtype == np.float16
    assert len(model.inputs) == 1
    model_input = model.inputs[0]
    assert model_input.shape == (2, 3, 4, 6)
    assert model_input.dtype == torch.float32
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    zero_expected = ((torch.zeros_like(mean) - mean) / std).expand_as(model_input[0:1])
    one_expected = ((torch.ones_like(mean) - mean) / std).expand_as(model_input[1:2])
    torch.testing.assert_close(model_input[0:1], zero_expected)
    torch.testing.assert_close(model_input[1:2], one_expected)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (np.zeros((1, 4, 6, 3), dtype=np.float32), "uint8"),
        (np.zeros((4, 6, 3), dtype=np.uint8), "shape"),
        (np.zeros((1, 5, 6, 3), dtype=np.uint8), "shape"),
        (np.zeros((0, 4, 6, 3), dtype=np.uint8), "non-empty"),
    ],
)
def test_invalid_frames_fail_before_model_entry(
    frame_spec: DinoTokenSpec,
    frames: np.ndarray,
    message: str,
) -> None:
    encoder, model = _encoder(frame_spec)

    with pytest.raises(ModelInputContractError, match=message):
        encoder(cast(Any, frames))

    assert not model.inputs


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("missing", "missing required"),
        ("wrong_shape", "must have shape"),
        ("nonfinite", "non-finite"),
        ("integer", "floating tensor"),
    ],
)
def test_invalid_model_output_is_rejected(
    frame_spec: DinoTokenSpec,
    output: str,
    message: str,
) -> None:
    encoder, model = _encoder(frame_spec, output=output)

    with pytest.raises(ModelOutputContractError, match=message):
        encoder(np.zeros((1, 4, 6, 3), dtype=np.uint8))

    assert len(model.inputs) == 1


def _runtime_config(
    spec: DinoTokenSpec,
    *,
    device: str = "cpu",
) -> SLCSPrecomputeConfig:
    return cast(
        SLCSPrecomputeConfig,
        SimpleNamespace(
            data=SimpleNamespace(pipeline=SimpleNamespace(dino_spec=spec)),
            device=device,
            repository_path=Path("/repository"),
            checkpoint_path=Path("/checkpoint.pth"),
            strict=True,
        ),
    )


def test_factory_selects_and_binds_backbone_before_execution(
    frame_spec: DinoTokenSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeBackbone(frame_spec)
    calls: list[dict[str, object]] = []

    def load(**kwargs: object) -> _FakeBackbone:
        calls.append(kwargs)
        return model

    monkeypatch.setattr("src.tasks.slcs.model_io.factory.load_dinov3_backbone", load)

    encoder = create_slcs_frame_token_encoder(_runtime_config(frame_spec))

    assert encoder.model is model
    assert not model.training
    assert calls == [
        {
            "repository_path": Path("/repository"),
            "checkpoint_path": Path("/checkpoint.pth"),
            "backbone_name": "fake",
            "strict": True,
        }
    ]
    assert not model.inputs


def test_factory_rejects_backbone_spec_mismatch_before_execution(
    frame_spec: DinoTokenSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeBackbone(frame_spec, embed_dim=frame_spec.embed_dim + 1)
    monkeypatch.setattr(
        "src.tasks.slcs.model_io.factory.load_dinov3_backbone",
        lambda **_: model,
    )

    with pytest.raises(ModelAdapterMismatchError, match="expects backbone"):
        create_slcs_frame_token_encoder(_runtime_config(frame_spec))

    assert not model.inputs


def test_factory_rejects_unavailable_explicit_cuda_before_backbone_load(
    frame_spec: DinoTokenSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime_config(frame_spec, device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loads: list[object] = []
    monkeypatch.setattr(
        "src.tasks.slcs.model_io.factory.load_dinov3_backbone",
        lambda **kwargs: loads.append(kwargs),
    )

    with pytest.raises(DeviceSelectionError, match="CUDA is unavailable"):
        create_slcs_frame_token_encoder(runtime)

    assert loads == []


def test_adapter_rejects_missing_backbone_surface_before_execution(
    frame_spec: DinoTokenSpec,
) -> None:
    adapter = SLCSFrameTokenIOAdapter(frame_spec, torch.device("cpu"))

    with pytest.raises(ModelAdapterMismatchError, match="requires a backbone"):
        adapter.validate_model(object())
