"""CPU-only unit tests for the DINO court adapter using a tiny fake model."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import src.tasks.court_alignment.models.dino_detector as dino_detector_module
from src.tasks.court_alignment.models.dino_detector import (
    COURT_CLASS_COUNT,
    DinoCourtDetector,
    lora_parameter_count,
)
from src.utils.models.lora import LoRALinear


class _FakeNestedTensor:
    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        self.tensors = tensors
        self.mask = mask


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_embed = nn.ModuleList()


class _FakeAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.qkv(values))


class _FakeTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layer_count: int,
        query_count: int,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.d_model = hidden_dim
        self.num_decoder_layers = layer_count
        self.query_count = query_count
        self.use_checkpoint = use_checkpoint
        self.input_projection = nn.Linear(3, hidden_dim)
        self.block = _FakeAttentionBlock(hidden_dim)
        self.decoder = _FakeDecoder()
        self.enc_out_class_embed = nn.Linear(hidden_dim, 91)
        self.enc_out_bbox_embed = nn.Linear(hidden_dim, 4)

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], None]:
        pooled = images.mean(dim=(-2, -1))
        projected = self.input_projection(pooled)
        if self.use_checkpoint:
            hidden = checkpoint(self.block, projected, use_reentrant=True)
        else:
            hidden = self.block(projected)
        queries = hidden[:, None, :].expand(-1, self.query_count, -1)
        layers = [
            queries + float(index) for index in range(self.num_decoder_layers)
        ]
        return layers, None


class _FakeDino(nn.Module):
    def __init__(self, *, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.num_classes = 91
        self.hidden_dim = 8
        self.dn_labelbook_size = 91
        self.transformer = _FakeTransformer(
            hidden_dim=self.hidden_dim,
            layer_count=3,
            query_count=5,
            use_checkpoint=use_checkpoint,
        )
        # Shared decoder heads match the released Swin-L DINO configuration.
        class_projection = nn.Linear(self.hidden_dim, 91)
        self.class_embed = nn.ModuleList([class_projection] * 3)
        self.transformer.decoder.class_embed = self.class_embed
        self.bbox_embed = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 4) for _ in range(3)]
        )
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.label_enc = nn.Embedding(92, self.hidden_dim)
        self.output_dn_meta = {"pad_size": 4}
        self.last_nested_input: _FakeNestedTensor | None = None

    def forward(
        self,
        images: object,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        del targets
        if not isinstance(images, _FakeNestedTensor):
            raise TypeError("Fake official DINO requires a NestedTensor input.")
        self.last_nested_input = images
        hidden_states, _ = self.transformer(images.tensors)
        logits = torch.stack(
            tuple(head(hidden) for head, hidden in zip(self.class_embed, hidden_states, strict=True))
        )
        boxes = torch.stack(
            tuple(
                head(hidden).sigmoid()
                for head, hidden in zip(self.bbox_embed, hidden_states, strict=True)
            )
        )
        return {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "aux_outputs": [
                {"pred_logits": layer_logits, "pred_boxes": layer_boxes}
                for layer_logits, layer_boxes in zip(logits[:-1], boxes[:-1], strict=True)
            ],
            "dn_meta": self.output_dn_meta,
        }


def _build_model(
    *,
    mode: str = "repeat_rgb",
    use_checkpoint: bool = False,
) -> DinoCourtDetector:
    return DinoCourtDetector(
        _FakeDino(use_checkpoint=use_checkpoint),
        input_mode=mode,  # type: ignore[arg-type]
        short_side=8,
        max_long_side=8,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        lora_target_modules=("qkv",),
        nested_tensor_factory=_FakeNestedTensor,
    )


def test_official_decoder_state_list_is_stacked_by_layer() -> None:
    states = [torch.full((2, 5, 8), float(index)) for index in range(3)]

    stacked = dino_detector_module._stack_decoder_hidden_states(
        (states, object()),
        hidden_dim=8,
    )

    assert stacked.shape == (3, 2, 5, 8)
    torch.testing.assert_close(stacked[2], states[2])


@pytest.mark.parametrize(
    ("decoder_states", "message"),
    [
        (torch.zeros(3, 2, 5, 8), "non-empty list"),
        ([], "non-empty list"),
        (
            [torch.zeros(2, 5, 8), torch.zeros(2, 4, 8)],
            "share batch/query/hidden shapes",
        ),
        ([torch.zeros(2, 5, 7)], "hidden size"),
    ],
)
def test_invalid_official_decoder_state_contract_fails_explicitly(
    decoder_states: object,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        dino_detector_module._stack_decoder_hidden_states(
            (decoder_states, object()),
            hidden_dim=8,
        )


def test_forward_preserves_dino_boxes_and_adds_scale_axis_to_every_layer() -> None:
    model = _build_model()

    output = model(torch.zeros(2, 1, 8, 8))

    assert output["pred_logits"].shape == (2, 5, COURT_CLASS_COUNT)
    assert output["pred_boxes"].shape == (2, 5, 4)
    assert output["pred_court_boxes"].shape == (2, 5, 3)
    auxiliary = output["aux_outputs"]
    assert isinstance(auxiliary, list)
    assert len(auxiliary) == 2
    assert all(layer["pred_court_boxes"].shape == (2, 5, 3) for layer in auxiliary)
    torch.testing.assert_close(
        output["pred_court_boxes"],
        torch.tensor((0.0, 1.0, 0.0)).expand(2, 5, 3),
    )
    assert output["dn_meta"] is cast(_FakeDino, model.dino).output_dn_meta
    nested_input = cast(_FakeDino, model.dino).last_nested_input
    assert nested_input is not None
    assert not bool(nested_input.mask.any())


def test_only_lora_new_heads_and_learnable_input_adapter_are_trainable() -> None:
    model = _build_model(mode="learnable_1x1")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    lora_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_a, module.lora_b)
    }
    fake_dino = cast(_FakeDino, model.dino)
    class_head_ids = {
        id(parameter)
        for module in (
            fake_dino.class_embed,
            fake_dino.transformer.enc_out_class_embed,
            fake_dino.label_enc,
        )
        for parameter in module.parameters()
    }
    court_head_ids = {id(parameter) for parameter in model.court_head.parameters()}
    input_ids = {id(parameter) for parameter in model.input_adapter.parameters()}
    bbox_head_ids = {
        id(parameter)
        for module in (
            fake_dino.bbox_embed,
            cast(nn.Module, fake_dino.transformer.decoder.bbox_embed),
            fake_dino.transformer.enc_out_bbox_embed,
        )
        for parameter in module.parameters()
    }

    assert trainable_ids == (
        lora_ids | class_head_ids | bbox_head_ids | court_head_ids | input_ids
    )
    assert lora_parameter_count(model) > 0
    assert model.lora_module_names == ("transformer.block.qkv",)
    assert model.trainable_parameter_names()
    assert fake_dino.bbox_embed is fake_dino.transformer.decoder.bbox_embed
    assert all(
        not parameter.requires_grad
        for parameter in fake_dino.transformer.input_projection.parameters()
    )


@pytest.mark.parametrize("mode", ["repeat_rgb", "red_only"])
def test_checkpointed_frozen_block_retains_lora_gradients(mode: str) -> None:
    model = _build_model(mode=mode, use_checkpoint=True).train()

    output = model(torch.full((2, 1, 8, 8), 0.25))
    output["pred_boxes"].sum().backward()

    lora_layers = [module for module in model.modules() if isinstance(module, LoRALinear)]
    assert len(lora_layers) == 1
    gradient = lora_layers[0].lora_b.grad
    assert gradient is not None
    assert bool(torch.count_nonzero(gradient))


def test_direct_nested_tensor_preserves_learnable_adapter_gradient() -> None:
    model = _build_model(mode="learnable_1x1", use_checkpoint=True).train()

    output = model(torch.full((2, 1, 8, 8), 0.25))
    nested_input = cast(_FakeDino, model.dino).last_nested_input
    assert nested_input is not None
    assert nested_input.tensors.requires_grad
    output["pred_boxes"].sum().backward()

    projection = model.input_adapter.projection
    assert projection is not None
    assert projection.weight.grad is not None
    assert bool(torch.count_nonzero(projection.weight.grad))


def test_repeat_and_red_only_modes_add_no_input_parameters() -> None:
    for mode in ("repeat_rgb", "red_only"):
        model = _build_model(mode=mode)
        assert list(model.input_adapter.parameters()) == []


def test_zero_lora_target_match_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        DinoCourtDetector(
            _FakeDino(),
            input_mode="repeat_rgb",
            short_side=8,
            max_long_side=8,
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            lora_target_modules=("does_not_exist",),
        )


def test_non_coco_base_is_rejected_before_mutation() -> None:
    dino = _FakeDino()
    dino.num_classes = 1
    with pytest.raises(ValueError, match="91-class COCO"):
        DinoCourtDetector(
            dino,
            input_mode="repeat_rgb",
            short_side=8,
            max_long_side=8,
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            lora_target_modules=("qkv",),
        )


def test_missing_cuda_extension_has_actionable_build_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path.resolve()
    required = (
        repository / "models/__init__.py",
        repository / "util/__init__.py",
        repository / "config/DINO/DINO_4scale_swin.py",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def fake_load_package(package_name: str, _init_path: Path) -> ModuleType:
        if package_name == "util":
            return ModuleType("util")
        raise ModuleNotFoundError(
            "No module named 'MultiScaleDeformableAttention'",
            name="MultiScaleDeformableAttention",
        )

    monkeypatch.setattr(dino_detector_module, "_load_package", fake_load_package)

    with pytest.raises(RuntimeError, match="build_ext --inplace") as error:
        dino_detector_module._load_official_dino_components(repository)
    assert "PYTHONPATH" in str(error.value)
    assert "TENNIS_LAB_BUILD_CUDA_OPS=1" in str(error.value)
    assert str(repository.parent.parent) in str(error.value)
