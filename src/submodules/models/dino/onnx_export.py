"""Export DINO with standard GridSample and a variable image batch."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

from src.submodules.models.dino.architecture import (
    build_dino,
    dino_4scale_swin_args,
    load_dino_build_function,
    load_dino_state_dict,
)


class _DinoBindings(nn.Module):
    def __init__(self, model: nn.Module, nested_tensor: Callable[..., Any]) -> None:
        super().__init__()
        self.model = model
        self.nested_tensor = nested_tensor

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(
            (pixels.shape[0], *pixels.shape[-2:]),
            dtype=torch.bool,
            device=pixels.device,
        )
        result = self.model(self.nested_tensor(pixels, mask))
        return result["pred_logits"], result["pred_boxes"]


def _window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int
) -> torch.Tensor:
    # The upstream int() fixes the traced batch size. Only H and W are fixed.
    channels = windows.shape[-1]
    tiles = windows.reshape(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    return tiles.permute(0, 1, 3, 2, 4, 5).reshape(-1, height, width, channels)


def export_dino_onnx(
    checkpoint: Path, repository: Path, path: Path, height: int, width: int
) -> None:
    onnx = import_module("onnx")
    convert = import_module("onnxruntime.transformers.float16").convert_float_to_float16
    sort_graph = import_module(
        "onnxruntime.transformers.onnx_model"
    ).OnnxModel.graph_topological_sort
    model, criterion = build_dino(
        repository, dino_4scale_swin_args("cpu", use_checkpoint=False)
    )
    del criterion
    model.load_state_dict(load_dino_state_dict(checkpoint), strict=True)
    wrapper = _DinoBindings(model, import_module("util.misc").NestedTensor).eval()

    prefix = load_dino_build_function(repository).__module__.removesuffix(".dino.dino")
    attention = import_module(f"{prefix}.dino.ops.modules.ms_deform_attn")
    reference = import_module(
        f"{prefix}.dino.ops.functions.ms_deform_attn_func"
    ).ms_deform_attn_core_pytorch
    swin = import_module(f"{prefix}.dino.swin_transformer")

    def grid_sample(
        value: torch.Tensor,
        geometry: torch.Tensor,
        _starts: torch.Tensor,
        locations: torch.Tensor,
        weights: torch.Tensor,
        _step: int,
    ) -> torch.Tensor:
        fixed_shapes = [(h, w) for h, w in geometry.tolist()]
        output: torch.Tensor = reference(value, fixed_shapes, locations, weights)
        return output

    with TemporaryDirectory(prefix=".dino-export-", dir=path.parent) as raw:
        source = Path(raw) / "fp32.onnx"
        with (
            torch.no_grad(),
            patch.object(swin, "window_reverse", _window_reverse),
            patch.object(
                attention, "MSDeformAttnFunction", SimpleNamespace(apply=grid_sample)
            ),
        ):
            torch.onnx.export(
                wrapper,
                (torch.zeros(1, 3, height, width),),
                str(source),
                dynamo=False,
                opset_version=19,
                external_data=True,
                input_names=["pixels"],
                output_names=["pred_logits", "pred_boxes"],
                dynamic_axes={
                    name: {0: "batch"}
                    for name in ("pixels", "pred_logits", "pred_boxes")
                },
            )
        del wrapper, model
        typed = convert(onnx.load(source), keep_io_types=True)
        sort_graph(typed.graph)
        onnx.save_model(
            typed, path, save_as_external_data=True, location=f"{path.name}.data"
        )
