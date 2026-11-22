"""Export the released GVHMR denoiser with a dynamic time axis."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.onnx import symbolic_helper, symbolic_opset9, symbolic_opset11

from src.submodules.vendor.gvhmr.network.relative_transformer import NetworkEncoderRoPE
from src.submodules.vendor.gvhmr.utils.net_utils import length_to_mask

ATTENTION_WINDOW = 120
ROTARY_MAX_FRAMES = 4096
OPSET = 19


# The legacy exporter needs two explicit lowerings for TensorRT's typed graph.
def _einsum(graph: Any, equation: str, operands: Any, _path: Any = None) -> Any:
    left, right = symbolic_helper._unpack_list(operands)
    if equation == "abcd,abed->abce":
        right = graph.op("Transpose", right, perm_i=[0, 1, 3, 2])
    elif equation != "abcd,abde->abce":
        raise ValueError(f"Unexpected GVHMR attention equation: {equation}")
    return graph.op("MatMul", left, right)


def _index_put(
    graph: Any, data: Any, indices: Any, value: Any, accumulate: bool
) -> Any:
    keys = (
        symbolic_helper._unpack_list(indices)
        if symbolic_helper._is_packed_list(indices)
        else [indices]
    )
    rank = symbolic_helper._get_tensor_rank
    if (
        not accumulate
        and len(keys) == 1
        and symbolic_helper._is_bool(keys[0])
        and rank(data) == 4
        and rank(keys[0]) == 3
        and rank(value) == 0
    ):
        # obs[~visible] = 0 preserves x, y and confidence along the last axis.
        mask = symbolic_helper._unsqueeze_helper(graph, keys[0], [-1])
        return symbolic_opset9.masked_fill(graph, data, mask, value)
    return symbolic_opset11.index_put(graph, data, indices, value, accumulate)


def _attention_mask(length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    last_start = (positions[-1] + 1 - ATTENTION_WINDOW).clamp_min(0)
    starts = torch.minimum((positions - ATTENTION_WINDOW // 2).clamp_min(0), last_start)
    ends = (positions + ATTENTION_WINDOW // 2).clamp_min(ATTENTION_WINDOW)
    return (positions[None, :] < starts[:, None]) | (
        positions[None, :] >= ends[:, None]
    )


@contextmanager
def _dynamic_attention(network: NetworkEncoderRoPE) -> Iterator[None]:
    if network.max_len != ATTENTION_WINDOW:
        raise ValueError("GVHMR local attention window changed")
    mask: torch.Tensor | None = None

    def set_mask(_network: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        nonlocal mask
        obs = args[1]
        mask = _attention_mask(obs.shape[1], obs.device)

    def use_mask(
        _block: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, {**kwargs, "attn_mask": mask}

    network.max_len = ROTARY_MAX_FRAMES
    handles = [network.register_forward_pre_hook(set_mask)]
    handles.extend(
        block.register_forward_pre_hook(use_mask, with_kwargs=True)
        for block in network.blocks
    )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        network.max_len = ATTENTION_WINDOW


class _DenoiserBindings(nn.Module):
    def __init__(self, network: NetworkEncoderRoPE) -> None:
        super().__init__()
        self.network = network

    def forward(
        self,
        length: torch.Tensor,
        obs: torch.Tensor,
        f_cliffcam: torch.Tensor,
        f_cam_angvel: torch.Tensor,
        f_imgseq: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        data = self.network(length, obs, f_cliffcam, f_cam_angvel, f_imgseq)
        mask = length_to_mask(length, obs.shape[1])
        return (
            data["pred_x"],
            data["pred_cam"],
            data["static_conf_logits"],
            mask,
            mask.sum(dim=1),
        )


def export_gvhmr_onnx(checkpoint: Path, path: Path) -> None:
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "pipeline.denoiser3d."
    network = NetworkEncoderRoPE().eval()
    network.load_state_dict(
        {
            name.removeprefix(prefix): value
            for name, value in checkpoint_state["state_dict"].items()
            if name.startswith(prefix)
        },
        strict=True,
    )
    del checkpoint_state
    model = _DenoiserBindings(network).half().eval()
    frames = ATTENTION_WINDOW + 1
    inputs = ("length", "obs", "f_cliffcam", "f_cam_angvel", "f_imgseq")
    outputs = ("pred_x", "pred_cam", "static_conf_logits", "mask_out", "length_out")
    args = (
        torch.tensor([frames]),
        *(
            torch.zeros((1, frames, *shape), dtype=torch.float16)
            for shape in ((17, 3), (3,), (6,), (1024,))
        ),
    )
    symbolics = {
        "aten::einsum": symbolic_helper.parse_args("s", "v", "is")(_einsum),
        "aten::index_put": symbolic_helper.parse_args("v", "v", "v", "b")(_index_put),
    }
    for operator, lowering in symbolics.items():
        torch.onnx.register_custom_op_symbolic(operator, lowering, OPSET)
    try:
        with torch.no_grad(), _dynamic_attention(network):
            torch.onnx.export(
                model,
                args,
                str(path),
                dynamo=False,
                opset_version=OPSET,
                input_names=list(inputs),
                output_names=list(outputs),
                dynamic_axes={
                    name: {1: "frames"} for name in (*inputs[1:], *outputs[:-1])
                },
            )
    finally:
        for operator in symbolics:
            torch.onnx.unregister_custom_op_symbolic(operator, OPSET)
