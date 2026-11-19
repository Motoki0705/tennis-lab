"""Official Deformable DETR MSDA with explicit CUDA/FP32 execution."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch
from torch import Tensor, nn

from src.utils.models.components.ops.msda.vendor.modules.ms_deform_attn import (
    MSDeformAttn,
)

FunctionT = TypeVar("FunctionT", bound=Callable[..., Any])


def _without_compilation(function: FunctionT) -> FunctionT:
    return cast(FunctionT, torch.compiler.disable(function))


class OfficialMSDeformAttn(nn.Module):
    """Keep official projections, offsets and bilinear CUDA sampling unchanged.

    The entire official module executes in FP32 under disabled autocast.
    Construction and state_dict loading work on CPU; execution requires CUDA.
    No reference or CPU path is selected by this boundary.
    """

    def __init__(self, dim: int, levels: int, heads: int, points: int, *, im2col_step: int = 64) -> None:
        super().__init__()
        if type(heads) is not int or heads < 1 or dim % heads or dim // heads != 64:
            raise ValueError("MSDA requires head width 64")
        if any(type(x) is not int or x < 1 for x in (dim, levels, points, im2col_step)):
            raise ValueError("MSDA dimensions and im2col_step must be positive integers")
        self.op = MSDeformAttn(dim, levels, heads, points)
        self.op.im2col_step = im2col_step

    @_without_compilation
    def forward(self, query: Tensor, reference: Tensor, values: Tensor, shapes: Tensor,
                starts: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if query.device.type != "cuda" or values.device != query.device or reference.device != query.device:
            raise ValueError("Official MSDA execution requires CUDA on one device")
        if query.dtype not in (torch.float32, torch.bfloat16) or values.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("MSDA boundary accepts FP32 or BF16, explicitly casts to FP32")
        if shapes.dtype != torch.int64 or starts.dtype != torch.int64 or shapes.device != query.device or starts.device != query.device:
            raise ValueError("MSDA shapes/starts must be CUDA int64")
        if query.ndim != 3 or values.ndim != 3 or query.shape[0] != values.shape[0] or query.shape[-1] != self.op.d_model or values.shape[-1] != self.op.d_model:
            raise ValueError("MSDA requires batch, queries/tokens, dim")
        if shapes.shape != (self.op.n_levels, 2) or starts.shape != (self.op.n_levels,):
            raise ValueError("MSDA spatial_shapes require levels,(H,W)")
        expected_starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
        if bool((shapes < 1).any()) or int(shapes.prod(1).sum()) != values.shape[1] or not torch.equal(starts, expected_starts):
            raise ValueError("MSDA starts/token count do not match spatial_shapes")
        if reference.shape != (query.shape[0], query.shape[1], self.op.n_levels, 2):
            raise ValueError("MSDA reference points require batch,queries,levels,(x,y)")
        if padding_mask is not None and (padding_mask.dtype != torch.bool or padding_mask.shape != values.shape[:2] or padding_mask.device != query.device):
            raise ValueError("MSDA padding_mask must be batch,tokens boolean on CUDA")
        step = min(query.shape[0], self.op.im2col_step)
        if query.shape[0] % step:
            raise ValueError("MSDA batch must be divisible by min(batch,im2col_step)")
        with torch.autocast("cuda", enabled=False):
            return self.op(query.float().contiguous(), reference.float().contiguous(),
                           values.float().contiguous(), shapes.contiguous(), starts.contiguous(), padding_mask)
