"""Fixed positive/negative luminance differences; no trainable RGB features."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


def mdd_coefficients(a: float, b: float) -> tuple[float, float]:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("MDD parameters must be finite")
    return 5.0 / (0.45 * abs(math.tanh(a)) + 1e-6), 0.6 * math.tanh(b)


def luminance_to_mdd(luminance: Tensor, *, gain: float, offset: float) -> Tensor:
    """B,T,H,W -> B,2,T,H,W. The first frame has no invented predecessor."""
    if luminance.ndim != 4 or luminance.shape[1] < 1:
        raise ValueError("Expected nonempty B,T,H,W luminance")
    diff = luminance[:, 1:] - luminance[:, :-1]
    output = luminance.new_zeros((len(luminance), 2, *luminance.shape[1:]))
    for channel, delta in enumerate((diff, -diff)):
        output[:, channel, 1:] = ((delta.clamp_min(0) - offset) * gain).clamp(-80, 80).sigmoid()
    return output


class RGBToMDD(nn.Module):
    """RGB uint8 B,T,3,H,W -> FP32 MDD B,2,T,H,W on the input device.

    The complete transform is part of the model forward/compile graph. AMP must
    not round the RGB/luminance differences before their fixed sigmoid mapping.
    Coefficients are immutable configuration, saved in the v3 input contract.
    """

    def __init__(self, a: float = .2, b: float = .15) -> None:
        super().__init__()
        if type(a) not in (float, int) or type(b) not in (float, int):
            raise ValueError("MDD coefficients must be fixed numeric configuration")
        self.a, self.b = a, b
        self.gain, self.offset = mdd_coefficients(a, b)

    def input_contract(self) -> dict[str, Any]:
        return dict(schema="rgb_uint8_native_mdd.v1", layout="B,T,3,H,W", color_order="RGB",
                    dtype="uint8", value_range=[0, 255], computation_dtype="float32",
                    normalization="divide_255", luminance="0.114*B + 0.587*G + 0.299*R",
                    first_mdd_frame="zero", mdd_a=self.a, mdd_b=self.b)

    @classmethod
    def from_contract(cls, contract: dict[str, Any]) -> RGBToMDD:
        if contract.get("schema") != "rgb_uint8_native_mdd.v1":
            raise ValueError("Expected an explicit native RGB/MDD input contract")
        model = cls(a=contract["mdd_a"], b=contract["mdd_b"])
        if contract != model.input_contract():
            raise ValueError("Unsupported RGB order, dtype, normalization or MDD contract")
        return model

    def forward(self, rgb: Tensor) -> Tensor:
        # Metadata guards are traceable; uint8 itself guarantees finite [0,255].
        if rgb.dtype != torch.uint8 or rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("Native MDD requires RGB uint8 B,T,3,H,W")
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            pixels = rgb.to(torch.float32) / 255
            # Preserve the old BGR reader's arithmetic order, now with RGB indices.
            gray = .114 * pixels[:, :, 2] + .587 * pixels[:, :, 1] + .299 * pixels[:, :, 0]
            return luminance_to_mdd(gray, gain=self.gain, offset=self.offset)
