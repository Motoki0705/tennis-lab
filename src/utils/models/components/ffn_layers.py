from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def default_ffn_dim(hidden_dim: int) -> int:
    """Default FFN width shared across transformer-style blocks."""
    ffn_dim = int((8 * hidden_dim) / 3)
    return (ffn_dim + 63) // 64 * 64


class SwiGLU(nn.Module):
    """
    Standard SwiGLU FFN:
        y = W2( silu(W1 x) * (W3 x) )
    """

    def __init__(self, dim: int, ffn_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, ffn_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    """Standard 2-layer GELU MLP FFN."""

    def __init__(self, dim: int, ffn_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_input = torch.randn(2, 3, 16, device=demo_device)

    demo_swiglu = SwiGLU(dim=16, ffn_dim=32).eval().to(demo_device)
    demo_mlp = MLP(dim=16, ffn_dim=32).eval().to(demo_device)

    with torch.no_grad():
        print(demo_swiglu(demo_input))
        print(demo_mlp(demo_input))
