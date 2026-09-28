"""Validated conditional GMM on normalized image coordinates plus presence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.base.model_io.tensors import TensorSpec

FLOAT_DTYPES = frozenset({torch.float32, torch.float64})


def conditional_log_density(
    uv: Tensor, means: Tensor, scale_tril: Tensor, logits: Tensor
) -> Tensor:
    """Density kernel for validated tensors, including selected frame subsets."""
    delta = (uv.unsqueeze(-2) - means).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(scale_tril, delta, upper=False)
    mahalanobis = whitened.square().sum(dim=(-2, -1))
    log_det = scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    component_log_prob = -math.log(2 * math.pi) - log_det - 0.5 * mahalanobis
    return torch.logsumexp(
        torch.log_softmax(logits, dim=-1) + component_log_prob,
        dim=-1,
    )


@dataclass(frozen=True)
class BallGMM2D:
    """One ball's alternative locations, conditional on amodal presence.

    B,T,K are batch, source frame and mixture axes. Means use source-grid UV;
    the density lives on R² (not a truncated/renormalized image rectangle).
    A positive-diagonal Cholesky factor encodes full covariance. Logits retain
    stable training likelihoods even when exported probabilities saturate.
    """

    means: Tensor  # B,T,K,2
    scale_tril: Tensor  # B,T,K,2,2
    mixture_logits: Tensor  # B,T,K
    presence_logits: Tensor  # B,T

    def __post_init__(self) -> None:
        TensorSpec((None, None, None, 2), FLOAT_DTYPES).validate("means", self.means)
        b, t, k, _ = self.means.shape
        if min(b, t, k) <= 0:
            raise ValueError("GMM batch, time and component axes must be nonempty")
        for name, value, shape in (
            ("scale_tril", self.scale_tril, (b, t, k, 2, 2)),
            ("mixture_logits", self.mixture_logits, (b, t, k)),
            ("presence_logits", self.presence_logits, (b, t)),
        ):
            TensorSpec(shape, frozenset({self.means.dtype})).validate(name, value)
            if value.device != self.means.device:
                raise ValueError(f"{name} must share means' device")
        for value in (
            self.means,
            self.scale_tril,
            self.mixture_logits,
            self.presence_logits,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("GMM parameters must be finite")
        if bool(((self.means < 0) | (self.means > 1)).any()):
            raise ValueError("GMM means must use normalized image UV in [0,1]")
        if bool((self.scale_tril[..., 0, 1] != 0).any()):
            raise ValueError("scale_tril must be lower triangular")
        if bool((self.scale_tril.diagonal(dim1=-2, dim2=-1) <= 0).any()):
            raise ValueError("scale_tril must have a strictly positive diagonal")

    @property
    def covariance(self) -> Tensor:
        with torch.autocast(device_type=self.means.device.type, enabled=False):
            return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    @property
    def weights(self) -> Tensor:
        return torch.softmax(self.mixture_logits, dim=-1)

    @property
    def presence_probability(self) -> Tensor:
        return torch.sigmoid(self.presence_logits)

    def log_prob(self, uv: Tensor) -> Tensor:
        """Conditional log density in normalized UV², excluding presence."""
        TensorSpec(
            (*self.presence_logits.shape, 2),
            frozenset({self.means.dtype}),
        ).validate("uv", uv)
        if uv.device != self.means.device or not bool(torch.isfinite(uv).all()):
            raise ValueError("uv must be finite and on the GMM device")
        return conditional_log_density(
            uv, self.means, self.scale_tril, self.mixture_logits
        )

    def pixel_moments(self, source_size_wh: Tensor) -> tuple[Tensor, Tensor]:
        """Component means/covariances in source px/px²; weights are unchanged.

        Each batch row may have its own source image size. The Jacobian for
        pixel log density is log((W-1)*(H-1)), subtracted from UV log density.
        """
        b = self.means.shape[0]
        TensorSpec((b, 2), FLOAT_DTYPES).validate("source_size_wh", source_size_wh)
        if source_size_wh.device != self.means.device:
            raise ValueError("source_size_wh must share the GMM device")
        if not bool(torch.isfinite(source_size_wh).all()) or bool(
            (source_size_wh <= 1).any()
        ):
            raise ValueError("Source image dimensions must be finite and exceed 1")
        if bool((source_size_wh != source_size_wh.round()).any()):
            raise ValueError("Source image dimensions must be integers")
        scale = (source_size_wh.to(self.means.dtype) - 1)[:, None, None, :]
        return self.means * scale, self.covariance * scale.unsqueeze(
            -1
        ) * scale.unsqueeze(-2)
