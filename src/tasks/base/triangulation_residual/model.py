"""Permutation-equivariant camera attention and temporal RoPE residual models."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.base.triangulation_residual.configuration import ModelConfig
from src.tasks.base.triangulation_residual.contracts import feature_dimension
from src.utils.models.components.block import TransformerBlock, TransformerBlockConfig
from src.utils.models.components.rope import RotaryFrequencyComputer


class GeometricResidualModel(nn.Module):
    """Explicit profile: PLCS hip-root + relative pose; BLCS ball XYZ only."""

    def __init__(self, task: str, config: ModelConfig) -> None:
        super().__init__()
        if task not in ("plcs", "blcs"):
            raise ValueError("Unknown geometric residual task")
        self.task = task
        self.joints = 17 if task == "plcs" else 1
        self.root_indices = (11, 12) if task == "plcs" else (0,)
        self.input_dim = feature_dimension(self.joints)
        dim, heads = config.hidden_dim, config.num_heads
        self.head_dim = dim // heads
        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        block = TransformerBlockConfig(
            dim=dim,
            n_heads=heads,
            ffn_dim=config.ffn_dim,
            head_dim=self.head_dim,
            rope_dim=self.head_dim,
            attn_dropout=config.dropout,
            attention_type="mha",
            n_kv_heads=None,
            rope_base=config.rope_base,
            ffn_type="swiglu",
        )
        self.camera_layers = nn.ModuleList(
            [TransformerBlock(block) for _ in range(config.num_layers)]
        )
        self.time_layers = nn.ModuleList(
            [TransformerBlock(block) for _ in range(config.num_layers)]
        )
        self.time_rope = RotaryFrequencyComputer(
            dim=self.head_dim, base=config.rope_base, n_axes=1
        )
        self.pool = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.root_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3)
        )
        self.relative_head = (
            nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 51))
            if task == "plcs"
            else None
        )
        # The untrained model is exactly the explicit geometric seed.
        for head in (self.root_head, self.relative_head):
            if head is not None:
                final = head[-1]
                if not isinstance(final, nn.Linear):
                    raise TypeError("Residual head must terminate in a linear layer")
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def forward(
        self, features: Tensor, view_valid: Tensor, time_positions: Tensor
    ) -> dict[str, Tensor]:
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError("Expected features [B,V,T,F] for this residual profile")
        batch, views, frames, _ = features.shape
        if (
            view_valid.shape != features.shape[:3]
            or view_valid.dtype != torch.bool
            or time_positions.shape != (batch, frames)
        ):
            raise ValueError("Invalid residual view/time contract")
        if (
            features.device != view_valid.device
            or features.device != time_positions.device
        ):
            raise ValueError("Inputs must share a device")
        x = self.embed(features)
        camera_valid = view_valid.permute(0, 2, 1).reshape(batch * frames, views)
        camera_mask = camera_valid[:, None, :].expand(-1, views, -1)
        temporal_valid = view_valid.reshape(batch * views, frames)
        time_mask = temporal_valid[:, None, :].expand(-1, frames, -1)
        # Camera identity is expressed by K/R/C, not an arbitrary camera index.
        camera_freq = torch.ones(
            (1, views, 1, self.head_dim // 2),
            device=features.device,
            dtype=torch.complex64,
        )
        time_freq = self.time_rope(time_positions.float()[..., None])
        time_freq = (
            time_freq[:, None]
            .expand(batch, views, frames, 1, self.head_dim // 2)
            .reshape(batch * views, frames, 1, self.head_dim // 2)
        )
        for camera_layer, time_layer in zip(
            self.camera_layers, self.time_layers, strict=True
        ):
            y = x.permute(0, 2, 1, 3).reshape(batch * frames, views, -1)
            y = camera_layer(y, freqs_cis=camera_freq, attn_mask=camera_mask)
            x = y.reshape(batch, frames, views, -1).permute(0, 2, 1, 3)
            x = time_layer(
                x.reshape(batch * views, frames, -1),
                freqs_cis=time_freq,
                attn_mask=time_mask,
            ).reshape(batch, views, frames, -1)
        logits = self.pool(x).squeeze(-1).float().masked_fill(~view_valid, -1e4)
        weights = logits.softmax(dim=1) * view_valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = self.norm((x * weights[..., None]).sum(dim=1))
        root_delta = self.root_head(pooled)
        if self.relative_head is None:
            return {"position_residual": root_delta}
        relative_delta = self.relative_head(pooled).reshape(batch, frames, 17, 3)
        relative_delta = relative_delta - relative_delta[:, :, [11, 12]].mean(
            dim=2, keepdim=True
        )
        return {"root_residual": root_delta, "relative_residual": relative_delta}


def reconstruct_world(
    output: dict[str, Tensor], root_init: Tensor, relative_init: Tensor, *, task: str
) -> tuple[Tensor, Tensor, Tensor]:
    expected = (
        {"root_residual", "relative_residual"}
        if task == "plcs"
        else {"position_residual"}
    )
    if set(output) != expected:
        raise ValueError("Residual output keys do not match task")
    delta = output["root_residual" if task == "plcs" else "position_residual"]
    if (
        delta.shape != root_init.shape
        or relative_init.shape[:-2] != root_init.shape[:-1]
    ):
        raise ValueError("Residual root/time shapes do not match initializer")
    root = root_init.float() + delta.float()
    relative = relative_init.float()
    if task == "plcs":
        if output["relative_residual"].shape != relative.shape:
            raise ValueError("Relative pose residual shape mismatch")
        relative = relative + output["relative_residual"].float()
    return root[:, :, None] + relative, root, relative
