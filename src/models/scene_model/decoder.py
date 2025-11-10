"""Recurrent temporal deformable decoder."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .positional import AbsTimePE, RelTimePE, fixed_2d_sincos


class QueryMergeLayer(nn.Module):
    """Softmax convex combination between previous and learned queries."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_prev = nn.Linear(dim, 1)
        self.w_init = nn.Linear(dim, 1)
        self.ctx_proj = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self, q_prev: Tensor, q_init: Tensor, ctx: Tensor, e_abs: Tensor
    ) -> Tensor:
        """Blend recurrent state with the learned query anchor."""
        attn_prev = self.w_prev(q_prev)
        attn_init = self.w_init(q_init)
        w = torch.softmax(torch.cat([attn_prev, attn_init], dim=-1), dim=-1)
        base = w[..., 0:1] * q_prev + w[..., 1:2] * q_init
        ctx_bias = self.ctx_proj(ctx).unsqueeze(1)
        time_bias = self.time_proj(e_abs).unsqueeze(1)
        return cast(Tensor, self.out_norm(base + ctx_bias + time_bias))


class RecurrentTemporalDeformableDecoder(nn.Module):
    """Decoder composed of recurrent query updates and deformable attention."""

    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_layers: int,
        num_points: int,
        k: int,
        offset_mode: str,
        tbptt_detach: bool,
        abs_pe: AbsTimePE,
        abs_proj: nn.Linear,
        rel_pe: RelTimePE,
        rel_proj: nn.Linear,
        fps: float,
        img_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.D = dim
        self.Q = num_queries
        self.L = num_layers
        self.M = num_points
        self.k = k
        self.offset_mode = offset_mode
        self.tbptt = tbptt_detach
        self.abs_pe: AbsTimePE = abs_pe
        self.abs_proj: nn.Linear = abs_proj
        self.rel_pe: RelTimePE = rel_pe
        self.rel_proj: nn.Linear = rel_proj
        self.fps = fps
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.register_buffer(
            "pos2d",
            fixed_2d_sincos(self.grid_h, self.grid_w, dim),
            persistent=False,
        )
        self.init_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.ctx_proj = nn.Linear(dim, dim)
        self.merge = QueryMergeLayer(dim)
        self.layers = nn.ModuleList(
            [
                TemporalDeformableDecoderLayer(dim, num_points, offset_mode)
                for _ in range(num_layers)
            ]
        )

    def forward(self, patch_tokens: Tensor, temporal_tokens: Tensor) -> Tensor:
        """Decode temporal tokens into query trajectories."""
        B, T, Np, D = patch_tokens.shape
        if Np != self.grid_h * self.grid_w:
            msg = "Number of patches mismatch"
            raise ValueError(msg)
        prev = self.init_queries.unsqueeze(0).expand(B, self.Q, D)
        outs = []
        for t in range(T):
            if self.tbptt:
                prev = prev.detach()
            ctx = self.ctx_proj(temporal_tokens[:, t])
            t_tensor = torch.full(
                (B,),
                float(t) / self.fps,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            abs_vec = self.abs_proj(self.abs_pe(t_tensor))
            q_init = self.init_queries.unsqueeze(0).expand_as(prev)
            q0 = self.merge(prev, q_init, ctx, abs_vec)
            kv_feats, kv_mask, delta_sec = self.build_kv(patch_tokens, t)
            q = q0
            for layer in self.layers:
                q = layer(q, kv_feats, kv_mask, self.rel_pe, self.rel_proj, delta_sec)
            outs.append(q.unsqueeze(1))
            prev = q
        return torch.cat(outs, dim=1)

    def build_kv(
        self, patch_tokens: Tensor, t_center: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Assemble the deformable window around a reference timestep."""
        B, T, Np, D = patch_tokens.shape
        device = patch_tokens.device
        dtype = patch_tokens.dtype
        idxs = list(range(t_center - self.k, t_center + self.k + 1))
        feats = []
        masks = []
        pos2d = cast(Tensor, self.pos2d).to(
            device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        for tau in idxs:
            if 0 <= tau < T:
                base = patch_tokens[:, tau] + pos2d
                tau_tensor = torch.full(
                    (B,), tau / self.fps, device=device, dtype=dtype
                )
                abs_vec = self.abs_proj(self.abs_pe(tau_tensor))
                base = base + abs_vec.view(B, 1, D)
                feats.append(base.view(B, self.grid_h, self.grid_w, D))
                masks.append(
                    torch.zeros(
                        B, self.grid_h, self.grid_w, dtype=torch.bool, device=device
                    )
                )
            else:
                feats.append(
                    torch.zeros(
                        B, self.grid_h, self.grid_w, D, device=device, dtype=dtype
                    )
                )
                masks.append(
                    torch.ones(
                        B, self.grid_h, self.grid_w, dtype=torch.bool, device=device
                    )
                )
        kv_feats = torch.stack(feats, dim=1)
        kv_mask = torch.stack(masks, dim=1)
        delta_vals = torch.tensor(
            [(tau - t_center) / self.fps for tau in idxs],
            device=device,
            dtype=dtype,
        )
        delta_sec = delta_vals.view(1, 1, -1).expand(B, self.Q, -1)
        return kv_feats, kv_mask, delta_sec


class TemporalDeformableDecoderLayer(nn.Module):
    """Single deformable attention layer operating on temporal windows."""

    def __init__(self, dim: int, num_points: int, offset_mode: str = "per_tau") -> None:
        super().__init__()
        self.mode = offset_mode
        self.M = num_points
        self.ref_xy = nn.Linear(dim, 2)
        self.delta_mlp = nn.Linear(dim * 2, 3 * num_points)
        self.alpha_mlp = nn.Linear(dim * 2, num_points)
        self.lam = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        q: Tensor,
        kv_feats: Tensor,
        kv_mask: Tensor,
        rel_pe: nn.Module,
        rel_proj: nn.Module,
        delta_sec: Tensor,
    ) -> Tensor:
        """Apply temporal deformable sampling and a feed-forward update."""
        B, Q, D = q.shape
        window_len = kv_feats.shape[1]
        qn = self.norm1(q)
        rel_feat = rel_proj(rel_pe(delta_sec))
        if self.mode == "per_tau":
            psi = torch.cat(
                [qn.unsqueeze(2).expand(-1, -1, window_len, -1), rel_feat], dim=-1
            )
        else:
            pooled = rel_feat.mean(dim=2, keepdim=True)
            psi = torch.cat([qn.unsqueeze(2), pooled], dim=-1).expand(
                -1, -1, window_len, -1
            )
        delta = self.delta_mlp(psi).view(B, Q, window_len, self.M, 3)
        logits = self.alpha_mlp(psi).view(B, Q, window_len, self.M)
        ref = torch.sigmoid(self.ref_xy(qn)).unsqueeze(2).expand(-1, -1, window_len, -1)
        coords = ref.unsqueeze(-2) + torch.tanh(delta[..., :2])
        coords = coords.clamp(0.0, 1.0)
        delta_t = torch.tanh(delta[..., 2])
        sampled = sample_trilinear(kv_feats, kv_mask, coords, delta_t)
        lam = F.softplus(self.lam)
        weights = torch.softmax(
            logits - lam * delta_sec.abs().unsqueeze(-1),
            dim=-1,
        )
        agg = (weights.unsqueeze(-1) * sampled).sum(dim=-2).sum(dim=2)
        proj = cast(Tensor, self.proj(agg))
        out = q + proj
        ffn_out = cast(Tensor, self.ffn(self.norm2(out)))
        return out + ffn_out


def sample_trilinear(
    kv_feats: Tensor,
    kv_mask: Tensor,
    coords: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """Sample spatial features with bilinear + temporal linear interpolation."""
    B, window_len, H, W, D = kv_feats.shape
    device = kv_feats.device
    dtype = kv_feats.dtype
    base = torch.arange(window_len, device=device, dtype=dtype).view(
        1, 1, window_len, 1
    )
    target = base + delta_t
    t0 = torch.floor(target).clamp(0, window_len - 1)
    t1 = (t0 + 1).clamp(0, window_len - 1)
    alpha = (target - t0).unsqueeze(-1)
    t0 = t0.long()
    t1 = t1.long()
    sample0 = _sample_spatial(kv_feats, kv_mask, coords, t0)
    sample1 = _sample_spatial(kv_feats, kv_mask, coords, t1)
    return sample0 * (1 - alpha) + sample1 * alpha


def _sample_spatial(
    kv_feats: Tensor,
    kv_mask: Tensor,
    coords: Tensor,
    frame_idx: Tensor,
) -> Tensor:
    B, window_len, H, W, D = kv_feats.shape
    Q = coords.shape[1]
    M = coords.shape[3]
    device = kv_feats.device
    coords_norm = coords * 2 - 1
    flat_coords = coords_norm.view(-1, 1, 1, 2)
    flat_feats = (
        kv_feats.permute(0, 1, 4, 2, 3).contiguous().view(B * window_len, D, H, W)
    )
    mask = (~kv_mask).float().unsqueeze(-1)
    flat_mask = mask.permute(0, 1, 4, 2, 3).contiguous().view(B * window_len, 1, H, W)
    flat_feats = flat_feats * flat_mask
    batch_offsets = torch.arange(B, device=device).view(B, 1, 1, 1) * window_len
    flat_indices = (frame_idx + batch_offsets).view(-1)
    selected = flat_feats.index_select(0, flat_indices)
    grid = flat_coords
    sampled = F.grid_sample(
        selected,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.view(B, Q, window_len, M, D)
    return sampled
