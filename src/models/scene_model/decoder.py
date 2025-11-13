"""Recurrent temporal deformable decoder."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util as importlib_util
from importlib.abc import Loader
from types import ModuleType
from typing import cast

import torch
from torch import Tensor, nn

try:
    import MultiScaleDeformableAttention as _MSDA
except ImportError:  # pragma: no cover - handled by runtime checks
    _MSDA = None

from .positional import AbsTimePE, RelTimePE, fixed_2d_sincos


def _load_ms_deform_attn_cls() -> type[nn.Module]:
    if _MSDA is None:  # pragma: no cover - runtime dependency
        msg = (
            "MultiScaleDeformableAttention extension is not available. "
            "Ensure CUDA kernels are built before using the decoder."
        )
        raise RuntimeError(msg)
    try:
        import sys
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(ROOT))
        from third_party.Deformable_DETR.models.ops.modules import ms_deform_attn

        msda_cls = getattr(ms_deform_attn, "MSDeformAttn", None)
        if msda_cls is None:
            msg = "MSDeformAttn class not found in third-party module"
            raise RuntimeError(msg)
        return cast(type[nn.Module], msda_cls)
    except ModuleNotFoundError as err:
        module_path = (
            Path(__file__).resolve().parents[3]
            / "third_party"
            / "Deformable-DETR"
            / "models"
            / "ops"
            / "modules"
            / "ms_deform_attn.py"
        )
        spec = importlib_util.spec_from_file_location(
            "third_party.deformable_detr.ms_deform_attn", module_path
        )
        if spec is None or spec.loader is None:
            msg = f"Failed to load MSDeformAttn module from {module_path}"
            raise RuntimeError(msg) from err
        module = importlib_util.module_from_spec(spec)
        assert isinstance(module, ModuleType)
        loader = spec.loader
        assert isinstance(loader, Loader)
        loader.exec_module(module)
        msda_cls = getattr(module, "MSDeformAttn", None)
        if msda_cls is None:
            msg = f"MSDeformAttn class not found in {module_path}"
            raise RuntimeError(msg) from err
        return cast(type[nn.Module], msda_cls)


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
        num_heads: int,
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
        self.num_heads = num_heads
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
        window_len = 2 * k + 1
        self.layers = nn.ModuleList(
            [
                TemporalDeformableDecoderLayer(
                    dim=dim,
                    num_points=num_points,
                    num_heads=num_heads,
                    window_len=window_len,
                    offset_mode=offset_mode,
                )
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
            value_flat, pad_mask, spatial_shapes, level_start_index, delta_sec = (
                self.build_kv(patch_tokens, t)
            )
            q = q0
            for layer in self.layers:
                q = layer(
                    q,
                    value_flat,
                    pad_mask,
                    spatial_shapes,
                    level_start_index,
                    self.rel_pe,
                    self.rel_proj,
                    delta_sec,
                )
            outs.append(q.unsqueeze(1))
            prev = q
        return torch.cat(outs, dim=1)

    def build_kv(
        self, patch_tokens: Tensor, t_center: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build flatten values and masks for deformable attention."""
        if _MSDA is None:  # pragma: no cover - runtime dependency
            msg = (
                "MultiScaleDeformableAttention extension is not available. "
                "Ensure CUDA kernels are built before using the decoder."
            )
            raise RuntimeError(msg)

        B, T, Np, D = patch_tokens.shape
        device = patch_tokens.device
        dtype = patch_tokens.dtype
        idxs = list(range(t_center - self.k, t_center + self.k + 1))
        window_len = len(idxs)

        pos2d = cast(Tensor, self.pos2d).to(device=device, dtype=dtype)

        feats = []
        masks = []
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

        feats_cat = torch.stack(feats, dim=1)  # (B, W, H, W, D)
        value_flat = feats_cat.view(B, window_len * self.grid_h * self.grid_w, D)
        mask_cat = torch.stack(masks, dim=1)
        pad_mask_flat = mask_cat.view(B, window_len * self.grid_h * self.grid_w)

        spatial_shapes = torch.tensor(
            [[self.grid_h, self.grid_w] for _ in range(window_len)],
            device=device,
            dtype=torch.long,
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                (spatial_shapes.prod(1).cumsum(0)[:-1]),
            )
        )

        delta_vals = torch.tensor(
            [(tau - t_center) / self.fps for tau in idxs],
            device=device,
            dtype=dtype,
        )
        delta_sec = delta_vals.view(1, 1, -1).expand(B, self.Q, -1)

        return value_flat, pad_mask_flat, spatial_shapes, level_start_index, delta_sec


@dataclass
class _TemporalMSDAInputs:
    value: Tensor
    padding_mask: Tensor
    spatial_shapes: Tensor
    level_start_index: Tensor


class TemporalMSDeformAttn(nn.Module):
    """Wrapper around MultiScaleDeformableAttention for temporal windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
    ) -> None:
        super().__init__()
        cls = _load_ms_deform_attn_cls()
        self.ms_attn = cls(
            d_model=dim,
            n_levels=num_levels,
            n_heads=num_heads,
            n_points=num_points,
        )

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        inputs: _TemporalMSDAInputs,
    ) -> Tensor:
        """Apply deformable attention to the query sequence."""
        return cast(
            Tensor,
            self.ms_attn(
                query,
                reference_points,
                inputs.value,
                inputs.spatial_shapes,
                inputs.level_start_index,
                inputs.padding_mask,
            ),
        )


class TemporalDeformableDecoderLayer(nn.Module):
    """Single deformable attention layer operating on temporal windows."""

    def __init__(
        self,
        dim: int,
        num_points: int,
        num_heads: int,
        window_len: int,
        offset_mode: str = "per_tau",
    ) -> None:
        super().__init__()
        self.mode = offset_mode
        self.window_len = window_len
        self.rel_bias_proj = nn.Linear(dim, dim)
        self.ref_xy = nn.Linear(dim, 2)
        self.temporal_attn = TemporalMSDeformAttn(
            dim=dim,
            num_heads=num_heads,
            num_levels=window_len,
            num_points=num_points,
        )
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
        value_flat: Tensor,
        padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        rel_pe: nn.Module,
        rel_proj: nn.Module,
        delta_sec: Tensor,
    ) -> Tensor:
        """Apply temporal deformable attention and a feed-forward update."""
        B, Q, D = q.shape
        qn = self.norm1(q)

        rel_feat = rel_proj(rel_pe(delta_sec))  # (B, Q, window_len, D)
        if self.mode != "per_tau":
            rel_feat = rel_feat.mean(dim=2, keepdim=True)
        rel_bias = rel_feat.mean(dim=2)
        query = qn + self.rel_bias_proj(rel_bias)

        ref = torch.sigmoid(self.ref_xy(qn)).unsqueeze(2)
        reference_points = ref.expand(-1, -1, self.window_len, -1)

        attn_inputs = _TemporalMSDAInputs(
            value=value_flat,
            padding_mask=padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        attn_out = self.temporal_attn(query, reference_points, attn_inputs)
        proj = cast(Tensor, self.proj(attn_out))
        out = q + proj
        ffn_out = cast(Tensor, self.ffn(self.norm2(out)))
        return out + ffn_out
