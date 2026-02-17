"""Temporal memory-attention detector for ball position/visibility prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.common.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    MSDeformCrossAttnBlock,
    MSDeformCrossAttnBlockConfig,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectorModel(nn.Module):
    """Ball detector with deformable frame-memory attention and space-time attention."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_dim: int = 256,
        base_channels: int = 64,
        num_scales: int = 3,
        num_heads: int = 8,
        pad_size: int = 8,
        max_spatial_tokens: int = 4096,
        num_frame_cross_layers: int = 2,
        num_spatiotemporal_layers: int = 2,
        num_query_cross_layers: int = 1,
        num_query_temporal_layers: int = 1,
        num_queries: int = 1,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        causal: bool = False,
        mlp_inter_dim: int | None = None,
        msda_num_points: int = 4,
        msda_use_cuda_kernel: bool = True,
        msda_allow_fallback: bool = True,
        msda_offset_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.num_queries = int(num_queries)
        self.max_seq_len = int(max_seq_len)
        self.num_scales = int(num_scales)
        self.causal = bool(causal)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.max_spatial_tokens <= 0:
            raise ValueError("max_spatial_tokens must be positive.")
        if self.num_scales <= 0:
            raise ValueError("num_scales must be positive.")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        head_dim = self.hidden_dim // self.num_heads
        mlp_inter_dim_value = (
            int(mlp_inter_dim) if mlp_inter_dim is not None else int((8 * self.hidden_dim) / 3)
        )

        self.input_pad = nn.ConstantPad2d(int(pad_size), 0.0)
        self.input_proj = nn.Sequential(
            nn.Conv2d(int(in_channels), int(base_channels), kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        channels = [int(base_channels)]
        self.downsample_blocks = nn.ModuleList()
        for _ in range(self.num_scales):
            in_ch = channels[-1]
            out_ch = in_ch * 2
            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
            )
            channels.append(out_ch)

        # Memory uses only two scales: stem (0) and middle depth (num_scales // 2).
        self.memory_scale_indices = sorted({0, self.num_scales // 2})
        self.num_memory_scales = len(self.memory_scale_indices)

        self.memory_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[idx], self.hidden_dim, kernel_size=1),
                    nn.GELU(),
                )
                for idx in self.memory_scale_indices
            ]
        )
        self.scale_type_embedding = nn.Embedding(self.num_memory_scales, self.hidden_dim)

        self.token_compressor = nn.Sequential(
            nn.Conv2d(channels[-1], self.hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.spatial_token_id_embedding = nn.Embedding(self.max_spatial_tokens, self.hidden_dim)

        self.frame_cross_layers = nn.ModuleList(
            [
                MSDeformCrossAttnBlock(
                    MSDeformCrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=self.num_heads,
                        n_levels=self.num_memory_scales,
                        n_points=int(msda_num_points),
                        mlp_inter_dim=mlp_inter_dim_value,
                        attn_dropout=float(dropout),
                        use_cuda_kernel=bool(msda_use_cuda_kernel),
                        allow_fallback=bool(msda_allow_fallback),
                        offset_scale=float(msda_offset_scale),
                    )
                )
                for _ in range(int(num_frame_cross_layers))
            ]
        )

        self.spatial_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=self.num_heads,
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=0,
                        attn_dropout=float(dropout),
                    )
                )
                for _ in range(int(num_spatiotemporal_layers))
            ]
        )
        self.temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=self.num_heads,
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=head_dim,
                        rope_base=float(rope_theta),
                        attn_dropout=float(dropout),
                    )
                )
                for _ in range(int(num_spatiotemporal_layers))
            ]
        )

        self.query_base = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_dim) * 0.02)
        self.query_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=self.num_heads,
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=0,
                        attn_dropout=float(dropout),
                    )
                )
                for _ in range(int(num_query_cross_layers))
            ]
        )
        self.query_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=self.num_heads,
                        mlp_inter_dim=mlp_inter_dim_value,
                        head_dim=head_dim,
                        rope_dim=head_dim,
                        rope_base=float(rope_theta),
                        attn_dropout=float(dropout),
                    )
                )
                for _ in range(int(num_query_temporal_layers))
            ]
        )

        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.xy_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
            nn.Sigmoid(),
        )
        self.vis_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            seqlen=self.max_seq_len,
            base=float(rope_theta),
            device=None,
        )
        self.register_buffer("temporal_freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(
            valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1]
        )
        return attn_mask, valid_fixed

    def _encode_backbone(self, x: Tensor) -> list[Tensor]:
        feats = []
        x = self.input_proj(self.input_pad(x))
        feats.append(x)
        for block in self.downsample_blocks:
            x = block(x)
            feats.append(x)
        return feats

    def _build_memory_levels(self, features: list[Tensor]) -> list[Tensor]:
        levels: list[Tensor] = []
        for scale_id, (feat_idx, proj) in enumerate(
            zip(self.memory_scale_indices, self.memory_projections, strict=True)
        ):
            feat = proj(features[feat_idx])
            feat = feat + self.scale_type_embedding.weight[scale_id].view(1, -1, 1, 1)
            levels.append(feat)
        return levels

    def _build_frame_tokens(self, deepest_feature: Tensor, *, bsz: int, seq_len: int) -> Tensor:
        token = self.token_compressor(deepest_feature)
        token = token.flatten(2).transpose(1, 2).contiguous()
        spatial_tokens = int(token.shape[1])
        if spatial_tokens > self.max_spatial_tokens:
            raise ValueError(
                f"Spatial token count {spatial_tokens} exceeds max_spatial_tokens={self.max_spatial_tokens}. "
                "Increase max_spatial_tokens or reduce input resolution/num_scales."
            )
        token = token.view(bsz, seq_len, spatial_tokens, self.hidden_dim)
        return token

    def _run_spatial_then_temporal(
        self,
        token_bt: Tensor,
        *,
        frame_valid: Tensor,
        freqs_cis_t: Tensor,
    ) -> Tensor:
        bsz, seq_len, spatial_tokens, _ = token_bt.shape
        token_ids = torch.arange(spatial_tokens, device=token_bt.device, dtype=torch.long)
        token_bt = token_bt + self.spatial_token_id_embedding(token_ids).view(1, 1, spatial_tokens, -1)

        temporal_valid = frame_valid[:, None, :].expand(bsz, spatial_tokens, seq_len)
        temporal_valid = temporal_valid.reshape(bsz * spatial_tokens, seq_len)
        temporal_mask, _ = self._build_self_attn_mask(temporal_valid)

        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers, strict=True):
            spatial_in = token_bt.reshape(bsz * seq_len, spatial_tokens, self.hidden_dim)
            spatial_out, _ = spatial_layer(
                spatial_in,
                residual=None,
                start_pos=0,
                freqs_cis=None,
                attn_mask=None,
                is_causal=False,
            )
            token_bt = spatial_out.view(bsz, seq_len, spatial_tokens, self.hidden_dim)

            temporal_in = token_bt.permute(0, 2, 1, 3).reshape(
                bsz * spatial_tokens,
                seq_len,
                self.hidden_dim,
            )
            temporal_out, _ = temporal_layer(
                temporal_in,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cis_t,
                attn_mask=temporal_mask,
                is_causal=self.causal,
            )
            token_bt = temporal_out.view(bsz, spatial_tokens, seq_len, self.hidden_dim).permute(0, 2, 1, 3)

        return token_bt

    def _run_query_decode(
        self,
        token_bt: Tensor,
        *,
        frame_valid: Tensor,
        freqs_cis_t: Tensor,
    ) -> Tensor:
        bsz, seq_len, spatial_tokens, _ = token_bt.shape

        token_btf = token_bt.reshape(bsz * seq_len, spatial_tokens, self.hidden_dim)
        query = self.query_base.expand(bsz * seq_len, -1, -1)
        for cross_layer in self.query_cross_layers:
            query = cross_layer(
                query,
                token_btf,
                key_valid=None,
                freqs_q_cis=None,
                freqs_k_cis=None,
            )

        query_btqd = query.view(bsz, seq_len, self.num_queries, self.hidden_dim)
        temporal_valid = frame_valid[:, None, :].expand(bsz, self.num_queries, seq_len)
        temporal_valid = temporal_valid.reshape(bsz * self.num_queries, seq_len)
        temporal_mask, _ = self._build_self_attn_mask(temporal_valid)

        query_temporal = query_btqd.permute(0, 2, 1, 3).reshape(
            bsz * self.num_queries,
            seq_len,
            self.hidden_dim,
        )
        for layer in self.query_temporal_layers:
            query_temporal, _ = layer(
                query_temporal,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cis_t,
                attn_mask=temporal_mask,
                is_causal=self.causal,
            )
        query_btqd = query_temporal.view(bsz, self.num_queries, seq_len, self.hidden_dim).permute(0, 2, 1, 3)
        return query_btqd

    def forward(self, frames: Tensor, frame_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            frames: (B, T, C, H, W) or (B, C, H, W).
            frame_mask: Optional (B, T) valid mask (1=valid).

        Returns:
            Dict with `xy` and `visibility_logit`.
        """
        squeeze_time = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)
            squeeze_time = True
        if frames.dim() != 5:
            raise ValueError(f"frames must be rank 5 or 4, got shape={tuple(frames.shape)}")

        bsz, seq_len, _, _, _ = frames.shape
        if seq_len > self.max_seq_len:
            frames = frames[:, : self.max_seq_len]
            if frame_mask is not None:
                frame_mask = frame_mask[:, : self.max_seq_len]
            seq_len = self.max_seq_len

        frame_valid = torch.ones(bsz, seq_len, dtype=torch.bool, device=frames.device)
        if frame_mask is not None:
            frame_valid = frame_mask > 0

        x = frames.reshape(bsz * seq_len, *frames.shape[2:])
        features = self._encode_backbone(x)

        memory_levels = self._build_memory_levels(features)
        token_bt = self._build_frame_tokens(features[-1], bsz=bsz, seq_len=seq_len)
        spatial_tokens = int(token_bt.shape[2])

        token_btf = token_bt.reshape(bsz * seq_len, spatial_tokens, self.hidden_dim)
        for layer in self.frame_cross_layers:
            token_btf = layer(token_btf, memory_levels)
        token_bt = token_btf.view(bsz, seq_len, spatial_tokens, self.hidden_dim)

        freqs_cis_t = self.temporal_freqs_cis[:seq_len]
        if freqs_cis_t.device != frames.device:
            freqs_cis_t = freqs_cis_t.to(frames.device)

        token_bt = self._run_spatial_then_temporal(
            token_bt,
            frame_valid=frame_valid,
            freqs_cis_t=freqs_cis_t,
        )
        query_btqd = self._run_query_decode(
            token_bt,
            frame_valid=frame_valid,
            freqs_cis_t=freqs_cis_t,
        )

        fused = query_btqd.mean(dim=2)
        fused = self.final_norm(fused)

        xy = self.xy_head(fused)
        vis_logit = self.vis_head(fused).squeeze(-1)

        if squeeze_time:
            xy = xy[:, 0]
            vis_logit = vis_logit[:, 0]

        return {
            "xy": xy,
            "visibility_logit": vis_logit,
        }

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> BallDetectorModel:
        """Build model from Hydra/OmegaConf-like config."""
        cfg = config or {}
        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else {}
        data_cfg = cfg.get("data", {}) if hasattr(cfg, "get") else {}

        return cls(
            in_channels=int(model_cfg.get("in_channels", 3)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            num_scales=int(model_cfg.get("num_scales", 3)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            pad_size=int(model_cfg.get("pad_size", 8)),
            max_spatial_tokens=int(model_cfg.get("max_spatial_tokens", 4096)),
            num_frame_cross_layers=int(model_cfg.get("num_frame_cross_layers", 2)),
            num_spatiotemporal_layers=int(model_cfg.get("num_spatiotemporal_layers", 2)),
            num_query_cross_layers=int(model_cfg.get("num_query_cross_layers", 1)),
            num_query_temporal_layers=int(model_cfg.get("num_query_temporal_layers", 1)),
            num_queries=int(model_cfg.get("num_queries", 1)),
            max_seq_len=int(model_cfg.get("max_seq_len", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            causal=bool(model_cfg.get("causal", False)),
            mlp_inter_dim=model_cfg.get("mlp_inter_dim"),
            msda_num_points=int(model_cfg.get("msda_num_points", 4)),
            msda_use_cuda_kernel=bool(model_cfg.get("msda_use_cuda_kernel", True)),
            msda_allow_fallback=bool(model_cfg.get("msda_allow_fallback", True)),
            msda_offset_scale=float(model_cfg.get("msda_offset_scale", 0.5)),
        )
