"""DINOv3 DETR-style heatmap model for patch-token inputs with DETR-segm-like head.

Key idea (DETRsegm-style):
- Build a query-conditioned 2D attention map over the patch grid (like MHAttentionMap).
- Convert tokens to a 2D feature map [B*T, C, Hg, Wg].
- Produce heatmap logits using a small conv head with progressive upsampling and skip fusion
  (FPN-like, but using token_map (ViT) as the skip source).

Inputs:
- tokens: [B, T, N, C]  (or images if use_backbone=True)
Outputs:
- heatmap logits: [B, frames_out, H_out, W_out]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16

LOGGER = logging.getLogger(__name__)


def _safe_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm that always divides channels."""
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGNReLU(nn.Module):
    """Conv2d -> GroupNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad)
        self.gn = _safe_gn(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.conv.weight, a=1.0)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention (no self-attn) + FFN block (Pre-LN)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Args:
        q:  [B, Lq, C] (here Lq=1)
        kv: [B, Lm, C]

        Returns:
        q:  [B, Lq, C]
        """
        q_ln = self.norm_q(q)
        kv_ln = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_ln, kv_ln, kv_ln, need_weights=False)
        q = q + self.drop_attn(attn_out)

        ffn_out = self.ffn(self.norm_ffn(q))
        q = q + self.drop_ffn(ffn_out)
        return q


class TokenMHAttentionMap(nn.Module):
    """2D attention map over a token grid (DETRsegm's MHAttentionMap analogue).

    Returns *attention weights* (no value multiplication).

    Input:
      q: [B, Lq, C]  (typically Lq=1)
      k_map: [B, C, H, W]
      mask: [B, H, W] where True indicates padding/invalid (optional)

    Output:
      weights: [B, Lq, num_heads, H, W]
    """

    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)

        self.normalize_fact = float(hidden_dim // num_heads) ** -0.5

    def forward(self, q: Tensor, k_map: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        bt, lq, c = q.shape
        if k_map.dim() != 4:
            raise ValueError(f"k_map must be [B,C,H,W], got {tuple(k_map.shape)}")
        if k_map.shape[0] != bt:
            raise ValueError(f"Batch mismatch: q.B={bt}, k_map.B={k_map.shape[0]}")
        if k_map.shape[1] != c:
            # We assume q dim matches k_map channels; project outside if needed.
            raise ValueError(f"Channel mismatch: q.C={c}, k_map.C={k_map.shape[1]}")

        q_proj = self.q_linear(q)  # [B, Lq, hidden]
        # Apply k_linear as a 1x1 conv over the 2D map
        k_proj = F.conv2d(
            k_map,
            self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
            self.k_linear.bias,
        )  # [B, hidden, H, W]

        head_dim = self.hidden_dim // self.num_heads
        qh = q_proj.view(bt, lq, self.num_heads, head_dim)  # [B, Lq, Hh, Dh]
        kh = k_proj.view(bt, self.num_heads, head_dim, k_proj.shape[-2], k_proj.shape[-1])  # [B,Hh,Dh,H,W]

        weights = torch.einsum("bqhd,bhdxy->bqhxy", qh * self.normalize_fact, kh)  # [B,Lq,Hh,H,W]

        if mask is not None:
            if mask.shape != (bt, k_proj.shape[-2], k_proj.shape[-1]):
                raise ValueError(f"mask must be [B,H,W]={bt,k_proj.shape[-2],k_proj.shape[-1]}, got {tuple(mask.shape)}")
            weights = weights.masked_fill(mask[:, None, None, :, :], float("-inf"))

        # Softmax over spatial per head (more intuitive for heatmaps than flattening head+space together)
        w = weights.flatten(-2)  # [B,Lq,Hh,H*W]
        w = F.softmax(w, dim=-1).view_as(weights)
        w = self.dropout(w)
        return w


def _auto_stage_hw(
    grid_hw: Tuple[int, int],
    out_hw: Tuple[int, int],
    max_stages: int = 4,
) -> List[Tuple[int, int]]:
    """Create progressive upsample sizes (roughly doubling) until reaching out_hw."""
    gh, gw = grid_hw
    oh, ow = out_hw
    sizes: List[Tuple[int, int]] = []

    cur_h, cur_w = gh, gw
    for _ in range(max_stages):
        if (cur_h, cur_w) == (oh, ow):
            break
        nxt_h = min(oh, cur_h * 2)
        nxt_w = min(ow, cur_w * 2)
        if (nxt_h, nxt_w) == (cur_h, cur_w):
            break
        sizes.append((nxt_h, nxt_w))
        cur_h, cur_w = nxt_h, nxt_w

    if sizes and sizes[-1] != (oh, ow):
        sizes.append((oh, ow))
    if not sizes and (gh, gw) != (oh, ow):
        sizes = [(oh, ow)]
    return sizes


class TokenFPNHeatmapHead(nn.Module):
    """DETRsegm-like small conv head with progressive upsampling + token-skip fusion.

    We do NOT require a CNN backbone FPN; instead we use token_map (ViT) as the skip source.
    This still helps because refinement happens in multi-channel feature space, not 1ch-only.

    Inputs:
      token_map: [B, C, Hg, Wg]          (context features)
      attn_map:  [B, Hh, Hg, Wg]         (query-conditioned attention per head)
    Output:
      logits:    [B, 1, H_out, W_out]
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        grid_hw: Tuple[int, int],
        out_hw: Tuple[int, int],
        dropout: float = 0.0,
        max_upsample_stages: int = 4,
        min_dim: int = 32,
    ) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.out_hw = out_hw

        # Progressive stage sizes excluding the starting grid (Hg,Wg)
        self.stage_hw = _auto_stage_hw(grid_hw, out_hw, max_stages=max_upsample_stages)

        in_dim = token_dim + num_heads

        # Channel schedule (roughly like DETRsegm's inter_dims)
        # Keep it simple: token_dim -> token_dim/2 -> token_dim/4 ...
        dims: List[int] = [token_dim]
        for i in range(1, 2 + len(self.stage_hw)):
            dims.append(max(min_dim, token_dim // (2**i)))

        self.lay1 = ConvGNReLU(in_dim, dims[0], k=3, dropout=dropout)
        self.lay2 = ConvGNReLU(dims[0], dims[1], k=3, dropout=dropout)

        # Adapters + conv blocks per upsample stage
        self.adapters = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()
        cur_dim = dims[1]

        for si, _hw in enumerate(self.stage_hw):
            # project token skip (interpolated token_map) to current channel dim
            self.adapters.append(nn.Conv2d(token_dim, cur_dim, kernel_size=1))
            nn.init.kaiming_uniform_(self.adapters[-1].weight, a=1.0)
            nn.init.constant_(self.adapters[-1].bias, 0.0)

            # refine after fusion
            nxt_dim = dims[min(2 + si, len(dims) - 1)]
            self.stage_blocks.append(
                nn.Sequential(
                    ConvGNReLU(cur_dim, nxt_dim, k=3, dropout=dropout),
                    ConvGNReLU(nxt_dim, nxt_dim, k=3, dropout=dropout),
                )
            )
            cur_dim = nxt_dim

        self.out_lay = nn.Conv2d(cur_dim, 1, kernel_size=3, padding=1)
        nn.init.kaiming_uniform_(self.out_lay.weight, a=1.0)
        nn.init.constant_(self.out_lay.bias, 0.0)

    def forward(self, token_map: Tensor, attn_map: Tensor) -> Tensor:
        if token_map.dim() != 4:
            raise ValueError(f"token_map must be [B,C,H,W], got {tuple(token_map.shape)}")
        if attn_map.dim() != 4:
            raise ValueError(f"attn_map must be [B,Hh,H,W], got {tuple(attn_map.shape)}")
        if token_map.shape[0] != attn_map.shape[0]:
            raise ValueError("Batch mismatch between token_map and attn_map")
        if token_map.shape[-2:] != attn_map.shape[-2:]:
            raise ValueError("Spatial mismatch between token_map and attn_map")

        x = torch.cat([token_map, attn_map], dim=1)  # [B, C+Hh, Hg, Wg]
        x = self.lay1(x)
        x = self.lay2(x)

        for adapter, block, hw in zip(self.adapters, self.stage_blocks, self.stage_hw, strict=True):
            # Skip from tokens (interpolate in feature space, then adapt channels)
            skip = F.interpolate(token_map, size=hw, mode="bilinear", align_corners=False)
            skip = adapter(skip)

            # Upsample current x, fuse, refine
            x = F.interpolate(x, size=hw, mode="nearest")
            x = x + skip
            x = block(x)

        logits = self.out_lay(x)  # [B,1,H_out,W_out] (if stage_hw ends at out_hw)
        if logits.shape[-2:] != self.out_hw:
            logits = F.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)
        return logits


class DinoV3DETRHeatmapFPN(nn.Module):
    """Temporal DETR-style heatmap model with DETRsegm-like (attention + FPN-ish) head."""

    def __init__(self, cfg: DictConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.config = cfg

        # Backbone (optional)
        self.use_backbone = bool(cfg.get("use_backbone", False))
        if self.use_backbone:
            checkpoint_path = cfg.get("backbone_checkpoint", None)
            self.backbone = get_dinov3_vits16(checkpoint_path=checkpoint_path)
        else:
            self.backbone = None

        # Embedding dim
        self.embed_dim = int(cfg.get("embed_dim", 384))
        if self.backbone is not None and hasattr(self.backbone, "embed_dim"):
            self.embed_dim = int(self.backbone.embed_dim)

        # Query token
        self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Transformer-ish layers
        self.num_layers = int(cfg.get("num_layers", 4))
        self.num_heads = int(cfg.get("num_heads", 6))
        self.ffn_dim = int(cfg.get("ffn_dim", self.embed_dim * 4))
        self.dropout = float(cfg.get("dropout", 0.1))

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.temporal_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ffn_dim,
                    dropout=self.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Time positional embedding
        self.max_frames = int(cfg.get("max_frames", 256))
        self.time_pos = nn.Parameter(torch.randn(1, self.max_frames, self.embed_dim) * 0.02)
        self.temporal_causal = bool(cfg.get("temporal_causal", False))

        # Whether to add time pos every layer (default: once at the beginning)
        self.time_pos_each_layer = bool(cfg.get("time_pos_each_layer", False))

        # Grid sizes
        self.patch_grid_hw = cfg.get("patch_grid_hw", None)
        if self.patch_grid_hw is None:
            raise ValueError("patch_grid_hw must be provided.")
        self.output_hw = cfg.get("heatmap_hw", None)
        if self.output_hw is None:
            self.output_hw = self.patch_grid_hw

        self.frames_out = int(cfg.get("frames_out", 1))

        grid_h, grid_w = int(self.patch_grid_hw[0]), int(self.patch_grid_hw[1])
        out_h, out_w = int(self.output_hw[0]), int(self.output_hw[1])

        # Token map projection (tokens -> 2D feature map)
        self.token_proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        nn.init.kaiming_uniform_(self.token_proj.weight, a=1.0)
        nn.init.constant_(self.token_proj.bias, 0.0)

        # Attention map (query -> token grid)
        self.bbox_attention = TokenMHAttentionMap(
            query_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,  # DETRsegm uses 0.0 here
        )

        # DETRsegm-like head (progressive upsample + skip fusion)
        self.mask_head = TokenFPNHeatmapHead(
            token_dim=self.embed_dim,
            num_heads=self.num_heads,
            grid_hw=(grid_h, grid_w),
            out_hw=(out_h, out_w),
            dropout=float(cfg.get("head_dropout", 0.0)),
            max_upsample_stages=int(cfg.get("max_upsample_stages", 4)),
            min_dim=int(cfg.get("head_min_dim", 32)),
        )

    def forward(self, frames: Tensor) -> Tensor:
        """Args:
        frames:
          - If backbone is None: tokens [B, T, N, C] (or [B, N, C])
          - If backbone exists: images [B, C, H, W] or [B, T, C, H, W]

        Returns:
          heatmap logits: [B, frames_out, H_out, W_out]
        """
        tokens = self._resolve_tokens(frames)
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B,T,N,C], got {tuple(tokens.shape)}")
        b, t, n, c = tokens.shape
        if c != self.embed_dim:
            raise ValueError(f"Token dim C={c} != embed_dim={self.embed_dim}")
        if t > self.max_frames:
            raise ValueError(f"T={t} exceeds max_frames={self.max_frames}")

        grid_h, grid_w = int(self.patch_grid_hw[0]), int(self.patch_grid_hw[1])
        if n != grid_h * grid_w:
            raise ValueError(f"N={n} must equal grid_h*grid_w={grid_h*grid_w} for view(H,W).")

        # [B*T, N, C]
        tokens_bt = tokens.reshape(b * t, n, c)

        # Query init [B*T,1,C]
        q_bt = self.query.expand(b * t, 1, c).contiguous()

        # Temporal causal mask (optional)
        temporal_mask: Optional[Tensor] = None
        if self.temporal_causal:
            temporal_mask = torch.triu(
                torch.ones(t, t, device=tokens.device, dtype=torch.bool),
                diagonal=1,
            )

        # Add time pos once (default) before the temporal stack
        if not self.time_pos_each_layer:
            q = q_bt.squeeze(1).reshape(b, t, c)  # [B,T,C]
            q = q + self.time_pos[:, :t, :]
            q_bt = q.reshape(b * t, 1, c)

        for cross_blk, temp_blk in zip(self.cross_blocks, self.temporal_blocks, strict=True):
            # (1) Cross-attn per-frame
            q_bt = cross_blk(q_bt, tokens_bt)  # [B*T,1,C]

            # (2) Temporal self-attn over time
            q = q_bt.squeeze(1).reshape(b, t, c)  # [B,T,C]
            if self.time_pos_each_layer:
                q = q + self.time_pos[:, :t, :]
            q = temp_blk(q, src_mask=temporal_mask)  # [B,T,C]
            q_bt = q.reshape(b * t, 1, c)

        # Final queries per frame: [B*T,1,C]
        q_final_bt = q_bt

        # Build token map [B*T,C,Hg,Wg]
        token_map = tokens_bt.transpose(1, 2).reshape(b * t, c, grid_h, grid_w)
        token_map = self.token_proj(token_map)

        # Attention weights (query -> token grid): [B*T,1,heads,Hg,Wg] -> [B*T,heads,Hg,Wg]
        attn = self.bbox_attention(q_final_bt, token_map, mask=None).squeeze(1)

        # Mask/heatmap head: [B*T,1,Hout,Wout]
        logits_bt = self.mask_head(token_map, attn)

        # [B,T,Hout,Wout]
        logits = logits_bt.reshape(b, t, logits_bt.shape[-2], logits_bt.shape[-1])
        return logits[:, -self.frames_out :, :, :]

    def _resolve_tokens(self, frames: Tensor) -> Tensor:
        """Resolve input to tokens [B, T, N, C]."""
        if self.backbone is None:
            # tokens path
            if frames.dim() == 3:
                return frames.unsqueeze(1)  # [B,1,N,C]
            return frames

        # backbone path (images -> tokens)
        if frames.dim() == 4:
            tokens = self._encode_frames(frames)  # [B,N,C]
            return tokens.unsqueeze(1)            # [B,1,N,C]

        if frames.dim() == 5:
            b, t, c, h, w = frames.shape
            frames_bt = frames.reshape(b * t, c, h, w)
            tokens_bt = self._encode_frames(frames_bt)  # [B*T,N,C]
            return tokens_bt.reshape(b, t, tokens_bt.shape[1], tokens_bt.shape[2])

        return frames

    def _encode_frames(self, frames: Tensor) -> Tensor:
        """Encode images to patch tokens [B, N, C] using DINOv3 backbone."""
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized for image input.")
        outputs = self.backbone.get_intermediate_layers(
            frames,
            n=1,
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        tokens = outputs[-1] if isinstance(outputs, (tuple, list)) else outputs
        if tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens [B,N,C], got {tuple(tokens.shape)}")
        return tokens

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: torch.device | str | None = "cpu",
    ) -> None:
        """Load pre-trained DINOv3 weights into the backbone."""
        _ = map_location
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized.")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        if checkpoint_path.is_dir():
            LOGGER.info("Backbone checkpoint path %s is a directory; skipping weight loading.", checkpoint_path)
            return

        loaded_backbone = get_dinov3_vits16(checkpoint_path=str(checkpoint_path))
        self.backbone.load_state_dict(loaded_backbone.state_dict(), strict=True)
        LOGGER.info("Loaded DINOv3 backbone parameters from %s", checkpoint_path)


    def freeze_backbone(self) -> None:
        """Disable gradient updates for the DINOv3 backbone."""
        self._backbone_train_mode = self.backbone.training
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True
        LOGGER.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Re-enable gradient updates for the DINOv3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        if self._backbone_train_mode is not None:
            if self._backbone_train_mode:
                self.backbone.train()
            self._backbone_train_mode = None
        self._backbone_frozen = False
        LOGGER.info("Backbone unfrozen")
