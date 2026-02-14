"""Sequential PLCS model with per-timestep CLS tokens.

Token layout:
    [court(20), frame_0(2+17), frame_1(2+17), ..., frame_{T-1}(2+17)]
where each frame block is:
    [CLS_po_t, CLS_ro_t, player_kp_0..player_kp_16]

Total token length:
    S = NUM_COURT_KP + T * (2 + NUM_HUMAN_KP)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    MoEConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.common.models.embeddings import CourtKPUVEmbedding, InvisibleTokenEmbedding, PlayerKPUVEmbedding
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSSequenceModel(nn.Module):
    """Sequential PLCS model using decoder-style Transformer blocks."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.yarn = yarn

        self.frame_block_tokens = 2 + NUM_HUMAN_KP
        self.max_tokens = int(NUM_COURT_KP + self.max_seq_len * self.frame_block_tokens)

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")
        if moe_config is not None and moe_config.dim != hidden_dim:
            raise ValueError(f"moe_config.dim={moe_config.dim} must match hidden_dim={hidden_dim}")

        self.invisible_token = InvisibleTokenEmbedding(dim=hidden_dim, init_std=invisible_init_std)
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.player_embed = PlayerKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )

        self.cls_po_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_ro_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_po_token, std=0.02)
        nn.init.trunc_normal_(self.cls_ro_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        rope_base=self.rope_theta,
                        yarn=self.yarn,
                        use_moe=use_moe,
                        moe_config=moe_config,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)

        self.position_head = PositionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.rotation_head = RotationHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
        )

        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_tokens,
            base=self.rope_theta,
            yarn=self.yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSSequenceModel:
        """Create model from hydra config."""
        model_cfg = config.get("model", {})

        yarn_cfg = model_cfg.get("yarn", None)
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len", None) is not None:
                yarn = YaRNConfig(**yarn_cfg)

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_cfg = model_cfg.get("moe_config", None)
        moe_config: MoEConfig | None = None
        if use_moe and moe_cfg is not None:
            moe_config = MoEConfig(dim=int(model_cfg.get("hidden_dim", 256)), **dict(moe_cfg))

        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 8),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            yarn=yarn,
            use_moe=use_moe,
            moe_config=moe_config,
            max_seq_len=model_cfg.get("max_seq_len", 120),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def _normalize_court_inputs(
        self,
        court_kp: Tensor,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Normalize court tensors to scene-level shapes: (B,20,2)/(B,20)."""
        if court_kp.dim() == 4:
            court_kp = court_kp[:, 0, :, :]
        elif court_kp.dim() == 3:
            if court_kp.size(1) == NUM_COURT_KP and court_kp.size(2) == 2:
                pass
            elif court_kp.size(2) == NUM_COURT_KP * 2:
                court_kp = court_kp[:, 0, :]
            else:
                raise ValueError(
                    f"Unsupported court_kp shape {tuple(court_kp.shape)}. "
                    "Expected (B,40), (B,20,2), (B,T,40), or (B,T,20,2)."
                )

        if court_vis is not None and court_vis.dim() == 3:
            court_vis = court_vis[:, 0, :]

        return court_kp, court_vis

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp:
                Human 2D keypoints in normalized image UV.
                Shape: (B, T, 34) or (B, T, 17, 2).
            court_kp:
                Court 2D keypoints in normalized image UV.
                Supported shapes:
                - (B, 40), (B, 20, 2) as scene-level court keypoints
                - (B, T, 40), (B, T, 20, 2) as per-frame court keypoints
                For per-frame input, frame 0 is used as scene-level court.
            human_vis:
                Human keypoint visibility flags aligned with `human_kp`.
                Shape: (B, T, 17). Each element is interpreted as visible if > 0
                (bool/0-1). Optional; if None, all human keypoints are treated
                as visible.
            court_vis:
                Court keypoint visibility flags aligned with `court_kp`.
                Shape: (B, 20) or (B, T, 20). Each element is interpreted as
                visible if > 0. For per-frame input, frame 0 is used.
                Optional; if None, all court keypoints are treated as visible.
            human_mask:
                Padding mask. Supported shapes:
                - (B, N, T): unified mask (reduced to any camera)
                - (B, T): sequence validity
                Optional; if None, all frames are valid.

        Returns:
            dict with:
              - position: (B, T, 3) normalized court-space xyz per frame
              - rotation: (B, T, 2) per frame as (sin(yaw), cos(yaw))
        """
        if human_kp.dim() == 5:  # (B, N, T, 17, 2)
            human_kp = human_kp[:, 0]
        if human_vis is not None and human_vis.dim() == 4:  # (B, N, T, 17)
            human_vis = human_vis[:, 0]
        if court_kp.dim() == 5:  # (B, N, T, 20, 2)
            court_kp = court_kp[:, 0]
        if court_vis is not None and court_vis.dim() == 4:  # (B, N, T, 20)
            court_vis = court_vis[:, 0]

        seq_mask: Tensor | None = None
        if human_mask is not None:
            if human_mask.dim() == 3:  # (B, N, T)
                seq_mask = (human_mask > 0).any(dim=1)
            elif human_mask.dim() == 2:  # (B, T)
                seq_mask = human_mask > 0
            else:
                raise ValueError(
                    "human_mask for sequence models must be (B,N,T) or (B,T), "
                    f"got shape {tuple(human_mask.shape)}"
                )

        B, T = human_kp.shape[:2]

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds configured max_seq_len={self.max_seq_len}."
            )

        court_kp, court_vis = self._normalize_court_inputs(court_kp, court_vis)

        # court: (B, 20, D)
        court_tok = self.court_embed(court_kp, court_vis)

        # player: (B, T, 17, D)
        if human_kp.dim() == 3:
            human_kp = human_kp.view(B, T, NUM_HUMAN_KP, 2)
        human_kp_flat = human_kp.reshape(B * T, NUM_HUMAN_KP, 2)

        human_vis_flat: Tensor | None = None
        if human_vis is not None:
            human_vis_flat = human_vis.reshape(B * T, NUM_HUMAN_KP)

        player_tok = self.player_embed(human_kp_flat, human_vis_flat)
        player_tok = player_tok.view(B, T, NUM_HUMAN_KP, self.hidden_dim)

        cls_po = self.cls_po_token.expand(B, T, -1, -1)
        cls_ro = self.cls_ro_token.expand(B, T, -1, -1)
        frame_tokens = torch.cat([cls_po, cls_ro, player_tok], dim=2)  # (B,T,19,D)
        frame_tokens = frame_tokens.reshape(B, T * self.frame_block_tokens, self.hidden_dim)

        x = torch.cat([court_tok, frame_tokens], dim=1)
        S = x.size(1)

        if S > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Token length S={S} exceeds cached RoPE length {self.freqs_cis.shape[0]}. "
                "Increase max_seq_len."
            )

        freqs_cis = self.freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if seq_mask is not None:
            if seq_mask.dim() == 1:
                seq_mask = seq_mask.unsqueeze(0)
            if seq_mask.shape != (B, T):
                raise ValueError(
                    f"seq_mask must have shape {(B, T)}, got {tuple(seq_mask.shape)}"
                )
            seq_valid = seq_mask > 0
            court_valid = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            frame_valid = seq_valid.unsqueeze(-1).expand(B, T, self.frame_block_tokens)
            frame_valid = frame_valid.reshape(B, T * self.frame_block_tokens)
            token_valid = torch.cat([court_valid, frame_valid], dim=1)
            attn_mask = token_valid[:, None, :] & token_valid[:, :, None]

        residual = None
        for blk in self.blocks:
            x, residual = blk(
                x,
                residual,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=False,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)

        time_offsets = NUM_COURT_KP + torch.arange(
            T,
            device=x.device,
            dtype=torch.long,
        ) * self.frame_block_tokens
        po_idx = time_offsets
        ro_idx = time_offsets + 1

        po_feat = x.gather(1, po_idx.view(1, T, 1).expand(B, T, self.hidden_dim))
        ro_feat = x.gather(1, ro_idx.view(1, T, 1).expand(B, T, self.hidden_dim))

        po_flat = po_feat.reshape(B * T, self.hidden_dim)
        ro_flat = ro_feat.reshape(B * T, self.hidden_dim)

        position = self.position_head(po_flat).view(B, T, 3)
        rotation = self.rotation_head(ro_flat).view(B, T, 2)

        return {
            "position": position,
            "rotation": rotation,
        }

    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PLCSSequenceModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_seq_len=16,
    )

    B = 2
    T = 8
    human_kp = torch.randn(B, T, NUM_HUMAN_KP, 2)
    court_kp = torch.randn(B, NUM_COURT_KP, 2)
    human_vis = (torch.rand(B, T, NUM_HUMAN_KP) > 0.2).to(torch.float32)
    court_vis = (torch.rand(B, NUM_COURT_KP) > 0.1).to(torch.float32)

    with torch.no_grad():
        out = model(human_kp=human_kp, court_kp=court_kp, human_vis=human_vis, court_vis=court_vis)

    print("PLCSSequenceModel:")
    for key, value in out.items():
        print(f"  {key}: {tuple(value.shape)}")
