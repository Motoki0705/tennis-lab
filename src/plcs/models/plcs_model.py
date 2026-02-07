"""Main PLCS model implementation.

Player Localization in Court System: estimates player position and
rotation in tennis court coordinates from 2D pose observations.

Architecture:
    - Decoder-only Transformer with MHA (+ optional RoPE) + SDPA
    - Court keypoints (20) are tokenized as individual tokens
    - Player keypoints (17) are tokenized as individual tokens
    - Both court and player tokens are processed together
    - Outputs are computed from the CLS token

Notes:
    PLCS input is a *set* of fixed-identity tokens (court/human keypoints), not a
    sequence where order has semantic meaning. This model optionally supports:
      - Register tokens: extra learnable tokens inserted after CLS to stabilize
        representations under missing/noisy observations.
      - KP-ID embeddings: explicit embeddings for each keypoint index to reduce
        reliance on token order / RoPE for identity.
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
from src.common.models.embeddings import (
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.plcs.models.components.heads import PositionHead, RotationHead
from src.utils.geometry import NUM_COURT_KP, NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSModel(nn.Module):
    """PLCS: Player Localization in Court System.

    Llama-style architecture with:
    - Multi-Head Self-Attention (MHA) with SDPA for efficiency
    - Rotary Position Embedding (RoPE)
    - SwiGLU MLP and RMSNorm

    This model takes 2D keypoints (human pose + court landmarks) from a
    camera view and predicts the player's 3D position and rotation in
    the court coordinate system.

    Tokens = [CLS, court_tokens(NUM_COURT_KP), player_tokens(NUM_HUMAN_KP)]
    Predicts 3D position and rotation from the CLS token.

    Input:
        - human_kp: Human 2D keypoints (COCO 17), shape (B, 34) or (B, 17, 2)
        - court_kp: Court 2D keypoints (20 landmarks), shape (B, 40) or (B, 20, 2)
        - human_vis: Human visibility mask, shape (B, 17). Optional.
        - court_vis: Court visibility mask, shape (B, 20). Optional.

    Output:
        - position: Normalized (x, y, z) in court coordinates, shape (B, 3)
        - rotation: (sin(yaw), cos(yaw)), shape (B, 2)

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        num_register_tokens: int = 4,
        use_kp_id_embedding: bool = True,
        use_rope: bool = False,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        invisible_init_std: float = 0.02,
    ) -> None:
        """Initialize the PLCS model.

        Args:
            hidden_dim: Hidden dimension for all components.
            num_layers: Number of Transformer blocks.
            num_heads: Number of query attention heads.
            ffn_dim: FFN intermediate dimension. Defaults to 8/3 * hidden_dim.
            dropout: Dropout probability.
            rope_dim: RoPE dimension. Defaults to head_dim.
            rope_theta: RoPE theta parameter.
            yarn: Optional YaRN config for long-context extrapolation.
            num_register_tokens: Number of register tokens inserted after CLS.
            use_kp_id_embedding: Whether to add explicit KP-ID embeddings.
            use_rope: Whether to apply RoPE in attention.
            use_moe: Use Mixture-of-Experts FFN in each Transformer block.
            moe_config: MoE configuration (required when use_moe=True).
            invisible_init_std: Initialization std for invisible tokens.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.yarn = yarn
        self.num_register_tokens = int(num_register_tokens)
        self.use_kp_id_embedding = bool(use_kp_id_embedding)
        self.use_rope = bool(use_rope)
        self.max_tokens = int(NUM_COURT_KP + NUM_HUMAN_KP)

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else rope_dim
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64  # Round to multiple of 64

        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")
        if moe_config is not None and moe_config.dim != hidden_dim:
            raise ValueError(f"moe_config.dim={moe_config.dim} must match hidden_dim={hidden_dim}")
        if self.num_register_tokens < 0:
            raise ValueError(f"num_register_tokens must be >= 0, got {self.num_register_tokens}")

        # Token embeddings
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
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

        # Type embedding: 0 = court, 1 = player
        self.type_embed = nn.Embedding(2, hidden_dim)

        # CLS token (no RoPE applied)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Register tokens (prefix tokens; no RoPE applied)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Optional KP-ID embeddings
        if self.use_kp_id_embedding:
            self.court_id_embed = nn.Embedding(NUM_COURT_KP, hidden_dim)
            self.player_id_embed = nn.Embedding(NUM_HUMAN_KP, hidden_dim)

        # Transformer blocks
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

        # Output heads
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

        if self.use_rope:
            freqs_cis = precompute_freqs_cis(
                dim=self.rope_dim,
                seqlen=self.max_tokens,
                base=self.rope_theta,
                yarn=self.yarn,
                device=None,  # initialized on CPU; moved by `model.to(device)`
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PLCSModel: Initialized model.

        """
        model_cfg = config.get("model", {})

        yarn_cfg = model_cfg.get("yarn", None)
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len") is not None:
                yarn = YaRNConfig(**yarn_cfg)

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_cfg = model_cfg.get("moe_config", None)
        moe_config: MoEConfig | None = None
        if use_moe and moe_cfg is not None:
            moe_config = MoEConfig(dim=int(model_cfg.get("hidden_dim", 256)), **dict(moe_cfg))

        return cls(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=model_cfg.get("dropout", 0.1),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            yarn=yarn,
            num_register_tokens=int(model_cfg.get("num_register_tokens", 4)),
            use_kp_id_embedding=bool(model_cfg.get("use_kp_id_embedding", True)),
            use_rope=bool(model_cfg.get("use_rope", False)),
            use_moe=use_moe,
            moe_config=moe_config,
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def _build_body_tokens(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Build body tokens (court + player), excluding CLS/register tokens."""
        # Tokenize court and player keypoints
        court_tok = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        player_tok = self.player_embed(human_kp, human_vis)  # (B, 17, D)

        # Add type embeddings
        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, 20, D)
        player_type = self.type_embed(
            torch.ones(NUM_HUMAN_KP, device=human_kp.device, dtype=torch.long)
        )[None, :, :]  # (1, 17, D)

        if self.use_kp_id_embedding:
            court_id = self.court_id_embed(
                torch.arange(NUM_COURT_KP, device=human_kp.device, dtype=torch.long)
            )[None, :, :]
            player_id = self.player_id_embed(
                torch.arange(NUM_HUMAN_KP, device=human_kp.device, dtype=torch.long)
            )[None, :, :]
            court_tok = court_tok + court_id
            player_tok = player_tok + player_id

        token_body = torch.cat(
            [court_tok + court_type, player_tok + player_type], dim=1
        )  # (B, 37, D)
        return token_body

    def _add_prefix_tokens(self, token_body: Tensor) -> Tensor:
        """Add CLS/register prefix tokens to body tokens."""
        B = token_body.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        if self.num_register_tokens > 0:
            reg = self.register_tokens.expand(B, -1, -1)
            return torch.cat([cls, reg, token_body], dim=1)  # (B, 1+R+37, D)
        return torch.cat([cls, token_body], dim=1)  # (B, 38, D)

    def _forward_transformer(self, x: Tensor, S_body: int) -> Tensor:
        """Run transformer stack and final normalization."""
        prefix_len = x.size(1) - S_body

        freqs_cis: Tensor | None = None
        if self.use_rope:
            if S_body > self.freqs_cis.shape[0]:
                raise ValueError(
                    f"Sequence length S={S_body} exceeds cached freqs_cis length {self.freqs_cis.shape[0]}. "
                    "Increase max_tokens."
                )
            freqs_cis_body = self.freqs_cis[:S_body]
            if freqs_cis_body.device != x.device:
                freqs_cis_body = freqs_cis_body.to(x.device)

            prefix_freqs = torch.ones(
                prefix_len, freqs_cis_body.shape[1], device=x.device, dtype=freqs_cis_body.dtype
            )
            freqs_cis = torch.cat([prefix_freqs, freqs_cis_body], dim=0)

        attn_mask: Tensor | None = None

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
        return x

    def _encode_tokens(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> tuple[Tensor, int]:
        """Encode tokens and return contextualized outputs and player start index."""
        token_body = self._build_body_tokens(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )
        S_body = token_body.shape[1]
        x = self._add_prefix_tokens(token_body)
        x = self._forward_transformer(x=x, S_body=S_body)
        player_start_idx = 1 + self.num_register_tokens + NUM_COURT_KP
        return x, player_start_idx

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints, shape (B, 34) or (B, 17, 2).
            court_kp: Court keypoints, shape (B, 40) or (B, 20, 2).
            human_vis: Human visibility mask, shape (B, 17). Optional.
            court_vis: Court visibility mask, shape (B, 20). Optional.

        Returns:
            dict: Dictionary with 'position' (B, 3) and 'rotation' (B, 2).

        """
        x, _ = self._encode_tokens(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )

        # Extract CLS token
        cls_out = x[:, 0, :]  # (B, D)

        # Apply output heads
        position = self.position_head(cls_out)  # (B, 3)
        rotation = self.rotation_head(cls_out)  # (B, 2)

        return {
            "position": position,
            "rotation": rotation,
        }

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PLCSModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    B = 2
    human_kp = torch.randn(B, NUM_HUMAN_KP, 2)
    court_kp = torch.randn(B, NUM_COURT_KP, 2)
    human_vis = (torch.rand(B, NUM_HUMAN_KP) > 0.2).to(torch.float32)
    court_vis = (torch.rand(B, NUM_COURT_KP) > 0.1).to(torch.float32)

    with torch.no_grad():
        out = model(human_kp=human_kp, court_kp=court_kp, human_vis=human_vis, court_vis=court_vis)

    print("PLCSModel:")
    for key, value in out.items():
        print(f"  {key}: {tuple(value.shape)}")
