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

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import (
    CanonicalPoseHead,
    PositionHead,
    RotationHead,
)
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
    resolve_rope_bases,
)
from src.utils.models.embeddings import (
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
    PlayerKPUVEmbedding,
)
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.plcs.configuration import PLCSModelConfig


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
        - rotation: (cos(yaw), sin(yaw)), shape (B, 2)

    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        rope_theta_time: float,
        rope_theta_camera: float,
        rope_theta_type: float,
        num_register_tokens: int,
        use_kp_id_embedding: bool,
        use_rope: bool,
        ffn_type: Literal["swiglu", "mlp"],
        predict_canonical_pose: bool,
        invisible_init_std: float,
        num_court_tokens: int,
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
            num_register_tokens: Number of register tokens inserted after CLS.
            use_kp_id_embedding: Whether to add explicit KP-ID embeddings.
            use_rope: Whether to apply RoPE in attention.
            invisible_init_std: Initialization std for invisible tokens.

        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_register_tokens = int(num_register_tokens)
        self.use_kp_id_embedding = bool(use_kp_id_embedding)
        self.use_rope = bool(use_rope)
        self.predict_canonical_pose = bool(predict_canonical_pose)
        self.num_court_tokens = int(num_court_tokens)
        self.max_tokens = int(self.num_court_tokens + NUM_HUMAN_KP)

        head_dim = hidden_dim // num_heads
        self.rope_dim = rope_dim
        self.rope_theta = float(rope_theta)
        self.rope_bases = resolve_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
            rope_theta_type=rope_theta_type,
        )

        self._validate_init_args(num_register_tokens=self.num_register_tokens)

        # Token embeddings
        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )
        self.player_embed = PlayerKPUVEmbedding(
            dim=hidden_dim,
            invisible_token=self.invisible_token,
        )

        # CLS token (no RoPE applied)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Register tokens (prefix tokens; no RoPE applied)
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, self.num_register_tokens, hidden_dim)
            )
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Optional KP-ID embeddings
        if self.use_kp_id_embedding:
            self.court_id_embed = nn.Embedding(self.num_court_tokens, hidden_dim)
            self.player_id_embed = nn.Embedding(NUM_HUMAN_KP, hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
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
        self.canonical_pose_head = None
        if self.predict_canonical_pose:
            self.canonical_pose_head = CanonicalPoseHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                num_layers=2,
                dropout=dropout,
                num_keypoints=NUM_HUMAN_KP,
            )

        if self.use_rope:
            freqs_cis = precompute_freqs_cis_nd(
                dim=self.rope_dim,
                pos=self._build_body_rope_positions(),
                base=self.rope_bases,
            )
            self.register_buffer("freqs_cis_body", freqs_cis, persistent=False)

    @staticmethod
    def _validate_init_args(*, num_register_tokens: int) -> None:
        if num_register_tokens < 0:
            raise ValueError(
                f"num_register_tokens must be >= 0, got {num_register_tokens}"
            )

    @classmethod
    def from_config(
        cls, config: PLCSModelConfig, *, num_court_tokens: int
    ) -> PLCSModel:
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PLCSModel: Initialized model.

        """
        return cls(
            hidden_dim=config.integer("hidden_dim"),
            num_layers=config.integer("num_layers"),
            num_heads=config.integer("num_heads"),
            ffn_dim=config.integer("ffn_dim"),
            dropout=config.number("dropout"),
            rope_dim=config.integer("rope_dim"),
            rope_theta=config.number("rope_theta"),
            rope_theta_time=config.number("rope_theta_time"),
            rope_theta_camera=config.number("rope_theta_camera"),
            rope_theta_type=config.number("rope_theta_type"),
            num_register_tokens=config.integer("num_register_tokens"),
            use_kp_id_embedding=config.boolean("use_kp_id_embedding"),
            use_rope=config.boolean("use_rope"),
            ffn_type=cast(Literal["swiglu", "mlp"], config.string("ffn_type")),
            predict_canonical_pose=config.boolean("predict_canonical_pose"),
            invisible_init_std=config.number("invisible_init_std"),
            num_court_tokens=num_court_tokens,
        )

    def _build_body_rope_positions(self) -> Tensor:
        """Build 3-axis RoPE positions for `[court, player]` body tokens."""
        court_idx = torch.arange(self.num_court_tokens, dtype=torch.long)
        player_idx = torch.arange(NUM_HUMAN_KP, dtype=torch.long)

        court_pos = torch.stack(
            [
                court_idx,
                torch.zeros_like(court_idx),
                torch.zeros_like(court_idx),
            ],
            dim=-1,
        )
        player_pos = torch.stack(
            [
                player_idx,
                torch.zeros_like(player_idx),
                torch.ones_like(player_idx),
            ],
            dim=-1,
        )
        return torch.cat([court_pos, player_pos], dim=0)

    def _build_body_tokens(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            human_kp:
                Human 2D keypoints in normalized image UV.
                Shape: (B, 34) or (B, 17, 2).
            court_kp:
                Court 2D keypoints in normalized image UV.
                Shape: (B, 40) or (B, 20, 2).
            human_vis:
                Human keypoint visibility flags aligned with `human_kp`.
                Shape: (B, 17). Each element is interpreted as visible if > 0
                (e.g., bool, 0/1 float, or 0/1 int). Optional; if None, all
                human keypoints are treated as visible.
            court_vis:
                Court keypoint visibility flags aligned with `court_kp`.
                Shape: (B, 20). Each element is interpreted as visible if > 0.
                Optional; if None, all court keypoints are treated as visible.

        Returns:
            dict:
                - position: (B, 3) normalized court-space xyz
                - rotation: (B, 2) as (cos(yaw), sin(yaw))

        """
        # Tokenize court and player keypoints
        court_tok = self.court_embed(court_kp, court_vis)  # (B, K, D)
        player_tok = self.player_embed(human_kp, human_vis)  # (B, 17, D)

        K = court_tok.shape[1]

        if self.use_kp_id_embedding:
            court_id = self.court_id_embed(
                torch.arange(K, device=human_kp.device, dtype=torch.long)
            )[None, :, :]
            player_id = self.player_id_embed(
                torch.arange(NUM_HUMAN_KP, device=human_kp.device, dtype=torch.long)
            )[None, :, :]
            court_tok = court_tok + court_id
            player_tok = player_tok + player_id

        return torch.cat([court_tok, player_tok], dim=1)  # (B, 37, D)

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
            freqs_cis_body = cast(Tensor, self.freqs_cis_body)
            if S_body > freqs_cis_body.shape[0]:
                raise ValueError(
                    f"Sequence length S={S_body} exceeds cached freqs_cis length {freqs_cis_body.shape[0]}. "
                    "Increase max_tokens."
                )
            freqs_cis_body = freqs_cis_body[:S_body]
            if freqs_cis_body.device != x.device:
                freqs_cis_body = freqs_cis_body.to(x.device)

            prefix_freqs = torch.ones(
                prefix_len,
                freqs_cis_body.shape[1],
                device=x.device,
                dtype=freqs_cis_body.dtype,
            )
            freqs_cis = torch.cat([prefix_freqs, freqs_cis_body], dim=0)

        attn_mask: Tensor | None = None

        for blk in self.blocks:
            x = blk(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)
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
        player_start_idx = 1 + self.num_register_tokens + self.num_court_tokens
        return x, player_start_idx

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for frame model.

        Expected shapes:
        - ``human_kp``: ``(B,17,2)`` or ``(B,34)``
        - ``court_kp``: ``(B,20,2)`` or ``(B,40)``
        - ``human_vis``: ``(B,17)``, optional
        - ``court_vis``: ``(B,20)``, optional
        - ``human_mask``: ``(B,)`` or ``None`` (unused by frame model)
        """
        self._validate_forward_inputs(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            human_mask=human_mask,
            court_vis=court_vis,
        )

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

        out = {
            "position": position,
            "rotation": rotation,
        }
        if self.predict_canonical_pose and self.canonical_pose_head is not None:
            out["canonical_pose"] = self.canonical_pose_head(cls_out)
        return out

    @staticmethod
    def _validate_forward_inputs(
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None,
        human_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> None:
        if human_kp.dim() not in {2, 3}:
            raise ValueError(
                "PLCSModel expects human_kp as (B,17,2) or (B,34), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() not in {2, 3}:
            raise ValueError(
                "PLCSModel expects court_kp as (B,20,2) or (B,40), "
                f"got shape {tuple(court_kp.shape)}"
            )
        if human_vis is not None and human_vis.dim() != 2:
            raise ValueError(
                "PLCSModel expects human_vis as (B,17), "
                f"got shape {tuple(human_vis.shape)}"
            )
        if court_vis is not None and court_vis.dim() != 2:
            raise ValueError(
                "PLCSModel expects court_vis as (B,20), "
                f"got shape {tuple(court_vis.shape)}"
            )
        if human_mask is not None and human_mask.dim() != 1:
            raise ValueError(
                "PLCSModel expects human_mask as (B,) or None, "
                f"got shape {tuple(human_mask.shape)}"
            )

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
