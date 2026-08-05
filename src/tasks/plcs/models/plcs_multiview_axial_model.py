"""Axial multi-view PLCS model with one token per camera/time element."""

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
    resolve_axial_rope_bases,
)
from src.utils.models.axial_multiview_mixin import AxialMultiViewMixin
from src.utils.models.embeddings import (
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.plcs.configuration import PLCSModelConfig


class PLCSMultiViewAxialModel(AxialMultiViewMixin, nn.Module):
    """PLCS multiview model with alternating camera/time self-attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rope_dim: int,
        rope_theta_time: float,
        rope_theta_camera: float,
        ffn_type: Literal["swiglu", "mlp"],
        predict_canonical_pose: bool,
        max_views: int,
        max_seq_len: int,
        invisible_init_std: float,
        num_court_tokens: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.predict_canonical_pose = bool(predict_canonical_pose)
        self.max_views = int(max_views)
        self.max_seq_len = int(max_seq_len)
        self.num_court_tokens = int(num_court_tokens)

        self._validate_init_args(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_views=self.max_views,
            max_seq_len=self.max_seq_len,
        )

        head_dim = self.hidden_dim // num_heads
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)

        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_bases = resolve_axial_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
        )

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embed = CourtPlayerGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_tokens,
        )

        self.camera_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_bases[1],
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.time_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_bases[0],
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        self.position_head = PositionHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.rotation_head = RotationHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
        )
        self.canonical_pose_head = None
        if self.predict_canonical_pose:
            self.canonical_pose_head = CanonicalPoseHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                num_layers=2,
                dropout=dropout,
                num_keypoints=NUM_HUMAN_KP,
            )

        token_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_token_positions(
                seq_len=self.max_seq_len,
                n_cams=self.max_views,
            ),
            base=self.rope_bases,
        )
        self.register_buffer("token_freqs_cis", token_freqs, persistent=False)

    @staticmethod
    def _validate_init_args(
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_views: int,
        max_seq_len: int,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if max_views <= 0:
            raise ValueError(f"max_views must be positive, got {max_views}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    @classmethod
    def from_config(
        cls, config: PLCSModelConfig, *, num_court_tokens: int
    ) -> PLCSMultiViewAxialModel:
        """Create model from hydra config."""
        return cls(
            hidden_dim=config.integer("hidden_dim"),
            num_layers=config.integer("num_layers"),
            num_heads=config.integer("num_heads"),
            ffn_dim=config.integer("ffn_dim"),
            dropout=config.number("dropout"),
            rope_dim=config.integer("rope_dim"),
            rope_theta_time=config.number("rope_theta_time"),
            rope_theta_camera=config.number("rope_theta_camera"),
            ffn_type=cast(Literal["swiglu", "mlp"], config.string("ffn_type")),
            predict_canonical_pose=config.boolean("predict_canonical_pose"),
            max_views=config.integer("max_views"),
            max_seq_len=config.integer("max_seq_len"),
            invisible_init_std=config.number("invisible_init_std"),
            num_court_tokens=num_court_tokens,
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for multiview PLCS inputs."""
        batch_size, n_cams, seq_len_in = self._validate_forward_inputs(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            human_mask=human_mask,
            court_vis=court_vis,
        )

        if human_vis is not None:
            human_kp = human_kp * (human_vis > 0).unsqueeze(-1).to(dtype=human_kp.dtype)
        if court_vis is not None:
            court_kp = court_kp * (court_vis > 0).unsqueeze(-1).to(dtype=court_kp.dtype)

        if human_mask is not None:
            token_valid = human_mask > 0
        else:
            token_valid = torch.ones(
                batch_size,
                n_cams,
                seq_len_in,
                dtype=torch.bool,
                device=human_kp.device,
            )

        court_flat = court_kp.reshape(
            batch_size * n_cams * seq_len_in, self.num_court_tokens, 2
        )
        human_flat = human_kp.reshape(batch_size * n_cams * seq_len_in, NUM_HUMAN_KP, 2)
        group_vis = token_valid.reshape(batch_size * n_cams * seq_len_in)
        x = (
            self.group_embed(court_flat, human_flat, group_vis)
            .reshape(
                batch_size,
                n_cams,
                seq_len_in,
                self.hidden_dim,
            )
            .permute(0, 2, 1, 3)
        )

        token_valid_t = token_valid.permute(0, 2, 1)
        camera_valid = token_valid_t.reshape(batch_size * seq_len_in, n_cams)
        time_valid = token_valid_t.permute(0, 2, 1).reshape(
            batch_size * n_cams, seq_len_in
        )
        camera_mask, _ = self._build_self_attn_mask(camera_valid)
        time_mask, _ = self._build_self_attn_mask(time_valid)
        camera_freqs = self._camera_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
        )

        for camera_layer, time_layer in zip(
            self.camera_layers,
            self.time_layers,
            strict=True,
        ):
            x_camera = x.reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
            x_camera = camera_layer(
                x_camera,
                freqs_cis=camera_freqs,
                attn_mask=camera_mask,
            )
            x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(
                batch_size * n_cams, seq_len_in, self.hidden_dim
            )
            x_time = time_layer(
                x_time,
                freqs_cis=time_freqs,
                attn_mask=time_mask,
            )
            x = x_time.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim).permute(
                0, 2, 1, 3
            )

        x = x[:, :, 0, :]
        x = self.final_norm(x)

        out = {
            "position": self.position_head(x),
            "rotation": self.rotation_head(x),
        }
        if self.predict_canonical_pose and self.canonical_pose_head is not None:
            out["canonical_pose"] = self.canonical_pose_head(x)
        return out

    def _validate_forward_inputs(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None,
        human_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[int, int, int]:
        if human_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewAxialModel expects human_kp as (B,N,T,17,2), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() != 5:
            raise ValueError(
                "PLCSMultiViewAxialModel expects court_kp as "
                f"(B,N,T,{self.num_court_tokens},2), "
                f"got shape {tuple(court_kp.shape)}"
            )
        if court_kp.shape[-2] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_kp with K={self.num_court_tokens}, got K={court_kp.shape[-2]}."
            )
        if human_vis is not None and human_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewAxialModel expects human_vis as (B,N,T,17), "
                f"got shape {tuple(human_vis.shape)}"
            )
        if court_vis is not None and court_vis.dim() != 4:
            raise ValueError(
                "PLCSMultiViewAxialModel expects court_vis as "
                f"(B,N,T,{self.num_court_tokens}), "
                f"got shape {tuple(court_vis.shape)}"
            )
        if court_vis is not None and court_vis.shape[-1] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_vis with K={self.num_court_tokens}, got K={court_vis.shape[-1]}."
            )

        batch_size, n_cams, seq_len_in = human_kp.shape[:3]
        if n_cams > self.max_views:
            raise ValueError(
                f"Number of views N={n_cams} exceeds max_views={self.max_views}."
            )
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={seq_len_in} exceeds max_seq_len={self.max_seq_len}."
            )
        if human_mask is not None and (
            human_mask.dim() != 3
            or human_mask.shape != (batch_size, n_cams, seq_len_in)
        ):
            raise ValueError(
                "human_mask for multiview models must be (B,N,T), "
                f"got {tuple(human_mask.shape)}"
            )
        return batch_size, n_cams, seq_len_in
