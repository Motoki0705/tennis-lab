"""Axial single-view BLCS model with one court-ball token per timestep."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
)
from src.utils.models.components.ops.time_local import build_local_attention_keep_mask
from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSAxialModel(nn.Module):
    """BLCS single-view model with temporal self-attention over group tokens.

    Each timestep is represented by one token built from the court keypoints and
    the observed ball UV coordinate. This keeps the standard single-view BLCS
    input/output contract while avoiding full attention over separate court and
    ball token sequences.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_type: Literal["mha", "gqa"] = "mha",
        num_kv_heads: int | None = None,
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
        time_window_radius: int = 16,
        time_global_layer_mask: Sequence[bool] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)
        self.time_window_radius = int(time_window_radius)
        self.time_global_layer_mask = self._normalize_layer_mask(
            time_global_layer_mask,
            num_layers=self.num_layers,
        )

        self._validate_init_args(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            time_window_radius=self.time_window_radius,
            time_global_layer_mask=self.time_global_layer_mask,
        )

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(
            rope_theta if rope_theta_time is None else rope_theta_time
        )

        if ffn_dim is None:
            ffn_dim = int((8 * self.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.group_embed = CourtBallGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_tokens,
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
                        attention_type=attention_type,
                        n_kv_heads=num_kv_heads,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if self.predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

        freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_time_positions(seq_len=self.max_seq_len),
            base=self.rope_theta,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

    @staticmethod
    def _validate_init_args(
        *,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        num_layers: int,
        time_window_radius: int,
        time_global_layer_mask: tuple[bool, ...],
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if time_window_radius < 0:
            raise ValueError(
                f"time_window_radius must be non-negative, got {time_window_radius}"
            )
        if len(time_global_layer_mask) != num_layers:
            raise ValueError(
                "time_global_layer_mask length must equal num_layers, got "
                f"{len(time_global_layer_mask)} and {num_layers}"
            )

    @staticmethod
    def _validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

    @staticmethod
    def _normalize_layer_mask(
        values: Sequence[bool] | None,
        *,
        num_layers: int,
    ) -> tuple[bool, ...]:
        if values is None:
            return tuple(False for _ in range(num_layers))
        return tuple(bool(value) for value in values)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSAxialModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        global_mask = model_cfg.get(
            "time_global_layer_mask",
            model_cfg.get("time_global_stage_mask", None),
        )

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            attention_type=cast(
                Literal["mha", "gqa"], str(model_cfg.get("attention_type", "mha"))
            ),
            num_kv_heads=model_cfg.get("num_kv_heads", None),
            ffn_dim=model_cfg.get("ffn_dim", None),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
            time_window_radius=int(model_cfg.get("time_window_radius", 16)),
            time_global_layer_mask=global_mask,
        )

    @staticmethod
    def _build_time_positions(*, seq_len: int) -> Tensor:
        return torch.arange(seq_len, dtype=torch.long) + 1

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(
            valid_fixed.shape[0],
            valid_fixed.shape[1],
            valid_fixed.shape[1],
        )
        return attn_mask, valid_fixed

    @staticmethod
    def _build_sliding_attn_mask(valid: Tensor, radius: int) -> Tensor:
        return build_local_attention_keep_mask(valid, radius)

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass for single-view BLCS inputs."""
        court_kp, ball_vis, ball_mask, seq_len_in = (
            self._prepare_forward_inputs(
                ball_uv=ball_uv,
                court_kp=court_kp,
                ball_vis=ball_vis,
                ball_mask=ball_mask,
                court_vis=court_vis,
            )
        )

        x = self.group_embed(court_kp, ball_uv, ball_vis)
        token_valid = ball_mask > 0
        time_full_mask, time_valid = self._build_self_attn_mask(token_valid)
        sliding_mask = self._build_sliding_attn_mask(
            time_valid,
            radius=self.time_window_radius,
        )
        freqs_cis = self._freqs_for_sequence(x=x, seq_len=seq_len_in)

        for layer_index, time_layer in enumerate(self.time_layers):
            use_global_attention = self.time_global_layer_mask[layer_index]
            x = time_layer(
                x,
                freqs_cis=freqs_cis,
                attn_mask=time_full_mask if use_global_attention else sliding_mask,
            )

        x = self.final_norm(x)
        out: dict[str, Tensor] = {"position": self.position_head(x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(x)
        return out

    def _prepare_forward_inputs(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor, int]:
        if ball_uv.dim() != 3:
            raise ValueError(
                f"ball_uv must have shape (B, T, 2), got {tuple(ball_uv.shape)}"
            )
        batch_size, seq_len_in, coord_dim = ball_uv.shape
        if coord_dim != 2:
            raise ValueError(
                f"ball_uv must have shape (B, T, 2), got {tuple(ball_uv.shape)}"
            )
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len_in} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )

        if court_kp.dim() == 2 or (
            court_kp.dim() == 3
            and tuple(court_kp.shape[-2:]) == (self.num_court_tokens, 2)
        ):
            court_kp = court_kp.unsqueeze(1).expand(-1, seq_len_in, *court_kp.shape[1:])
        if court_kp.dim() not in {3, 4}:
            raise ValueError(
                "court_kp must have shape "
                f"(B, {self.num_court_tokens}, 2), "
                f"(B, {self.num_court_tokens * 2}), "
                f"(B, T, {self.num_court_tokens}, 2), or "
                f"(B, T, {self.num_court_tokens * 2}), "
                f"got {tuple(court_kp.shape)}"
            )
        if court_kp.shape[:2] != (batch_size, seq_len_in):
            raise ValueError(
                f"court_kp leading shape must be {(batch_size, seq_len_in)}, "
                f"got {tuple(court_kp.shape[:2])}"
            )

        if court_vis is not None:
            if court_vis.dim() == 2:
                court_vis = court_vis.unsqueeze(1).expand(-1, seq_len_in, -1)
            if court_vis.shape != (batch_size, seq_len_in, self.num_court_tokens):
                raise ValueError(
                    "court_vis must have shape "
                    f"(B, {self.num_court_tokens}) or "
                    f"(B, T, {self.num_court_tokens}), "
                    f"got {tuple(court_vis.shape)}"
                )

        if ball_vis is not None and ball_vis.shape != (batch_size, seq_len_in):
            raise ValueError(
                f"ball_vis must have shape {(batch_size, seq_len_in)}, "
                f"got {tuple(ball_vis.shape)}"
            )
        if ball_mask is None:
            ball_mask = torch.ones(
                batch_size,
                seq_len_in,
                device=ball_uv.device,
                dtype=torch.bool,
            )
        if ball_mask.shape != (batch_size, seq_len_in):
            raise ValueError(
                f"ball_mask must have shape {(batch_size, seq_len_in)}, "
                f"got {tuple(ball_mask.shape)}"
            )
        return court_kp, ball_vis, ball_mask, seq_len_in

    def _freqs_for_sequence(self, *, x: Tensor, seq_len: int) -> Tensor:
        freqs_cis = cast(Tensor, self.freqs_cis)
        if freqs_cis.shape[0] < seq_len:
            raise ValueError(
                f"Sequence length T={seq_len} exceeds cached freqs_cis length {freqs_cis.shape[0]}. "
                "Increase max_seq_len."
            )
        freqs_cis = freqs_cis[:seq_len]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)
        return freqs_cis

    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
