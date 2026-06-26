"""Conditional-position variants of the split axial PLCS model.

The base split model keeps rotation and position trunks separate.  These
variants preserve that isolation, then let the position readout query
rotation/shared context through a small conditional adapter.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.plcs_multiview_axial_split_model import (
    PLCSMultiViewAxialSplitModel,
)
from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    RMSNorm,
    SwiGLU,
    default_ffn_dim,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


ContextSource = Literal["shared", "rotation", "both"]
ConditionalArchitecture = Literal[
    "reference_gated_cross_attn",
    "conditional_pos_decoder",
    "head_specific_conditional_bands",
]


def _flatten_tokens(x: Tensor) -> Tensor:
    """Flatten axial tokens from (B, T, N, D) to (B, T*N, D)."""
    batch_size, seq_len, n_cams, hidden_dim = x.shape
    return x.reshape(batch_size, seq_len * n_cams, hidden_dim)


class ReferenceGatedCrossAttention(nn.Module):
    """Cross-attend from pose tokens to context with a near-zero residual gate."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_dim: int | None,
        head_dim: int,
        rope_dim: int,
        dropout: float,
        ffn_type: Literal["swiglu", "mlp"],
        gate_init_bias: float = -4.0,
    ) -> None:
        super().__init__()
        self.block = CrossAttnBlock(
            CrossAttnBlockConfig(
                dim=dim,
                n_heads=num_heads,
                ffn_dim=ffn_dim,
                head_dim=head_dim,
                rope_dim=rope_dim,
                attn_dropout=dropout,
                ffn_type=ffn_type,
            )
        )
        self.gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    def forward(self, pose_feat: Tensor, context: Tensor, key_valid: Tensor) -> Tensor:
        updated = self.block(pose_feat, context, key_valid=key_valid)
        gate = torch.sigmoid(self.gate(pose_feat))
        return pose_feat + gate * (updated - pose_feat)


class ConditionalPosDecoder(nn.Module):
    """Small cross-attention decoder for the final position representation."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_dim: int | None,
        head_dim: int,
        rope_dim: int,
        dropout: float,
        ffn_type: Literal["swiglu", "mlp"],
        num_layers: int,
        query_init_std: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("ConditionalPosDecoder requires num_layers > 0.")
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.query, mean=0.0, std=query_init_std)
        self.pose_to_query = nn.Linear(dim, dim)
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, pose_feat: Tensor, context: Tensor, key_valid: Tensor) -> Tensor:
        query = self.pose_to_query(pose_feat) + self.query.expand(
            pose_feat.shape[0], pose_feat.shape[1], -1
        )
        for layer in self.layers:
            query = layer(query, context, key_valid=key_valid)
        return self.norm(query)


class HeadSpecificConditionalBands(nn.Module):
    """Cross-attention with per-head predicted time/camera reference bands."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_dim: int | None,
        head_dim: int,
        dropout: float,
        ffn_type: Literal["swiglu", "mlp"],
        gate_init_bias: float = -4.0,
        min_bandwidth: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.min_bandwidth = float(min_bandwidth)

        self.q_norm = RMSNorm(dim)
        self.kv_norm = RMSNorm(dim)
        self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.ref_offset = nn.Linear(dim, num_heads * 2)
        self.bandwidth = nn.Linear(dim, num_heads * 2)
        self.gate = nn.Linear(dim, dim)
        self.ffn_norm = RMSNorm(dim)
        resolved_ffn_dim = default_ffn_dim(dim) if ffn_dim is None else int(ffn_dim)
        self.ffn = (
            SwiGLU(dim, resolved_ffn_dim)
            if ffn_type == "swiglu"
            else nn.Sequential(
                nn.Linear(dim, resolved_ffn_dim),
                nn.GELU(),
                nn.Linear(resolved_ffn_dim, dim),
            )
        )

        nn.init.zeros_(self.ref_offset.weight)
        nn.init.zeros_(self.ref_offset.bias)
        nn.init.zeros_(self.bandwidth.weight)
        nn.init.constant_(self.bandwidth.bias, -1.4)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init_bias)

    def _shape(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def forward(
        self,
        pose_feat: Tensor,
        context: Tensor,
        key_valid: Tensor,
        *,
        seq_len: int,
        n_cams: int,
    ) -> Tensor:
        batch_size, q_len, _ = pose_feat.shape
        key_len = context.shape[1]
        if key_valid.shape != (batch_size, key_len):
            raise ValueError(
                f"key_valid must have shape {(batch_size, key_len)}, got {tuple(key_valid.shape)}"
            )

        key_keep = key_valid > 0
        fully_masked = ~key_keep.any(dim=1)
        context_norm = self.kv_norm(context)
        if fully_masked.any():
            key_keep = key_keep.clone()
            key_keep[fully_masked, 0] = True
            context_norm = context_norm.clone()
            context_norm[fully_masked] = 0.0

        q_norm = self.q_norm(pose_feat)
        q = self._shape(self.wq(q_norm)).transpose(1, 2)
        k = self._shape(self.wk(context_norm)).transpose(1, 2)
        v = self._shape(self.wv(context_norm)).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + self._band_bias(
            q_norm,
            seq_len=seq_len,
            n_cams=n_cams,
            key_len=key_len,
        )
        scores = scores.masked_fill(
            ~key_keep[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attn = torch.softmax(scores, dim=-1)
        attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.num_heads * self.head_dim)
        )
        out = self.wo(out)

        gate = torch.sigmoid(self.gate(pose_feat))
        x = pose_feat + gate * out
        return x + self.ffn(self.ffn_norm(x))

    def _band_bias(
        self,
        query: Tensor,
        *,
        seq_len: int,
        n_cams: int,
        key_len: int,
    ) -> Tensor:
        device = query.device
        dtype = query.dtype
        q_len = query.shape[1]

        time_base = torch.linspace(0.0, 1.0, q_len, device=device, dtype=dtype)
        query_pos = torch.stack(
            [time_base, torch.zeros_like(time_base)],
            dim=-1,
        )
        time_pos = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
        cam_pos = torch.linspace(0.0, 1.0, n_cams, device=device, dtype=dtype)
        key_pos = torch.stack(
            [
                time_pos[:, None].expand(seq_len, n_cams),
                cam_pos[None, :].expand(seq_len, n_cams),
            ],
            dim=-1,
        ).reshape(seq_len * n_cams, 2)
        if key_pos.shape[0] != key_len:
            if key_len % key_pos.shape[0] != 0:
                raise ValueError(
                    f"key_len={key_len} is incompatible with seq_len={seq_len}, n_cams={n_cams}"
                )
            key_pos = key_pos.repeat(key_len // key_pos.shape[0], 1)

        offsets = 0.25 * torch.tanh(self.ref_offset(query)).view(
            query.shape[0], q_len, self.num_heads, 2
        )
        ref = query_pos[None, :, None, :] + offsets
        bandwidth = self.min_bandwidth + nn.functional.softplus(
            self.bandwidth(query).view(query.shape[0], q_len, self.num_heads, 2)
        )
        delta = (ref[:, :, :, None, :] - key_pos[None, None, None, :, :]) / bandwidth[
            :, :, :, None, :
        ]
        bias = -0.5 * (delta * delta).sum(dim=-1)
        return bias.permute(0, 2, 1, 3)


class PLCSMultiViewAxialConditionalPosModel(PLCSMultiViewAxialSplitModel):
    """Split axial PLCS model with conditional position-context adapters."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 0,
        num_task_layers: int = 6,
        rot_num_task_layers: int | None = None,
        pose_num_task_layers: int | None = None,
        canonical_on_rotation_branch: bool = True,
        aux_position_on_rotation_branch: bool = True,
        detach_pose_branch: bool = False,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        predict_canonical_pose: bool = False,
        max_views: int = 8,
        max_seq_len: int = 120,
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
        conditional_architecture: ConditionalArchitecture = "reference_gated_cross_attn",
        context_source: ContextSource = "rotation",
        decoder_layers: int = 1,
        gate_init_bias: float = -4.0,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_task_layers=num_task_layers,
            rot_num_task_layers=rot_num_task_layers,
            pose_num_task_layers=pose_num_task_layers,
            canonical_on_rotation_branch=canonical_on_rotation_branch,
            aux_position_on_rotation_branch=aux_position_on_rotation_branch,
            detach_pose_branch=detach_pose_branch,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rope_dim=rope_dim,
            rope_theta=rope_theta,
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
            ffn_type=ffn_type,
            predict_canonical_pose=predict_canonical_pose,
            max_views=max_views,
            max_seq_len=max_seq_len,
            invisible_init_std=invisible_init_std,
            num_court_tokens=num_court_tokens,
        )
        self.conditional_architecture = conditional_architecture
        self.context_source = context_source

        adapter_kwargs = {
            "dim": self.hidden_dim,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "head_dim": self.head_dim,
            "rope_dim": self.rope_dim,
            "dropout": dropout,
            "ffn_type": ffn_type,
        }
        if conditional_architecture == "reference_gated_cross_attn":
            self.pos_context_adapter = ReferenceGatedCrossAttention(
                **adapter_kwargs,
                gate_init_bias=gate_init_bias,
            )
        elif conditional_architecture == "conditional_pos_decoder":
            self.pos_context_adapter = ConditionalPosDecoder(
                **adapter_kwargs,
                num_layers=decoder_layers,
                query_init_std=query_init_std,
            )
        elif conditional_architecture == "head_specific_conditional_bands":
            self.pos_context_adapter = HeadSpecificConditionalBands(
                dim=self.hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                head_dim=self.head_dim,
                dropout=dropout,
                ffn_type=ffn_type,
                gate_init_bias=gate_init_bias,
            )
        else:
            raise ValueError(
                f"Unsupported conditional_architecture={conditional_architecture!r}"
            )

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSMultiViewAxialConditionalPosModel:
        """Create the conditional-position split model from a Hydra config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        pos_context_cfg = model_cfg.get("pos_context", {}) or {}

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 0)),
            num_task_layers=int(model_cfg.get("num_task_layers", 6)),
            rot_num_task_layers=(
                int(model_cfg["rot_num_task_layers"])
                if model_cfg.get("rot_num_task_layers", None) is not None
                else None
            ),
            pose_num_task_layers=(
                int(model_cfg["pose_num_task_layers"])
                if model_cfg.get("pose_num_task_layers", None) is not None
                else None
            ),
            canonical_on_rotation_branch=bool(
                model_cfg.get("canonical_on_rotation_branch", True)
            ),
            aux_position_on_rotation_branch=bool(
                model_cfg.get("aux_position_on_rotation_branch", True)
            ),
            detach_pose_branch=bool(model_cfg.get("detach_pose_branch", False)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            predict_canonical_pose=bool(model_cfg.get("predict_canonical_pose", False)),
            max_views=int(model_cfg.get("max_views", 8)),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
            conditional_architecture=cast(
                ConditionalArchitecture,
                str(pos_context_cfg.get("architecture", "reference_gated_cross_attn")),
            ),
            context_source=cast(
                ContextSource,
                str(pos_context_cfg.get("source", "rotation")),
            ),
            decoder_layers=int(pos_context_cfg.get("decoder_layers", 1)),
            gate_init_bias=float(pos_context_cfg.get("gate_init_bias", -4.0)),
            query_init_std=float(pos_context_cfg.get("query_init_std", 0.02)),
        )

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with conditional context readout for position."""
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
            .reshape(batch_size, n_cams, seq_len_in, self.hidden_dim)
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
            batch_size=batch_size, seq_len=seq_len_in, n_cams=n_cams
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size, seq_len=seq_len_in, n_cams=n_cams
        )

        stack_kwargs = {
            "batch_size": batch_size,
            "seq_len": seq_len_in,
            "n_cams": n_cams,
            "camera_freqs": camera_freqs,
            "time_freqs": time_freqs,
            "camera_mask": camera_mask,
            "time_mask": time_mask,
        }

        x_shared = self._run_axial_stack(
            x, self.camera_layers, self.time_layers, **stack_kwargs
        )
        x_rot = self._run_axial_stack(
            x_shared, self.rot_camera_layers, self.rot_time_layers, **stack_kwargs
        )
        pose_input = x_shared.detach() if self.detach_pose_branch else x_shared
        x_pose = self._run_axial_stack(
            pose_input, self.pose_camera_layers, self.pose_time_layers, **stack_kwargs
        )

        rot_feat = self.rot_final_norm(x_rot[:, :, 0, :])
        pose_feat = self.pose_final_norm(x_pose[:, :, 0, :])
        context = self._context_tokens(x_shared=x_shared, x_rot=x_rot)
        key_valid = self._context_valid(token_valid_t)
        pose_feat = self._apply_pos_context(
            pose_feat,
            context,
            key_valid,
            seq_len=seq_len_in,
            n_cams=n_cams,
        )

        out = {
            "position": self.position_head(pose_feat),
            "rotation": self.rotation_head(rot_feat),
        }
        if self.predict_canonical_pose and self.canonical_pose_head is not None:
            canonical_feat = (
                rot_feat if self.canonical_on_rotation_branch else pose_feat
            )
            out["canonical_pose"] = self.canonical_pose_head(canonical_feat)
        if self.aux_position_head is not None:
            out["aux_position"] = self.aux_position_head(rot_feat)
        return out

    def _context_tokens(self, *, x_shared: Tensor, x_rot: Tensor) -> Tensor:
        if self.context_source == "shared":
            return _flatten_tokens(x_shared)
        if self.context_source == "rotation":
            return _flatten_tokens(x_rot)
        if self.context_source == "both":
            return torch.cat([_flatten_tokens(x_shared), _flatten_tokens(x_rot)], dim=1)
        raise ValueError(f"Unsupported context_source={self.context_source!r}")

    def _context_valid(self, token_valid_t: Tensor) -> Tensor:
        valid = token_valid_t.reshape(token_valid_t.shape[0], -1)
        if self.context_source == "both":
            return torch.cat([valid, valid], dim=1)
        return valid

    def _apply_pos_context(
        self,
        pose_feat: Tensor,
        context: Tensor,
        key_valid: Tensor,
        *,
        seq_len: int,
        n_cams: int,
    ) -> Tensor:
        if isinstance(self.pos_context_adapter, HeadSpecificConditionalBands):
            return self.pos_context_adapter(
                pose_feat,
                context,
                key_valid,
                seq_len=seq_len,
                n_cams=n_cams,
            )
        return self.pos_context_adapter(pose_feat, context, key_valid)
