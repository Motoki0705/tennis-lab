"""Separate-trunk PLCS axial model (issue #518 EX10 architecture).

``PLCSMultiViewAxialSplitModel`` extends :class:`PLCSMultiViewAxialModel` with
*task-specific axial trunks*: after an optional shared stack, the rotation task
and the pose (position + canonical) task each run their own stack of
camera/time layers, instead of reading a single shared trunk.

Why a separate trunk (experiments/README.md exp3-exp10, issues #518/#525):
    On a single shared trunk, position and rotation compete for capacity -- the
    dominant gradient owns the readout feature, so scalar loss reweighting can
    only trade one task off against the other. Splitting the readout into
    independent trunks removes the competition. The winning recipe (exp10):

      - rotation trunk carries the rotation head **and** the canonical-pose head
        (``canonical_on_rotation_branch``) so the hard rotation task gets the 3D
        geometry regularisation it needs, plus an **auxiliary position head**
        (``aux_position_on_rotation_branch``) that teaches the multiview
        triangulation / cross-view correspondence rotation depends on;
      - a dedicated pose trunk produces the precise final position.

    Result: rotation ~9.98 deg AND position ~0.238 m, beating both the shared
    trunk and (issue #525) parameter-matched shared trunks -- the gain is
    architectural, not a parameter-count effect.

This lives in its own module so the shared-trunk base
(``plcs_multiview_axial_model.py``) stays a clean, single-responsibility model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import PositionHead
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    default_ffn_dim,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSMultiViewAxialSplitModel(PLCSMultiViewAxialModel):
    """Multiview axial PLCS model with separate rotation and pose trunks."""

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
    ) -> None:
        if num_task_layers <= 0:
            raise ValueError(
                "PLCSMultiViewAxialSplitModel requires num_task_layers > 0; "
                "use PLCSMultiViewAxialModel for a shared trunk."
            )
        # Asymmetric-capacity option (issue #525/#535 follow-up): the rotation and
        # pose trunks may each be deeper than the shared ``num_task_layers``
        # default. Because the two trunks are isolated, spending extra depth on one
        # cannot contaminate the other (the shared-trunk depth-collapse seen in
        # #525 does not apply). ``None`` keeps the symmetric default
        # (== num_task_layers), so existing configs are unchanged.
        if rot_num_task_layers is None:
            rot_num_task_layers = num_task_layers
        if pose_num_task_layers is None:
            pose_num_task_layers = num_task_layers
        if rot_num_task_layers <= 0 or pose_num_task_layers <= 0:
            raise ValueError("rot/pose_num_task_layers must be > 0 when set.")
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
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

        self.num_task_layers = int(num_task_layers)
        self.rot_num_task_layers = int(rot_num_task_layers)
        self.pose_num_task_layers = int(pose_num_task_layers)
        self.canonical_on_rotation_branch = bool(canonical_on_rotation_branch)
        self.aux_position_on_rotation_branch = bool(aux_position_on_rotation_branch)
        self.detach_pose_branch = bool(detach_pose_branch)

        if ffn_dim is None:
            ffn_dim = default_ffn_dim(self.hidden_dim)

        def _axial_stack(rope_base: float, depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    TransformerBlock(
                        TransformerBlockConfig(
                            dim=self.hidden_dim,
                            n_heads=num_heads,
                            ffn_dim=ffn_dim,
                            head_dim=self.head_dim,
                            rope_dim=self.rope_dim,
                            attn_dropout=dropout,
                            rope_base=rope_base,
                            ffn_type=ffn_type,
                        )
                    )
                    for _ in range(depth)
                ]
            )

        # rope_bases = (time_base, camera_base). Rotation and pose trunks can each
        # carry an independent depth (asymmetric capacity, issue #525/#535).
        self.rot_camera_layers = _axial_stack(self.rope_bases[1], self.rot_num_task_layers)
        self.rot_time_layers = _axial_stack(self.rope_bases[0], self.rot_num_task_layers)
        self.pose_camera_layers = _axial_stack(self.rope_bases[1], self.pose_num_task_layers)
        self.pose_time_layers = _axial_stack(self.rope_bases[0], self.pose_num_task_layers)
        self.rot_final_norm = RMSNorm(self.hidden_dim)
        self.pose_final_norm = RMSNorm(self.hidden_dim)

        # Auxiliary position head on the rotation trunk: supervises it with the
        # multiview-triangulation signal rotation depends on (the precise final
        # position still comes from the dedicated pose trunk).
        self.aux_position_head: PositionHead | None = None
        if self.aux_position_on_rotation_branch:
            self.aux_position_head = PositionHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

    @classmethod
    def from_config(cls, config: DictConfig) -> PLCSMultiViewAxialSplitModel:
        """Create the split model from a hydra config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

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
        )

    def _run_axial_stack(
        self,
        x: Tensor,
        camera_layers: nn.ModuleList,
        time_layers: nn.ModuleList,
        *,
        batch_size: int,
        seq_len: int,
        n_cams: int,
        camera_freqs: Tensor,
        time_freqs: Tensor,
        camera_mask: Tensor,
        time_mask: Tensor,
    ) -> Tensor:
        """Run alternating camera/time self-attention over ``x`` (B,T,N,hidden)."""
        for camera_layer, time_layer in zip(camera_layers, time_layers, strict=True):
            x_camera = x.reshape(batch_size * seq_len, n_cams, self.hidden_dim)
            x_camera = camera_layer(x_camera, freqs_cis=camera_freqs, attn_mask=camera_mask)
            x = x_camera.reshape(batch_size, seq_len, n_cams, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(
                batch_size * n_cams, seq_len, self.hidden_dim
            )
            x_time = time_layer(x_time, freqs_cis=time_freqs, attn_mask=time_mask)
            x = x_time.reshape(batch_size, n_cams, seq_len, self.hidden_dim).permute(
                0, 2, 1, 3
            )
        return x

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass with shared encoder + separate rotation/pose trunks."""
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

        # Optional shared trunk (num_layers may be 0 -> identity), then split.
        x = self._run_axial_stack(x, self.camera_layers, self.time_layers, **stack_kwargs)
        x_rot = self._run_axial_stack(
            x, self.rot_camera_layers, self.rot_time_layers, **stack_kwargs
        )
        pose_input = x.detach() if self.detach_pose_branch else x
        x_pose = self._run_axial_stack(
            pose_input, self.pose_camera_layers, self.pose_time_layers, **stack_kwargs
        )

        rot_feat = self.rot_final_norm(x_rot[:, :, 0, :])
        pose_feat = self.pose_final_norm(x_pose[:, :, 0, :])

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
