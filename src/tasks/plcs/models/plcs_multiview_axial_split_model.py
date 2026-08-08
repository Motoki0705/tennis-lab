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

import torch.nn as nn
from torch import Tensor

from src.tasks.plcs.models.components.heads import CanonicalPoseHead, PositionHead
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.schema.player import NUM_HUMAN_KP

if TYPE_CHECKING:
    from src.tasks.plcs.configuration import PLCSModelConfig


class PLCSMultiViewAxialSplitModel(PLCSMultiViewAxialModel):
    """Multiview axial PLCS model with separate rotation and pose trunks."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_task_layers: int,
        rot_num_task_layers: int,
        pose_num_task_layers: int,
        canonical_on_rotation_branch: bool,
        aux_position_on_rotation_branch: bool,
        detach_pose_branch: bool,
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
        if num_task_layers <= 0:
            raise ValueError(
                "PLCSMultiViewAxialSplitModel requires num_task_layers > 0; "
                "use PLCSMultiViewAxialModel for a shared trunk."
            )
        # The validated model contract resolves symmetric/asymmetric branch
        # depths before construction; the constructor never supplies a default.
        if rot_num_task_layers <= 0 or pose_num_task_layers <= 0:
            raise ValueError("rot/pose_num_task_layers must be > 0 when set.")
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rope_dim=rope_dim,
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
                            attention_type="mha",
                            n_kv_heads=None,
                            rope_base=rope_base,
                            ffn_type=ffn_type,
                        )
                    )
                    for _ in range(depth)
                ]
            )

        # rope_bases = (time_base, camera_base). Rotation and pose trunks can each
        # carry an independent depth (asymmetric capacity, issue #525/#535).
        self.rot_camera_layers = _axial_stack(
            self.rope_bases[1], self.rot_num_task_layers
        )
        self.rot_time_layers = _axial_stack(
            self.rope_bases[0], self.rot_num_task_layers
        )
        self.pose_camera_layers = _axial_stack(
            self.rope_bases[1], self.pose_num_task_layers
        )
        self.pose_time_layers = _axial_stack(
            self.rope_bases[0], self.pose_num_task_layers
        )
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
        self._pose_branch_input = (
            self._detach_pose_branch
            if self.detach_pose_branch
            else self._share_pose_branch
        )
        self._canonical_feature = (
            self._rotation_canonical_feature
            if self.canonical_on_rotation_branch
            else self._pose_canonical_feature
        )
        output_profile = (
            self.predict_canonical_pose,
            self.aux_position_on_rotation_branch,
        )
        self._decode_split_outputs = {
            (False, False): self._decode_split_outputs_basic,
            (True, False): self._decode_split_outputs_with_canonical_pose,
            (False, True): self._decode_split_outputs_with_auxiliary_position,
            (True, True): self._decode_split_outputs_with_all_heads,
        }[output_profile]

    @classmethod
    def from_config(
        cls, config: PLCSModelConfig, *, num_court_tokens: int
    ) -> PLCSMultiViewAxialSplitModel:
        """Create the split model from a hydra config."""
        num_task_layers = config.integer("num_task_layers")
        return cls(
            hidden_dim=config.integer("hidden_dim"),
            num_layers=config.integer("num_layers"),
            num_task_layers=num_task_layers,
            rot_num_task_layers=config.integer("rot_num_task_layers"),
            pose_num_task_layers=config.integer("pose_num_task_layers"),
            canonical_on_rotation_branch=config.boolean("canonical_on_rotation_branch"),
            aux_position_on_rotation_branch=config.boolean(
                "aux_position_on_rotation_branch"
            ),
            detach_pose_branch=config.boolean("detach_pose_branch"),
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
            x_camera = camera_layer(
                x_camera, freqs_cis=camera_freqs, attn_mask=camera_mask
            )
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
        human_vis: Tensor,
        human_mask: Tensor,
        court_vis: Tensor,
        camera_attention_mask: Tensor,
        time_attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass with shared encoder + separate rotation/pose trunks."""
        batch_size, n_cams, seq_len_in = human_kp.shape[:3]

        human_kp = human_kp * (human_vis > 0).unsqueeze(-1).to(dtype=human_kp.dtype)
        court_kp = court_kp * (court_vis > 0).unsqueeze(-1).to(dtype=court_kp.dtype)

        token_valid = human_mask > 0

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

        camera_freqs = self._camera_freqs(
            batch_size=batch_size, seq_len=seq_len_in, n_cams=n_cams
        )
        time_freqs = self._time_freqs(
            batch_size=batch_size, seq_len=seq_len_in, n_cams=n_cams
        )

        # Optional shared trunk (num_layers may be 0 -> identity), then split.
        x = self._run_axial_stack(
            x,
            self.camera_layers,
            self.time_layers,
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
            camera_freqs=camera_freqs,
            time_freqs=time_freqs,
            camera_mask=camera_attention_mask,
            time_mask=time_attention_mask,
        )
        x_rot = self._run_axial_stack(
            x,
            self.rot_camera_layers,
            self.rot_time_layers,
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
            camera_freqs=camera_freqs,
            time_freqs=time_freqs,
            camera_mask=camera_attention_mask,
            time_mask=time_attention_mask,
        )
        pose_input = self._pose_branch_input(x)
        x_pose = self._run_axial_stack(
            pose_input,
            self.pose_camera_layers,
            self.pose_time_layers,
            batch_size=batch_size,
            seq_len=seq_len_in,
            n_cams=n_cams,
            camera_freqs=camera_freqs,
            time_freqs=time_freqs,
            camera_mask=camera_attention_mask,
            time_mask=time_attention_mask,
        )

        rot_feat = self.rot_final_norm(x_rot[:, :, 0, :])
        pose_feat = self.pose_final_norm(x_pose[:, :, 0, :])

        return self._decode_split_outputs(rot_feat, pose_feat)

    def _decode_split_outputs_basic(
        self, rot_feat: Tensor, pose_feat: Tensor
    ) -> dict[str, Tensor]:
        return {
            "position": self.position_head(pose_feat),
            "rotation": self.rotation_head(rot_feat),
        }

    def _decode_split_outputs_with_canonical_pose(
        self, rot_feat: Tensor, pose_feat: Tensor
    ) -> dict[str, Tensor]:
        output = self._decode_split_outputs_basic(rot_feat, pose_feat)
        canonical_head = cast(CanonicalPoseHead, self.canonical_pose_head)
        output["canonical_pose"] = canonical_head(
            self._canonical_feature(rot_feat, pose_feat)
        )
        return output

    def _decode_split_outputs_with_auxiliary_position(
        self, rot_feat: Tensor, pose_feat: Tensor
    ) -> dict[str, Tensor]:
        output = self._decode_split_outputs_basic(rot_feat, pose_feat)
        auxiliary_head = cast(PositionHead, self.aux_position_head)
        output["aux_position"] = auxiliary_head(rot_feat)
        return output

    def _decode_split_outputs_with_all_heads(
        self, rot_feat: Tensor, pose_feat: Tensor
    ) -> dict[str, Tensor]:
        output = self._decode_split_outputs_with_canonical_pose(rot_feat, pose_feat)
        auxiliary_head = cast(PositionHead, self.aux_position_head)
        output["aux_position"] = auxiliary_head(rot_feat)
        return output

    @staticmethod
    def _detach_pose_branch(x: Tensor) -> Tensor:
        return x.detach()

    @staticmethod
    def _share_pose_branch(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def _rotation_canonical_feature(rot_feat: Tensor, pose_feat: Tensor) -> Tensor:
        del pose_feat
        return rot_feat

    @staticmethod
    def _pose_canonical_feature(rot_feat: Tensor, pose_feat: Tensor) -> Tensor:
        del rot_feat
        return pose_feat
