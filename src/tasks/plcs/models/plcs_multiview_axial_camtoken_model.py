"""Per-camera readout-token PLCS axial model (issue #576).

``PLCSMultiViewAxialCamTokenModel`` keeps the *shared* trunk of
:class:`PLCSMultiViewAxialModel` untouched and changes **only the readout**: the
pose (position) head reads the camera-0 token while the rotation head reads the
camera-1 token, instead of both heads reading the same camera-0 token.

Why (issues #545/#560):
    On a single shared trunk, position and rotation compete for capacity because
    both heads read the *same* readout vector (``x[:, :, 0, :]``); the dominant
    gradient owns that feature, so the fully-shared trunk loses position
    (#560 ``run-i560-nocanon-s0-h12``: position ~0.283 m). Separate trunks
    (``PLCSMultiViewAxialSplitModel``) remove the competition but cost a second
    trunk.

    After the alternating camera/time axial stack, **every camera token has
    already attended across all cameras and all time steps**, so the camera-0 and
    camera-1 tokens are equally valid "summary" tokens for the world-coordinate,
    per-time targets (position / rotation do not depend on which camera token we
    read). Handing each head a *distinct* camera token lets the shared trunk
    allocate different readout features per task -- decoupling them cheaply,
    without a second trunk.

This subclass overrides ``forward`` only; ``__init__``, the heads, and
``from_config`` are inherited unchanged so the model is config-compatible with
``plcs_multiview_axial`` apart from ``model.name``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.utils.schema.player import NUM_HUMAN_KP


class PLCSMultiViewAxialCamTokenModel(PLCSMultiViewAxialModel):
    """Shared-trunk axial PLCS model with per-camera readout tokens."""

    #: Readout camera index for the pose (position) head.
    POSE_CAM_IDX = 0
    #: Readout camera index for the rotation head.
    ROT_CAM_IDX = 1

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass: pose reads camera-0 token, rotation reads camera-1 token."""
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

        # x: (B, T, N, hidden). Read a distinct camera token per task. Each token
        # has already attended across all cameras/time, so both are valid summary
        # tokens for the camera-invariant per-time targets. Fall back to camera 0
        # for the rotation head when only a single view is present.
        rot_cam_idx = (
            self.ROT_CAM_IDX if n_cams > self.ROT_CAM_IDX else self.POSE_CAM_IDX
        )
        pose_feat = self.final_norm(x[:, :, self.POSE_CAM_IDX, :])
        rot_feat = self.final_norm(x[:, :, rot_cam_idx, :])

        out = {
            "position": self.position_head(pose_feat),
            "rotation": self.rotation_head(rot_feat),
        }
        # Mirror the split-model EX10 recipe: the canonical-pose head (3D geometry
        # regularisation) rides the rotation branch.
        if self.predict_canonical_pose and self.canonical_pose_head is not None:
            out["canonical_pose"] = self.canonical_pose_head(rot_feat)
        return out
