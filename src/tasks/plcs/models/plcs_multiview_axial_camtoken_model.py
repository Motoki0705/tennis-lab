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
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
        camera_attention_mask: Tensor,
        time_attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass: pose reads camera-0 token, rotation reads camera-1 token."""
        batch_size, n_cams, seq_len_in = human_kp.shape[:3]

        human_kp = human_kp * (human_vis > 0).unsqueeze(-1).to(dtype=human_kp.dtype)
        court_kp = court_kp * (court_vis > 0).unsqueeze(-1).to(dtype=court_kp.dtype)

        token_valid = ~padding_mask

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
                attn_mask=camera_attention_mask,
            )
            x = x_camera.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim)

            x_time = x.permute(0, 2, 1, 3).reshape(
                batch_size * n_cams, seq_len_in, self.hidden_dim
            )
            x_time = time_layer(
                x_time,
                freqs_cis=time_freqs,
                attn_mask=time_attention_mask,
            )
            x = x_time.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim).permute(
                0, 2, 1, 3
            )

        # x: (B, T, N, hidden). Read a distinct camera token per task. Each token
        pose_feat = self.final_norm(x[:, :, self.POSE_CAM_IDX, :])
        rot_feat = self.final_norm(x[:, :, self.ROT_CAM_IDX, :])

        # Mirror the split-model EX10 recipe: canonical geometry rides the
        # rotation readout when that head was selected during construction.
        return self._decode_readouts(pose_feat, rot_feat)
