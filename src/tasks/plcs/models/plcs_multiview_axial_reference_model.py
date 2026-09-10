"""Reference-conditioned axial PLCS with an explicit camera readout."""

from __future__ import annotations

import torch
from torch import Tensor

from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel
from src.utils.models import precompute_freqs_cis_nd


class PLCSMultiViewAxialReferenceModel(PLCSMultiViewAxialModel):
    """Use (time, camera, reference-selector) RoPE and the selected view head.

    The paired adapter validates the selector and reference provenance before
    compiled forward. Frequencies are precomputed for every possible reference;
    forward contains no host extraction of the per-sample selector.
    """

    def _register_token_frequencies(self) -> None:
        if self.rope_dim < 6:
            raise ValueError("Axial reference RoPE requires rope_dim >= 6.")
        positions = self._build_token_positions(
            seq_len=self.max_seq_len,
            n_cams=self.max_views,
        )
        reference = torch.arange(self.max_views)[:, None, None]
        cameras = torch.arange(self.max_views)[None, None, :]
        selector = (cameras != reference).expand(-1, self.max_seq_len, -1)
        coordinates = torch.cat(
            (
                positions[None].expand(self.max_views, -1, -1, -1),
                selector[..., None].long(),
            ),
            dim=-1,
        )
        frequencies = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=coordinates,
            base=(*self.rope_bases, self.rope_bases[1]),
        )
        self.register_buffer("token_freqs_cis", frequencies, persistent=False)
        self.register_buffer(
            "_axial_reference_contract_marker", torch.tensor(1, dtype=torch.uint8)
        )

    def forward(  # type: ignore[override]
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        padding_mask: Tensor,
        court_vis: Tensor,
        camera_attention_mask: Tensor,
        time_attention_mask: Tensor,
        reference_view_index: Tensor,
    ) -> dict[str, Tensor]:
        batch, views, frames = human_kp.shape[:3]
        frequencies = self.token_freqs_cis[reference_view_index, :frames, :views]
        camera_freqs = frequencies.reshape(batch * frames, views, 1, self.rope_dim // 2)
        time_freqs = frequencies.permute(0, 2, 1, 3, 4).reshape(
            batch * views,
            frames,
            1,
            self.rope_dim // 2,
        )
        features = self._encode_views(
            human_kp,
            court_kp,
            human_vis,
            padding_mask,
            court_vis,
            camera_attention_mask,
            time_attention_mask,
            camera_freqs,
            time_freqs,
        )
        readout = features.gather(
            2,
            reference_view_index[:, None, None, None].expand(
                batch, frames, 1, self.hidden_dim
            ),
        ).squeeze(2)
        readout = self.final_norm(readout)
        frame_valid = ~padding_mask.all(dim=1)
        return self._decode_readouts(readout, readout, frame_valid)
