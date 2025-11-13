"""Top level scene model module."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from .backbone import ViTBackbone
from .config import SceneConfig
from .decoder import RecurrentTemporalDeformableDecoder
from .head import PredictionHead
from .positional import AbsTimePE, RelTimePE
from .temporal import TemporalEncoderRoPE


class SceneModel(nn.Module):
    """End-to-end model that follows the architecture spec."""

    def __init__(self, cfg: SceneConfig) -> None:
        super().__init__()
        D = cfg.D_model
        self.backbone = ViTBackbone(
            cfg.img_size,
            cfg.patch_size,
            D,
            cfg.vit_depth,
            cfg.vit_heads,
        )
        self.temporal = TemporalEncoderRoPE(D, cfg.temporal_depth, cfg.temporal_heads)
        self.abs_pe = AbsTimePE(cfg.absK, cfg.fps, cfg.abs_Thorizon)
        self.abs_proj = nn.Linear(2 * cfg.absK, D)
        self.rel_pe = RelTimePE(cfg.relQ, cfg.rel_wmin, cfg.rel_wmax)
        self.rel_proj = nn.Linear(2 * cfg.relQ, D)
        self.decoder = RecurrentTemporalDeformableDecoder(
            dim=D,
            num_queries=cfg.num_queries,
            num_layers=cfg.decoder_layers,
            num_points=cfg.num_points,
            num_heads=cfg.decoder_heads,
            k=cfg.window_k,
            offset_mode=cfg.offset_mode,
            tbptt_detach=cfg.tbptt_detach,
            abs_pe=self.abs_pe,
            abs_proj=self.abs_proj,
            rel_pe=self.rel_pe,
            rel_proj=self.rel_proj,
            fps=cfg.fps,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
        )
        self.head = PredictionHead(D, cfg.smpl_param_dim)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        """Run the full scene model pipeline."""
        patch_tokens, cls_tokens = cast(tuple[Tensor, Tensor], self.backbone(frames))
        temporal_tokens = cast(Tensor, self.temporal(cls_tokens))
        decoded = cast(Tensor, self.decoder(patch_tokens, temporal_tokens))
        return cast(dict[str, Tensor], self.head(decoded))
