"""Trainable projections of four frozen DINO features to a fixed downstream width."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class CourtFeatureAdapter(nn.Module):
    """Independent pointwise linear maps; no spatial resizing or normalization."""

    def __init__(self, input_channels: Sequence[int], output_channels: int) -> None:
        super().__init__()
        if len(input_channels) != 4 or min(input_channels) <= 0 or output_channels <= 0:
            raise ValueError(
                "Feature adapter requires four positive input widths and a positive output width"
            )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(channels, output_channels, kernel_size=1)
                for channels in input_channels
            ]
        )

    def forward(
        self,
        features: tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if any(feature is None for feature in features):
            raise ValueError("Feature adapter requires all four DINO feature maps")
        outputs = [
            projection(feature)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        return outputs[0], outputs[1], outputs[2], outputs[3]
