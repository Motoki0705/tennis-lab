"""Input augmentation settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorruptionConfig:
    event_probability: float
    isolated_probability: float
    gap_min: int
    gap_max: int
    noise_p95_px: float
    jitter_sigma_px: float
    outlier_probability: float
    triangulation_steps: int

    def __post_init__(self) -> None:
        if (
            not 0 <= self.event_probability <= 1
            or not 0 <= self.isolated_probability <= 1
        ):
            raise ValueError("Missingness probabilities must be in [0,1]")
        if not 1 <= self.gap_min < self.gap_max:
            raise ValueError("Require distinct positive left/right gap widths")
        if self.noise_p95_px == 0:
            if self.jitter_sigma_px != 0 or self.outlier_probability != 0:
                raise ValueError(
                    "Zero noise requires zero jitter and zero outlier probability"
                )
        elif not 0.05 < self.outlier_probability < 1:
            raise ValueError(
                "Outlier probability must exceed 5% for the specified P95 mixture"
            )
        elif not 0 <= self.jitter_sigma_px < self.noise_p95_px / 5:
            raise ValueError("Jitter must be small relative to the positive noise P95")
        if self.triangulation_steps < 0:
            raise ValueError("triangulation_steps must be nonnegative")
