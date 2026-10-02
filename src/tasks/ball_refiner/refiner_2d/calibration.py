"""Explicit checkpoint-bound covariance scaling; no implicit calibration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.utils.checksum import dual_sha256


@dataclass(frozen=True)
class CovarianceCalibration:
    covariance_multiplier: float
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if (isinstance(self.covariance_multiplier, bool)
                or not math.isfinite(self.covariance_multiplier) or self.covariance_multiplier <= 0):
            raise ValueError("Covariance multiplier must be finite and positive")
        if re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256) is None:
            raise ValueError("Calibration requires a checkpoint SHA256")

    def apply(self, prediction: BallGMM2D) -> BallGMM2D:
        """Scale Sigma by s (L by sqrt(s)), retaining every mean and logit."""
        return BallGMM2D(
            means=prediction.means,
            scale_tril=prediction.scale_tril * math.sqrt(self.covariance_multiplier),
            mixture_logits=prediction.mixture_logits,
            presence_logits=prediction.presence_logits,
        )


def load_covariance_calibration(
    path: Path, *, expected_sha256: str, checkpoint_sha256: str,
) -> CovarianceCalibration:
    """Require both an explicit artifact digest and the loaded checkpoint identity."""
    if dual_sha256(path) != expected_sha256:
        raise ValueError("Covariance calibration SHA256 mismatch")
    raw = json.loads(path.read_text())
    if (set(raw) != {"schema", "covariance_multiplier", "checkpoint_sha256", "provenance"}
            or raw["schema"] != "ball_refiner_2d.covariance_calibration.v1"
            or not isinstance(raw["provenance"], dict) or not raw["provenance"]):
        raise ValueError("Unknown/incomplete covariance calibration artifact")
    if raw["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Covariance calibration checkpoint mismatch")
    if type(raw["covariance_multiplier"]) not in (float, int):
        raise ValueError("Covariance multiplier must be numeric")
    return CovarianceCalibration(raw["covariance_multiplier"], raw["checkpoint_sha256"])
