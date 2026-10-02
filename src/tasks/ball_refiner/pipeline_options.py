"""Explicit, pinned ball-path candidates; never discover or select runtime assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.tasks.ball_refiner.deployment import InferenceBundle
from src.tasks.ball_refiner.refiner_2d.calibration import (
    CovarianceCalibration,
    load_covariance_calibration,
)


@dataclass(frozen=True)
class BallPathIdentity:
    detector_sha256: str
    checkpoint_sha256: str
    bundle_manifest_sha256: str
    calibration_sha256: str


E9_ANCHORED_S42 = BallPathIdentity(
    detector_sha256="37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36",
    checkpoint_sha256="985308b02d4b1b33c3bcfb40dbf2a846b900e730ee6d5dfc07c3a0ed561cd36c",
    bundle_manifest_sha256="39cf3f2a7e8e4a1e5434b3d0174b432074bd97fef8be2921c49d82893da6ea7a",
    calibration_sha256="197f9e6490e6c91f2ff699de27b490df2669c4ff79ea599d4dd8b484d9e122f5",
)
BALL_PATHS = ("bundle", "e9_anchored_s42_covariance")


@dataclass(frozen=True)
class CalibrationArtifact:
    path: Path
    sha256: str
    checkpoint_sha256: str

    def load(self) -> CovarianceCalibration:
        if not self.path.is_file():
            raise FileNotFoundError(f"Required covariance calibration artifact is missing: {self.path}")
        return load_covariance_calibration(
            self.path, expected_sha256=self.sha256, checkpoint_sha256=self.checkpoint_sha256,
        )


def select_ball_path(
    name: str, bundle: InferenceBundle, calibration_artifact: Path | None,
) -> CalibrationArtifact | None:
    """The manifest pins exported weights, model/input config and checkpoint provenance."""
    if name == "bundle":
        if calibration_artifact is not None:
            raise ValueError("The bundle ball path is uncalibrated; explicitly select a calibrated ball path")
        return None
    if name != "e9_anchored_s42_covariance":
        raise ValueError(f"Unknown ball path: {name}")
    if calibration_artifact is None:
        raise ValueError("e9_anchored_s42_covariance requires an explicit covariance calibration artifact")
    expected = E9_ANCHORED_S42
    if (bundle.manifest_sha256 != expected.bundle_manifest_sha256
            or bundle.detector.checkpoint_sha256 != expected.detector_sha256):
        raise ValueError("e9_anchored_s42_covariance bundle/detector SHA256 mismatch")
    artifact = CalibrationArtifact(calibration_artifact, expected.calibration_sha256, expected.checkpoint_sha256)
    artifact.load()  # Fail before either execute or load-only can access the store.
    return artifact
