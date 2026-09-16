"""Typed contracts for the cross-model court-alignment benchmark.

Every artifact in this package is derived from one of these dataclasses so
that the manifest, the per-sample predictions, and the report tables can be
compared without re-interpreting raw JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

DomainName: TypeAlias = Literal["real_validation", "synthetic_test"]
ModelName: TypeAlias = Literal["ours", "tcd"]

DOMAIN_NAMES: tuple[DomainName, ...] = ("real_validation", "synthetic_test")
MODEL_NAMES: tuple[ModelName, ...] = ("ours", "tcd")

# One channel per ordered TennisCourtDetector keypoint; the synthetic V3
# ``target_court`` contract publishes the same fourteen semantic entries.
KEYPOINT_COUNT = 14

# Strata axes are the only keys a manifest may attach to a sample.  A domain
# fills the axes it owns and leaves the others absent instead of inventing a
# placeholder value.
STRATA_AXES: tuple[str, ...] = ("scene", "coverage_mode", "visible_kp_count")

MANIFEST_SCHEMA = "court_alignment_manifest_v1"
PREDICTION_SCHEMA = "court_alignment_predictions_v1"
METRICS_SCHEMA = "court_alignment_metrics_v1"


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class SampleRef:
    """One validated benchmark sample identity shared by every model.

    A reference is pure identity plus provenance: ground-truth-derived strata
    stay on :class:`LoadedSample` so the manifest cannot drift from the labels
    that are actually scored.
    """

    domain: DomainName
    sample_id: str
    scene_id: str
    trajectory_group_id: str | None
    split: str
    source_target_sha256: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.domain not in DOMAIN_NAMES:
            raise ValueError(f"Unsupported benchmark domain: {self.domain!r}.")
        _require_identifier(self.sample_id, name="sample_id")
        _require_identifier(self.scene_id, name="scene_id")
        if self.trajectory_group_id is not None:
            _require_identifier(self.trajectory_group_id, name="trajectory_group_id")
        _require_identifier(self.split, name="split")
        digest = self.source_target_sha256
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("source_target_sha256 must be a lowercase sha256 digest.")
        _require_positive_int(self.width, name="width")
        _require_positive_int(self.height, name="height")

    @property
    def diagonal_px(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.scene_id, self.trajectory_group_id or self.sample_id)

    def to_json(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "sample_id": self.sample_id,
            "scene_id": self.scene_id,
            "trajectory_group_id": self.trajectory_group_id,
            "split": self.split,
            "source_target_sha256": self.source_target_sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_json(cls, value: object) -> SampleRef:
        if not isinstance(value, Mapping):
            raise ValueError("Manifest sample entries must be mappings.")
        expected = {
            "domain",
            "sample_id",
            "scene_id",
            "trajectory_group_id",
            "split",
            "source_target_sha256",
            "width",
            "height",
        }
        if set(value) != expected:
            raise ValueError("Manifest sample entry fields changed.")
        return cls(
            domain=cast("DomainName", value["domain"]),
            sample_id=cast("str", value["sample_id"]),
            scene_id=cast("str", value["scene_id"]),
            trajectory_group_id=cast("str | None", value["trajectory_group_id"]),
            split=cast("str", value["split"]),
            source_target_sha256=cast("str", value["source_target_sha256"]),
            width=cast("int", value["width"]),
            height=cast("int", value["height"]),
        )


@dataclass(frozen=True, slots=True)
class KeypointPrediction:
    """Fourteen image-space keypoints with an explicit validity mask."""

    keypoints_xy: NDArray[np.float64]  # [14, 2]; NaN where invalid
    scores: NDArray[np.float64]  # [14]
    valid: NDArray[np.bool_]  # [14]

    def __post_init__(self) -> None:
        points = np.asarray(self.keypoints_xy, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if points.shape != (KEYPOINT_COUNT, 2):
            raise ValueError(
                f"Keypoint predictions must have shape ({KEYPOINT_COUNT}, 2), "
                f"got {points.shape}."
            )
        if scores.shape != (KEYPOINT_COUNT,) or valid.shape != (KEYPOINT_COUNT,):
            raise ValueError("Keypoint scores and validity must have shape (14,).")
        if not np.isfinite(scores).all():
            raise ValueError("Keypoint scores must be finite.")
        if np.any(valid & ~np.isfinite(points).all(axis=1)):
            raise ValueError("Valid keypoints must have finite coordinates.")
        object.__setattr__(self, "keypoints_xy", points)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """One model's decoded prediction for one sample."""

    model: ModelName
    keypoints: KeypointPrediction
    elapsed_seconds: float
    extras: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.model not in MODEL_NAMES:
            raise ValueError(f"Unsupported benchmark model: {self.model!r}.")
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("Prediction elapsed_seconds must be finite and >= 0.")
        object.__setattr__(self, "extras", MappingProxyType(dict(self.extras)))


@dataclass(frozen=True, slots=True)
class LoadedSample:
    """Ground truth and RGB for one manifest entry."""

    ref: SampleRef
    image_rgb: NDArray[np.uint8]  # [H, W, 3]
    template_xy: NDArray[np.float64]  # [14, 2] canonical court template per channel
    gt_keypoints_xy: NDArray[np.float64]  # [14, 2]
    gt_visible: NDArray[np.bool_]  # [14]
    strata: Mapping[str, str]

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Loaded sample RGB must be uint8 [H, W, 3].")
        if image.shape[:2] != (self.ref.height, self.ref.width):
            raise ValueError(
                "Loaded sample RGB resolution disagrees with its manifest entry."
            )
        points = np.asarray(self.gt_keypoints_xy, dtype=np.float64)
        template = np.asarray(self.template_xy, dtype=np.float64)
        visible = np.asarray(self.gt_visible, dtype=bool)
        if points.shape != (KEYPOINT_COUNT, 2) or template.shape != (
            KEYPOINT_COUNT,
            2,
        ):
            raise ValueError("Loaded sample keypoints and template must be [14, 2].")
        if visible.shape != (KEYPOINT_COUNT,):
            raise ValueError("Loaded sample visibility must have shape (14,).")
        if not np.isfinite(template).all():
            raise ValueError("Loaded sample court template must be finite.")
        if np.any(visible & ~np.isfinite(points).all(axis=1)):
            raise ValueError("Visible ground-truth keypoints must be finite.")
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(self, "gt_keypoints_xy", points)
        object.__setattr__(self, "template_xy", template)
        object.__setattr__(self, "gt_visible", visible)
        copied = {str(key): str(item) for key, item in self.strata.items()}
        unknown = sorted(set(copied) - set(STRATA_AXES))
        if unknown:
            raise ValueError(f"Unknown strata axes: {unknown}.")
        for key, value in copied.items():
            _require_identifier(value, name=f"strata value for {key!r}")
        object.__setattr__(self, "strata", MappingProxyType(copied))


__all__ = [
    "DOMAIN_NAMES",
    "KEYPOINT_COUNT",
    "MANIFEST_SCHEMA",
    "METRICS_SCHEMA",
    "MODEL_NAMES",
    "PREDICTION_SCHEMA",
    "STRATA_AXES",
    "DomainName",
    "KeypointPrediction",
    "LoadedSample",
    "ModelName",
    "ModelPrediction",
    "SampleRef",
]
