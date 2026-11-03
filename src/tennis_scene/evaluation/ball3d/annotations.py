"""Thin scoring-array adapter for the shared, strict user-reference reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ground_truth import (
    ReferenceIdentity,
    ReferenceSelection,
    UserReference,
    load_user_reference,
)
from src.tennis_scene.pipeline.artifacts import json_value

from .bundle import FixedInputBundle
from .contracts import MethodInput


@dataclass(frozen=True)
class Ball2DAnnotations:
    """Scoring only. Missing coordinates remain NaN and never become observations.

    JSON/source/provenance validation belongs exclusively to ground_truth.
    Keep the original reference, including every event row and estimation segment.
    """

    reference: UserReference
    selection: ReferenceSelection = field(init=False)

    def __post_init__(self) -> None:
        identity = self.reference.identity
        object.__setattr__(
            self,
            "selection",
            self.reference.select(
                camera_ids=identity.camera_ids, frame_indices=range(identity.num_frames)
            ),
        )

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return cast(tuple[str, ...], self.selection.camera_ids)

    @property
    def uv_px(self) -> NDArray[np.float64]:
        return self.selection.uv_px

    @property
    def valid_for_scoring(self) -> NDArray[np.bool_]:
        return self.selection.valid_for_scoring

    @property
    def reference_kind(self) -> NDArray[np.str_]:
        return self.selection.reference_kind

    @property
    def observed(self) -> NDArray[np.bool_]:
        return self.valid_for_scoring & (self.reference_kind == "observed")

    @property
    def estimated(self) -> NDArray[np.bool_]:
        return self.valid_for_scoring & (self.reference_kind == "estimated")

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            **json_value(self.reference.metadata["provenance"]),
            "file_sha256": self.reference.sha256,
        }

    def state_counts(self) -> dict[str, dict[str, int]]:
        return {
            camera: {
                str(kind): int((self.reference_kind[i] == kind).sum())
                for kind in np.unique(self.reference_kind[i])
            }
            for i, camera in enumerate(self.camera_ids)
        }

    def validate_for_inputs(self, inputs: MethodInput) -> None:
        identity = self.reference.identity
        if (
            identity.clip_id != inputs.clip_id
            or identity.num_frames != inputs.num_frames
            or self.camera_ids != inputs.camera_ids
            or not np.isclose(identity.fps, inputs.fps, atol=1e-6, rtol=0)
            or identity.source_size_wh != inputs.source_size_wh
            or self.selection.frame_indices != tuple(inputs.frame_indices)
        ):
            raise ValueError("Scoring annotation camera/frame/coordinate axes disagree")

    def validate_for_bundle(self, bundle: FixedInputBundle) -> None:
        """Scoring must bind the reference to trusted video bytes, not names alone."""
        self.validate_for_inputs(bundle.inputs)
        if self.reference.identity.video_sha256 != {
            video.camera_id: video.sha256 for video in bundle.source.videos
        }:
            raise ValueError("Scoring annotation video hash identity disagrees")


def load_annotations(
    path: Path, bundle: FixedInputBundle, *, expected_sha256: str
) -> Ball2DAnnotations:
    """Bind reader identity to the verified bundle, never to the labels themselves."""
    inputs = bundle.inputs
    expected = ReferenceIdentity(
        inputs.clip_id,
        inputs.camera_ids,
        inputs.num_frames,
        inputs.fps,
        inputs.source_size_wh,
        {v.camera_id: v.sha256 for v in bundle.source.videos},
    )
    return Ball2DAnnotations(
        load_user_reference(path, expected=expected, expected_sha256=expected_sha256)
    )
