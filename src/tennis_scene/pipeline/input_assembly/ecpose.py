"""Bind the combined camera artifact to the two standard person stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from src.tennis_scene.pipeline.components.ecpose import (
    ECPoseFeaturesInput,
    ECPoseOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext


class ECPoseDetectionsAssembler:
    version = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> ECPoseOutput:
        return cast(ECPoseOutput, artifacts["joint"])


class ECPoseFeaturesAssembler:
    version = 1

    def assemble(
        self, context: AssemblyContext, artifacts: Mapping[str, Any]
    ) -> ECPoseFeaturesInput:
        if context.camera_id is None:
            raise ValueError("ECPose features require a camera scope")
        return ECPoseFeaturesInput(
            context.source.video(context.camera_id),
            artifacts["detections"],
            artifacts["joint"],
        )
