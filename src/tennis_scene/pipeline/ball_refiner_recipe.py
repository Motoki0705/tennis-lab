"""Opt-in detector-only distribution recipe using the shared component store."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from src.tasks.ball_refiner.deployment import (
    CENTRE_SELECTION,
    DetectorRequirements,
    load_inference_bundle,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionInput,
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.components.ball_refiner import BallRefiner2DModule
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
from src.tennis_scene.pipeline.input_assembly.ball_refiner import (
    BallRefiner2DInputAssembler,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    BallDetectionInputAssembler,
)
from src.tennis_scene.pipeline.runner import ComponentNode
from src.utils.checksum import dual_sha256


class RefinerEvidenceModule(BallDetectionModule):
    """Bind actual predictor preprocessing/window length to the training contract."""

    def __init__(self, config: BallDetectionConfig, requirements: DetectorRequirements) -> None:
        super().__init__(config)
        self.requirements = requirements

    def load(self) -> None:
        if dual_sha256(self.config.checkpoint) != self.requirements.checkpoint_sha256:
            raise ValueError("Refiner detector checkpoint checksum mismatch")
        super().load()
        if self._pipeline is None:
            raise RuntimeError("Ball predictor did not load")
        if (self._pipeline.configured_frames != self.requirements.window_length
                or self._pipeline.image_normalization != self.requirements.normalization):
            raise ValueError("Actual detector preprocessing/window contract differs from refiner training")

    def process(self, inputs: BallDetectionInput) -> BallDetectionOutput:
        if dual_sha256(inputs.video.path) != inputs.video.sha256:
            raise ValueError("Refiner source video checksum mismatch")
        output = super().process(inputs)
        if (dual_sha256(inputs.video.path) != inputs.video.sha256
                or dual_sha256(self.config.checkpoint) != self.requirements.checkpoint_sha256):
            raise ValueError("Refiner source or detector changed during inference")
        return output


def ball_refiner_definition(
    source: ClipSource, *, detector_config: BallDetectionConfig, bundle_directory: Path,
    batch_size: int, code_identity: str, execution_source: str,
) -> tuple[ComponentNode, ...]:
    """One independent detector -> refiner chain per camera, no 3D or point fallback.

    This explicitly selected research recipe leaves the standard scene recipe's
    deployment choice to the subsequent full evaluation and probabilistic 3D task.
    """
    if detector_config.device not in {"cpu", "cuda"}:
        raise ValueError("Refiner recipe requires an explicit cpu/cuda device")
    bundle = load_inference_bundle(bundle_directory)
    required = bundle.detector
    if (dual_sha256(detector_config.checkpoint) != required.checkpoint_sha256
            or detector_config.image_size != required.image_size_hw
            or detector_config.normalize_imagenet != required.normalization.enabled
            or detector_config.subpixel_refine != required.subpixel_refine
            or detector_config.candidates != required.candidates
            or detector_config.window_stride != required.stride
            or detector_config.tail_policy != "backfill"
            or detector_config.overlap_aggregation != CENTRE_SELECTION or not detector_config.checkpoint_strict):
        raise ValueError("Pipeline detector settings do not match the refiner's frozen training evidence")
    detector_settings = json_value({
        "config": {field.name: getattr(detector_config, field.name)
                   for field in fields(detector_config) if field.name != "resolver"},
        "requirements": required,
    })
    refiner_settings = json_value({
        "bundle_manifest_sha256": bundle.manifest_sha256, "weights_sha256": bundle.weights_sha256,
        "batch_size": batch_size, "device": detector_config.device,
    })
    nodes: list[ComponentNode] = []
    for camera in source.camera_ids:
        context = AssemblyContext(source, camera)
        detector = RefinerEvidenceModule(detector_config, required)
        refiner = BallRefiner2DModule(bundle, device=detector_config.device, batch_size=batch_size)
        nodes.extend((
            ComponentNode(f"ball_detection/{camera}", detector, detector.io, BallDetectionInputAssembler(), {},
                          context, detector_settings, code_identity, execution_source),
            ComponentNode(f"ball_refiner_2d/{camera}", refiner, refiner.io, BallRefiner2DInputAssembler(bundle),
                          {"detections": f"ball_detection/{camera}"}, context, refiner_settings, code_identity, execution_source),
        ))
    return tuple(nodes)
