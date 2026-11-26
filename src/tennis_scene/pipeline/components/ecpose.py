"""Camera-local ECPose inference and views with the existing person schemas."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models.ecpose.inference import decode_predictions, preprocess_images
from src.submodules.models.ecpose.releases import IMAGE_SIZE, SOURCE_COMMIT
from src.submodules.models.ecpose.runtime import (
    INPUT_NAME,
    ECPoseConfig,
    EdgeCrafterInference,
    load_backend,
    read_engine,
)
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionInput,
    PersonDetectionOutput,
)
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
    iter_source_frames,
    require_alignment,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import file_identity
from src.tennis_scene.pipeline.recipes import ECPOSE_RECIPE
from src.utils.device import resolve_device
from src.utils.inference_recording import span

ECPOSE_PORT = InputPort("person_detection_and_pose", 1)


@dataclass(frozen=True)
class ECPoseOutput:
    detections: PersonDetectionOutput
    poses: PersonPoseFeaturesOutput

    def __post_init__(self) -> None:
        if (
            self.detections.camera_id != self.poses.camera_id
            or not np.array_equal(
                self.detections.frame_offsets, self.poses.frame_offsets
            )
            or not np.array_equal(self.detections.source_rows, self.poses.source_rows)
        ):
            raise ValueError(
                "ECPose detection and pose must share camera, frames and rows"
            )


class ECPoseModule(InferenceComponent[PersonDetectionInput, ECPoseOutput]):
    io = ComponentIO(
        ECPOSE_RECIPE,
        PersonDetectionInput,
        ECPoseOutput,
        {},
        ECPOSE_PORT.schema,
        version=ECPOSE_PORT.version,
    )

    def __init__(self, config: ECPoseConfig) -> None:
        self.config = config

    def cache_identity(self) -> dict[str, Any]:
        return {
            "scope": "full_frame",
            "image_size_hw": IMAGE_SIZE,
            "box": "in_image_coco17_envelope",
            "joint_confidence": "in_image_validity",
            "confidence_exclusive": self.config.confidence,
            "box_margin": self.config.box_margin,
        }

    @cached_property
    def spec(self) -> ModelSpec[EdgeCrafterInference]:
        config = self.config
        settings = {
            **self.cache_identity(),
            "upstream_commit": SOURCE_COMMIT,
            "backend": config.backend,
            "batch_size": config.batch_size,
        }
        assets = {"checkpoint": config.checkpoint, **config.source_assets}
        if config.engine is not None:

            def digest(path: Path) -> str:
                return str(file_identity(path)["sha256"])

            bundle = read_engine(config.engine, config.model_id, digest=digest)
            profile = bundle.engine.profiles[INPUT_NAME]
            for count in (1, config.batch_size):
                profile.require((count, 3, *IMAGE_SIZE), tensor=INPUT_NAME)
            assets.update({f"engine/{path.name}": path for path in bundle.files})
            settings["engine"] = bundle.engine.model_dump(mode="json")
        return ModelSpec(
            kind=ECPOSE_RECIPE,
            weights=assets,
            architecture=config.model_id,
            device=config.device,
            dtype="float32" if config.backend == "pt32" else "float16",
            settings=settings,
            factory=self._load,
        )

    def model_specs(self) -> dict[str, ModelSpec[EdgeCrafterInference]]:
        return {ECPOSE_RECIPE: self.spec}

    def _load(self) -> EdgeCrafterInference:
        config = self.config
        return load_backend(
            config.repository,
            config.checkpoint,
            config.model_id,
            config.backend,
            resolve_device(config.device),
            config.engine,
        )

    def process_with_models(
        self, inputs: PersonDetectionInput, models: ModelBorrower
    ) -> ECPoseOutput:
        video, config = inputs.video, self.config
        infer: EdgeCrafterInference = models.borrow(ECPOSE_RECIPE)
        offsets = [0]
        boxes: list[NDArray[np.float32]] = []
        scores: list[NDArray[np.float32]] = []
        poses: list[NDArray[np.float32]] = []
        with closing(iter_source_frames(video)) as frames, torch.inference_mode():
            while batch := list(islice(frames, config.batch_size)):
                with span("preprocess", model=ECPOSE_RECIPE):
                    pixels, sizes = preprocess_images([frame for _, frame in batch])
                with span("transfer", model=ECPOSE_RECIPE, operation="to_device"):
                    pixels, sizes = pixels.to(config.device), sizes.to(config.device)
                with span("inference", model=ECPOSE_RECIPE):
                    output = infer(pixels)
                with span("postprocess", model=ECPOSE_RECIPE):
                    rows = decode_predictions(
                        config.release,
                        output,
                        sizes,
                        confidence=config.confidence,
                        pose_box_margin=config.box_margin,
                    )
                with span("transfer", model=ECPOSE_RECIPE, operation="to_host"):
                    for row in rows:
                        if row.keypoints_xyc is None:
                            raise ValueError(
                                "ECPose is missing aligned COCO17 keypoints"
                            )
                        boxes.append(row.boxes_xyxy.cpu().numpy())
                        scores.append(row.scores.cpu().numpy())
                        poses.append(row.keypoints_xyc.cpu().numpy())
                        offsets.append(offsets[-1] + len(row.scores))
        frame_offsets = np.asarray(offsets, dtype=np.int64)
        source_rows: NDArray[np.int64] = np.arange(offsets[-1], dtype=np.int64)
        detection = PersonDetectionOutput(
            video.camera_id,
            frame_offsets,
            np.concatenate(boxes),
            np.concatenate(scores),
            source_rows,
        )
        pose = PersonPoseFeaturesOutput(
            video.camera_id,
            frame_offsets,
            source_rows,
            np.concatenate(poses),
            models.identity(ECPOSE_RECIPE),
            document_digest(self.cache_identity()),
        )
        require_alignment(video, detection, pose)
        return ECPoseOutput(detection, pose)


class ECPoseDetectionsModule:
    io = ComponentIO(
        "person_detection",
        ECPoseOutput,
        PersonDetectionOutput,
        {"joint": ECPOSE_PORT},
        "person_detections",
        version=2,
    )

    def process(self, joint: ECPoseOutput) -> PersonDetectionOutput:
        return joint.detections


@dataclass(frozen=True)
class ECPoseFeaturesInput:
    video: SourceVideo
    detections: PersonDetectionOutput
    joint: ECPoseOutput


class ECPoseFeaturesModule:
    io = ComponentIO(
        "person_pose_features",
        ECPoseFeaturesInput,
        PersonPoseFeaturesOutput,
        {"joint": ECPOSE_PORT, "detections": InputPort("person_detections", 2)},
        "person_pose_features",
        version=2,
    )

    def process(self, inputs: ECPoseFeaturesInput) -> PersonPoseFeaturesOutput:
        require_alignment(inputs.video, inputs.detections, inputs.joint.poses)
        expected = inputs.joint.detections
        if not np.array_equal(
            inputs.detections.boxes_xyxy, expected.boxes_xyxy
        ) or not np.array_equal(inputs.detections.confidence, expected.confidence):
            raise ValueError("Pose features must use the same ECPose detections")
        return inputs.joint.poses
