"""Camera-local, row-addressed inference artifacts, separate from CPU tracking."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import Pose2DFrameSequenceRequest, ViTPosePose2D
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import (
    DEFAULT_CONFIG,
    AppearanceCrop,
    FeatureConfig,
    UnpromptedEncoder,
    appearance_batches,
    infer_appearance,
    iter_appearance_crops,
    pose_boxes,
    scatter_appearance,
)
from src.tasks.person_tracking.sequence import CLIP_ENCODER
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.model_resources import vitpose_model
from src.utils.video import OpenCVVideoFrameReader

CLIP_DIMENSION = 1280


def _validate_axis(camera_id: str, offsets: NDArray[np.int64], rows: NDArray[np.int64]) -> None:
    if not camera_id or offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype != np.int64 \
            or offsets[0] != 0 or offsets[-1] != len(rows) or (np.diff(offsets) < 0).any() \
            or rows.ndim != 1 or rows.dtype != np.int64 or (rows < 0).any() or (np.diff(rows) <= 0).any():
        raise ValueError('Feature artifacts require a complete timeline and unique increasing source rows')


@dataclass(frozen=True)
class PersonPoseFeaturesOutput:
    camera_id: str
    frame_offsets: NDArray[np.int64]
    source_rows: NDArray[np.int64]
    keypoints: NDArray[np.float32]  # N,17,3; source pixels and raw heatmap peaks

    def __post_init__(self) -> None:
        _validate_axis(self.camera_id, self.frame_offsets, self.source_rows)
        if self.keypoints.shape != (len(self.source_rows), 17, 3) or self.keypoints.dtype != np.float32 \
                or not np.isfinite(self.keypoints).all():
            raise ValueError('Pose features require finite float32 COCO17 values for every source row')


@dataclass(frozen=True)
class PersonAppearanceFeaturesOutput:
    camera_id: str
    frame_offsets: NDArray[np.int64]
    source_rows: NDArray[np.int64]
    embeddings: NDArray[np.float32]  # N,1280; masked zero
    appearance_valid: NDArray[np.bool_]
    encoder: str
    encoder_identity: str

    def __post_init__(self) -> None:
        _validate_axis(self.camera_id, self.frame_offsets, self.source_rows)
        if not isinstance(self.encoder_identity, str) or len(self.encoder_identity) != 64 or any(c not in '0123456789abcdef' for c in self.encoder_identity):
            raise ValueError('Appearance features require an explicit content/settings identity')
        n = len(self.source_rows)
        if self.encoder != CLIP_ENCODER or self.embeddings.shape != (n, CLIP_DIMENSION) \
                or self.embeddings.dtype != np.float32 or not np.isfinite(self.embeddings).all() \
                or self.appearance_valid.shape != (n,) or self.appearance_valid.dtype != np.bool_:
            raise ValueError('Appearance features require the explicit CLIP profile and aligned float32 embeddings')
        if (self.embeddings[~self.appearance_valid] != 0).any() \
                or not np.allclose(np.linalg.norm(self.embeddings[self.appearance_valid], axis=1), 1., atol=1e-4):
            raise ValueError('Appearance features require normalized valid rows and masked zeros')


def require_alignment(video: SourceVideo, detections: PersonDetectionOutput,
                      *features: PersonPoseFeaturesOutput | PersonAppearanceFeaturesOutput) -> NDArray[np.int64]:
    if detections.camera_id != video.camera_id or len(detections.frame_offsets) != video.num_frames + 1:
        raise ValueError('Feature input camera/timeline mismatch')
    if detections.source_rows is None:
        raise ValueError('Features require person_detections v2 source rows; regenerate the artifact')
    for item in features:
        if item.camera_id != video.camera_id or not np.array_equal(item.frame_offsets, detections.frame_offsets) \
                or not np.array_equal(item.source_rows, detections.source_rows):
            raise ValueError('Feature camera/timeline/source row mismatch; regenerate upstream artifacts')
    return detections.source_rows


@dataclass(frozen=True)
class PersonFeaturesInput:
    video: SourceVideo
    detections: PersonDetectionOutput


@dataclass(frozen=True)
class PersonAppearanceFeaturesInput:
    video: SourceVideo
    detections: PersonDetectionOutput
    poses: PersonPoseFeaturesOutput


def _frames(video: SourceVideo) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    count = 0
    for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
        if packet.index != count:
            raise ValueError('Feature video frame order changed')
        if packet.frame.shape != (video.height, video.width, 3) or packet.frame.dtype != np.uint8:
            raise ValueError('Feature video frame size or dtype changed')
        yield count, packet.frame
        count += 1
    if count != video.num_frames:
        raise ValueError('Feature pass did not decode the complete source timeline')


class PersonPoseFeaturesModule(InferenceComponent[PersonFeaturesInput, PersonPoseFeaturesOutput]):
    io = ComponentIO('person_pose_features', PersonFeaturesInput, PersonPoseFeaturesOutput,
                     {'detections': InputPort('person_detections', 2)}, 'person_pose_features', version=1)

    def __init__(self, people: PeopleModelConfig, config: FeatureConfig = DEFAULT_CONFIG, *, enabled: bool = True) -> None:
        self.people, self.config, self.enabled = people, config, enabled

    def model_specs(self) -> dict[str, ModelSpec[ViTPosePose2D]]:
        return {'pose': vitpose_model(self.people, factory=ViTPosePose2D)} if self.enabled else {}

    def process_with_models(self, inputs: PersonFeaturesInput, models: ModelBorrower) -> PersonPoseFeaturesOutput:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections)
        poses: NDArray[np.float32] = np.zeros((len(rows), 17, 3), np.float32)
        if not self.enabled and len(rows):
            raise ValueError('Disabled pose features require empty detections')
        if len(rows):
            if (detections.boxes_xyxy[:, 2:] <= detections.boxes_xyxy[:, :2]).any():
                raise ValueError('Feature boxes must have positive area')
            # Only boxes/indices and numeric outputs span the camera. The model's
            # shared crop iterator retains one decoded image and one crop batch.
            import torch
            indices: NDArray[np.int64] = np.repeat(np.arange(video.num_frames, dtype=np.int64), np.diff(detections.frame_offsets))
            frames = _frames(video)
            model: ViTPosePose2D = models.borrow('pose')
            result = model.predict(Pose2DFrameSequenceRequest(frames, pose_boxes(detections.boxes_xyxy, self.config),
                                                            torch.from_numpy(indices)))
            # ViTPose stops at the last requested detection. Validate the rest of
            # the camera too, including empty trailing frames, without retaining RGB.
            for _ in frames:
                pass
            if result.keypoints.shape != poses.shape:
                raise ValueError('Pose model changed the detection row axis')
            poses = result.keypoints.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(poses).all():
                bad = np.argwhere(~np.isfinite(poses))
                details = [(int(indices[i]), int(rows[i]), int(j), int(k), str(poses[i, j, k])) for i, j, k in bad]
                raise ValueError(f'Nonfinite pose: (frame, detection row, joint, channel, value)={details}')
        return PersonPoseFeaturesOutput(video.camera_id, detections.frame_offsets, rows, poses)


class PersonAppearanceFeaturesModule(InferenceComponent[PersonAppearanceFeaturesInput, PersonAppearanceFeaturesOutput]):
    io = ComponentIO('person_appearance_features', PersonAppearanceFeaturesInput, PersonAppearanceFeaturesOutput,
                     {'detections': InputPort('person_detections', 2), 'poses': InputPort('person_pose_features', 1)},
                     'person_appearance_features', version=2)

    def __init__(self, encoder: ModelSpec[AppearanceEncoder] | None, config: FeatureConfig = DEFAULT_CONFIG, *,
                 enabled: bool = True, batch_size: int | None = None) -> None:
        from dataclasses import replace
        self.encoder = encoder
        self.config = config if batch_size is None else replace(config, appearance_batch_size=batch_size)
        self.enabled = enabled
        if enabled and encoder is None:
            raise ValueError('Appearance features require an explicit encoder declaration')

    def model_specs(self) -> dict[str, ModelSpec[AppearanceEncoder]]:
        return {'encoder': self.encoder} if self.enabled and self.encoder is not None else {}

    def process_with_models(self, inputs: PersonAppearanceFeaturesInput, models: ModelBorrower) -> PersonAppearanceFeaturesOutput:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections, inputs.poses)
        from src.tennis_scene.pipeline.artifacts import document_digest
        identity = models.identity('encoder') if self.enabled else document_digest({'enabled': False})
        embeddings: NDArray[np.float32] = np.zeros((len(rows), CLIP_DIMENSION), np.float32)
        valid: NDArray[np.bool_] = np.zeros(len(rows), bool)
        if not self.enabled and len(rows):
            raise ValueError('Disabled appearance features require empty detections')
        if len(rows):
            model: AppearanceEncoder = models.borrow('encoder')
            if model.name != CLIP_ENCODER:
                raise ValueError('Appearance encoder differs from the configured CLIP profile')
            encoder = UnpromptedEncoder(model, CLIP_DIMENSION)
            row_indices = {int(row): i for i, row in enumerate(rows)}
            def crops() -> Iterator[AppearanceCrop]:
                for frame, image in _frames(video):
                    start, end = detections.frame_offsets[frame:frame + 2]
                    # Same input validation as FeatureExtractor, before cropping.
                    DetectionFeatures(frame, rows[start:end], detections.boxes_xyxy[start:end], detections.confidence[start:end],
                        inputs.poses.keypoints[start:end], embeddings[start:end], valid[start:end])
                    yield from iter_appearance_crops(frame, image, rows[start:end], detections.boxes_xyxy[start:end],
                                                     None, encoder, self.config)
            for batch in appearance_batches(crops(), self.config.appearance_batch_size):
                scatter_appearance(batch, infer_appearance(batch, encoder), row_indices, embeddings, valid)
        return PersonAppearanceFeaturesOutput(video.camera_id, detections.frame_offsets, rows, embeddings, valid, CLIP_ENCODER, identity)
