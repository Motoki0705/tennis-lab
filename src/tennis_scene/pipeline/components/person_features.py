"""Camera-local, row-addressed inference artifacts, separate from CPU tracking."""
from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import Pose2DFrameSequenceRequest, ViTPosePose2D
from src.submodules.models._base.crops import iter_frame_person_crops
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
from src.tennis_scene.pipeline.appearance_media import (
    APPEARANCE_MEDIA,
    appearance_crop_settings,
    require_appearance_media,
    require_current_appearance_crop,
)
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchAddress,
    BatchSource,
    BatchUnit,
    SourceMemory,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig, file_identity
from src.tennis_scene.pipeline.model_resources import model_identities, vitpose_model
from src.tennis_scene.pipeline.storage.clip_store import artifact_memory_bytes
from src.utils.video import OpenCVVideoFrameReader
from src.utils.video.reader import OpenCVDecodeIdentity

CLIP_DIMENSION = 1280
APPEARANCE_FEATURES_PORT = InputPort('person_appearance_features', 4)
_IMAGE_DIGEST_SIZE = 32


def pose_feature_settings(config: FeatureConfig, enabled: bool = True) -> dict[str, Any]:
    return {'bbox_enlarge': config.bbox_enlarge, 'enabled': enabled, 'output_precision': 'float32'}


def appearance_feature_settings(config: FeatureConfig, enabled: bool = True) -> dict[str, Any]:
    return {'min_appearance_height_px': config.min_appearance_height_px,
            'batch_size': config.appearance_batch_size, 'enabled': enabled,
            'crop': appearance_crop_settings(),
            'pose_prompt': False, 'output_precision': 'float32'}


def feature_model_identity(spec: ModelSpec[Any] | None) -> str:
    # Use the signature-checked asset cache, rather than rereading large weights
    # for every camera's CPU tracker. This has exactly ModelBorrower.identity's key.
    return str(document_digest({'enabled': False} if spec is None else
                               spec._key_with(lambda path: file_identity(path)['sha256'])))


def require_feature_identity(model: str | None, preprocessing: str | None,
                             spec: ModelSpec[Any] | None, settings: Mapping[str, Any]) -> None:
    if not matches_feature_identity(model, preprocessing, spec, settings):
        raise ValueError('Stored feature model/preprocessing identity differs from the expected declaration; regenerate features')


def matches_feature_identity(model: str | None, preprocessing: str | None,
                             spec: ModelSpec[Any] | None, settings: Mapping[str, Any]) -> bool:
    return model == feature_model_identity(spec) and preprocessing == document_digest(settings)


def appearance_image_digest(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Hash the decoded BGR pixels. Camera, frame and dimensions are bound separately."""
    return np.frombuffer(sha256(np.ascontiguousarray(image)).digest(), dtype=np.uint8)


def _validate_saved_metadata(payload: Mapping[str, Any], settings: Mapping[str, Any], *,
                             specs: Mapping[str, ModelSpec[Any]], model_field: str,
                             preprocessing: Mapping[str, Any]) -> None:
    # Imports/load-only nodes never borrow a model. Still verify the complete
    # bound declaration, including current asset SHA and dtype, before adoption.
    if json_value(settings.get('models')) != json_value(model_identities(specs, file_identity)):
        raise ValueError('Stored feature requires the expected bound ModelSpec declaration')
    require_feature_identity(payload.get(model_field), payload.get('preprocessing_identity'),
                             next(iter(specs.values()), None), preprocessing)


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


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
    model_identity: str
    preprocessing_identity: str

    def __post_init__(self) -> None:
        _validate_axis(self.camera_id, self.frame_offsets, self.source_rows)
        if not _is_digest(self.model_identity) or not _is_digest(self.preprocessing_identity):
            raise ValueError('Pose features require explicit model and preprocessing identities')
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
    preprocessing_identity: str
    frame_sha256: NDArray[np.uint8]  # T,32; zero when proof is disabled or the frame has no detections

    def __post_init__(self) -> None:
        _validate_axis(self.camera_id, self.frame_offsets, self.source_rows)
        if not _is_digest(self.encoder_identity) or not _is_digest(self.preprocessing_identity):
            raise ValueError('Appearance features require an explicit content/settings identity')
        if self.frame_sha256.shape != (len(self.frame_offsets) - 1, _IMAGE_DIGEST_SIZE) or self.frame_sha256.dtype != np.uint8:
            raise ValueError('Appearance features require SHA-256 bytes aligned to the source timeline')
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
    expected_decoder: OpenCVDecodeIdentity | None = None


def _frames(video: SourceVideo, expected_decoder: OpenCVDecodeIdentity | None = None
            ) -> Generator[tuple[int, NDArray[np.uint8]], None, None]:
    count = 0
    for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames, expected_decoder=expected_decoder):
        if packet.index != count:
            raise ValueError('Feature video frame order changed')
        if packet.frame.shape != (video.height, video.width, 3) or packet.frame.dtype != np.uint8:
            raise ValueError('Feature video frame size or dtype changed')
        yield count, packet.frame
        count += 1
    if count != video.num_frames:
        raise ValueError('Feature pass did not decode the complete source timeline')


class PersonPoseFeaturesModule(InferenceComponent[PersonFeaturesInput, PersonPoseFeaturesOutput]):
    batch_model = 'vitpose'
    batch_alias = 'pose'
    io = ComponentIO('person_pose_features', PersonFeaturesInput, PersonPoseFeaturesOutput,
                     {'detections': InputPort('person_detections', 2)}, 'person_pose_features', version=2)

    def __init__(self, people: PeopleModelConfig, config: FeatureConfig = DEFAULT_CONFIG, *, enabled: bool = True) -> None:
        self.people, self.config, self.enabled = people, config, enabled

    def model_specs(self) -> dict[str, ModelSpec[ViTPosePose2D]]:
        return {'pose': vitpose_model(self.people, factory=ViTPosePose2D)} if self.enabled else {}

    def cache_identity(self) -> dict[str, Any]:
        return pose_feature_settings(self.config, self.enabled)

    def validate_artifact(self, descriptor: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
        _validate_saved_metadata(descriptor['payload'], settings, specs=self.model_specs(),
                                 model_field='model_identity', preprocessing=self.cache_identity())

    def validate_output(self, output: PersonPoseFeaturesOutput, settings: Mapping[str, Any]) -> None:
        _validate_saved_metadata({'model_identity': output.model_identity, 'preprocessing_identity': output.preprocessing_identity},
                                 settings, specs=self.model_specs(), model_field='model_identity', preprocessing=self.cache_identity())

    def process_with_models(self, inputs: PersonFeaturesInput, models: ModelBorrower) -> PersonPoseFeaturesOutput:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections)
        identity = models.identity('pose') if self.enabled else feature_model_identity(None)
        poses: NDArray[np.float32] = np.zeros((len(rows), 17, 3), np.float32)
        if not self.enabled and len(rows):
            raise ValueError('Disabled pose features require empty detections')
        if len(rows):
            if (detections.boxes_xyxy[:, 2:] <= detections.boxes_xyxy[:, :2]).any():
                raise ValueError('Feature boxes must have positive area')
            # Only boxes/indices and numeric outputs span the camera. The model's
            # shared crop iterator retains one decoded image and one crop batch.
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
        return PersonPoseFeaturesOutput(video.camera_id, detections.frame_offsets, rows, poses,
                                        identity, document_digest(self.cache_identity()))

    def batch_adapter(self) -> BatchAdapter | None:
        return self if self.enabled else None

    def open_batch(self, clip_id: str, inputs: PersonFeaturesInput, models: ModelBorrower,
                   memory: SourceMemory) -> BatchSource:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections)
        identity = models.identity(self.batch_alias)
        # The crop iterator holds a decoded frame, RGB/half-resolution images
        # and one normalized crop while suspended. Track result rows as well.
        if len(rows):
            memory.reserve(video.height * video.width * 10 + 3 * 256 * 256 * 8)
        memory.reserve(artifact_memory_bytes(detections) + len(rows) * (17 * 3 * 5 + 72))
        poses: NDArray[np.float32] = np.empty((len(rows), 17, 3), np.float32)
        written: NDArray[np.bool_] = np.zeros(len(rows), bool)
        flip = self.people.runtime.vitpose.flip_test
        crop_bytes = 3 * 256 * 256 * 4
        heatmap_bytes = 17 * 64 * 48 * 4
        item_bytes = crop_bytes * (7 if flip else 4) + heatmap_bytes * (5 if flip else 2)

        def units() -> Generator[BatchUnit, None, None]:
            if not len(rows):
                return
            if (detections.boxes_xyxy[:, 2:] <= detections.boxes_xyxy[:, :2]).any():
                raise ValueError('Feature boxes must have positive area')
            indices: NDArray[np.int64] = np.repeat(np.arange(video.num_frames, dtype=np.int64), np.diff(detections.frame_offsets))
            frames = _frames(video)
            crops = iter_frame_person_crops(frames, pose_boxes(detections.boxes_xyxy, self.config),
                                            torch.from_numpy(indices), batch_size=1)
            with closing(frames), closing(crops):
                for frame, row, crop in zip(indices, rows, crops, strict=True):
                    yield BatchUnit(BatchAddress(clip_id, video.camera_id, int(frame), int(row)),
                                    'imagenet256', crop, item_bytes)
                # Preserve validation of empty trailing frames.
                for _ in frames:
                    pass

        def write(unit: BatchUnit, result: NDArray[np.float32]) -> None:
            index = _detection_index(rows, detections.frame_offsets, unit.address)
            if written[index] or not np.isfinite(result).all():
                raise ValueError(f'Duplicate or nonfinite pose for {unit.address}')
            poses[index] = result
            written[index] = True

        def finish() -> PersonPoseFeaturesOutput:
            if not written.all():
                raise ValueError('Pose batch did not write every detection row')
            return PersonPoseFeaturesOutput(video.camera_id, detections.frame_offsets, rows, poses, identity,
                                            document_digest(self.cache_identity()))

        return BatchSource(units(), write, finish)

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]:
        pose: ViTPosePose2D = models.borrow(self.batch_alias)
        images, boxes = zip(*(unit.payload for unit in units), strict=True)
        result = pose.predict_crops(torch.cat(images), torch.cat(boxes)).keypoints.detach().cpu().numpy()
        if result.shape != (len(units), 17, 3) or result.dtype != np.float32:
            raise ValueError('Pose model changed the detection row contract')
        return list(result)


class PersonAppearanceFeaturesModule(InferenceComponent[PersonAppearanceFeaturesInput, PersonAppearanceFeaturesOutput]):
    batch_model = 'clip'
    batch_alias = 'encoder'
    io = ComponentIO('person_appearance_features', PersonAppearanceFeaturesInput, PersonAppearanceFeaturesOutput,
                     {'detections': InputPort('person_detections', 2)}, APPEARANCE_FEATURES_PORT.schema, APPEARANCE_FEATURES_PORT.version)

    def __init__(self, encoder: ModelSpec[AppearanceEncoder] | None, config: FeatureConfig = DEFAULT_CONFIG, *,
                 enabled: bool = True, batch_size: int | None = None, capture_frame_sha256: bool = False) -> None:
        from dataclasses import replace
        self.encoder = encoder
        self.config = config if batch_size is None else replace(config, appearance_batch_size=batch_size)
        self.enabled = enabled
        self.capture_frame_sha256 = capture_frame_sha256
        if enabled and encoder is None:
            raise ValueError('Appearance features require an explicit encoder declaration')

    def model_specs(self) -> dict[str, ModelSpec[AppearanceEncoder]]:
        return {'encoder': self.encoder} if self.enabled and self.encoder is not None else {}

    def cache_identity(self) -> dict[str, Any]:
        return {**self.preprocessing_settings(), 'capture_frame_sha256': self.capture_frame_sha256}

    def preprocessing_settings(self) -> dict[str, Any]:
        return appearance_feature_settings(self.config, self.enabled)

    def validate_artifact(self, descriptor: Mapping[str, Any], settings: Mapping[str, Any]) -> None:
        require_appearance_media(descriptor, settings.get(APPEARANCE_MEDIA))
        _validate_saved_metadata(descriptor['payload'], settings, specs=self.model_specs(),
                                 model_field='encoder_identity', preprocessing=self.preprocessing_settings())

    def validate_output(self, output: PersonAppearanceFeaturesOutput, settings: Mapping[str, Any]) -> None:
        require_current_appearance_crop(settings)
        _validate_saved_metadata({'encoder_identity': output.encoder_identity, 'preprocessing_identity': output.preprocessing_identity},
                                 settings, specs=self.model_specs(), model_field='encoder_identity', preprocessing=self.preprocessing_settings())

    def process_with_models(self, inputs: PersonAppearanceFeaturesInput, models: ModelBorrower) -> PersonAppearanceFeaturesOutput:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections)
        identity = models.identity('encoder') if self.enabled else feature_model_identity(None)
        embeddings: NDArray[np.float32] = np.zeros((len(rows), CLIP_DIMENSION), np.float32)
        valid: NDArray[np.bool_] = np.zeros(len(rows), bool)
        frame_sha256 = np.zeros((video.num_frames, _IMAGE_DIGEST_SIZE), np.uint8)
        if not self.enabled and len(rows):
            raise ValueError('Disabled appearance features require empty detections')
        if len(rows):
            model: AppearanceEncoder = models.borrow('encoder')
            encoder = _appearance_encoder(model)
            row_indices = {int(row): i for i, row in enumerate(rows)}
            with closing(self._crops(inputs, rows, embeddings, valid, frame_sha256, encoder)) as crops:
                for batch in appearance_batches(crops, self.config.appearance_batch_size):
                    scatter_appearance(batch, infer_appearance(batch, encoder), row_indices, embeddings, valid)
        return PersonAppearanceFeaturesOutput(video.camera_id, detections.frame_offsets, rows, embeddings, valid, CLIP_ENCODER, identity,
                                               document_digest(self.preprocessing_settings()), frame_sha256)

    def batch_adapter(self) -> BatchAdapter | None:
        return self if self.enabled else None

    def open_batch(self, clip_id: str, inputs: PersonAppearanceFeaturesInput, models: ModelBorrower,
                   memory: SourceMemory) -> BatchSource:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections)
        identity = models.identity(self.batch_alias)
        memory.reserve(artifact_memory_bytes(detections) + len(rows) * (CLIP_DIMENSION * 12 + 128)
                       + video.num_frames * _IMAGE_DIGEST_SIZE)
        if len(rows):
            memory.reserve(video.height * video.width * 9)
        embeddings: NDArray[np.float32] = np.zeros((len(rows), CLIP_DIMENSION), np.float32)
        valid: NDArray[np.bool_] = np.zeros(len(rows), bool)
        frame_sha256 = np.zeros((video.num_frames, _IMAGE_DIGEST_SIZE), np.uint8)

        def units() -> Generator[BatchUnit, None, None]:
            if not len(rows):
                return
            encoder = _appearance_encoder(models.borrow(self.batch_alias))
            with closing(self._crops(inputs, rows, embeddings, valid, frame_sha256, encoder)) as crops:
                for crop in crops:
                    item_bytes = crop.pixels.numel() * crop.pixels.element_size() * 5 + CLIP_DIMENSION * 12
                    yield BatchUnit(BatchAddress(clip_id, video.camera_id, crop.frame, crop.row),
                                    tuple(crop.pixels.shape), crop, item_bytes)

        def write(unit: BatchUnit, result: NDArray[np.float32]) -> None:
            index = _detection_index(rows, detections.frame_offsets, unit.address)
            if valid[index]:
                raise ValueError(f'Appearance source row was inferred more than once: {unit.address}')
            embeddings[index] = result
            valid[index] = True

        def finish() -> PersonAppearanceFeaturesOutput:
            return PersonAppearanceFeaturesOutput(video.camera_id, detections.frame_offsets, rows, embeddings, valid,
                                                   CLIP_ENCODER, identity, document_digest(self.preprocessing_settings()),
                                                   frame_sha256)

        return BatchSource(units(), write, finish)

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]:
        encoder = _appearance_encoder(models.borrow(self.batch_alias))
        return list(infer_appearance([unit.payload for unit in units], encoder))

    def _crops(self, inputs: PersonAppearanceFeaturesInput, rows: NDArray[np.int64],
               embeddings: NDArray[np.float32], valid: NDArray[np.bool_], frame_sha256: NDArray[np.uint8],
               encoder: UnpromptedEncoder) -> Generator[AppearanceCrop, None, None]:
        detections = inputs.detections
        with closing(_frames(inputs.video, inputs.expected_decoder)) as frames:
            for frame, image in frames:
                start, end = detections.frame_offsets[frame:frame + 2]
                DetectionFeatures(frame, rows[start:end], detections.boxes_xyxy[start:end], detections.confidence[start:end],
                    None, embeddings[start:end], valid[start:end])
                if self.capture_frame_sha256 and start != end:
                    frame_sha256[frame] = appearance_image_digest(image)
                yield from iter_appearance_crops(frame, image, rows[start:end], detections.boxes_xyxy[start:end],
                                                 None, encoder, self.config)


def _detection_index(rows: NDArray[np.int64], offsets: NDArray[np.int64], address: BatchAddress) -> int:
    if address.detection_row is None:
        raise RuntimeError('A person feature requires a source detection row')
    index = int(np.searchsorted(rows, address.detection_row))
    start, end = offsets[address.source_frame:address.source_frame + 2]
    if not start <= index < end or rows[index] != address.detection_row:
        raise ValueError(f'Feature batch returned an unknown source detection: {address}')
    return index


def _appearance_encoder(model: AppearanceEncoder) -> UnpromptedEncoder:
    if model.name != CLIP_ENCODER:
        raise ValueError('Appearance encoder differs from the configured CLIP profile')
    return UnpromptedEncoder(model, CLIP_DIMENSION)
