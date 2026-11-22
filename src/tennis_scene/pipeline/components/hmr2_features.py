"""Persist HMR2 image evidence with explicit clip/person/segment/frame identity."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.submodules.models import (
    Hmr2FeatureExtractor,
    ImageFeatureRequest,
    iter_frame_person_crops,
)
from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchAddress,
    BatchSource,
    BatchUnit,
    SourceMemory,
)
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.contracts import (
    ClipSource,
    ComponentIO,
    InputPort,
    SourceVideo,
)
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.tennis_scene.pipeline.model_resources import hmr2_tensorrt
from src.tennis_scene.pipeline.tensorrt_assets import tensorrt_identity
from src.utils.video import OpenCVVideoFrameReader


@dataclass(frozen=True)
class HMR2FeatureSegment:
    segment_index: int
    source_frames: NDArray[np.int64]
    features: NDArray[np.float32]  # (L,1024), CPU float32

    def __post_init__(self) -> None:
        frames = self.source_frames
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("HMR2 segment index must be a nonnegative integer")
        if frames.ndim != 1 or frames.dtype != np.int64 or len(frames) < 2 or (frames < 0).any() or (np.diff(frames) <= 0).any():
            raise ValueError("HMR2 features require ordered source frames")
        if self.features.shape != (len(frames), 1024) or self.features.dtype != np.float32 or not np.isfinite(self.features).all():
            raise ValueError("HMR2 features must be finite float32 (L,1024)")


@dataclass(frozen=True)
class HMR2BodyFeatures:
    person_id: int
    camera_id: str
    video_sha256: str
    segments: tuple[HMR2FeatureSegment, ...]

    def __post_init__(self) -> None:
        if type(self.person_id) is not int or not self.camera_id or not self.video_sha256:
            raise ValueError("HMR2 features require person/camera/source identity")
        if tuple(s.segment_index for s in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("HMR2 segment indices must preserve selection order")
        if any(a.source_frames[-1] >= b.source_frames[0] for a, b in zip(self.segments, self.segments[1:], strict=False)):
            raise ValueError("HMR2 segments must have disjoint ordered source frames")


@dataclass(frozen=True)
class HMR2FeaturesOutput:
    clip_id: str
    bodies: tuple[HMR2BodyFeatures, ...]

    def __post_init__(self) -> None:
        if not self.clip_id or len({b.person_id for b in self.bodies}) != len(self.bodies):
            raise ValueError("HMR2 features require a clip and unique person identities")


@dataclass(frozen=True)
class HMR2FeaturesInput:
    source: ClipSource
    selection: BodyViewSelectionOutput


def validate_body_selection(source: ClipSource, selection: BodyViewSelectionOutput) -> None:
    """Reject mismatched sources before any image or temporal inference starts."""
    if len({s.person_id for s in selection.selections}) != len(selection.selections):
        raise ValueError("Body selection contains duplicate person identities")
    for selected in selection.selections:
        if selected.camera_id not in source.camera_ids:
            raise ValueError("Body selection camera does not belong to this clip")
        video = source.video(selected.camera_id)
        if len(selected.requests) != len(selected.observed_samples):
            raise ValueError("Body selection segment/observation counts disagree")
        previous = -1
        for request, observed in zip(selected.requests, selected.observed_samples, strict=True):
            if request.video_path != video.path or request.size != source.size:
                raise ValueError("Body selection video/size does not match this clip")
            if request.source_frames[0] <= previous or request.source_frames[-1] >= video.num_frames:
                raise ValueError("Body selection source frames overlap or exceed this clip")
            if type(observed) is not int or not 0 <= observed <= len(request.source_frames):
                raise ValueError("Body selection observed count is invalid")
            previous = int(request.source_frames[-1])


class HMR2FeaturesModule(InferenceComponent[HMR2FeaturesInput, HMR2FeaturesOutput]):
    batch_model = "hmr2"
    batch_alias = "features"
    io = ComponentIO("hmr2_features", HMR2FeaturesInput, HMR2FeaturesOutput,
        {"selection": InputPort("body_view_selection")}, "hmr2_features")

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def model_specs(self) -> dict[str, ModelSpec[Hmr2FeatureExtractor]]:
        config = self.config
        if not self.enabled:
            return {}
        runtime = config.runtime.hmr2
        spec = hmr2_tensorrt(config)
        assets = dict(spec.assets)
        settings: dict[str, Any] = {"runtime": {"batch_size": runtime.batch_size}}
        dtype = "float32"
        kwargs = {}
        if runtime.backend.engines:
            engine = spec.bind(runtime.backend.engines[0], device=config.runtime.device, batch_size=runtime.batch_size)
            engine_assets, identity = tensorrt_identity((engine,), batch_size=runtime.batch_size)
            assets.update(engine_assets)
            settings.update(identity)
            dtype = "float16"
            kwargs["engine"] = engine
        return {
            "features": ModelSpec("hmr2", assets, spec.model, config.runtime.device, dtype, settings,
                lambda: Hmr2FeatureExtractor(config.hmr2_checkpoint, device=config.runtime.device,
                    batch_size=runtime.batch_size, mean_params_path=config.smpl_assets.hmr2_mean_params, **kwargs),
                initialize=lambda model: model.load(), release=lambda model: model.unload()),
        }

    def process_with_models(self, inputs: HMR2FeaturesInput, models: ModelBorrower) -> HMR2FeaturesOutput:
        if not self.enabled:
            return HMR2FeaturesOutput(inputs.source.clip_id, ())
        validate_body_selection(inputs.source, inputs.selection)
        bodies = []
        for selected in inputs.selection.selections:
            segments = []
            for index, request in enumerate(selected.requests):
                model: Hmr2FeatureExtractor = models.borrow("features")
                result = model.predict(ImageFeatureRequest(request.video_path,
                    torch.from_numpy(np.array(request.boxes_xys, copy=True)),
                    frame_indices=torch.from_numpy(np.array(request.source_frames, copy=True))))
                segments.append(HMR2FeatureSegment(index, request.source_frames,
                    result.features.detach().cpu().numpy()))
            bodies.append(HMR2BodyFeatures(selected.person_id, selected.camera_id,
                inputs.source.video(selected.camera_id).sha256, tuple(segments)))
        return HMR2FeaturesOutput(inputs.source.clip_id, tuple(bodies))

    def batch_adapter(self) -> BatchAdapter | None:
        return self if self.enabled else None

    def open_batch(self, clip_id: str, inputs: HMR2FeaturesInput, models: ModelBorrower,
                   memory: SourceMemory) -> BatchSource:
        source, selection = inputs.source, inputs.selection
        if clip_id != source.clip_id:
            raise ValueError("HMR2 batch clip does not match the body selection")
        validate_body_selection(source, selection)
        request_rows = sum(len(request.source_frames) for body in selection.selections for request in body.requests)
        memory.reserve(request_rows * (1024 * 5 + 544))
        if request_rows:
            memory.reserve(source.size[0] * source.size[1] * 10 + 3 * 256 * 256 * 8)
        routes: dict[str, list[tuple[int, int, int, NDArray[np.float32]]]] = {}
        features: dict[tuple[int, int], NDArray[np.float32]] = {}
        written: dict[tuple[int, int], NDArray[np.bool_]] = {}
        requests = {}
        cameras = {}
        for body in selection.selections:
            cameras[body.person_id] = body.camera_id
            for segment, request in enumerate(body.requests):
                key = (body.person_id, segment)
                requests[key] = request
                features[key] = np.empty((len(request.source_frames), 1024), np.float32)
                written[key] = np.zeros(len(request.source_frames), bool)
                routes.setdefault(body.camera_id, []).extend(
                    (int(frame), body.person_id, segment, box)
                    for frame, box in zip(request.source_frames, request.boxes_xys, strict=True))

        def frames(video: SourceVideo) -> Generator[tuple[int, NDArray[np.uint8]], None, None]:
            for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
                if packet.frame.shape != (video.height, video.width, 3) or packet.frame.dtype != np.uint8:
                    raise ValueError("HMR2 video frame size or dtype changed")
                yield packet.index, packet.frame

        def units() -> Generator[BatchUnit, None, None]:
            for camera, rows in routes.items():
                video = source.video(camera)
                rows.sort(key=lambda row: row[:3])
                frame_indices = torch.tensor([row[0] for row in rows], dtype=torch.int64)
                boxes = torch.from_numpy(np.stack([row[3] for row in rows]))

                with closing(frames(video)) as decoded, closing(
                    iter_frame_person_crops(decoded, boxes, frame_indices, batch_size=1)
                ) as crops:
                    for (frame, person, segment, _), (pixels, _) in zip(rows, crops, strict=True):
                        yield BatchUnit(BatchAddress(clip_id, camera, frame, person_id=person, segment=segment),
                                        "imagenet256", pixels, 3 * 256 * 256 * 4 * 3 + 1024 * 8)

        def write(unit: BatchUnit, value: NDArray[np.float32]) -> None:
            address = unit.address
            if address.person_id is None or address.segment is None:
                raise RuntimeError("HMR2 requires person and segment identities")
            key = address.person_id, address.segment
            frames = requests[key].source_frames
            index = int(np.searchsorted(frames, address.source_frame))
            if (cameras[address.person_id] != address.camera_id or index == len(frames)
                    or frames[index] != address.source_frame or written[key][index]):
                raise ValueError(f"HMR2 batch returned an unknown or duplicate body frame: {address}")
            features[key][index] = value
            written[key][index] = True

        def finish() -> HMR2FeaturesOutput:
            if not all(mask.all() for mask in written.values()):
                raise ValueError("HMR2 batch did not write every selected body frame")
            return HMR2FeaturesOutput(clip_id, tuple(
                HMR2BodyFeatures(body.person_id, body.camera_id, source.video(body.camera_id).sha256, tuple(
                    HMR2FeatureSegment(segment, request.source_frames, features[body.person_id, segment])
                    for segment, request in enumerate(body.requests)))
                for body in selection.selections))

        return BatchSource(units(), write, finish)

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]:
        model: Hmr2FeatureExtractor = models.borrow(self.batch_alias)
        result = model.predict_crops(torch.cat([unit.payload for unit in units])).features.detach().cpu().numpy()
        if result.shape != (len(units), 1024) or result.dtype != np.float32:
            raise ValueError("HMR2 model changed the feature row contract")
        return list(result)
