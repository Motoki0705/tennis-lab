"""One shared MDD model instance serving new clips at native resolution and real PTS."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    canonical_bytes,
)
from src.tasks.base.annotation.media import check_clip, decode_range
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import (
    BallPredictions,
    BallProposal,
    MDDCheckpointSpec,
    MDDInferenceConfig,
)
from .windows import window_owners, window_starts

if TYPE_CHECKING:
    from src.tasks.ball_detection.model_io.predict import CoordinateModel


class MDDPredictor:
    def __init__(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        config: MDDInferenceConfig,
    ) -> None:
        import torch

        from src.tasks.ball_detection.checkpoints.coordinates import (
            restore_coordinate_model,
        )
        from src.tasks.ball_detection.models.mdd import (
            DeepMDDQueryDetector,
            MDDQueryDetector,
        )

        spec = read_document(layout, context.model.configuration, MDDCheckpointSpec)
        saved = torch.load(
            verify_reference(layout, context.model.inference_weights),
            map_location="cpu",
            weights_only=True,
        )
        if saved.get("schema") != spec.checkpoint_schema or canonical_bytes(
            saved.get("model_config")
        ) != canonical_bytes(spec.architecture):
            raise ValueError(
                "MDD checkpoint architecture differs from the frozen model spec"
            )
        model, recipe, manifest = restore_coordinate_model(saved)
        if not isinstance(model, (DeepMDDQueryDetector, MDDQueryDetector)):
            raise ValueError(
                "annotation adapter supports explicit MDD-only coordinate models; pose is not substituted"
            )
        if (
            canonical_bytes(recipe["input_contract"])
            != canonical_bytes(spec.input_contract)
            or manifest != spec.training_manifest_digest
        ):
            raise ValueError(
                "MDD checkpoint preprocessing/training provenance differs from the frozen spec"
            )
        if canonical_bytes(model.mdd.input_contract()) != canonical_bytes(
            spec.input_contract
        ):
            raise ValueError(
                "restored preprocessing differs from the frozen model spec"
            )
        self.device = torch.device(config.device)
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        if config.precision == "bfloat16" and (
            config.device != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 inference requires a supported CUDA device")
        self.precision = config.precision
        self.model: CoordinateModel = model.to(self.device).eval()

    def __call__(
        self, rgb: NDArray[np.uint8], seconds: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        import torch

        from src.tasks.ball_detection.model_io.predict import predict_coordinates

        if rgb.shape[:2] != (32, 3) or rgb.dtype != np.uint8 or min(rgb.shape[-2:]) < 8:
            raise ValueError(
                "MDD expects 32 native RGB uint8 CHW frames with dimensions at least 8"
            )
        if (
            seconds.shape != (32,)
            or seconds.dtype != np.float32
            or not np.all(np.diff(seconds) > 0)
        ):
            raise ValueError("MDD requires strictly increasing real timestamps")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.precision == "bfloat16",
            ),
        ):
            output = predict_coordinates(
                self.model,
                {
                    "rgb": torch.from_numpy(rgb)[None],
                    "timestamps": torch.from_numpy(seconds)[None],
                },
                self.device,
            )
        values: NDArray[np.float32] = output[0].float().cpu().numpy()
        if (
            values.shape != (32, 2)
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
        ):
            raise ValueError("MDD returned invalid normalized coordinates")
        return values


class BallPredictionAdapter:
    task = "ball"

    def __init__(self) -> None:
        self._predictor: MDDPredictor | None = None
        self._identity: tuple[str, str] | None = None

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        config = read_document(layout, context.configuration, MDDInferenceConfig)
        starts = window_starts(len(manifest.frames), config.stride)
        owners = window_owners(len(manifest.frames), config.stride)
        path = directory / "coordinates.json"
        if path.exists():
            reference = annotation_reference(layout, path)
            self.validate(layout, context, manifest, reference)
            return reference
        video = verify_reference(layout, manifest.media)
        timeline = check_clip(video, manifest)
        identity = (context.model.sha256, context.configuration.digest)
        if self._identity is not None and self._identity != identity:
            raise ValueError(
                "a shared inference adapter cannot silently switch its model/configuration"
            )
        if self._predictor is None:
            self._predictor = MDDPredictor(layout, context, config)
            self._identity = identity
        rows: list[BallProposal | None] = [None] * len(manifest.frames)
        clock = Fraction(manifest.time_base)
        for start in starts:
            rgb = np.empty((32, 3, manifest.height, manifest.width), dtype=np.uint8)
            for local, frame in enumerate(
                decode_range(video, timeline, start, start + 32)
            ):
                rgb[local] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
            seconds = np.asarray(
                [
                    (frame.source_pts - manifest.frames[start].source_pts) * clock
                    for frame in manifest.frames[start : start + 32]
                ],
                dtype=np.float32,
            )
            values = self._predictor(rgb, seconds)
            for local, (u, v) in enumerate(values):
                index = start + local
                if owners[index] != start:
                    continue
                frame = manifest.frames[index]
                rows[index] = BallProposal(
                    frame_index=index,
                    source_frame_index=frame.source_frame_index,
                    source_pts=frame.source_pts,
                    clip_pts=frame.clip_pts,
                    x=float(u) * (manifest.width - 1),
                    y=float(v) * (manifest.height - 1),
                    window_start=start,
                )
        if any(row is None for row in rows):
            raise ValueError("MDD inference did not cover every real clip frame")
        return write_document(
            layout,
            path,
            BallPredictions(
                context_digest=context.sha256,
                rows=tuple(row for row in rows if row is not None),
            ),
        )

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None:
        from .validation import validate_predictions

        validate_predictions(layout, context, manifest, result)
