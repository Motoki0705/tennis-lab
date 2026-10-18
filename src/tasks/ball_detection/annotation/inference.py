"""One pinned model instance serving new clips at native resolution and real PTS."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import closing
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
from src.tasks.base.annotation.media import (
    Timeline,
    check_clip,
    decode_clip,
    decode_range,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import (
    BallPredictions,
    BallProposal,
    HybridInferenceConfig,
    HybridPredictions,
    HybridProposal,
    MDDCheckpointSpec,
    MDDInferenceConfig,
    read_inference,
)
from .hybrid import HybridPredictor, HybridWindow
from .windows import window_owners, window_starts

if TYPE_CHECKING:
    from src.tasks.ball_detection.model_io.predict import CoordinateModel


def _rgb_windows(
    video: Path,
    timeline: Timeline,
    starts: tuple[int, ...],
    manifest: ClipManifest | None = None,
) -> Generator[tuple[int, NDArray[np.uint8]], None, None]:
    """Borrow one rolling RGB buffer until the next window; decode each frame once."""
    rgb: NDArray[np.uint8] = np.empty(
        (32, 3, timeline.height, timeline.width), dtype=np.uint8
    )
    previous_start: int | None = None
    frames = (
        decode_range(video, timeline, 0, len(timeline.pts))
        if manifest is None or manifest.jpeg is None
        else decode_clip(video, manifest, 0, len(timeline.pts))
    )
    with closing(frames) as decoded:
        for start in starts:
            retained = 0 if previous_start is None else 32 - (start - previous_start)
            # Move each retained frame separately: overlapping array assignment
            # may allocate another full window at native resolution.
            for local in range(retained):
                rgb[local] = rgb[32 - retained + local]
            for local in range(retained, 32):
                frame = next(decoded)
                rgb[local] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
            yield start, rgb
            previous_start = start
        # Resume after the final yield to run the decoder's PTS/count validation.
        if next(decoded, None) is not None:
            raise ValueError("decoder returned frames beyond the clip manifest")


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
        self._predictor: MDDPredictor | HybridPredictor | None = None
        self._identity: tuple[str, str] | None = None

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        config = read_inference(layout, context.configuration)
        config.validate_media(manifest)
        starts = window_starts(len(manifest.frames), config.stride)
        owners = window_owners(len(manifest.frames), config.stride)
        hybrid = isinstance(config, HybridInferenceConfig)
        path = directory / ("hybrid.json" if hybrid else "coordinates.json")
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
            self._predictor = (
                HybridPredictor(layout, context, config)
                if isinstance(config, HybridInferenceConfig)
                else MDDPredictor(layout, context, config)
            )
            self._identity = identity
        rows: list[BallProposal | HybridProposal | None] = [None] * len(manifest.frames)
        heatmap_size = ((manifest.height + 3) // 4, (manifest.width + 3) // 4)
        clock = Fraction(manifest.time_base)
        with closing(_rgb_windows(video, timeline, starts, manifest)) as windows:
            for start, rgb in windows:
                seconds = np.asarray(
                    [
                        (frame.source_pts - manifest.frames[start].source_pts) * clock
                        for frame in manifest.frames[start : start + 32]
                    ],
                    dtype=np.float32,
                )
                values = self._predictor(rgb, seconds)
                if isinstance(values, HybridWindow) != hybrid:
                    raise ValueError("prediction output differs from the explicit model family")
                if isinstance(values, HybridWindow) and values.heatmap_size != heatmap_size:
                    raise ValueError("hybrid heatmap grid differs from native /4 dimensions")
                coordinates = values.coordinates if isinstance(values, HybridWindow) else values
                for local, (u, v) in enumerate(coordinates):
                    index = start + local
                    if owners[index] != start:
                        continue
                    mapped = manifest.frames[index]
                    identity_fields = dict(
                        frame_index=index,
                        source_frame_index=mapped.source_frame_index,
                        source_pts=mapped.source_pts,
                        clip_pts=mapped.clip_pts,
                        window_start=start,
                    )
                    if isinstance(values, HybridWindow):
                        present = bool(values.present[local])
                        rows[index] = HybridProposal(
                            **identity_fields,
                            presence="present" if present else "absent",
                            presence_logit=float(values.presence_logits[local]),
                            presence_score=float(values.presence_scores[local]),
                            x=float(u) * (manifest.width - 1) if present else None,
                            y=float(v) * (manifest.height - 1) if present else None,
                            heatmap_peak=float(values.heatmap_peaks[local]),
                        )
                    else:
                        rows[index] = BallProposal(
                            **identity_fields,
                            x=float(u) * (manifest.width - 1),
                            y=float(v) * (manifest.height - 1),
                        )
        if any(row is None for row in rows):
            raise ValueError("ball inference did not cover every real clip frame")
        document = (
            HybridPredictions(
                context_digest=context.sha256,
                heatmap_size=heatmap_size,
                rows=tuple(row for row in rows if isinstance(row, HybridProposal)),
            )
            if hybrid else BallPredictions(
                context_digest=context.sha256,
                rows=tuple(row for row in rows if isinstance(row, BallProposal)),
            )
        )
        return write_document(
            layout,
            path,
            document,
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
