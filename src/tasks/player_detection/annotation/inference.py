"""Use the exported player DINO predictor without running tracking, ReID or pose."""

from __future__ import annotations

import subprocess
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
    content_digest,
)
from src.tasks.base.annotation.media import check_clip, decode_range
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.predictions import PredictionContext

from .contracts import (
    Box,
    DinoCheckpointSpec,
    DinoInferenceConfig,
    PlayerFrameProposals,
    PlayerPredictions,
    PlayerProposal,
)

if TYPE_CHECKING:
    from src.tasks.player_detection.inference import PlayerDetectionResult


def observation_id(
    context: PredictionContext, frame_index: int, candidate_index: int
) -> str:
    identity: str = content_digest([context.request_id, frame_index, candidate_index])
    return identity


class DinoPredictor:
    def __init__(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        config: DinoInferenceConfig,
    ) -> None:
        from src.tasks.player_detection.inference import DinoPlayerDetector

        spec = read_document(layout, context.model.configuration, DinoCheckpointSpec)
        repository = layout.repository / "third_party/DINO"
        if repository.resolve() != repository or not repository.is_dir():
            raise ValueError(
                "DINO must be the nonsymlink shared repository under third_party/DINO"
            )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != spec.repository_commit:
            raise ValueError(
                "DINO repository revision differs from the frozen model spec"
            )
        clean = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=repository, check=False
        )
        if clean.returncode != 0:
            raise ValueError("DINO tracked source differs from its frozen revision")
        self.detector = DinoPlayerDetector(
            verify_reference(layout, context.model.inference_weights),
            repository,
            device=config.device,
            confidence=config.confidence_threshold,
            short_side=config.short_side,
            max_long_side=config.max_long_side,
        )
        # Explicit load exposes and verifies the exported/training-weight pairing once.
        self.detector.load()
        info = self.detector.checkpoint_info
        if info is None or (
            info.source_checkpoint_sha256,
            info.epoch,
            info.global_step,
        ) != (context.model.training_weights.digest, spec.epoch, spec.global_step):
            self.detector.unload()
            raise ValueError(
                "player export does not match the frozen training checkpoint/epoch/step"
            )

    def __call__(self, bgr: NDArray[np.uint8]) -> PlayerDetectionResult:
        from src.tasks.player_detection.inference import PlayerDetectionRequest

        result: PlayerDetectionResult = self.detector.predict(
            PlayerDetectionRequest(frame_bgr=bgr)
        )
        return result

    def close(self) -> None:
        self.detector.unload()


class PlayerPredictionAdapter:
    task = "player"

    def __init__(self) -> None:
        self._predictor: DinoPredictor | None = None
        self._identity: tuple[str, str] | None = None

    def close(self) -> None:
        if self._predictor is not None:
            self._predictor.close()
        self._predictor, self._identity = None, None

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
        config = read_document(layout, context.configuration, DinoInferenceConfig)
        path = directory / "boxes.json"
        if path.exists():
            reference = annotation_reference(layout, path)
            self.validate(layout, context, manifest, reference)
            return reference
        video = verify_reference(layout, manifest.media)
        timeline = check_clip(video, manifest)
        identity = (context.model.sha256, context.configuration.digest)
        if self._identity is not None and self._identity != identity:
            raise ValueError(
                "a shared player predictor cannot silently change its model/configuration"
            )
        if self._predictor is None:
            self._predictor = DinoPredictor(layout, context, config)
            self._identity = identity
        rows: list[PlayerFrameProposals] = []
        for index, frame in enumerate(
            decode_range(video, timeline, 0, len(manifest.frames))
        ):
            result = self._predictor(frame.to_ndarray(format="bgr24"))
            if (
                result.boxes_xyxy.shape != (len(result.scores), 4)
                or result.scores.ndim != 1
            ):
                raise ValueError("DINO returned misaligned boxes and scores")
            if (
                not np.isfinite(result.boxes_xyxy).all()
                or not np.isfinite(result.scores).all()
            ):
                raise ValueError("DINO returned nonfinite boxes or scores")
            observations = tuple(
                PlayerProposal(
                    observation_id=observation_id(context, index, rank),
                    candidate_index=rank,
                    box=Box(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    confidence=float(score),
                )
                for rank, (box, score) in enumerate(
                    zip(result.boxes_xyxy, result.scores, strict=True)
                )
            )
            mapped = manifest.frames[index]
            rows.append(
                PlayerFrameProposals(
                    frame_index=index,
                    source_frame_index=mapped.source_frame_index,
                    source_pts=mapped.source_pts,
                    clip_pts=mapped.clip_pts,
                    observations=observations,
                )
            )
        document = PlayerPredictions(context_digest=context.sha256, rows=tuple(rows))
        self._validate_rows(document, context, manifest, config)
        return write_document(layout, path, document)

    def validate(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        result: ArtifactRef,
    ) -> None:
        document = read_document(layout, result, PlayerPredictions)
        config = read_document(layout, context.configuration, DinoInferenceConfig)
        self._validate_rows(document, context, manifest, config)

    @staticmethod
    def _validate_rows(
        document: PlayerPredictions,
        context: PredictionContext,
        manifest: ClipManifest,
        config: DinoInferenceConfig,
    ) -> None:
        if document.context_digest != context.sha256 or len(document.rows) != len(
            manifest.frames
        ):
            raise ValueError(
                "player predictions differ from their source/config/frame count"
            )
        for row, frame in zip(document.rows, manifest.frames, strict=True):
            if (
                row.frame_index,
                row.source_frame_index,
                row.source_pts,
                row.clip_pts,
            ) != (
                frame.frame_index,
                frame.source_frame_index,
                frame.source_pts,
                frame.clip_pts,
            ):
                raise ValueError(
                    "player frame/PTS identity differs from the clip manifest"
                )
            previous = 1.0
            for rank, observation in enumerate(row.observations):
                if (observation.candidate_index, observation.observation_id) != (
                    rank,
                    observation_id(context, row.frame_index, rank),
                ):
                    raise ValueError(
                        "player observation identity differs from its original candidate"
                    )
                if (
                    not float(np.float32(config.confidence_threshold))
                    <= observation.confidence
                    <= previous
                ):
                    raise ValueError(
                        "player scores must satisfy the frozen threshold and descending order"
                    )
                previous = observation.confidence
                if (
                    observation.box.x2 > manifest.width
                    or observation.box.y2 > manifest.height
                ):
                    raise ValueError(
                        "player box is outside the native image edge coordinates"
                    )
