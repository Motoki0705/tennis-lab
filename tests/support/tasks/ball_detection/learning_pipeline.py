"""Actual tiny CPU model stages behind a synthetic queue; no GPU is allocated."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from src.tasks.ball_detection.annotation.contracts import MDDCheckpointSpec
from src.tasks.ball_detection.data.windows.learning.holdout import freeze_holdout
from src.tasks.ball_detection.training.annotation.adapter import BallLearningAdapter
from src.tasks.ball_detection.training.annotation.contracts import (
    BallLearningRecipe,
    BallSelectionPolicy,
    LearningRuntime,
)
from src.tasks.ball_detection.training.annotation.inputs import output_reference
from src.tasks.ball_detection.training.annotation.stages import execute
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Document,
    ModelRef,
    ProductOutcome,
)
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageExit,
    StageLaunch,
)
from src.tasks.base.annotation.execution.runner import stage_directory
from src.tasks.base.annotation.learning import Learning
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    FixedEvaluation,
    LearningIntent,
    StageCost,
    TaskProducts,
)
from src.tasks.base.annotation.learning.contracts import (
    LearningRuntime as SharedRuntime,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LearningPlan
from tests.support.tasks.ball_detection.annotation_learning import (
    holdout,
    policy,
    publish,
)
from tests.support.tasks.ball_detection.learning import full_model_checkpoint


def save(layout: AnnotationLayout, document: Document) -> ArtifactRef:
    return write_document(
        layout, layout.path("models", "fixtures", document.sha256 + ".json"), document
    )


class CPUExecutor:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout
        self.submitted: list[ArtifactRef] = []

    def submit(self, launch: StageLaunch) -> dict[str, Any]:
        ref = write_document(
            self.layout, stage_directory(self.layout, launch) / "launch.json", launch
        )
        if ref not in self.submitted:
            self.submitted.append(ref)
        write_asset(
            self.layout,
            stage_directory(self.layout, launch) / "queue.accepted",
            ref.canonical_bytes(),
        )
        return dict(state="queued")

    def status(self, ref: ArtifactRef) -> dict[str, Any]:
        launch = read_document(self.layout, ref, StageLaunch)
        path = stage_directory(self.layout, launch) / "exit.json"
        if path.exists():
            exit_ref = annotation_reference(self.layout, path)
            receipt = read_document(self.layout, exit_ref, StageExit)
            return dict(
                state=receipt.status, exit=exit_ref.model_dump(), reason=receipt.reason
            )
        return dict(
            state="queued" if (path.parent / "queue.accepted").exists() else "pending"
        )

    def finish(self, ref: ArtifactRef) -> ArtifactRef:
        launch = read_document(self.layout, ref, StageLaunch)
        assert launch.input is not None
        with patch.dict(
            os.environ, {"TENNIS_ANNOTATION_STAGE_OPERATION": launch.input.digest}
        ):
            result = execute(self.layout, launch.input, Path(launch.command[-1]))
        self.record_exit(ref)
        return result

    def record_exit(self, ref: ArtifactRef, *, failed: bool = False) -> ArtifactRef:
        launch = read_document(self.layout, ref, StageLaunch)
        directory = stage_directory(self.layout, launch)
        out = write_asset(
            self.layout, directory / "stdout.log", b"synthetic queue, real CPU stage"
        )
        err = write_asset(self.layout, directory / "stderr.log", b"")
        return write_document(
            self.layout,
            directory / "exit.json",
            StageExit(
                launch=ref,
                status="failed" if failed else "succeeded",
                exit_code=1 if failed else 0,
                started_at="2026-01-01T00:00:00Z",
                ended_at="2026-01-01T00:00:01Z",
                elapsed_seconds=1.0,
                stdout=out,
                stderr=err,
                reason="synthetic interruption" if failed else None,
            ),
        )


class LearningEnvironment:
    def __init__(
        self, root: Path, *, selection_policy: BallSelectionPolicy | None = None
    ) -> None:
        self.layout = AnnotationLayout.testing(root / "annotation")

        def setup(layout: AnnotationLayout, base: ArtifactRef) -> dict[str, Any]:
            parent = full_model_checkpoint(root / "outputs/ball_detection/train/parent")
            weights = output_reference(layout, parent)
            saved = torch.load(parent, weights_only=True, map_location="cpu")
            spec = MDDCheckpointSpec(
                checkpoint_schema=saved["schema"],
                architecture=saved["model_config"],
                input_contract=saved["recipe"]["input_contract"],
                training_manifest_digest=saved["recipe"]["manifest_sha256"],
            )
            self.parent = ModelRef(
                task="ball",
                version="fixture-parent",
                training_weights=weights,
                inference_weights=weights,
                configuration=save(layout, spec),
            )
            augmentation = (
                Path(__file__).resolve().parents[4]
                / "src/tasks/ball_detection/configs/augmentation/mdd_posttraining.yaml"
            )
            augmentation_ref = write_asset(
                layout,
                layout.path("models", "fixture-augmentation.yaml"),
                augmentation.read_bytes(),
            )
            self.recipe = BallLearningRecipe(
                windows=policy(),
                augmentation=augmentation_ref,
                runtime=LearningRuntime(
                    device="cpu",
                    precision="fp32",
                    num_workers=0,
                    pin_memory=False,
                    prefetch_factor=1,
                    cpu_threads=1,
                    compile_mode="off",
                    jpeg_decoder="opencv",
                    image_prefetch=False,
                ),
                epochs=2,
                freeze_epochs=1,
                windows_per_epoch=1,
                batch_size=1,
                seed=42,
                learning_rate=0.001,
                encoder_lr_ratio=0.1,
                warmup_updates=0,
                log_every=1,
                preview_clips=0,
                stress_evaluation=False,
            )
            fixed = FixedEvaluation(
                task="ball",
                validation=freeze_holdout(layout, holdout(layout, base)),
                test=freeze_holdout(layout, holdout(layout, base, split="test")),
            )
            learning = LearningPlan(
                recipe=save(layout, self.recipe),
                validation=save(layout, fixed),
                selection_policy=save(layout, selection_policy)
                if selection_policy is not None
                else None,
            )
            return dict(model=self.parent, learning=learning)

        self.review, publication, _ = publish(root, setup=setup)
        self.completion = save(
            self.layout,
            CampaignCompletion(
                plan=self.review.plan_ref,
                tasks=(
                    TaskProducts(
                        task="ball",
                        products=(
                            ProductOutcome(
                                product="frames",
                                status="published",
                                publication=publication,
                                reason=None,
                            ),
                        ),
                    ),
                ),
            ),
        )
        self.adapter = BallLearningAdapter()
        self.executor = CPUExecutor(self.layout)
        self.learning = Learning(self.layout, executor=self.executor)
        self.initial = self.learning.models.bootstrap(self.parent, self.review.blob)
        registration = self.learning.register(self.completion, self.adapter)
        assert registration.intent is not None
        self.intent_ref = registration.intent
        self.intent = read_document(self.layout, self.intent_ref, LearningIntent)
        self.runtime = SharedRuntime(
            scope=self.review.blob,
            queue=QueueConfiguration(
                script="/fixture/queue.sh",
                script_digest=self.review.blob.digest,
                directory="/fixture/queue",
                session="fixture",
                issue=1136,
            ),
            train=StageCost(
                stage="train",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
            evaluate=StageCost(
                stage="evaluate",
                pss_bytes=1,
                storage_bytes=1,
                gpu_seconds=60.0,
                timeout_seconds=40.0,
            ),
        )

    def tick(self):
        return self.learning.tick(self.intent_ref, self.adapter, self.runtime)
