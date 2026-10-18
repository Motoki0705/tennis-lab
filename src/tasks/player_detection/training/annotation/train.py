"""One owned training attempt; explicit retries may recover a completed run."""

from __future__ import annotations

from pathlib import Path

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    file_reference,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ModelRef
from src.tasks.base.annotation.learning.contracts import (
    LearningIntent,
    StageInput,
    TrainingReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.configuration import ExportConfig
from src.tasks.player_detection.model_export import export_player_checkpoint

from .checkpoints import (
    checkpoint_directory,
    checkpoint_value,
    normalized_configuration,
    selected_checkpoint,
    validate_resume,
    validate_training,
    verify_export,
)
from .contracts import PlayerTrainingProvenance, RunBinding, TrainingComplete
from .inputs import resolve_inputs, run_root
from .runner import AnnotationPlayerRunner


def train(
    layout: AnnotationLayout,
    reference: ArtifactRef,
    request: StageInput,
    intent: LearningIntent,
    result: Path,
) -> ArtifactRef:
    if result.exists():
        existing = annotation_reference(layout, result)
        receipt = read_document(layout, existing, TrainingReceipt)
        if receipt.request != reference:
            raise ValueError("player training result belongs to another attempt")
        validate_training(layout, intent, receipt)
        return existing
    inputs = resolve_inputs(layout, intent)
    output = run_root(layout, intent) / f"attempt-{request.attempt:03d}"
    if request.resume_checkpoint is not None:
        validate_resume(layout, intent, request.resume_checkpoint)
        _, prior_binding = checkpoint_value(
            layout, intent, request.resume_checkpoint, inputs
        )
        output = Path(prior_binding.output)
        if int(output.name.removeprefix("attempt-")) >= request.attempt:
            raise ValueError("player retry can resume only an earlier attempt")
    directory = layout.path(
        "models", "player", "training-runs", intent.sha256, output.name
    )
    initial = inputs.configuration(layout, intent, output, None)
    initial_ref = write_asset(
        layout,
        directory / "initial-config.json",
        normalized_configuration(initial) + b"\n",
    )
    binding = RunBinding(
        intent_digest=intent.sha256,
        recipe=intent.recipe,
        initial_weights=intent.parent.model.inference_weights,
        parent_model_digest=intent.parent.model.sha256,
        train_partition=inputs.train.manifest,
        validation_partition=inputs.fixed.validation.manifest,
        initial_configuration=initial_ref,
        output=str(output),
        total_updates=inputs.total_updates,
        epochs=inputs.training.trainer.max_epochs,
        repository_commit=inputs.spec.repository_commit,
    )
    binding_ref = write_document(layout, directory / "binding.json", binding)
    latest_path = checkpoint_directory(output) / "last.ckpt"
    complete = False
    if latest_path.exists():
        latest_ref = file_reference(layout, latest_path, role="artifact")
        latest, actual_binding = checkpoint_value(layout, intent, latest_ref, inputs)
        if actual_binding != binding:
            raise ValueError("existing player run belongs to another binding")
        complete = latest["global_step"] == binding.total_updates
        if not complete and (
            request.resume_checkpoint is None
            or (request.resume_checkpoint.digest, request.resume_checkpoint.size_bytes)
            != (latest_ref.digest, latest_ref.size_bytes)
        ):
            raise ValueError(
                "incomplete player run requires an explicit retry from its latest checkpoint"
            )
    if not complete:
        if (directory / "completed.json").exists():
            raise ValueError("completed player run lost its final checkpoint")
        AnnotationPlayerRunner(layout, inputs, binding).run(
            inputs.configuration(layout, intent, output, request.resume_checkpoint)
        )
    latest_ref = file_reference(layout, latest_path, role="artifact")
    latest, _ = checkpoint_value(layout, intent, latest_ref, inputs)
    if (
        latest["global_step"] != binding.total_updates
        or latest["epoch"] != binding.epochs - 1
    ):
        raise ValueError("player training stopped before its full update budget")
    selected_ref, score, _ = selected_checkpoint(layout, intent, latest_ref, inputs)
    export_path = layout.artifact_path(
        ArtifactRef(
            role="checkpoint",
            path=f"player_detection/annotation/{intent.sha256}/{output.name}/best.pth",
            digest="0" * 64,
            size_bytes=0,
        )
    )
    if not export_path.exists():
        export_player_checkpoint(
            ExportConfig(
                lightning_checkpoint=verify_reference(layout, selected_ref),
                destination=export_path,
                repository=layout.repository / "third_party/DINO",
            )
        )
    export_ref = file_reference(layout, export_path, role="checkpoint")
    spec = verify_export(layout, selected_ref, export_ref)
    spec_ref = write_document(layout, directory / "spec.json", spec)
    completed = write_document(
        layout,
        directory / "completed.json",
        TrainingComplete(
            binding=binding_ref,
            latest_checkpoint=latest_ref,
            selected_checkpoint=selected_ref,
            export=export_ref,
            best_validation_map=score,
        ),
    )
    provenance = write_document(
        layout,
        result.parent / "provenance.json",
        PlayerTrainingProvenance(
            request=reference,
            intent=request.intent,
            run_binding=binding_ref,
            initial_configuration=initial_ref,
            latest_checkpoint=latest_ref,
            selected_checkpoint=selected_ref,
            completed=completed,
        ),
    )
    receipt = TrainingReceipt(
        request=reference,
        provenance=provenance,
        candidate=ModelRef(
            task="player",
            version="annotation-" + intent.sha256,
            training_weights=selected_ref,
            inference_weights=export_ref,
            configuration=spec_ref,
        ),
    )
    validate_training(layout, intent, receipt)
    return write_document(layout, result, receipt)
