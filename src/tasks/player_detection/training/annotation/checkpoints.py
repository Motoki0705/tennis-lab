"""Full-state resume and selected Lightning/export pairing, without model inference."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.tasks.base.annotation.artifacts import (
    artifact_lock,
    file_reference,
    read_document,
    sync_directory,
    verify_reference,
)
from src.tasks.base.annotation.contracts import ArtifactRef, canonical_bytes
from src.tasks.base.annotation.learning.contracts import (
    LearningIntent,
    StageInput,
    TrainingReceipt,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.annotation.contracts import DinoCheckpointSpec
from src.tasks.player_detection.inference.checkpoint import inspect_player_checkpoint
from src.tasks.player_detection.model_export import extract_dino_state_dict

from .contracts import PlayerTrainingProvenance, RunBinding, TrainingComplete
from .inputs import Inputs, resolve_inputs, run_root


def checkpoint_directory(output: Path) -> Path:
    return output / "logs/learning/checkpoints"


def normalized_configuration(config: object) -> bytes:
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("player checkpoint configuration must be a resolved mapping")
    # Resume is execution state; every other configuration value stays fixed.
    value = dict(config)
    value["run"] = {**value["run"], "resume": None}
    result: bytes = canonical_bytes(value)
    return result


def binding_inputs(
    layout: AnnotationLayout,
    intent: LearningIntent,
    binding: RunBinding,
    inputs: Inputs,
) -> Path:
    output = Path(binding.output)
    if output.parent != run_root(layout, intent) or not re.fullmatch(
        r"attempt-[0-9]{3,}", output.name
    ):
        raise ValueError("player run binding is outside its intent output directory")
    expected_config = inputs.configuration(layout, intent, output, None)
    if (
        binding.intent_digest != intent.sha256
        or binding.recipe != intent.recipe
        or binding.initial_weights != intent.parent.model.inference_weights
        or binding.parent_model_digest != intent.parent.model.sha256
        or binding.train_partition != inputs.train.manifest
        or binding.validation_partition != inputs.fixed.validation.manifest
        or binding.total_updates != inputs.total_updates
        or binding.epochs != inputs.training.trainer.max_epochs
        or binding.repository_commit != inputs.spec.repository_commit
        or normalized_configuration(
            json.loads(
                verify_reference(layout, binding.initial_configuration).read_bytes()
            )
        )
        != normalized_configuration(expected_config)
    ):
        raise ValueError(
            "player run binding differs from its intent, recipe, inputs or initial weights"
        )
    return output


def checkpoint_value(
    layout: AnnotationLayout,
    intent: LearningIntent,
    reference: ArtifactRef,
    inputs: Inputs,
) -> tuple[dict[str, Any], RunBinding]:
    path = verify_reference(layout, reference)
    value: Any = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(value, dict) or not isinstance(
        value.get("annotation_learning"), dict
    ):
        raise ValueError("player checkpoint has no annotation learning binding")
    binding = RunBinding.model_validate_json(json.dumps(value["annotation_learning"]))
    output = binding_inputs(layout, intent, binding, inputs)
    directory = checkpoint_directory(output)
    frozen_resume = (
        path.parent == directory / "resume" and path.name == reference.digest + ".ckpt"
    )
    if (path.parent != directory and not frozen_resume) or reference.role != "artifact":
        raise ValueError("player checkpoint is outside its bound run")
    if normalized_configuration(
        value["hyper_parameters"]["config"]
    ) != normalized_configuration(
        json.loads(verify_reference(layout, binding.initial_configuration).read_bytes())
    ):
        raise ValueError("player checkpoint configuration changed")
    epoch, step = value.get("epoch"), value.get("global_step")
    per_epoch = inputs.total_updates // binding.epochs
    if (
        type(epoch) is not int
        or type(step) is not int
        or not 0 <= epoch < binding.epochs
        or step != (epoch + 1) * per_epoch
    ):
        raise ValueError(
            "player checkpoint is not a completed epoch of the declared update budget"
        )
    if (
        not isinstance(value.get("optimizer_states"), list)
        or len(value["optimizer_states"]) != 1
        or not isinstance(value.get("lr_schedulers"), list)
        or len(value["lr_schedulers"]) != 1
        or not isinstance(value.get("loops"), dict)
        or not isinstance(value.get("state_dict"), dict)
    ):
        raise ValueError("player checkpoint lacks full optimizer/schedule/loop state")
    return value, binding


def selected_checkpoint(
    layout: AnnotationLayout,
    intent: LearningIntent,
    latest: ArtifactRef,
    inputs: Inputs,
) -> tuple[ArtifactRef, float, RunBinding]:
    value, binding = checkpoint_value(layout, intent, latest, inputs)
    if Path(latest.path).name != "last.ckpt":
        raise ValueError(
            "player run completion requires the independent last checkpoint"
        )
    callbacks = value.get("callbacks")
    if not isinstance(callbacks, dict):
        raise ValueError("player latest checkpoint has no validation selection state")
    candidates = [
        item
        for item in callbacks.values()
        if isinstance(item, dict) and item.get("monitor") == "val/map"
    ]
    if len(candidates) != 1:
        raise ValueError("player best checkpoint selection is absent or ambiguous")
    chosen = candidates[0]
    score = float(chosen["best_model_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("player best validation mAP must be finite and defined")
    path = Path(chosen["best_model_path"])
    if (
        path.parent != checkpoint_directory(Path(binding.output))
        or path.name == "last.ckpt"
    ):
        raise ValueError("selected player checkpoint is outside the completed run")
    selected = file_reference(layout, path, role="artifact")
    selected_value, selected_binding = checkpoint_value(
        layout, intent, selected, inputs
    )
    if (
        selected_binding != binding
        or selected_value["global_step"] > value["global_step"]
    ):
        raise ValueError(
            "selected player checkpoint belongs to another run or future epoch"
        )
    selected_callbacks = [
        item
        for item in selected_value["callbacks"].values()
        if isinstance(item, dict) and item.get("monitor") == "val/map"
    ]
    if (
        len(selected_callbacks) != 1
        or float(selected_callbacks[0]["best_model_score"]) != score
        or float(chosen["best_k_models"].get(str(path), float("nan"))) != score
    ):
        raise ValueError(
            "selected player checkpoint score differs from latest top-1 state"
        )
    return selected, score, binding


def verify_export(
    layout: AnnotationLayout, selected: ArtifactRef, exported: ArtifactRef
) -> DinoCheckpointSpec:
    path = verify_reference(layout, exported)
    if exported.role != "checkpoint":
        raise ValueError("player inference export must use the checkpoint role")
    info = inspect_player_checkpoint(path)
    source: dict[str, Any] = torch.load(
        verify_reference(layout, selected),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if (info.source_checkpoint_sha256, info.epoch, info.global_step) != (
        selected.digest,
        source["epoch"],
        source["global_step"],
    ):
        raise ValueError(
            "player export does not match the selected training checkpoint"
        )
    payload: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    expected = extract_dino_state_dict(source["state_dict"])
    actual = payload.get("model")
    if (
        not isinstance(actual, dict)
        or set(actual) != set(expected)
        or any(
            not isinstance(actual[name], torch.Tensor)
            or actual[name].dtype != tensor.dtype
            or actual[name].shape != tensor.shape
            or not torch.equal(actual[name], tensor)
            for name, tensor in expected.items()
        )
    ):
        raise ValueError(
            "player export tensor state differs from the selected checkpoint"
        )
    if normalized_configuration(
        payload["tennis_lab"]["train_config"]
    ) != normalized_configuration(source["hyper_parameters"]["config"]):
        raise ValueError("player export configuration differs from selected checkpoint")
    binding = RunBinding.model_validate_json(json.dumps(source["annotation_learning"]))
    return DinoCheckpointSpec(
        repository_commit=binding.repository_commit,
        epoch=info.epoch,
        global_step=info.global_step,
    )


def validate_resume(
    layout: AnnotationLayout, intent: LearningIntent, checkpoint: ArtifactRef
) -> RunBinding:
    _, binding = checkpoint_value(
        layout, intent, checkpoint, resolve_inputs(layout, intent)
    )
    expected = (
        checkpoint_directory(Path(binding.output))
        / "resume"
        / (checkpoint.digest + ".ckpt")
    )
    if verify_reference(layout, checkpoint) != expected:
        raise ValueError("player resume requires a frozen latest-checkpoint copy")
    return binding


def freeze_resume_checkpoint(
    layout: AnnotationLayout, intent: LearningIntent, latest: ArtifactRef
) -> ArtifactRef:
    """Explicit I/O preparation: retain the exact input even when last.ckpt advances."""
    with artifact_lock(layout, "player-learning-" + intent.sha256, blocking=False):
        _, binding = checkpoint_value(
            layout, intent, latest, resolve_inputs(layout, intent)
        )
        source = verify_reference(layout, latest)
        if source != checkpoint_directory(Path(binding.output)) / "last.ckpt":
            raise ValueError(
                "only the run's latest checkpoint can be frozen for resume"
            )
        destination = source.parent / "resume" / (latest.digest + ".ckpt")
        reference = ArtifactRef(
            role="artifact",
            path=destination.relative_to(layout.repository / "outputs").as_posix(),
            digest=latest.digest,
            size_bytes=latest.size_bytes,
        )
        layout.artifact_path(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary: Path | None = None
            try:
                with (
                    tempfile.NamedTemporaryFile(
                        dir=destination.parent, delete=False
                    ) as output,
                    source.open("rb") as original,
                ):
                    temporary = Path(output.name)
                    shutil.copyfileobj(original, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                copied = file_reference(layout, temporary, role="artifact")
                if (copied.digest, copied.size_bytes) != (
                    latest.digest,
                    latest.size_bytes,
                ):
                    raise ValueError("player resume checkpoint changed while copying")
                verify_reference(layout, latest)
                os.replace(temporary, destination)
                sync_directory(destination.parent)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        verify_reference(layout, reference)
        return reference


def validate_training(
    layout: AnnotationLayout, intent: LearningIntent, receipt: TrainingReceipt
) -> PlayerTrainingProvenance:
    request = read_document(layout, receipt.request, StageInput)
    if (
        read_document(layout, request.intent, LearningIntent) != intent
        or request.stage != "train"
    ):
        raise ValueError("player training receipt belongs to another intent or stage")
    inputs = resolve_inputs(layout, intent)
    provenance = read_document(layout, receipt.provenance, PlayerTrainingProvenance)
    if (provenance.request, provenance.intent) != (receipt.request, request.intent):
        raise ValueError("player training provenance has a different request")
    selected, score, binding = selected_checkpoint(
        layout, intent, provenance.latest_checkpoint, inputs
    )
    value, _ = checkpoint_value(layout, intent, provenance.latest_checkpoint, inputs)
    if (
        value["epoch"] != binding.epochs - 1
        or value["global_step"] != binding.total_updates
    ):
        raise ValueError("player training did not complete its full update budget")
    if (
        read_document(layout, provenance.run_binding, RunBinding) != binding
        or provenance.initial_configuration != binding.initial_configuration
        or provenance.selected_checkpoint != selected
        or receipt.candidate.training_weights != selected
        or receipt.candidate.task != "player"
        or receipt.candidate.version != "annotation-" + intent.sha256
    ):
        raise ValueError("player candidate differs from selected run provenance")
    spec = verify_export(layout, selected, receipt.candidate.inference_weights)
    if (
        read_document(layout, receipt.candidate.configuration, DinoCheckpointSpec)
        != spec
    ):
        raise ValueError("player candidate specification differs from its export")
    completed = read_document(layout, provenance.completed, TrainingComplete)
    expected = TrainingComplete(
        binding=provenance.run_binding,
        latest_checkpoint=provenance.latest_checkpoint,
        selected_checkpoint=selected,
        export=receipt.candidate.inference_weights,
        best_validation_map=score,
    )
    if completed != expected:
        raise ValueError(
            "player completion receipt differs from the selected artifacts"
        )
    return provenance
