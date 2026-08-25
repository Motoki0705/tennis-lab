"""BLCS adapter for the shared strict reference-counterfactual evaluator."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import (
    CourtCoordinateNormalizationConfig,
    TrainingRuntimeConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.evaluation import (
    REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
    ReferenceCounterfactualError,
    ReferenceCounterfactualManifest,
    ReferenceCounterfactualPass,
    ReferenceCounterfactualPassRow,
    ReferenceCounterfactualQuantityArrays,
    ReferenceCounterfactualQuantitySchema,
    ReferenceCounterfactualReportPaths,
    ReferenceCounterfactualRunIdentity,
    ReferenceCounterfactualSide,
    array_payload_sha256,
    build_reference_counterfactual_manifest,
    canonicalize_reference_counterfactual_raw_payload,
    evaluate_reference_counterfactual,
    file_sha256,
    masked_counterfactual_quantity_for_digest,
    write_reference_counterfactual_report,
)
from src.tasks.base.generate_dataset import (
    court_points_target_to_physical,
    court_vectors_target_to_physical,
)
from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    validate_training_boundary,
)
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCS_DATASET_SCHEMA_ID
from src.tasks.blcs.model_io.checkpoints import (
    resolve_blcs_track_query_reference_contract,
    validate_checkpoint_path,
)
from src.tasks.blcs.model_io.training import compose_blcs_training
from src.tasks.blcs.training.tracking_lightning_module import (
    BLCSTrackingLightningModule,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import CourtCoordinateNormalization

_PASS_NAMES = ("same_side", "opposite_side")
_TRAINER_FIELDS = frozenset(
    {
        "accelerator",
        "devices",
        "precision",
        "deterministic",
        "enable_progress_bar",
        "enable_model_summary",
    }
)
_EVALUATION_FIELDS = frozenset(
    {"task", "checkpoint_path", "output_dir", "passes", "trainer"}
)
EvaluationPrecision: TypeAlias = Literal["32-true", "bf16-mixed"]


def _exact(
    value: object,
    fields: frozenset[str],
    *,
    location: str,
) -> Mapping[str, object]:
    mapping: Mapping[str, object] = as_config_mapping(value, path=location)
    missing = sorted(fields - set(mapping))
    unknown = sorted(set(mapping) - fields)
    if missing:
        raise MissingConfigurationKeyError(
            f"Missing required configuration key(s): "
            f"{', '.join(f'{location}.{key}' for key in missing)}."
        )
    if unknown:
        raise UnknownConfigurationKeyError(
            f"Unknown configuration key(s): "
            f"{', '.join(f'{location}.{key}' for key in unknown)}."
        )
    return mapping


def _resolved_container(config: DictConfig) -> dict[str, object]:
    value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ConfigurationTypeError(
            "BLCS evaluation config must resolve to a mapping."
        )
    return cast("dict[str, object]", value)


def _training_config(
    config: DictConfig, *, reference_camera_id: str | None
) -> DictConfig:
    value = _resolved_container(config)
    value.pop("evaluation", None)
    data = value.get("data")
    if not isinstance(data, dict):
        raise ConfigurationTypeError("BLCS evaluation data must be a mapping.")
    if reference_camera_id is not None:
        data["evaluation_reference_camera_id"] = reference_camera_id
    result = OmegaConf.create(value)
    if not isinstance(result, DictConfig):
        raise ConfigurationTypeError("BLCS task config did not compose as DictConfig.")
    validate_training_boundary(result)
    return result


@dataclass(frozen=True, slots=True)
class BLCSReferenceCounterfactualConfig:
    """Strict task-local boundary for two checkpoint-only Lightning test passes."""

    checkpoint_path: Path
    output_dir: Path
    accelerator: str
    devices: int
    precision: EvaluationPrecision
    deterministic: bool
    enable_progress_bar: bool
    enable_model_summary: bool

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSReferenceCounterfactualConfig:
        """Validate task, data, checkpoint, output, and Trainer settings exactly."""
        task_config = _training_config(config, reference_camera_id=None)
        root = as_config_mapping(config, path="configuration")
        evaluation = _exact(
            require_config_mapping(root, "evaluation", path="configuration"),
            _EVALUATION_FIELDS,
            location="evaluation",
        )
        trainer = _exact(
            evaluation["trainer"],
            _TRAINER_FIELDS,
            location="evaluation.trainer",
        )
        if evaluation["task"] != "blcs":
            raise SemanticConfigurationError("evaluation.task must be exactly 'blcs'.")
        raw_passes = evaluation["passes"]
        if not isinstance(raw_passes, Sequence) or isinstance(raw_passes, (str, bytes)):
            raise ConfigurationTypeError("evaluation.passes must be a sequence.")
        if tuple(raw_passes) != _PASS_NAMES:
            raise SemanticConfigurationError(
                "evaluation.passes must be exactly [same_side, opposite_side]."
            )
        for path, expected in (
            ("data.seq_len_range", (128, 128)),
            ("data.num_views_range", (6, 6)),
        ):
            section, key = path.split(".")
            section_mapping = require_config_mapping(
                root, section, path="configuration"
            )
            raw_value = section_mapping[key]
            if not isinstance(raw_value, Sequence) or isinstance(
                raw_value, (str, bytes)
            ):
                raise ConfigurationTypeError(f"{path} must be an integer sequence.")
            if tuple(raw_value) != expected:
                raise SemanticConfigurationError(
                    f"{path} must be exactly {list(expected)}."
                )
        data = require_config_mapping(root, "data", path="configuration")
        if data.get("camera_mode") != "first":
            raise SemanticConfigurationError(
                "data.camera_mode must be 'first' for ordered six-view evaluation."
            )
        if data.get("backend") != "default":
            raise SemanticConfigurationError(
                "data.backend must be 'default' for persisted test evaluation."
            )
        if data.get("evaluation_reference_camera_id") != "manifest_resolved_per_side":
            raise SemanticConfigurationError(
                "data.evaluation_reference_camera_id is owned by the persisted-side manifest."
            )
        normalization = require_config_mapping(
            root, "court_coordinate_normalization", path="configuration"
        )
        if normalization.get("version") != "v2":
            raise SemanticConfigurationError(
                "BLCS counterfactual evaluation requires normalization v2."
            )
        runtime = TrainingRuntimeConfig.from_config(
            task_config,
            repository_root=PROJECT_ROOT,
        )
        if (
            runtime.run.resume is not None
            or runtime.run.init_weights is not None
            or runtime.run.fast_dev_run
            or runtime.run.dry_run
        ):
            raise SemanticConfigurationError(
                "Checkpoint-only evaluation forbids resume/init_weights/fast_dev_run/dry_run."
            )
        if runtime.training.compile.enabled:
            raise SemanticConfigurationError(
                "Checkpoint-only evaluation requires training.compile.enabled=false."
            )
        checkpoint_value = evaluation["checkpoint_path"]
        output_value = evaluation["output_dir"]
        if type(checkpoint_value) is not str or not checkpoint_value.strip():
            raise ConfigurationTypeError(
                "evaluation.checkpoint_path must be a non-empty string."
            )
        if type(output_value) is not str or not output_value.strip():
            raise ConfigurationTypeError(
                "evaluation.output_dir must be a non-empty string."
            )
        checkpoint = runtime.resolver.resolve_configured(
            PathRole.CHECKPOINT,
            checkpoint_value,
        )
        if not checkpoint.is_file():
            raise SemanticConfigurationError(
                f"evaluation.checkpoint_path is not an existing file: {checkpoint}."
            )
        if "retry3" in checkpoint.as_posix().lower():
            raise SemanticConfigurationError(
                "Entropy-seeded retry3 checkpoints are excluded from Attempt-2 evidence."
            )
        repro = os.environ.get("TENNIS_REPRO_DIR")
        if not repro:
            raise SemanticConfigurationError(
                "TENNIS_REPRO_DIR is required for queue-registerable evaluation output."
            )
        repro_root = Path(repro)
        if not repro_root.is_absolute():
            raise SemanticConfigurationError(
                "TENNIS_REPRO_DIR must be an absolute path."
            )
        output_resolver = PathResolver(
            replace(runtime.resolver.roots, output_root=repro_root.resolve())
        )
        expected_output = output_resolver.resolve(PathRole.OUTPUT, "predictions")
        output = output_resolver.resolve_configured(PathRole.OUTPUT, output_value)
        if output != expected_output:
            raise SemanticConfigurationError(
                "evaluation.output_dir must be exactly $TENNIS_REPRO_DIR/predictions."
            )
        accelerator = trainer["accelerator"]
        if accelerator not in {"auto", "cpu", "gpu"}:
            raise SemanticConfigurationError(
                "evaluation.trainer.accelerator must be auto, cpu, or gpu."
            )
        devices = trainer["devices"]
        if type(devices) is not int or devices != 1:
            raise SemanticConfigurationError(
                "evaluation.trainer.devices must be exactly 1."
            )
        precision = trainer["precision"]
        if precision not in {"32-true", "bf16-mixed"}:
            raise SemanticConfigurationError(
                "evaluation.trainer.precision must be 32-true or bf16-mixed."
            )
        for key in ("deterministic", "enable_progress_bar", "enable_model_summary"):
            if type(trainer[key]) is not bool:
                raise ConfigurationTypeError(
                    f"evaluation.trainer.{key} must be exactly bool."
                )
        if trainer["deterministic"] is not True:
            raise SemanticConfigurationError(
                "evaluation.trainer.deterministic must be true."
            )
        return cls(
            checkpoint_path=checkpoint,
            output_dir=output,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            deterministic=cast("bool", trainer["deterministic"]),
            enable_progress_bar=cast("bool", trainer["enable_progress_bar"]),
            enable_model_summary=cast("bool", trainer["enable_model_summary"]),
        )


def _required_array(
    arrays: Mapping[str, np.ndarray[Any, Any]], key: str
) -> np.ndarray[Any, Any]:
    try:
        value = arrays[key]
    except KeyError as error:
        raise ReferenceCounterfactualError(
            f"BLCS raw prediction payload is missing {key!r}."
        ) from error
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ReferenceCounterfactualError(f"BLCS raw field {key!r} is invalid.")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    except (OSError, ValueError) as error:
        raise ReferenceCounterfactualError(
            f"Cannot load BLCS raw prediction payload {path}: {error}."
        ) from error


def _as_physical_position(
    value: np.ndarray[Any, Any],
    normalization: CourtCoordinateNormalization,
    provenance: Any,
) -> np.ndarray[Any, Any]:
    metres = normalization.denormalize_position(value)
    restored = court_points_target_to_physical(metres, provenance)
    return cast("np.ndarray[Any, Any]", np.asarray(restored))


def _as_physical_velocity(
    value: np.ndarray[Any, Any],
    normalization: CourtCoordinateNormalization,
    provenance: Any,
) -> np.ndarray[Any, Any]:
    metres = normalization.denormalize_velocity(value)
    restored = court_vectors_target_to_physical(metres, provenance)
    return cast("np.ndarray[Any, Any]", np.asarray(restored))


def build_blcs_counterfactual_pass(
    prediction_path: Path,
    *,
    side: ReferenceCounterfactualSide,
    identity: ReferenceCounterfactualRunIdentity,
    manifest: ReferenceCounterfactualManifest,
    normalization: CourtCoordinateNormalization,
    window_bounds: Mapping[str, tuple[int, int]],
) -> ReferenceCounterfactualPass:
    """Adapt one BLCS raw Lightning payload without reproducing pair metrics."""
    if identity.task != "blcs":
        raise ReferenceCounterfactualError(
            "BLCS adapter received another task identity."
        )
    arrays = canonicalize_reference_counterfactual_raw_payload(
        _load_arrays(prediction_path),
        manifest=manifest,
        task="blcs",
    )
    batch_size = len(manifest.scenes)
    scene_ids = _required_array(arrays, "scene_ids")
    local_order = _required_array(arrays, "view_camera_id_strings")
    ref_strings = _required_array(arrays, "reference_camera_id_string")
    ref_indices = _required_array(arrays, "reference_view_index")
    ref_codes = _required_array(arrays, "reference_camera_id")
    view_codes = _required_array(arrays, "view_camera_ids")
    ref_transforms = _required_array(arrays, "reference_from_physical")
    inverse_transforms = _required_array(arrays, "physical_from_reference")
    prediction_norm = _required_array(arrays, "pred_position")
    target_norm = _required_array(arrays, "target_position")
    target_velocity = _required_array(arrays, "target_velocity")
    target_presence = _required_array(arrays, "target_presence")
    frame_valid = _required_array(arrays, "frame_valid")
    if local_order.shape != (batch_size, 6):
        raise ReferenceCounterfactualError(
            "BLCS raw view_camera_id_strings must have exact shape (B, 6)."
        )
    if (
        prediction_norm.shape != target_norm.shape
        or prediction_norm.shape[0] != batch_size
    ):
        raise ReferenceCounterfactualError(
            "BLCS raw position prediction/target shape or row count differs."
        )
    if target_presence.shape != prediction_norm.shape[:-1]:
        raise ReferenceCounterfactualError(
            "BLCS target_presence must match position leading axes."
        )
    if frame_valid.shape != prediction_norm.shape[:2]:
        raise ReferenceCounterfactualError(
            "BLCS frame_valid must have exact (B, T) shape."
        )
    if (
        not np.issubdtype(target_presence.dtype, np.bool_)
        or frame_valid.dtype != np.bool_
    ):
        raise ReferenceCounterfactualError(
            "BLCS target_presence and frame_valid must use bool dtype."
        )
    valid_mask = np.asarray(target_presence & frame_valid[..., None], dtype=np.bool_)
    rows: list[ReferenceCounterfactualPassRow] = []
    target_metres = np.asarray(normalization.denormalize_position(target_norm))
    prediction_metres = np.asarray(normalization.denormalize_position(prediction_norm))
    for index in range(batch_size):
        scene_id = str(scene_ids[index])
        scene = manifest.scene(scene_id)
        ordering = tuple(str(value) for value in local_order[index])
        if any(not value for value in ordering) or len(set(ordering)) != 6:
            raise ReferenceCounterfactualError(
                f"BLCS raw row {index} has padded/duplicate canonical camera IDs."
            )
        if ordering != scene.local_ordering:
            raise ReferenceCounterfactualError(
                f"BLCS raw row {index} local camera order differs from manifest."
            )
        expected = scene.selection(side)
        reference_index = int(ref_indices[index])
        reference_id = str(ref_strings[index])
        if (
            reference_id != expected.camera_id
            or reference_index != expected.local_index
        ):
            raise ReferenceCounterfactualError(
                f"BLCS raw row {index} reference differs from {side} manifest selection."
            )
        scene.validate_camera_codes(
            tuple(int(value) for value in view_codes[index]),
            reference_camera_id=reference_id,
            reference_camera_id_code=int(ref_codes[index]),
        )
        if not np.array_equal(
            ref_transforms[index],
            np.asarray(expected.provenance.reference_from_physical),
        ) or not np.array_equal(
            inverse_transforms[index],
            np.asarray(expected.provenance.physical_from_reference),
        ):
            raise ReferenceCounterfactualError(
                f"BLCS raw row {index} persisted reference transforms differ."
            )
        for key, expected_marker in (
            ("court_keypoint_contract", identity.court_keypoint_contract),
            ("target_frame_contract", identity.target_frame_contract),
            ("track_query_rope_contract", identity.track_query_rope_contract),
            ("reference_selector_mode", identity.selector_mode),
        ):
            values = _required_array(arrays, key)
            if values.shape != (batch_size,) or str(values[index]) != expected_marker:
                raise ReferenceCounterfactualError(
                    f"BLCS raw row {index} has mismatched {key}."
                )
        try:
            start, stop = window_bounds[scene_id]
        except KeyError as error:
            raise ReferenceCounterfactualError(
                f"BLCS raw row {index} has no exact test-window identity."
            ) from error
        physical_target = _as_physical_position(
            target_norm[index], normalization, expected.provenance
        )
        physical_velocity = _as_physical_velocity(
            target_velocity[index], normalization, expected.provenance
        )
        row_valid = valid_mask[index]
        physical_target_digest = masked_counterfactual_quantity_for_digest(
            physical_target, row_valid
        )
        physical_velocity_digest = masked_counterfactual_quantity_for_digest(
            physical_velocity, row_valid
        )
        rows.append(
            ReferenceCounterfactualPassRow(
                key=scene.key,
                window_start=start,
                window_stop=stop,
                reference_camera_id=reference_id,
                reference_view_index=reference_index,
                provenance=expected.provenance,
                frame_digest=array_payload_sha256(
                    {
                        "frame_valid": frame_valid[index],
                        "padding_mask": _required_array(arrays, "padding_mask")[index],
                    }
                ),
                lifecycle_digest=array_payload_sha256(
                    {
                        "target_presence": target_presence[index],
                        "target_instance_id": _required_array(
                            arrays, "target_instance_id"
                        )[index],
                        "target_slot_mask": _required_array(arrays, "target_slot_mask")[
                            index
                        ],
                        "candidate_gt_index": _required_array(
                            arrays, "candidate_gt_index"
                        )[index],
                    }
                ),
                observation_digest=array_payload_sha256(
                    {
                        "ball_uv": _required_array(arrays, "ball_uv")[index],
                        "ball_vis": _required_array(arrays, "ball_vis")[index],
                        "clean_ball_uv": _required_array(arrays, "clean_ball_uv")[
                            index
                        ],
                        "clean_ball_vis": _required_array(arrays, "clean_ball_vis")[
                            index
                        ],
                    }
                ),
                target_digest=array_payload_sha256(
                    {
                        "position_physical_m": physical_target_digest,
                        "velocity_physical_mps": physical_velocity_digest,
                    }
                ),
            )
        )
    return ReferenceCounterfactualPass(
        schema_version=REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
        side=side,
        identity=identity,
        quantity_schema=ReferenceCounterfactualQuantitySchema.for_task("blcs"),
        rows=tuple(rows),
        valid_mask=valid_mask,
        position=ReferenceCounterfactualQuantityArrays(
            prediction=prediction_metres,
            target=target_metres,
            quantity="point",
        ),
    )


def _uniform_reference_id(
    manifest: ReferenceCounterfactualManifest,
    side: ReferenceCounterfactualSide,
) -> str:
    values: set[str] = {scene.selection(side).camera_id for scene in manifest.scenes}
    if len(values) != 1:
        raise ReferenceCounterfactualError(
            "BLCS DataModule requires one persisted-side stable camera ID shared "
            "by every test scene; per-scene fallback is forbidden."
        )
    return next(iter(values))


def _window_bounds(datamodule: Any) -> dict[str, tuple[int, int]]:
    dataset = datamodule.test_dataset
    headers = getattr(dataset, "scene_headers", None)
    config = getattr(dataset, "config", None)
    if not isinstance(headers, list) or config is None:
        raise ReferenceCounterfactualError(
            "BLCS test dataset does not expose exact persisted scene headers."
        )
    if tuple(config.seq_len_range) != (128, 128) or config.crop_mode != "center":
        raise ReferenceCounterfactualError(
            "BLCS counterfactual window must be deterministic centered T=128."
        )
    result: dict[str, tuple[int, int]] = {}
    for header in headers:
        scene_id = header.path.name
        start = (int(header.num_frames) - 128) // 2
        result[scene_id] = (start, start + 128)
    return result


def _run_pass(
    config: DictConfig,
    runtime: BLCSReferenceCounterfactualConfig,
    *,
    side: ReferenceCounterfactualSide,
    reference_camera_id: str,
    output_dir: Path,
) -> tuple[Path, dict[str, tuple[int, int]]]:
    task_config = _training_config(config, reference_camera_id=reference_camera_id)
    pl.seed_everything(int(task_config.run.seed), workers=True)
    composition = compose_blcs_training(task_config, generator_config=None)
    module = composition.lightning_module
    if not isinstance(module, BLCSTrackingLightningModule):
        raise ReferenceCounterfactualError(
            "BLCS evaluator requires the strict tracking Lightning runtime."
        )
    module.set_counterfactual_prediction_dir(output_dir)
    trainer = pl.Trainer(
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        precision=runtime.precision,
        deterministic=runtime.deterministic,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=runtime.enable_progress_bar,
        enable_model_summary=runtime.enable_model_summary,
    )
    trainer.test(
        module,
        datamodule=composition.datamodule,
        ckpt_path=str(runtime.checkpoint_path),
        verbose=False,
        weights_only=False,
    )
    prediction_path = output_dir / "pred_test.npz"
    if not prediction_path.is_file():
        raise ReferenceCounterfactualError(
            f"BLCS {side} test pass did not produce {prediction_path}."
        )
    return prediction_path, _window_bounds(composition.datamodule)


def run_blcs_reference_counterfactual(
    config: DictConfig,
) -> ReferenceCounterfactualReportPaths:
    """Execute two strict checkpoint-only BLCS passes and atomically join them."""
    runtime = BLCSReferenceCounterfactualConfig.from_config(config)
    if runtime.output_dir.exists():
        raise ReferenceCounterfactualError(
            f"BLCS counterfactual output refuses overwrite: {runtime.output_dir}."
        )
    task_config = _training_config(config, reference_camera_id=None)
    shared_runtime = TrainingRuntimeConfig.from_config(
        task_config, repository_root=PROJECT_ROOT
    )
    contract = resolve_blcs_track_query_reference_contract(task_config)
    selector = contract.reference_selector_mode
    if selector is None:
        raise ReferenceCounterfactualError(
            "BLCS counterfactual evaluation rejects legacy v1 checkpoints/configs."
        )
    normalization = CourtCoordinateNormalizationConfig.from_config(task_config).contract
    validate_checkpoint_path(
        runtime.checkpoint_path,
        normalization,
        runtime_court_keypoints=parse_court_keypoint_contract(task_config),
        runtime_track_query_reference=contract,
    )
    data_root = shared_runtime.resolver.resolve(
        PathRole.DATA, str(task_config.data.scene_dir)
    )
    test_split = data_root / "test.txt"
    if not test_split.is_file():
        raise ReferenceCounterfactualError(
            f"BLCS counterfactual test split is missing: {test_split}."
        )
    scene_ids = tuple(
        line.strip() for line in test_split.read_text().splitlines() if line.strip()
    )
    manifest = build_reference_counterfactual_manifest(
        data_root,
        expected_dataset_schema_id=BLCS_DATASET_SCHEMA_ID,
        scene_ids=scene_ids,
    )
    resolved_config = _resolved_container(config)
    identity = ReferenceCounterfactualRunIdentity.create(
        task="blcs",
        seed=int(task_config.run.seed),
        selector_mode=cast("Any", selector.value),
        resolved_config=resolved_config,
        checkpoint_sha256=file_sha256(runtime.checkpoint_path),
        manifest_digest=manifest.digest,
        court_keypoint_contract=contract.court_keypoint_contract,
        target_frame_contract=contract.target_frame_contract,
        track_query_rope_contract=contract.track_query_rope_contract.value,
    )
    runtime.output_dir.mkdir(parents=True, exist_ok=False)
    passes: dict[str, ReferenceCounterfactualPass] = {}
    for side in _PASS_NAMES:
        typed_side = cast("ReferenceCounterfactualSide", side)
        raw_dir = runtime.output_dir / f"raw_{side}"
        prediction_path, windows = _run_pass(
            config,
            runtime,
            side=typed_side,
            reference_camera_id=_uniform_reference_id(manifest, typed_side),
            output_dir=raw_dir,
        )
        passes[side] = build_blcs_counterfactual_pass(
            prediction_path,
            side=typed_side,
            identity=identity,
            manifest=manifest,
            normalization=normalization,
            window_bounds=windows,
        )
    report = evaluate_reference_counterfactual(
        manifest,
        passes["same_side"],
        passes["opposite_side"],
    )
    return write_reference_counterfactual_report(report, runtime.output_dir)


def _validate_boundary(config: DictConfig) -> None:
    BLCSReferenceCounterfactualConfig.from_config(config)


register_boundary_validator(
    "blcs.evaluate_reference_counterfactual", _validate_boundary
)


__all__ = [
    "BLCSReferenceCounterfactualConfig",
    "build_blcs_counterfactual_pass",
    "run_blcs_reference_counterfactual",
]
