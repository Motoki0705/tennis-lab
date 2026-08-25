"""PLCS adapter for the shared strict reference-counterfactual evaluator."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import as_config_mapping, require_config_mapping
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
    evaluate_reference_counterfactual,
    file_sha256,
    masked_counterfactual_quantity_for_digest,
    validate_reference_counterfactual_raw_payload,
    write_reference_counterfactual_report,
)
from src.tasks.base.generate_dataset import (
    court_headings_target_to_physical,
    court_points_target_to_physical,
    court_world_joints_target_to_physical,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import PLCS_GENERATED_DATASET_SCHEMA_ID
from src.tasks.plcs.model_io import (
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_court_keypoints,
    validate_plcs_checkpoint_normalization,
    validate_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
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
    mapping = as_config_mapping(value, path=location)
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
    return cast("Mapping[str, object]", mapping)


def _resolved_container(config: DictConfig) -> dict[str, object]:
    value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ConfigurationTypeError(
            "PLCS evaluation config must resolve to a mapping."
        )
    return cast("dict[str, object]", value)


def _training_config(
    config: DictConfig, *, reference_camera_id: str | None
) -> DictConfig:
    value = _resolved_container(config)
    value.pop("evaluation", None)
    data = value.get("data")
    if not isinstance(data, dict):
        raise ConfigurationTypeError("PLCS evaluation data must be a mapping.")
    if reference_camera_id is not None:
        data["evaluation_reference_camera_id"] = reference_camera_id
    result = OmegaConf.create(value)
    if not isinstance(result, DictConfig):
        raise ConfigurationTypeError("PLCS task config did not compose as DictConfig.")
    PLCSTrainingConfig.from_config(result)
    return result


@dataclass(frozen=True, slots=True)
class PLCSReferenceCounterfactualConfig:
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
    def from_config(cls, config: DictConfig) -> PLCSReferenceCounterfactualConfig:
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
        if evaluation["task"] != "plcs":
            raise SemanticConfigurationError("evaluation.task must be exactly 'plcs'.")
        raw_passes = evaluation["passes"]
        if not isinstance(raw_passes, Sequence) or isinstance(raw_passes, (str, bytes)):
            raise ConfigurationTypeError("evaluation.passes must be a sequence.")
        if tuple(raw_passes) != _PASS_NAMES:
            raise SemanticConfigurationError(
                "evaluation.passes must be exactly [same_side, opposite_side]."
            )
        data = require_config_mapping(root, "data", path="configuration")
        for key, expected in (
            ("seq_len_range", (128, 128)),
            ("num_views_range", (6, 6)),
        ):
            raw_value = data[key]
            if not isinstance(raw_value, Sequence) or isinstance(
                raw_value, (str, bytes)
            ):
                raise ConfigurationTypeError(f"data.{key} must be an integer sequence.")
            if tuple(raw_value) != expected:
                raise SemanticConfigurationError(
                    f"data.{key} must be exactly {list(expected)}."
                )
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
                "PLCS counterfactual evaluation requires normalization v2."
            )
        runtime = PLCSTrainingConfig.from_config(task_config)
        if (
            runtime.shared.run.resume is not None
            or runtime.shared.run.init_weights is not None
            or runtime.shared.run.fast_dev_run
            or runtime.shared.run.dry_run
        ):
            raise SemanticConfigurationError(
                "Checkpoint-only evaluation forbids resume/init_weights/fast_dev_run/dry_run."
            )
        if runtime.shared.training.compile.enabled:
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
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
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
        expected_output = (Path(repro).resolve() / "predictions").resolve()
        output = Path(output_value).resolve()
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
            f"PLCS raw prediction payload is missing {key!r}."
        ) from error
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ReferenceCounterfactualError(f"PLCS raw field {key!r} is invalid.")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
    except (OSError, ValueError) as error:
        raise ReferenceCounterfactualError(
            f"Cannot load PLCS raw prediction payload {path}: {error}."
        ) from error


def build_plcs_counterfactual_pass(
    prediction_path: Path,
    *,
    side: ReferenceCounterfactualSide,
    identity: ReferenceCounterfactualRunIdentity,
    manifest: ReferenceCounterfactualManifest,
    normalization: CourtCoordinateNormalization,
    window_bounds: Mapping[str, tuple[int, int]],
) -> ReferenceCounterfactualPass:
    """Adapt one PLCS raw Lightning payload without reproducing pair metrics."""
    if identity.task != "plcs":
        raise ReferenceCounterfactualError(
            "PLCS adapter received another task identity."
        )
    arrays = _load_arrays(prediction_path)
    batch_size = validate_reference_counterfactual_raw_payload(arrays, task="plcs")
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
    pred_heading = _required_array(arrays, "pred_rotation")
    target_heading = _required_array(arrays, "target_rotation")
    target_presence = _required_array(arrays, "target_presence")
    padding_mask = _required_array(arrays, "padding_mask")
    if local_order.shape != (batch_size, 6):
        raise ReferenceCounterfactualError(
            "PLCS raw view_camera_id_strings must have exact shape (B, 6)."
        )
    if (
        prediction_norm.shape != target_norm.shape
        or prediction_norm.shape[0] != batch_size
    ):
        raise ReferenceCounterfactualError(
            "PLCS raw position prediction/target shape or row count differs."
        )
    if (
        pred_heading.shape != target_heading.shape
        or pred_heading.shape[:-1] != prediction_norm.shape[:-1]
    ):
        raise ReferenceCounterfactualError(
            "PLCS learned heading prediction/target shape differs from position."
        )
    if target_presence.shape != prediction_norm.shape[:-1]:
        raise ReferenceCounterfactualError(
            "PLCS target_presence must match position leading axes."
        )
    if padding_mask.shape[:1] != (batch_size,) or padding_mask.ndim != 3:
        raise ReferenceCounterfactualError(
            "PLCS padding_mask must have exact (B, V, T) axes."
        )
    if target_presence.dtype != np.bool_ or padding_mask.dtype != np.bool_:
        raise ReferenceCounterfactualError(
            "PLCS target_presence and padding_mask must use bool dtype."
        )
    frame_valid = np.logical_not(padding_mask).any(axis=1)
    if frame_valid.shape != prediction_norm.shape[:2]:
        raise ReferenceCounterfactualError(
            "PLCS frame validity does not match predicted time axes."
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
                f"PLCS raw row {index} has padded/duplicate canonical camera IDs."
            )
        if ordering != scene.local_ordering:
            raise ReferenceCounterfactualError(
                f"PLCS raw row {index} local camera order differs from manifest."
            )
        expected = scene.selection(side)
        reference_index = int(ref_indices[index])
        reference_id = str(ref_strings[index])
        if (
            reference_id != expected.camera_id
            or reference_index != expected.local_index
        ):
            raise ReferenceCounterfactualError(
                f"PLCS raw row {index} reference differs from {side} manifest selection."
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
                f"PLCS raw row {index} persisted reference transforms differ."
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
                    f"PLCS raw row {index} has mismatched {key}."
                )
        try:
            start, stop = window_bounds[scene_id]
        except KeyError as error:
            raise ReferenceCounterfactualError(
                f"PLCS raw row {index} has no exact test-window identity."
            ) from error
        physical_target = np.asarray(
            court_points_target_to_physical(target_metres[index], expected.provenance)
        )
        physical_heading = np.asarray(
            court_headings_target_to_physical(
                target_heading[index], expected.provenance
            )
        )
        target_world = _required_array(arrays, "target_human_kp_3d")[index]
        physical_world = np.asarray(
            court_world_joints_target_to_physical(target_world, expected.provenance)
        )
        row_valid = valid_mask[index]
        physical_target_digest = masked_counterfactual_quantity_for_digest(
            physical_target, row_valid
        )
        physical_heading_digest = masked_counterfactual_quantity_for_digest(
            physical_heading, row_valid
        )
        physical_world_digest = masked_counterfactual_quantity_for_digest(
            physical_world, row_valid
        )
        canonical_pose = _required_array(
            arrays, "target_canonical_pose_3d"
        )[index]
        canonical_pose_digest = masked_counterfactual_quantity_for_digest(
            canonical_pose, row_valid
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
                        "padding_mask": padding_mask[index],
                    }
                ),
                lifecycle_digest=array_payload_sha256(
                    {
                        "target_presence": target_presence[index],
                        "target_instance_id": _required_array(
                            arrays, "target_instance_id"
                        )[index],
                        "detection_gt_index": _required_array(
                            arrays, "detection_gt_index"
                        )[index],
                    }
                ),
                observation_digest=array_payload_sha256(
                    {
                        "human_kp": _required_array(arrays, "human_kp")[index],
                        "human_vis": _required_array(arrays, "human_vis")[index],
                        "clean_human_kp": _required_array(arrays, "clean_human_kp")[
                            index
                        ],
                        "clean_human_vis": _required_array(arrays, "clean_human_vis")[
                            index
                        ],
                    }
                ),
                target_digest=array_payload_sha256(
                    {
                        "position_physical_m": physical_target_digest,
                        "heading_physical": physical_heading_digest,
                        "world_joints_physical_m": physical_world_digest,
                        "canonical_pose": canonical_pose_digest,
                    }
                ),
            )
        )
    return ReferenceCounterfactualPass(
        schema_version=REFERENCE_COUNTERFACTUAL_SCHEMA_VERSION,
        side=side,
        identity=identity,
        quantity_schema=ReferenceCounterfactualQuantitySchema.for_task("plcs"),
        rows=tuple(rows),
        valid_mask=valid_mask,
        position=ReferenceCounterfactualQuantityArrays(
            prediction=prediction_metres,
            target=target_metres,
            quantity="point",
        ),
        heading=ReferenceCounterfactualQuantityArrays(
            prediction=pred_heading,
            target=target_heading,
            quantity="heading",
        ),
    )


def _uniform_reference_id(
    manifest: ReferenceCounterfactualManifest,
    side: ReferenceCounterfactualSide,
) -> str:
    values: set[str] = {scene.selection(side).camera_id for scene in manifest.scenes}
    if len(values) != 1:
        raise ReferenceCounterfactualError(
            "PLCS DataModule requires one persisted-side stable camera ID shared "
            "by every test scene; per-scene fallback is forbidden."
        )
    return next(iter(values))


def _window_bounds(datamodule: Any) -> dict[str, tuple[int, int]]:
    dataset = datamodule.test_dataset
    headers = getattr(dataset, "scene_headers", None)
    config = getattr(dataset, "config", None)
    if not isinstance(headers, list) or config is None:
        raise ReferenceCounterfactualError(
            "PLCS test dataset does not expose exact persisted scene headers."
        )
    if tuple(config.seq_len_range) != (128, 128) or config.crop_mode != "center":
        raise ReferenceCounterfactualError(
            "PLCS counterfactual window must be deterministic centered T=128."
        )
    result: dict[str, tuple[int, int]] = {}
    for header in headers:
        scene_id = header.path.name
        start = (int(header.num_frames) - 128) // 2
        result[scene_id] = (start, start + 128)
    return result


def _run_pass(
    config: DictConfig,
    runtime: PLCSReferenceCounterfactualConfig,
    *,
    side: ReferenceCounterfactualSide,
    reference_camera_id: str,
    output_dir: Path,
) -> tuple[Path, dict[str, tuple[int, int]]]:
    task_config = _training_config(config, reference_camera_id=reference_camera_id)
    pl.seed_everything(int(task_config.run.seed), workers=True)
    datamodule = build_plcs_datamodule(task_config)
    module = build_plcs_lightning_module(task_config)
    if not isinstance(module, PLCSTrackingLightningModule):
        raise ReferenceCounterfactualError(
            "PLCS evaluator requires the strict tracking Lightning runtime."
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
        datamodule=datamodule,
        ckpt_path=str(runtime.checkpoint_path),
        verbose=False,
        weights_only=False,
    )
    prediction_path = output_dir / "pred_test.npz"
    if not prediction_path.is_file():
        raise ReferenceCounterfactualError(
            f"PLCS {side} test pass did not produce {prediction_path}."
        )
    return prediction_path, _window_bounds(datamodule)


def run_plcs_reference_counterfactual(
    config: DictConfig,
) -> ReferenceCounterfactualReportPaths:
    """Execute two strict checkpoint-only PLCS passes and atomically join them."""
    runtime = PLCSReferenceCounterfactualConfig.from_config(config)
    if runtime.output_dir.exists():
        raise ReferenceCounterfactualError(
            f"PLCS counterfactual output refuses overwrite: {runtime.output_dir}."
        )
    task_config = _training_config(config, reference_camera_id=None)
    plcs_runtime = PLCSTrainingConfig.from_config(task_config)
    contract = resolve_plcs_track_query_reference_contract(
        plcs_runtime.model,
        plcs_runtime.court_keypoint_contract,
    )
    selector = contract.reference_selector_mode
    if selector is None:
        raise ReferenceCounterfactualError(
            "PLCS counterfactual evaluation rejects legacy v1 checkpoints/configs."
        )
    checkpoint = torch.load(
        runtime.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ReferenceCounterfactualError(
            "PLCS counterfactual checkpoint root must be a mapping."
        )
    normalization = plcs_runtime.court_coordinate_normalization.contract
    validate_plcs_checkpoint_normalization(checkpoint, normalization)
    validate_plcs_checkpoint_court_keypoints(
        checkpoint,
        plcs_runtime.court_keypoint_contract,
    )
    validate_plcs_checkpoint_track_query_reference(checkpoint, contract)
    data_root = plcs_runtime.data.scene_dir
    test_split = data_root / "test.txt"
    if not test_split.is_file():
        raise ReferenceCounterfactualError(
            f"PLCS counterfactual test split is missing: {test_split}."
        )
    scene_ids = tuple(
        line.strip() for line in test_split.read_text().splitlines() if line.strip()
    )
    manifest = build_reference_counterfactual_manifest(
        data_root,
        expected_dataset_schema_id=PLCS_GENERATED_DATASET_SCHEMA_ID,
        scene_ids=scene_ids,
    )
    identity = ReferenceCounterfactualRunIdentity.create(
        task="plcs",
        seed=plcs_runtime.shared.run.seed,
        selector_mode=cast("Any", selector.value),
        resolved_config=_resolved_container(config),
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
        passes[side] = build_plcs_counterfactual_pass(
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
    PLCSReferenceCounterfactualConfig.from_config(config)


register_boundary_validator(
    "plcs.evaluate_reference_counterfactual", _validate_boundary
)


__all__ = [
    "PLCSReferenceCounterfactualConfig",
    "build_plcs_counterfactual_pass",
    "run_plcs_reference_counterfactual",
]
