"""Strict typed contracts for every player-detection runtime boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    ConfigMapping,
    TrainingRuntimeConfig,
    as_config_mapping,
    exact_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.utils.configuration import (
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
)
from src.utils.paths import PROJECT_ROOT

TASK_NAME = "player_detection"
ANNOTATION_STATUSES = frozenset({"completed", "partial"})


def _mapping(parent: ConfigMapping, key: str, path: str, keys: set[str]) -> ConfigMapping:
    return exact_config_mapping(
        require_config_mapping(parent, key, path=path),
        path=f"{path}.{key}",
        required_keys=keys,
    )


def _int(mapping: ConfigMapping, key: str, path: str, *, minimum: int) -> int:
    value = cast(int, require_config_value(mapping, key, int, path=path))
    if value < minimum:
        raise SemanticConfigurationError(f"{path}.{key} must be >= {minimum}, got {value}.")
    return value


def _float(
    mapping: ConfigMapping, key: str, path: str, *, low: float, high: float
) -> float:
    value = cast(float, require_config_value(mapping, key, (float, int), path=path))
    if not low <= float(value) <= high:
        raise SemanticConfigurationError(
            f"{path}.{key} must be in [{low}, {high}], got {value}."
        )
    return float(value)


def _bool(mapping: ConfigMapping, key: str, path: str) -> bool:
    return cast(bool, require_config_value(mapping, key, bool, path=path))


def _text(mapping: ConfigMapping, key: str, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise SemanticConfigurationError(f"{path}.{key} must be non-empty and trimmed.")
    return value


def _str_list(mapping: ConfigMapping, key: str, path: str) -> tuple[str, ...]:
    raw = require_config_value(mapping, key, list, path=path)
    values = cast(list[object], raw)
    if any(type(value) is not str or not value for value in values):
        raise SemanticConfigurationError(f"{path}.{key} must be a list of non-empty strings.")
    return tuple(cast(list[str], values))


def _resolver(config: ConfigMapping) -> PathResolver:
    return PathResolver(
        RuntimePathRoots.from_mapping(
            require_config_mapping(config, "paths", path="configuration"),
            repository_root=PROJECT_ROOT,
        )
    )


@dataclass(frozen=True, slots=True)
class InputSizeConfig:
    """The DINO resize rule; evaluation must equal the deployed detector's."""

    short_side: int
    max_long_side: int

    @classmethod
    def from_mapping(cls, mapping: ConfigMapping, path: str) -> InputSizeConfig:
        short_side = _int(mapping, "short_side", path, minimum=32)
        max_long_side = _int(mapping, "max_long_side", path, minimum=short_side)
        return cls(short_side, max_long_side)


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerateDatasetConfig:
    annotation_root: Path
    allowed_statuses: frozenset[str]
    dataset_dir: Path
    version: str
    jpeg_quality: int
    val_ratio: float
    test_ratio: float
    split_seed: int
    workers: int
    resolver: PathResolver

    @classmethod
    def from_config(cls, value: object) -> GenerateDatasetConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "source", "dataset", "split", "workers", "run"},
            optional_keys={"hydra"},
        )
        resolver = _resolver(config)
        source = _mapping(config, "source", "configuration", {"annotation_root", "allowed_statuses"})
        dataset = _mapping(config, "dataset", "configuration", {"version", "jpeg_quality"})
        split = _mapping(config, "split", "configuration", {"val_ratio", "test_ratio", "seed"})
        run = _mapping(config, "run", "configuration", {"output_dir"})
        statuses = frozenset(_str_list(source, "allowed_statuses", "source"))
        if not statuses or not statuses <= ANNOTATION_STATUSES:
            raise SemanticConfigurationError(
                f"source.allowed_statuses must be a non-empty subset of {sorted(ANNOTATION_STATUSES)}."
            )
        val_ratio = _float(split, "val_ratio", "split", low=0.0, high=1.0)
        test_ratio = _float(split, "test_ratio", "split", low=0.0, high=1.0)
        if val_ratio + test_ratio >= 1.0:
            raise SemanticConfigurationError("split ratios must sum to less than 1.")
        return cls(
            annotation_root=resolver.resolve(PathRole.OUTPUT, _text(source, "annotation_root", "source")),
            allowed_statuses=statuses,
            dataset_dir=resolver.resolve(PathRole.DATA, _text(run, "output_dir", "run")),
            version=_text(dataset, "version", "dataset"),
            jpeg_quality=_int(dataset, "jpeg_quality", "dataset", minimum=50),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=_int(split, "seed", "split", minimum=0),
            workers=_int(config, "workers", "configuration", minimum=1),
            resolver=resolver,
        )


def validate_generate_boundary(config: DictConfig) -> None:
    GenerateDatasetConfig.from_config(config)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    hflip_prob: float
    short_side_choices: tuple[int, ...]
    brightness: float
    contrast: float
    saturation: float

    @classmethod
    def from_mapping(cls, mapping: ConfigMapping, path: str, *, max_long_side: int) -> AugmentationConfig:
        raw_choices = require_config_value(mapping, "short_side_choices", list, path=path)
        choices = tuple(cast(list[int], raw_choices))
        if not choices or any(type(value) is not int or not 32 <= value <= max_long_side for value in choices):
            raise SemanticConfigurationError(
                f"{path}.short_side_choices must be integers in [32, {max_long_side}]."
            )
        return cls(
            hflip_prob=_float(mapping, "hflip_prob", path, low=0.0, high=1.0),
            short_side_choices=choices,
            brightness=_float(mapping, "brightness", path, low=0.0, high=1.0),
            contrast=_float(mapping, "contrast", path, low=0.0, high=1.0),
            saturation=_float(mapping, "saturation", path, low=0.0, high=1.0),
        )


@dataclass(frozen=True, slots=True)
class FrameSelectionConfig:
    """Which stored frames/instances become detection samples.

    ``require_located_players`` drops frames containing an ``unresolved``
    player: such a player is present without a box, so any detection on it
    would otherwise be trained as a false positive.
    """

    require_reviewed: bool
    require_located_players: bool
    min_visible_box_px: float

    @classmethod
    def from_mapping(cls, mapping: ConfigMapping, path: str) -> FrameSelectionConfig:
        return cls(
            require_reviewed=_bool(mapping, "require_reviewed", path),
            require_located_players=_bool(mapping, "require_located_players", path),
            min_visible_box_px=_float(mapping, "min_visible_box_px", path, low=0.0, high=1e4),
        )


@dataclass(frozen=True, slots=True)
class PlayerDataConfig:
    dataset_dir: Path
    batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    train_samples_per_epoch: int
    val_frame_stride: int
    test_frame_stride: int
    selection: FrameSelectionConfig
    input_size: InputSizeConfig
    augmentation: AugmentationConfig

    @classmethod
    def from_mapping(cls, value: ConfigMapping, resolver: PathResolver) -> PlayerDataConfig:
        path = "data"
        mapping = exact_config_mapping(
            value,
            path=path,
            required_keys={
                "dataset_dir", "batch_size", "eval_batch_size", "num_workers", "pin_memory",
                "train_samples_per_epoch", "val_frame_stride", "test_frame_stride",
                "selection", "input_size", "augmentation",
            },
        )
        input_size = InputSizeConfig.from_mapping(
            _mapping(mapping, "input_size", path, {"short_side", "max_long_side"}), "data.input_size"
        )
        return cls(
            dataset_dir=resolver.resolve(PathRole.DATA, _text(mapping, "dataset_dir", path)),
            batch_size=_int(mapping, "batch_size", path, minimum=1),
            eval_batch_size=_int(mapping, "eval_batch_size", path, minimum=1),
            num_workers=_int(mapping, "num_workers", path, minimum=0),
            pin_memory=_bool(mapping, "pin_memory", path),
            train_samples_per_epoch=_int(mapping, "train_samples_per_epoch", path, minimum=1),
            val_frame_stride=_int(mapping, "val_frame_stride", path, minimum=1),
            test_frame_stride=_int(mapping, "test_frame_stride", path, minimum=1),
            selection=FrameSelectionConfig.from_mapping(
                _mapping(mapping, "selection", path, {"require_reviewed", "require_located_players", "min_visible_box_px"}),
                "data.selection",
            ),
            input_size=input_size,
            augmentation=AugmentationConfig.from_mapping(
                _mapping(mapping, "augmentation", path, {"hflip_prob", "short_side_choices", "brightness", "contrast", "saturation"}),
                "data.augmentation",
                max_long_side=input_size.max_long_side,
            ),
        )


@dataclass(frozen=True, slots=True)
class DinoModelConfig:
    repository: Path
    init_checkpoint: Path
    use_checkpoint: bool
    backbone_freeze_keywords: tuple[str, ...]
    backbone_lr_scale: float

    @classmethod
    def from_mapping(cls, value: ConfigMapping, resolver: PathResolver) -> DinoModelConfig:
        path = "model"
        mapping = exact_config_mapping(
            value,
            path=path,
            required_keys={"name", "repository", "init_checkpoint", "use_checkpoint", "backbone_freeze_keywords", "backbone_lr_scale"},
        )
        if _text(mapping, "name", path) != "dino_4scale_swin_l":
            raise SemanticConfigurationError("model.name must be 'dino_4scale_swin_l'.")
        return cls(
            repository=resolver.resolve(PathRole.EXTERNAL_ASSET, _text(mapping, "repository", path)),
            init_checkpoint=resolver.resolve(PathRole.CHECKPOINT, _text(mapping, "init_checkpoint", path)),
            use_checkpoint=_bool(mapping, "use_checkpoint", path),
            backbone_freeze_keywords=_str_list(mapping, "backbone_freeze_keywords", path),
            backbone_lr_scale=_float(mapping, "backbone_lr_scale", path, low=0.0, high=1.0),
        )


@dataclass(frozen=True, slots=True)
class DetectionEvaluationConfig:
    """Operating point used for precision/recall besides threshold-free AP."""

    score_threshold: float
    iou_threshold: float
    max_detections: int

    @classmethod
    def from_mapping(cls, value: ConfigMapping, path: str) -> DetectionEvaluationConfig:
        mapping = exact_config_mapping(
            value, path=path, required_keys={"score_threshold", "iou_threshold", "max_detections"}
        )
        return cls(
            score_threshold=_float(mapping, "score_threshold", path, low=0.0, high=1.0),
            iou_threshold=_float(mapping, "iou_threshold", path, low=0.0, high=1.0),
            max_detections=_int(mapping, "max_detections", path, minimum=1),
        )


@dataclass(frozen=True, slots=True)
class PlayerTrainingConfig:
    shared: TrainingRuntimeConfig
    data: PlayerDataConfig
    model: DinoModelConfig
    evaluation: DetectionEvaluationConfig

    @classmethod
    def from_config(cls, value: object) -> PlayerTrainingConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "data", "model", "training", "run", "evaluation"},
            optional_keys={"hydra"},
        )
        shared = TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
        if shared.training.compile.enabled:
            raise SemanticConfigurationError(
                "training.compile.enabled must be false for DINO: batches have "
                "per-image resolutions and the deformable-attention op is an "
                "opaque custom CUDA kernel (see player_detection README)."
            )
        return cls(
            shared=shared,
            data=PlayerDataConfig.from_mapping(
                require_config_mapping(config, "data", path="configuration"), shared.resolver
            ),
            model=DinoModelConfig.from_mapping(
                require_config_mapping(config, "model", path="configuration"), shared.resolver
            ),
            evaluation=DetectionEvaluationConfig.from_mapping(
                require_config_mapping(config, "evaluation", path="configuration"), "evaluation"
            ),
        )


def validate_train_boundary(config: DictConfig) -> None:
    PlayerTrainingConfig.from_config(config)


# ---------------------------------------------------------------------------
# export / evaluate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Lightning checkpoint (output role) -> DINO-format checkpoint (checkpoint role)."""

    lightning_checkpoint: Path
    destination: Path
    repository: Path

    @classmethod
    def from_config(cls, value: object) -> ExportConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "export"},
            optional_keys={"hydra"},
        )
        resolver = _resolver(config)
        export = _mapping(config, "export", "configuration", {"lightning_checkpoint", "destination", "repository"})
        destination = resolver.resolve(PathRole.CHECKPOINT, _text(export, "destination", "export"))
        if destination.suffix != ".pth":
            raise SemanticConfigurationError("export.destination must name a .pth file.")
        return cls(
            lightning_checkpoint=resolver.resolve(
                PathRole.OUTPUT, _text(export, "lightning_checkpoint", "export")
            ),
            destination=destination,
            repository=resolver.resolve(PathRole.EXTERNAL_ASSET, _text(export, "repository", "export")),
        )


def validate_export_boundary(config: DictConfig) -> None:
    ExportConfig.from_config(config)


@dataclass(frozen=True, slots=True)
class EvaluateConfig:
    """Evaluate DINO-format checkpoints on one split of the player store."""

    checkpoints: Mapping[str, Path]
    repository: Path
    dataset_dir: Path
    split: str
    frame_stride: int
    selection: FrameSelectionConfig
    input_size: InputSizeConfig
    evaluation: DetectionEvaluationConfig
    num_workers: int
    overlay_frames: int
    output_dir: Path
    device: str

    @classmethod
    def from_config(cls, value: object) -> EvaluateConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "evaluate", "evaluation", "run"},
            optional_keys={"hydra"},
        )
        resolver = _resolver(config)
        path = "evaluate"
        evaluate = _mapping(
            config,
            "evaluate",
            "configuration",
            {"checkpoints", "repository", "dataset_dir", "split", "frame_stride", "selection",
             "input_size", "num_workers", "overlay_frames", "device"},
        )
        raw_checkpoints = require_config_mapping(evaluate, "checkpoints", path=path)
        if not raw_checkpoints:
            raise SemanticConfigurationError("evaluate.checkpoints must name at least one checkpoint.")
        checkpoints = {
            name: resolver.resolve(PathRole.CHECKPOINT, _text(raw_checkpoints, name, "evaluate.checkpoints"))
            for name in raw_checkpoints
        }
        split = _text(evaluate, "split", path)
        if split not in {"train", "val", "test"}:
            raise SemanticConfigurationError("evaluate.split must be train, val, or test.")
        run = exact_config_mapping(
            require_config_mapping(config, "run", path="configuration"), path="run", required_keys={"output_dir"}
        )
        return cls(
            checkpoints=checkpoints,
            repository=resolver.resolve(PathRole.EXTERNAL_ASSET, _text(evaluate, "repository", path)),
            dataset_dir=resolver.resolve(PathRole.DATA, _text(evaluate, "dataset_dir", path)),
            split=split,
            frame_stride=_int(evaluate, "frame_stride", path, minimum=1),
            selection=FrameSelectionConfig.from_mapping(
                _mapping(evaluate, "selection", path, {"require_reviewed", "require_located_players", "min_visible_box_px"}),
                "evaluate.selection",
            ),
            input_size=InputSizeConfig.from_mapping(
                _mapping(evaluate, "input_size", path, {"short_side", "max_long_side"}), "evaluate.input_size"
            ),
            evaluation=DetectionEvaluationConfig.from_mapping(
                require_config_mapping(config, "evaluation", path="configuration"), "evaluation"
            ),
            num_workers=_int(evaluate, "num_workers", path, minimum=0),
            overlay_frames=_int(evaluate, "overlay_frames", path, minimum=0),
            output_dir=resolver.resolve(PathRole.OUTPUT, _text(run, "output_dir", "run")),
            device=_text(evaluate, "device", path),
        )


def validate_evaluate_boundary(config: DictConfig) -> None:
    EvaluateConfig.from_config(config)


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    """Render stored frames with their annotations for visual verification."""

    dataset_dir: Path
    frames_per_split: int
    output_dir: Path

    @classmethod
    def from_config(cls, value: object) -> PreviewConfig:
        config = exact_config_mapping(
            as_config_mapping(value, path="configuration"),
            path="configuration",
            required_keys={"paths", "preview", "run"},
            optional_keys={"hydra"},
        )
        resolver = _resolver(config)
        preview = _mapping(config, "preview", "configuration", {"dataset_dir", "frames_per_split"})
        run = exact_config_mapping(
            require_config_mapping(config, "run", path="configuration"), path="run", required_keys={"output_dir"}
        )
        return cls(
            dataset_dir=resolver.resolve(PathRole.DATA, _text(preview, "dataset_dir", "preview")),
            frames_per_split=_int(preview, "frames_per_split", "preview", minimum=1),
            output_dir=resolver.resolve(PathRole.OUTPUT, _text(run, "output_dir", "run")),
        )


def validate_preview_boundary(config: DictConfig) -> None:
    PreviewConfig.from_config(config)
