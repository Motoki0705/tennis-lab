"""Strict Hydra boundary contracts for Court query profiling and ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from omegaconf import DictConfig

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.court_detection.configuration import (
    ConfigMapping,
    CourtQueryLossConfig,
    CourtQueryModelConfig,
    validate_paths_boundary,
)
from src.utils.configuration import PathResolver, PathRole

DecoderFamily: TypeAlias = Literal["dpt"]
DecoderSize: TypeAlias = Literal["tiny", "small", "base", "large"]
SupervisionName: TypeAlias = Literal["kp", "kp+pose", "all", "all+pose"]


def _exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
    if set(mapping) != keys:
        raise ValueError(f"{path} requires exactly {sorted(keys)}.")


def _string(mapping: ConfigMapping, key: str, *, path: str) -> str:
    value = cast(str, require_config_value(mapping, key, str, path=path))
    if not value or value != value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty trimmed string.")
    return value


def _integer(mapping: ConfigMapping, key: str, *, path: str) -> int:
    return cast(int, require_config_value(mapping, key, int, path=path))


def _bool(mapping: ConfigMapping, key: str, *, path: str) -> bool:
    return cast(bool, require_config_value(mapping, key, bool, path=path))


def _number(mapping: ConfigMapping, key: str, *, path: str) -> float:
    return float(
        cast(
            float | int,
            require_config_value(mapping, key, (float, int), path=path),
        )
    )


def _strings(mapping: ConfigMapping, key: str, *, path: str) -> tuple[str, ...]:
    values = cast(
        Sequence[object],
        require_config_value(mapping, key, (list, tuple), path=path),
    )
    if any(type(value) is not str or not value for value in values):
        raise ValueError(f"{path}.{key} must contain non-empty strings.")
    return tuple(cast(str, value) for value in values)


def _integers(mapping: ConfigMapping, key: str, *, path: str) -> tuple[int, ...]:
    values = cast(
        Sequence[object],
        require_config_value(mapping, key, (list, tuple), path=path),
    )
    if any(type(value) is not int for value in values):
        raise ValueError(f"{path}.{key} must contain integers.")
    return tuple(cast(int, value) for value in values)


@dataclass(frozen=True, slots=True)
class QueryProfileConfig:
    """Resolved profile entrypoint contract."""

    model: CourtQueryModelConfig
    loss: CourtQueryLossConfig
    output_path: Path
    device: Literal["cpu", "cuda"]
    allow_cpu_diagnostic: bool
    batch_size: int
    channels: int
    height: int
    width: int
    dtype: Literal["float32"]
    warmup: int
    repeats: int
    candidate_family: DecoderFamily
    candidate_size: DecoderSize

    @classmethod
    def from_config(cls, value: object) -> QueryProfileConfig:
        root, resolver = _root(value, sections={"model", "loss", "profile"})
        model = CourtQueryModelConfig.from_mapping(
            require_config_mapping(root, "model", path="configuration"),
            resolver=resolver,
        )
        loss = CourtQueryLossConfig.from_mapping(
            require_config_mapping(root, "loss", path="configuration")
        )
        profile = require_config_mapping(root, "profile", path="configuration")
        _exact(
            profile,
            {
                "output_path",
                "device",
                "allow_cpu_diagnostic",
                "input",
                "warmup",
                "repeats",
                "candidate",
            },
            path="profile",
        )
        tensor = require_config_mapping(profile, "input", path="profile")
        _exact(
            tensor,
            {"batch_size", "channels", "height", "width", "dtype"},
            path="profile.input",
        )
        candidate = require_config_mapping(profile, "candidate", path="profile")
        _exact(candidate, {"family", "size"}, path="profile.candidate")
        device = _string(profile, "device", path="profile")
        family = _string(candidate, "family", path="profile.candidate")
        size = _string(candidate, "size", path="profile.candidate")
        dtype = _string(tensor, "dtype", path="profile.input")
        result = cls(
            model=model,
            loss=loss,
            output_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(profile, "output_path", path="profile"),
            ),
            device=cast(Literal["cpu", "cuda"], device),
            allow_cpu_diagnostic=_bool(profile, "allow_cpu_diagnostic", path="profile"),
            batch_size=_integer(tensor, "batch_size", path="profile.input"),
            channels=_integer(tensor, "channels", path="profile.input"),
            height=_integer(tensor, "height", path="profile.input"),
            width=_integer(tensor, "width", path="profile.input"),
            dtype=cast(Literal["float32"], dtype),
            warmup=_integer(profile, "warmup", path="profile"),
            repeats=_integer(profile, "repeats", path="profile"),
            candidate_family=cast(DecoderFamily, family),
            candidate_size=cast(DecoderSize, size),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("profile.device must be cpu or cuda.")
        if self.device == "cpu" and not self.allow_cpu_diagnostic:
            raise ValueError(
                "CPU profiling must explicitly set allow_cpu_diagnostic=true."
            )
        if self.channels != 3:
            raise ValueError("Court query profiling requires three RGB channels.")
        if (
            self.dtype != "float32"
            or min(self.batch_size, self.height, self.width) <= 0
        ):
            raise ValueError("Profile input must be positive-size float32.")
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("Profile warmup/repeats are invalid.")
        if self.candidate_family != "dpt":
            raise ValueError("Profile candidate family must be dpt.")
        if self.candidate_size not in {"tiny", "small", "base", "large"}:
            raise ValueError("Profile candidate size is invalid.")
        if self.model.decoder.family != self.candidate_family:
            raise ValueError("Profile candidate family disagrees with model.decoder.")
        expected_width = {"tiny": 32, "small": 64, "base": 128, "large": 256}[
            self.candidate_size
        ]
        if self.model.decoder.width != expected_width:
            raise ValueError(
                "Profile candidate size disagrees with model.decoder.width."
            )


@dataclass(frozen=True, slots=True)
class QueryAblationCompositionConfig:
    source: str
    keypoint_court_scope: str
    augmentation: str
    model: str
    heads: str
    loss_pose: str
    loss_dense: str
    processing_kp: str
    processing_all: str

    @classmethod
    def from_mapping(cls, value: object) -> QueryAblationCompositionConfig:
        mapping = as_config_mapping(value, path="ablation.composition")
        keys = {
            "source",
            "keypoint_court_scope",
            "augmentation",
            "model",
            "heads",
            "loss_pose",
            "loss_dense",
            "processing_kp",
            "processing_all",
        }
        _exact(mapping, keys, path="ablation.composition")
        result = cls(
            **{key: _string(mapping, key, path="ablation.composition") for key in keys}
        )
        expected = cls(
            source="synthetic_court",
            keypoint_court_scope="target_court",
            augmentation="pose_safe",
            model="query_encoder_base",
            heads="query_base",
            loss_pose="query_pose",
            loss_dense="query_dense",
            processing_kp="kp",
            processing_all="all",
        )
        if result != expected:
            raise ValueError(
                "Ablation composition must retain the query model, V3 target-court, "
                "pose-safe augmentation, head, loss, and processing authorities."
            )
        return result


@dataclass(frozen=True, slots=True)
class EncoderFirstConfig:
    name: Literal["encoder_first"]
    order: Literal[1]
    depths: tuple[int, ...]
    hidden_dim: int
    num_heads: int
    decoder_family: Literal["dpt"]
    decoder_size: Literal["base"]
    supervision: Literal["kp+pose"]
    reference_depth: int
    tolerance_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> EncoderFirstConfig:
        path = "ablation.encoder_first"
        mapping = as_config_mapping(value, path=path)
        keys = {
            "name",
            "order",
            "depths",
            "hidden_dim",
            "num_heads",
            "decoder_family",
            "decoder_size",
            "supervision",
            "reference_depth",
            "tolerance_ratio",
        }
        _exact(mapping, keys, path=path)
        result = cls(
            name=cast(Literal["encoder_first"], _string(mapping, "name", path=path)),
            order=cast(Literal[1], _integer(mapping, "order", path=path)),
            depths=_integers(mapping, "depths", path=path),
            hidden_dim=_integer(mapping, "hidden_dim", path=path),
            num_heads=_integer(mapping, "num_heads", path=path),
            decoder_family=cast(
                Literal["dpt"], _string(mapping, "decoder_family", path=path)
            ),
            decoder_size=cast(
                Literal["base"], _string(mapping, "decoder_size", path=path)
            ),
            supervision=cast(
                Literal["kp+pose"], _string(mapping, "supervision", path=path)
            ),
            reference_depth=_integer(mapping, "reference_depth", path=path),
            tolerance_ratio=_number(mapping, "tolerance_ratio", path=path),
        )
        if result != cls(
            name="encoder_first",
            order=1,
            depths=(1, 8),
            hidden_dim=256,
            num_heads=8,
            decoder_family="dpt",
            decoder_size="base",
            supervision="kp+pose",
            reference_depth=8,
            tolerance_ratio=0.05,
        ):
            raise ValueError("Encoder-first preset must retain the frozen depth sweep.")
        return result


@dataclass(frozen=True, slots=True)
class DecoderSecondConfig:
    name: Literal["decoder_second"]
    order: Literal[2]
    families: tuple[DecoderFamily, ...]
    sizes: tuple[DecoderSize, ...]
    supervision: Literal["kp+pose"]
    reference_family: Literal["dpt"]
    reference_size: Literal["base"]
    tolerance_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> DecoderSecondConfig:
        path = "ablation.decoder_second"
        mapping = as_config_mapping(value, path=path)
        keys = {
            "name",
            "order",
            "families",
            "sizes",
            "supervision",
            "reference_family",
            "reference_size",
            "tolerance_ratio",
        }
        _exact(mapping, keys, path=path)
        result = cls(
            name=cast(Literal["decoder_second"], _string(mapping, "name", path=path)),
            order=cast(Literal[2], _integer(mapping, "order", path=path)),
            families=cast(
                tuple[DecoderFamily, ...], _strings(mapping, "families", path=path)
            ),
            sizes=cast(tuple[DecoderSize, ...], _strings(mapping, "sizes", path=path)),
            supervision=cast(
                Literal["kp+pose"], _string(mapping, "supervision", path=path)
            ),
            reference_family=cast(
                Literal["dpt"], _string(mapping, "reference_family", path=path)
            ),
            reference_size=cast(
                Literal["base"], _string(mapping, "reference_size", path=path)
            ),
            tolerance_ratio=_number(mapping, "tolerance_ratio", path=path),
        )
        if result != cls(
            name="decoder_second",
            order=2,
            families=("dpt",),
            sizes=("tiny", "small", "base", "large"),
            supervision="kp+pose",
            reference_family="dpt",
            reference_size="base",
            tolerance_ratio=0.05,
        ):
            raise ValueError("Decoder-second preset must retain the frozen DPT size sweep.")
        return result


@dataclass(frozen=True, slots=True)
class SupervisionThirdConfig:
    name: Literal["supervision_third"]
    order: Literal[3]
    variants: tuple[SupervisionName, ...]
    seg_line_semantics: Literal["all_courts"]

    @classmethod
    def from_mapping(cls, value: object) -> SupervisionThirdConfig:
        path = "ablation.supervision_third"
        mapping = as_config_mapping(value, path=path)
        keys = {"name", "order", "variants", "seg_line_semantics"}
        _exact(mapping, keys, path=path)
        result = cls(
            name=cast(
                Literal["supervision_third"], _string(mapping, "name", path=path)
            ),
            order=cast(Literal[3], _integer(mapping, "order", path=path)),
            variants=cast(
                tuple[SupervisionName, ...], _strings(mapping, "variants", path=path)
            ),
            seg_line_semantics=cast(
                Literal["all_courts"],
                _string(mapping, "seg_line_semantics", path=path),
            ),
        )
        if result != cls(
            name="supervision_third",
            order=3,
            variants=("kp", "kp+pose", "all", "all+pose"),
            seg_line_semantics="all_courts",
        ):
            raise ValueError(
                "Supervision-third preset must retain kp/kp+pose/all/all+pose."
            )
        return result


@dataclass(frozen=True, slots=True)
class QueryAblationConfig:
    """Resolved deterministic three-phase ablation manifest contract."""

    output_path: Path
    python_executable: str
    train_module: str
    seeds: tuple[int, ...]
    epochs: int
    image_height: int
    image_width: int
    isotropic_letterbox: bool
    preserve_fx_fy: bool
    hflip: bool
    affine: bool
    shear: bool
    perspective: bool
    selected_encoder_depth: int | None
    selected_decoder_family: DecoderFamily | None
    selected_decoder_size: DecoderSize | None
    composition: QueryAblationCompositionConfig
    encoder_first: EncoderFirstConfig
    decoder_second: DecoderSecondConfig
    supervision_third: SupervisionThirdConfig

    @classmethod
    def from_config(cls, value: object) -> QueryAblationConfig:
        root, resolver = _root(value, sections={"ablation"})
        ablation = require_config_mapping(root, "ablation", path="configuration")
        keys = {
            "output_path",
            "python_executable",
            "train_module",
            "seeds",
            "epochs",
            "input",
            "augmentation",
            "selected",
            "composition",
            "encoder_first",
            "decoder_second",
            "supervision_third",
        }
        _exact(ablation, keys, path="ablation")
        input_config = require_config_mapping(ablation, "input", path="ablation")
        _exact(input_config, {"height", "width"}, path="ablation.input")
        augmentation = require_config_mapping(ablation, "augmentation", path="ablation")
        _exact(
            augmentation,
            {
                "isotropic_letterbox",
                "preserve_fx_fy",
                "hflip",
                "affine",
                "shear",
                "perspective",
            },
            path="ablation.augmentation",
        )
        selected = require_config_mapping(ablation, "selected", path="ablation")
        _exact(
            selected,
            {"encoder_depth", "decoder_family", "decoder_size"},
            path="ablation.selected",
        )
        raw_depth = require_config_value(
            selected,
            "encoder_depth",
            (int, type(None)),
            path="ablation.selected",
        )
        raw_family = require_config_value(
            selected,
            "decoder_family",
            (str, type(None)),
            path="ablation.selected",
        )
        raw_size = require_config_value(
            selected,
            "decoder_size",
            (str, type(None)),
            path="ablation.selected",
        )
        result = cls(
            output_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(ablation, "output_path", path="ablation"),
            ),
            python_executable=_string(ablation, "python_executable", path="ablation"),
            train_module=_string(ablation, "train_module", path="ablation"),
            seeds=_integers(ablation, "seeds", path="ablation"),
            epochs=_integer(ablation, "epochs", path="ablation"),
            image_height=_integer(input_config, "height", path="ablation.input"),
            image_width=_integer(input_config, "width", path="ablation.input"),
            isotropic_letterbox=_bool(
                augmentation, "isotropic_letterbox", path="ablation.augmentation"
            ),
            preserve_fx_fy=_bool(
                augmentation, "preserve_fx_fy", path="ablation.augmentation"
            ),
            hflip=_bool(augmentation, "hflip", path="ablation.augmentation"),
            affine=_bool(augmentation, "affine", path="ablation.augmentation"),
            shear=_bool(augmentation, "shear", path="ablation.augmentation"),
            perspective=_bool(
                augmentation, "perspective", path="ablation.augmentation"
            ),
            selected_encoder_depth=cast(int | None, raw_depth),
            selected_decoder_family=cast(DecoderFamily | None, raw_family),
            selected_decoder_size=cast(DecoderSize | None, raw_size),
            composition=QueryAblationCompositionConfig.from_mapping(
                require_config_mapping(ablation, "composition", path="ablation")
            ),
            encoder_first=EncoderFirstConfig.from_mapping(
                require_config_mapping(ablation, "encoder_first", path="ablation")
            ),
            decoder_second=DecoderSecondConfig.from_mapping(
                require_config_mapping(ablation, "decoder_second", path="ablation")
            ),
            supervision_third=SupervisionThirdConfig.from_mapping(
                require_config_mapping(ablation, "supervision_third", path="ablation")
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.python_executable != ".venv/bin/python":
            raise ValueError("Ablation commands must use the repository .venv Python.")
        if self.train_module != "src.tasks.court_detection.scripts.train":
            raise ValueError("Ablation commands must target the Court Hydra trainer.")
        if self.seeds != (42,) or self.epochs != 15:
            raise ValueError("Ablation contract requires seed 42 and 15 epochs.")
        if (self.image_height, self.image_width) != (256, 256):
            raise ValueError("Ablation input contract must be exactly 256x256.")
        if (
            not self.isotropic_letterbox
            or not self.preserve_fx_fy
            or any((self.hflip, self.affine, self.shear, self.perspective))
        ):
            raise ValueError(
                "Ablation augmentation requires isotropic letterbox, preserve_fx_fy, "
                "and no hflip/affine/shear/perspective."
            )
        if self.selected_encoder_depth is not None and (
            self.selected_encoder_depth not in self.encoder_first.depths
        ):
            raise ValueError("Selected encoder depth is outside the encoder sweep.")
        family_missing = self.selected_decoder_family is None
        size_missing = self.selected_decoder_size is None
        if family_missing != size_missing:
            raise ValueError("Selected decoder family and size must resolve together.")
        if self.selected_decoder_family is not None and (
            self.selected_decoder_family not in self.decoder_second.families
            or self.selected_decoder_size not in self.decoder_second.sizes
        ):
            raise ValueError("Selected decoder is outside the decoder matrix.")


@dataclass(frozen=True, slots=True)
class QuerySummaryConfig:
    """Resolved fail-closed summary input/output and adoption contract."""

    manifest_path: Path
    results_path: Path
    output_dir: Path
    require_gpu_profiles: bool
    adopted_supervision: SupervisionName | None
    adoption_rationale: str | None

    @classmethod
    def from_config(cls, value: object) -> QuerySummaryConfig:
        root, resolver = _root(value, sections={"summary"})
        summary = require_config_mapping(root, "summary", path="configuration")
        _exact(
            summary,
            {
                "manifest_path",
                "results_path",
                "output_dir",
                "require_gpu_profiles",
                "adoption",
            },
            path="summary",
        )
        adoption = require_config_mapping(summary, "adoption", path="summary")
        _exact(adoption, {"supervision", "rationale"}, path="summary.adoption")
        raw_supervision = require_config_value(
            adoption,
            "supervision",
            (str, type(None)),
            path="summary.adoption",
        )
        raw_rationale = require_config_value(
            adoption,
            "rationale",
            (str, type(None)),
            path="summary.adoption",
        )
        result = cls(
            manifest_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "manifest_path", path="summary"),
            ),
            results_path=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "results_path", path="summary"),
            ),
            output_dir=resolver.resolve(
                PathRole.OUTPUT,
                _string(summary, "output_dir", path="summary"),
            ),
            require_gpu_profiles=_bool(summary, "require_gpu_profiles", path="summary"),
            adopted_supervision=cast(SupervisionName | None, raw_supervision),
            adoption_rationale=cast(str | None, raw_rationale),
        )
        if result.adopted_supervision is not None and (
            result.adopted_supervision not in {"kp", "kp+pose", "all", "all+pose"}
        ):
            raise ValueError("summary.adoption.supervision is invalid.")
        if result.adoption_rationale is not None and (
            not result.adoption_rationale
            or result.adoption_rationale != result.adoption_rationale.strip()
        ):
            raise ValueError(
                "summary.adoption.rationale must be non-empty and trimmed."
            )
        return result

    def require_adoption_decision(self) -> tuple[SupervisionName, str]:
        if self.adopted_supervision is None or self.adoption_rationale is None:
            raise ValueError(
                "Summary requires explicit adoption.supervision and adoption.rationale "
                "after reviewing complete experiment evidence."
            )
        return self.adopted_supervision, self.adoption_rationale


def _root(
    value: object,
    *,
    sections: set[str],
) -> tuple[ConfigMapping, PathResolver]:
    if not isinstance(value, DictConfig):
        raise TypeError("Experiment boundary requires a composed DictConfig.")
    root: ConfigMapping
    resolver: PathResolver
    root, resolver = validate_paths_boundary(value, expected_sections=sections)
    return root, resolver


def validate_profile_boundary(config: DictConfig) -> None:
    QueryProfileConfig.from_config(config)


def validate_ablation_boundary(config: DictConfig) -> None:
    QueryAblationConfig.from_config(config)


def validate_summary_boundary(config: DictConfig) -> None:
    QuerySummaryConfig.from_config(config)


__all__ = [
    "DecoderFamily",
    "DecoderSecondConfig",
    "DecoderSize",
    "EncoderFirstConfig",
    "QueryAblationConfig",
    "QueryProfileConfig",
    "QuerySummaryConfig",
    "SupervisionName",
    "SupervisionThirdConfig",
    "validate_ablation_boundary",
    "validate_profile_boundary",
    "validate_summary_boundary",
]
