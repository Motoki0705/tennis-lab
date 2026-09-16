"""Typed settings for the court-alignment benchmark.

The YAML file only ever describes *what* is measured.  Absolute locations
(repository root, output directory, checkpoints) arrive from the CLI so one
manifest and one metric definition can be replayed from any worktree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from src.tasks.court_detection.configuration import (
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.evaluation.contracts import (
    DOMAIN_NAMES,
    STRATA_AXES,
    DomainName,
)
from src.utils.configuration import PathResolver, RuntimePathRoots

_TOP_LEVEL_KEYS = {
    "paths",
    "derived_target_root",
    "real",
    "synthetic",
    "quality",
    "selection",
    "visualization",
}
_DOMAIN_KEYS = {"source", "split", "display_name"}
_QUALITY_KEYS = {
    "pck_fractions",
    "ransac_threshold_fraction",
    "line_samples_per_segment",
    "strata_axes",
}
_SELECTION_KEYS = {"seed", "max_samples_per_domain"}
_VISUALIZATION_KEYS = {"samples_per_domain"}


def _exact(mapping: Mapping[str, object], keys: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - keys)
    if unknown:
        raise ValueError(
            f"Unknown benchmark configuration key(s): {unknown} at {path}."
        )
    missing = sorted(keys - set(mapping))
    if missing:
        raise ValueError(
            f"Missing benchmark configuration key(s): {missing} at {path}."
        )


@dataclass(frozen=True, slots=True)
class BenchmarkQualitySettings:
    """Metric definitions fixed for every model and domain."""

    pck_fractions: tuple[float, ...]
    ransac_threshold_fraction: float
    line_samples_per_segment: int
    strata_axes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object) -> BenchmarkQualitySettings:
        if not isinstance(value, Mapping):
            raise ValueError("benchmark quality settings must be a mapping.")
        _exact(value, _QUALITY_KEYS, path="quality")
        raw_fractions = value["pck_fractions"]
        if (
            not isinstance(raw_fractions, Sequence)
            or isinstance(raw_fractions, (str, bytes))
            or not raw_fractions
        ):
            raise ValueError("quality.pck_fractions must be a non-empty list.")
        fractions = tuple(float(cast("float", item)) for item in raw_fractions)
        if any(not 0.0 < fraction < 1.0 for fraction in fractions):
            raise ValueError("quality.pck_fractions must lie in (0, 1).")
        if list(fractions) != sorted(fractions) or len(set(fractions)) != len(
            fractions
        ):
            raise ValueError("quality.pck_fractions must be unique and ascending.")
        threshold_fraction = float(cast("float", value["ransac_threshold_fraction"]))
        if not 0.0 < threshold_fraction < 1.0:
            raise ValueError("quality.ransac_threshold_fraction must lie in (0, 1).")
        samples_per_segment = value["line_samples_per_segment"]
        if type(samples_per_segment) is not int or samples_per_segment < 2:
            raise ValueError("quality.line_samples_per_segment must be an int >= 2.")
        raw_axes = value["strata_axes"]
        if (
            not isinstance(raw_axes, Sequence)
            or isinstance(raw_axes, (str, bytes))
            or not raw_axes
            or any(type(axis) is not str for axis in raw_axes)
        ):
            raise ValueError("quality.strata_axes must be a non-empty string list.")
        axes = tuple(cast("tuple[str, ...]", tuple(raw_axes)))
        unknown_axes = sorted(set(axes) - set(STRATA_AXES))
        if unknown_axes:
            raise ValueError(
                f"quality.strata_axes contains unknown axes: {unknown_axes}."
            )
        if len(set(axes)) != len(axes):
            raise ValueError("quality.strata_axes must not repeat an axis.")
        return cls(
            pck_fractions=fractions,
            ransac_threshold_fraction=threshold_fraction,
            line_samples_per_segment=samples_per_segment,
            strata_axes=axes,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSelectionSettings:
    """Deterministic sample selection contract."""

    seed: int
    max_samples_per_domain: int | None

    @classmethod
    def from_mapping(cls, value: object) -> BenchmarkSelectionSettings:
        if not isinstance(value, Mapping):
            raise ValueError("benchmark selection settings must be a mapping.")
        _exact(value, _SELECTION_KEYS, path="selection")
        seed = value["seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("selection.seed must be a non-negative integer.")
        raw_max = value["max_samples_per_domain"]
        if raw_max is None:
            maximum: int | None = None
        elif type(raw_max) is int and raw_max > 0:
            maximum = raw_max
        else:
            raise ValueError(
                "selection.max_samples_per_domain must be null or a positive integer."
            )
        return cls(seed=seed, max_samples_per_domain=maximum)


@dataclass(frozen=True, slots=True)
class BenchmarkVisualizationSettings:
    samples_per_domain: int

    @classmethod
    def from_mapping(cls, value: object) -> BenchmarkVisualizationSettings:
        if not isinstance(value, Mapping):
            raise ValueError("benchmark visualization settings must be a mapping.")
        _exact(value, _VISUALIZATION_KEYS, path="visualization")
        count = value["samples_per_domain"]
        if type(count) is not int or count <= 0:
            raise ValueError("visualization.samples_per_domain must be positive.")
        return cls(samples_per_domain=count)


@dataclass(frozen=True, slots=True)
class BenchmarkDomainSettings:
    """One labelled domain: source contract, split, and display name."""

    domain: DomainName
    display_name: str
    split: str
    source: TennisCourtDetectorSourceConfig | SyntheticCourtSourceConfig

    @property
    def source_root(self) -> Path:
        if isinstance(self.source, TennisCourtDetectorSourceConfig):
            return Path(self.source.root)
        return Path(self.source.workspace_root)

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        domain: DomainName,
        resolver: PathResolver,
    ) -> BenchmarkDomainSettings:
        if not isinstance(value, Mapping):
            raise ValueError(f"benchmark {domain} settings must be a mapping.")
        _exact(value, _DOMAIN_KEYS, path=domain)
        display_name = value["display_name"]
        if type(display_name) is not str or not display_name.strip():
            raise ValueError(f"{domain}.display_name must be a non-empty string.")
        split = value["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"{domain}.split must be train, val, or test.")
        raw_source = value["source"]
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"{domain}.source must be a mapping.")
        kind = raw_source.get("kind")
        if kind == "tennis_court_detector":
            source: TennisCourtDetectorSourceConfig | SyntheticCourtSourceConfig = (
                TennisCourtDetectorSourceConfig.from_mapping(
                    raw_source, resolver=resolver
                )
            )
            if cast("str", split) == "test":
                raise ValueError(
                    "The TennisCourtDetector domain has no test split; validation "
                    "must not be renamed as test."
                )
        elif kind == "synthetic_court":
            source = SyntheticCourtSourceConfig.from_mapping(
                raw_source, resolver=resolver
            )
        else:
            raise ValueError(f"{domain}.source.kind is unsupported: {kind!r}.")
        return cls(
            domain=domain,
            display_name=display_name,
            split=cast("str", split),
            source=source,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Complete replayable benchmark contract below the CLI overrides."""

    roots: RuntimePathRoots
    derived_target_root: Path
    domains: Mapping[DomainName, BenchmarkDomainSettings]
    quality: BenchmarkQualitySettings
    selection: BenchmarkSelectionSettings
    visualization: BenchmarkVisualizationSettings

    @classmethod
    def load(cls, path: str | Path, *, project_root: Path) -> BenchmarkSettings:
        raw = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
        return cls.from_mapping(raw, project_root=project_root)

    @classmethod
    def from_mapping(cls, value: object, *, project_root: Path) -> BenchmarkSettings:
        if not isinstance(value, Mapping):
            raise ValueError("benchmark configuration must be a mapping.")
        _exact(value, _TOP_LEVEL_KEYS, path="configuration")
        raw_paths = value["paths"]
        if not isinstance(raw_paths, Mapping):
            raise ValueError("benchmark paths must be a mapping.")
        paths = {str(key): str(item) for key, item in raw_paths.items()}
        paths["project_root"] = str(project_root)
        if not project_root.is_absolute():
            raise ValueError("benchmark project_root must be absolute.")
        roots = RuntimePathRoots.from_mapping(paths, repository_root=project_root)
        derived_target_root = Path(str(value["derived_target_root"]))
        if derived_target_root.is_absolute():
            raise ValueError(
                "benchmark derived_target_root must be relative to the cache root."
            )
        resolver = PathResolver(roots)
        domains: dict[DomainName, BenchmarkDomainSettings] = {}
        for domain, key in zip(DOMAIN_NAMES, ("real", "synthetic"), strict=True):
            domains[domain] = BenchmarkDomainSettings.from_mapping(
                value[key], domain=domain, resolver=resolver
            )
        return cls(
            roots=roots,
            derived_target_root=roots.cache_root / derived_target_root,
            domains=domains,
            quality=BenchmarkQualitySettings.from_mapping(value["quality"]),
            selection=BenchmarkSelectionSettings.from_mapping(value["selection"]),
            visualization=BenchmarkVisualizationSettings.from_mapping(
                value["visualization"]
            ),
        )


__all__ = [
    "BenchmarkDomainSettings",
    "BenchmarkQualitySettings",
    "BenchmarkSelectionSettings",
    "BenchmarkSettings",
    "BenchmarkVisualizationSettings",
]
