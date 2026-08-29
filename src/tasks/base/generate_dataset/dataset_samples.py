"""Shared contracts for stratified, human-readable dataset sample GIFs."""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib.animation import PillowWriter
from numpy.typing import NDArray
from omegaconf import DictConfig
from PIL import Image

from src.tasks.base.configuration import (
    ConfigMapping,
    as_config_mapping,
    exact_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    resolve_court_keypoint_contract,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
)
from src.utils.io import load_json, save_json_atomic
from src.utils.paths import PROJECT_ROOT

DatasetMode = Literal["single", "multi"]
DurationBand = Literal["short", "medium", "long"]
RenderSample = Callable[["SelectedDatasetSample", Path], "RenderedDatasetSample"]

_DURATION_BANDS: tuple[DurationBand, DurationBand, DurationBand] = (
    "short",
    "medium",
    "long",
)
_DATASET_SPEC_KEYS = frozenset({"path", "mode", "court_keypoints"})
_SAMPLE_CONFIG_KEYS = frozenset(
    {
        "datasets",
        "samples_per_stratum",
        "max_frames",
        "min_fps",
        "max_fps",
        "view",
        "figure_size",
        "overwrite",
    }
)
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_MANIFEST_SCHEMA = "tennis_lab_dataset_samples_v1"
_MIN_VISIBLE_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class DatasetSampleSpec:
    """One configured dataset root and its exact semantic contract."""

    relative_path: str
    root: Path
    mode: DatasetMode
    court_keypoint_contract: CourtKeypointContract


@dataclass(frozen=True, slots=True)
class DatasetSamplesConfig:
    """Strict runtime configuration shared by PLCS and BLCS sample scripts."""

    datasets: tuple[DatasetSampleSpec, ...]
    samples_per_stratum: int
    max_frames: int
    min_fps: int
    max_fps: int
    view: Literal["camera"]
    figure_size: tuple[float, float]
    overwrite: bool

    @classmethod
    def from_config(
        cls, value: object, *, task: Literal["plcs", "blcs"]
    ) -> DatasetSamplesConfig:
        """Validate a composed Hydra config before any filesystem mutation."""
        if not isinstance(value, DictConfig):
            raise ConfigurationTypeError(
                "Dataset sample generation requires a DictConfig."
            )
        root = exact_config_mapping(
            value,
            path="configuration",
            required_keys=frozenset({"paths", "samples"}),
        )
        paths = require_config_mapping(root, "paths", path="configuration")
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
        )
        samples = exact_config_mapping(
            require_config_mapping(root, "samples", path="configuration"),
            path="samples",
            required_keys=_SAMPLE_CONFIG_KEYS,
        )
        raw_datasets = require_config_value(
            samples, "datasets", (list, tuple), path="samples"
        )
        datasets = tuple(
            _parse_dataset_spec(raw, index=index, task=task, resolver=resolver)
            for index, raw in enumerate(cast("Sequence[object]", raw_datasets))
        )
        if not datasets:
            raise SemanticConfigurationError("samples.datasets must not be empty.")
        relative_paths = tuple(spec.relative_path for spec in datasets)
        if len(relative_paths) != len(set(relative_paths)):
            raise SemanticConfigurationError(
                "samples.datasets paths must be unique within one invocation."
            )

        samples_per_stratum = cast(
            "int",
            require_config_value(samples, "samples_per_stratum", int, path="samples"),
        )
        max_frames = cast(
            "int", require_config_value(samples, "max_frames", int, path="samples")
        )
        min_fps = cast(
            "int", require_config_value(samples, "min_fps", int, path="samples")
        )
        max_fps = cast(
            "int", require_config_value(samples, "max_fps", int, path="samples")
        )
        if min(samples_per_stratum, max_frames, min_fps, max_fps) < 1:
            raise SemanticConfigurationError(
                "samples_per_stratum, max_frames, min_fps, and max_fps must be positive."
            )
        if max_frames < 2:
            raise SemanticConfigurationError(
                "samples.max_frames must be at least 2 for animated GIF output."
            )
        if min_fps > max_fps:
            raise SemanticConfigurationError(
                "samples.min_fps must not exceed samples.max_fps."
            )

        view = cast("str", require_config_value(samples, "view", str, path="samples"))
        if view != "camera":
            raise SemanticConfigurationError(
                "samples.view must be 'camera' so camera-layout datasets remain inspectable."
            )
        figure_size = _parse_figure_size(samples)
        overwrite = cast(
            "bool", require_config_value(samples, "overwrite", bool, path="samples")
        )
        return cls(
            datasets=datasets,
            samples_per_stratum=samples_per_stratum,
            max_frames=max_frames,
            min_fps=min_fps,
            max_fps=max_fps,
            view="camera",
            figure_size=figure_size,
            overwrite=overwrite,
        )


@dataclass(frozen=True, slots=True)
class DatasetSampleCandidate:
    """Task-owned scene metrics projected into the shared selection contract."""

    scene_id: str
    primary_group: str
    duration_value: float
    visibility_value: float
    auxiliary_value: float
    camera_visibilities: tuple[float, ...]
    metrics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SelectedDatasetSample:
    """One scene selected for a unique primary-group × duration stratum."""

    candidate: DatasetSampleCandidate
    duration_band: DurationBand
    rank_in_stratum: int
    selection_score: float
    visibility_target_quantile: float
    auxiliary_target_quantile: float
    camera_index: int
    camera_visibility: float

    @property
    def stratum_key(self) -> str:
        return f"{self.candidate.primary_group}:{self.duration_band}"


@dataclass(frozen=True, slots=True)
class DatasetSampleSelection:
    """Complete deterministic selection plan and its population evidence."""

    selected: tuple[SelectedDatasetSample, ...]
    duration_boundaries: Mapping[str, tuple[float, float]]
    stratum_population: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class RenderedDatasetSample:
    """GIF timing and frame provenance returned by a task renderer."""

    source_num_frames: int
    rendered_num_frames: int
    source_fps: float
    encoded_fps: int
    frame_indices: tuple[int, ...]


def track_lifecycle_metrics(
    tracks: Sequence[Mapping[str, object]],
    *,
    num_frames: int,
    location: str,
) -> tuple[int, int]:
    """Return total active frames and peak concurrency for track lifecycles."""
    occupancy: NDArray[np.int64] = np.zeros(num_frames + 1, dtype=np.int64)
    total = 0
    for index, track in enumerate(tracks):
        path = f"{location}.track_instances[{index}]"
        birth = cast("int", require_config_value(track, "birth_frame", int, path=path))
        death = cast("int", require_config_value(track, "death_frame", int, path=path))
        if not 0 <= birth < death <= num_frames:
            raise ValueError(f"{path}: invalid lifecycle interval [{birth}, {death}).")
        total += death - birth
        occupancy[birth] += 1
        occupancy[death] -= 1
    maximum = int(np.cumsum(occupancy[:-1]).max())
    return total, maximum


def take_temporal_sample(
    value: object,
    *,
    indices: NDArray[np.int64],
    source_num_frames: int,
    location: str,
) -> NDArray[Any]:
    """Select one validated frame index set from a temporal array-like value."""
    array: NDArray[Any] = np.asarray(value)
    if array.ndim < 1 or array.shape[0] != source_num_frames:
        raise ValueError(
            f"{location} must start with T={source_num_frames}, got {array.shape}."
        )
    result: NDArray[Any] = np.take(array, indices, axis=0)
    return result


def validate_sample_frame_indices(
    indices: NDArray[np.int64],
    source_num_frames: int,
    *,
    task: Literal["PLCS", "BLCS"],
) -> None:
    """Require a sorted, unique, endpoint-inclusive sample timeline."""
    if (
        indices.ndim != 1
        or len(indices) < 2
        or not np.issubdtype(indices.dtype, np.integer)
        or int(indices[0]) != 0
        or int(indices[-1]) != source_num_frames - 1
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(
            f"{task} sample indices must be sorted, unique, and endpoint-inclusive."
        )


def remap_sample_track_instances(
    meta: dict[str, Any],
    present: NDArray[Any],
    *,
    task: Literal["PLCS", "BLCS"],
) -> None:
    """Remap track lifecycle metadata onto a sampled presence timeline."""
    raw_tracks = meta["track_instances"]
    if not isinstance(raw_tracks, list) or present.ndim != 2:
        raise ValueError(
            f"{task} multi sample requires list track metadata and (T,Q) presence."
        )
    remapped: list[dict[str, object]] = []
    for index, raw_track in enumerate(raw_tracks):
        track = dict(as_config_mapping(raw_track, path=f"track_instances[{index}]"))
        track_id = cast(
            "int",
            require_config_value(
                track, "track_id", int, path=f"track_instances[{index}]"
            ),
        )
        if not 0 <= track_id < present.shape[1]:
            raise ValueError(f"{task} track_id {track_id} is outside presence width.")
        active = np.flatnonzero(present[:, track_id])
        if len(active) == 0:
            raise ValueError(
                f"{task} temporal sampling removed every active frame for track {track_id}."
            )
        track["birth_frame"] = int(active[0])
        track["death_frame"] = int(active[-1]) + 1
        remapped.append(track)
    meta["track_instances"] = remapped


def _parse_dataset_spec(
    raw: object,
    *,
    index: int,
    task: Literal["plcs", "blcs"],
    resolver: PathResolver,
) -> DatasetSampleSpec:
    path = f"samples.datasets[{index}]"
    mapping = exact_config_mapping(
        raw,
        path=path,
        required_keys=_DATASET_SPEC_KEYS,
    )
    relative_path = cast("str", require_config_value(mapping, "path", str, path=path))
    if not relative_path.startswith(f"{task}/"):
        raise SemanticConfigurationError(
            f"{path}.path must be {task}/-relative; got {relative_path!r}."
        )
    mode = cast("str", require_config_value(mapping, "mode", str, path=path))
    if mode not in {"single", "multi"}:
        raise SemanticConfigurationError(
            f"{path}.mode must be 'single' or 'multi'; got {mode!r}."
        )
    selector = cast(
        "str", require_config_value(mapping, "court_keypoints", str, path=path)
    )
    return DatasetSampleSpec(
        relative_path=relative_path,
        root=resolver.resolve(PathRole.DATA, relative_path),
        mode=cast("DatasetMode", mode),
        court_keypoint_contract=resolve_court_keypoint_contract(selector),
    )


def _parse_figure_size(samples: ConfigMapping) -> tuple[float, float]:
    raw = require_config_value(samples, "figure_size", (list, tuple), path="samples")
    values = cast("Sequence[object]", raw)
    if len(values) != 2 or any(type(value) not in {float, int} for value in values):
        raise SemanticConfigurationError(
            "samples.figure_size must contain exactly two finite positive numbers."
        )
    width, height = (float(cast("float | int", value)) for value in values)
    if not all(math.isfinite(value) and value > 0.0 for value in (width, height)):
        raise SemanticConfigurationError(
            "samples.figure_size must contain exactly two finite positive numbers."
        )
    return width, height


def tercile_boundaries(
    values: Sequence[float], *, metric_name: str
) -> tuple[float, float]:
    """Return strict lower/upper tercile boundaries for a non-degenerate metric."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3 or not np.isfinite(array).all():
        raise ValueError(f"{metric_name} requires at least three finite values.")
    quantiles: NDArray[np.float64] = np.asarray(
        np.quantile(array, (1.0 / 3.0, 2.0 / 3.0)),
        dtype=np.float64,
    )
    if quantiles.shape != (2,):
        raise RuntimeError(
            f"{metric_name} tercile computation returned {quantiles.shape}."
        )
    lower = float(quantiles[0])
    upper = float(quantiles[1])
    if not lower < upper:
        raise ValueError(
            f"{metric_name} must have distinct tercile boundaries; got "
            f"{lower!r}, {upper!r}."
        )
    return lower, upper


def assign_tercile(
    value: float,
    boundaries: tuple[float, float],
    *,
    labels: tuple[str, str, str],
) -> str:
    """Assign one finite value to an inclusive-lower tercile band."""
    if not math.isfinite(value):
        raise ValueError("Tercile values must be finite.")
    lower, upper = boundaries
    if value <= lower:
        return labels[0]
    if value <= upper:
        return labels[1]
    return labels[2]


def select_stratified_samples(
    candidates: Sequence[DatasetSampleCandidate],
    *,
    primary_order: tuple[str, str, str],
    samples_per_stratum: int,
) -> DatasetSampleSelection:
    """Select every primary-group × duration-tercile cell deterministically.

    Duration is stratified within each primary group. Within a cell, the scene
    nearest its duration-band centre is preferred while visibility and a
    task-owned auxiliary metric follow offset Latin-square quantile targets.
    This creates deliberate low/mid/high coverage without random first-N bias.
    """
    if samples_per_stratum < 1:
        raise ValueError("samples_per_stratum must be positive.")
    if len(set(primary_order)) != 3:
        raise ValueError("primary_order must contain exactly three unique labels.")
    unknown = sorted(
        {candidate.primary_group for candidate in candidates} - set(primary_order)
    )
    if unknown:
        raise ValueError(f"Unknown primary group(s): {unknown!r}.")
    if len({candidate.scene_id for candidate in candidates}) != len(candidates):
        raise ValueError("Dataset sample candidate scene IDs must be unique.")
    _validate_candidates(candidates)

    visibility_values = tuple(candidate.visibility_value for candidate in candidates)
    auxiliary_values = tuple(candidate.auxiliary_value for candidate in candidates)
    duration_boundaries: dict[str, tuple[float, float]] = {}
    stratum_population: dict[str, int] = {}
    selected: list[SelectedDatasetSample] = []

    for row_index, primary in enumerate(primary_order):
        primary_candidates = [
            candidate for candidate in candidates if candidate.primary_group == primary
        ]
        boundaries = tercile_boundaries(
            [candidate.duration_value for candidate in primary_candidates],
            metric_name=f"{primary} duration",
        )
        duration_boundaries[primary] = boundaries
        primary_durations = tuple(
            candidate.duration_value for candidate in primary_candidates
        )
        for column_index, duration_band in enumerate(_DURATION_BANDS):
            cell = [
                candidate
                for candidate in primary_candidates
                if assign_tercile(
                    candidate.duration_value,
                    boundaries,
                    labels=_DURATION_BANDS,
                )
                == duration_band
            ]
            stratum_key = f"{primary}:{duration_band}"
            stratum_population[stratum_key] = len(cell)
            if len(cell) < samples_per_stratum:
                raise ValueError(
                    f"Stratum {stratum_key!r} contains {len(cell)} candidate(s), "
                    f"fewer than requested {samples_per_stratum}."
                )

            duration_target = (column_index + 0.5) / 3.0
            visibility_target = ((row_index + column_index) % 3 + 0.5) / 3.0
            auxiliary_target = ((row_index + 2 * column_index) % 3 + 0.5) / 3.0
            ranked = sorted(
                cell,
                key=lambda candidate: (
                    _selection_score(
                        candidate,
                        duration_values=primary_durations,
                        visibility_values=visibility_values,
                        auxiliary_values=auxiliary_values,
                        duration_target=duration_target,
                        visibility_target=visibility_target,
                        auxiliary_target=auxiliary_target,
                    ),
                    candidate.scene_id,
                ),
            )
            for rank, candidate in enumerate(ranked[:samples_per_stratum]):
                camera_index = choose_camera_index(
                    candidate.camera_visibilities,
                    target_quantile=visibility_target,
                )
                selected.append(
                    SelectedDatasetSample(
                        candidate=candidate,
                        duration_band=duration_band,
                        rank_in_stratum=rank,
                        selection_score=_selection_score(
                            candidate,
                            duration_values=primary_durations,
                            visibility_values=visibility_values,
                            auxiliary_values=auxiliary_values,
                            duration_target=duration_target,
                            visibility_target=visibility_target,
                            auxiliary_target=auxiliary_target,
                        ),
                        visibility_target_quantile=visibility_target,
                        auxiliary_target_quantile=auxiliary_target,
                        camera_index=camera_index,
                        camera_visibility=candidate.camera_visibilities[camera_index],
                    )
                )

    return DatasetSampleSelection(
        selected=tuple(selected),
        duration_boundaries=duration_boundaries,
        stratum_population=stratum_population,
    )


def _validate_candidates(candidates: Sequence[DatasetSampleCandidate]) -> None:
    if not candidates:
        raise ValueError("At least one dataset sample candidate is required.")
    for candidate in candidates:
        if not candidate.scene_id or not candidate.primary_group:
            raise ValueError(
                "Candidate scene IDs and primary groups must be non-empty."
            )
        numeric = (
            candidate.duration_value,
            candidate.visibility_value,
            candidate.auxiliary_value,
            *candidate.camera_visibilities,
        )
        if not numeric or not all(math.isfinite(value) for value in numeric):
            raise ValueError(
                f"Candidate {candidate.scene_id} contains non-finite metrics."
            )
        if candidate.duration_value <= 0.0 or not candidate.camera_visibilities:
            raise ValueError(
                f"Candidate {candidate.scene_id} needs positive duration and cameras."
            )


def _percentile(value: float, population: Sequence[float]) -> float:
    lower = sum(candidate < value for candidate in population)
    equal = sum(candidate == value for candidate in population)
    return (lower + equal * 0.5) / len(population)


def _selection_score(
    candidate: DatasetSampleCandidate,
    *,
    duration_values: Sequence[float],
    visibility_values: Sequence[float],
    auxiliary_values: Sequence[float],
    duration_target: float,
    visibility_target: float,
    auxiliary_target: float,
) -> float:
    return (
        abs(_percentile(candidate.duration_value, duration_values) - duration_target)
        + 0.35
        * abs(
            _percentile(candidate.visibility_value, visibility_values)
            - visibility_target
        )
        + 0.25
        * abs(
            _percentile(candidate.auxiliary_value, auxiliary_values) - auxiliary_target
        )
    )


def choose_camera_index(
    camera_visibilities: Sequence[float],
    *,
    target_quantile: float,
) -> int:
    """Choose a visible low/mid/high camera by within-scene visibility rank."""
    if not 0.0 <= target_quantile <= 1.0:
        raise ValueError("target_quantile must be within [0, 1].")
    eligible = sorted(
        (
            (index, float(visibility))
            for index, visibility in enumerate(camera_visibilities)
            if math.isfinite(float(visibility))
            and float(visibility) >= _MIN_VISIBLE_FRACTION
        ),
        key=lambda item: (item[1], item[0]),
    )
    if not eligible:
        raise ValueError(
            "A sample scene needs at least one camera with visibility >= "
            f"{_MIN_VISIBLE_FRACTION}."
        )
    position = int(math.floor(target_quantile * (len(eligible) - 1) + 0.5))
    return eligible[position][0]


def evenly_spaced_frame_indices(num_frames: int, max_frames: int) -> NDArray[np.int64]:
    """Return unique endpoint-inclusive frame indices bounded by ``max_frames``."""
    if num_frames < 2 or max_frames < 2:
        raise ValueError("Animated samples require num_frames and max_frames >= 2.")
    count = min(num_frames, max_frames)
    indices: NDArray[np.int64] = np.linspace(0, num_frames - 1, count, dtype=np.int64)
    if (
        len(np.unique(indices)) != count
        or indices[0] != 0
        or indices[-1] != num_frames - 1
    ):
        raise RuntimeError(
            "Even frame sampling failed its uniqueness/endpoints contract."
        )
    return indices


def bounded_playback_fps(
    *,
    source_fps: float,
    source_num_frames: int,
    rendered_num_frames: int,
    min_fps: int,
    max_fps: int,
) -> int:
    """Approximately preserve duration while bounding GIF playback usability."""
    if (
        not math.isfinite(source_fps)
        or source_fps <= 0.0
        or source_num_frames < 2
        or rendered_num_frames < 2
        or min_fps < 1
        or max_fps < min_fps
    ):
        raise ValueError("Invalid source timing or GIF fps bounds.")
    duration_preserving = (
        source_fps * (rendered_num_frames - 1) / (source_num_frames - 1)
    )
    return max(min_fps, min(max_fps, int(round(duration_preserving))))


def save_animation_gif(
    animation: Any,
    *,
    path: Path,
    fps: int,
    expected_frames: int,
) -> None:
    """Atomically save and validate one matplotlib animation GIF."""
    if path.suffix.lower() != ".gif":
        raise ValueError(f"Sample animation output must be .gif, got {path}.")
    if fps < 1 or expected_frames < 2:
        raise ValueError("GIF fps must be positive and expected_frames at least 2.")
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.gif")
    try:
        animation.save(str(temporary), writer=PillowWriter(fps=fps))
        with Image.open(temporary) as image:
            if (
                image.format != "GIF"
                or int(getattr(image, "n_frames", 1)) != expected_frames
            ):
                raise RuntimeError(
                    f"Encoded GIF {path.name} does not contain {expected_frames} frames."
                )
            if image.width <= 0 or image.height <= 0:
                raise RuntimeError(f"Encoded GIF {path.name} has invalid dimensions.")
        temporary.replace(path)
    finally:
        plt.close(animation._fig)
        temporary.unlink(missing_ok=True)


def materialize_dataset_samples(
    *,
    task: Literal["plcs", "blcs"],
    spec: DatasetSampleSpec,
    config: DatasetSamplesConfig,
    selection: DatasetSampleSelection,
    strategy: Mapping[str, object],
    render_sample: RenderSample,
) -> Path:
    """Render one selection plan and atomically write its evidence manifest."""
    samples_dir = spec.root / "samples"
    if samples_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Sample output already exists: {samples_dir}. Set samples.overwrite=true."
            )
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=False)

    entries: list[dict[str, object]] = []
    for selected in selection.selected:
        filename = _sample_filename(selected)
        rendered = render_sample(selected, samples_dir / filename)
        entries.append(
            {
                "scene_id": selected.candidate.scene_id,
                "output_file": filename,
                "stratum": {
                    "primary": selected.candidate.primary_group,
                    "duration": selected.duration_band,
                    "rank": selected.rank_in_stratum,
                },
                "selection": {
                    "score": selected.selection_score,
                    "visibility_target_quantile": selected.visibility_target_quantile,
                    "auxiliary_target_quantile": selected.auxiliary_target_quantile,
                },
                "camera": {
                    "index": selected.camera_index,
                    "visibility": selected.camera_visibility,
                },
                "metrics": dict(selected.candidate.metrics),
                "render": {
                    "source_num_frames": rendered.source_num_frames,
                    "rendered_num_frames": rendered.rendered_num_frames,
                    "source_fps": rendered.source_fps,
                    "encoded_fps": rendered.encoded_fps,
                    "frame_indices": list(rendered.frame_indices),
                },
            }
        )

    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "task": task,
        "dataset": spec.relative_path,
        "mode": spec.mode,
        "court_keypoints": {
            "selector": spec.court_keypoint_contract.selector,
            "contract_id": spec.court_keypoint_contract.contract_id,
            "target_frame_id": spec.court_keypoint_contract.target_frame_id,
        },
        "selection_strategy": dict(strategy),
        "selection_contract": {
            "samples_per_stratum": config.samples_per_stratum,
            "primary_groups": sorted(selection.duration_boundaries),
            "duration_bands": list(_DURATION_BANDS),
            "duration_boundaries_by_primary": {
                key: list(value) for key, value in selection.duration_boundaries.items()
            },
            "stratum_population": dict(selection.stratum_population),
        },
        "render_contract": {
            "view": config.view,
            "max_frames": config.max_frames,
            "min_fps": config.min_fps,
            "max_fps": config.max_fps,
            "figure_size_inches": list(config.figure_size),
            "minimum_camera_visibility": _MIN_VISIBLE_FRACTION,
        },
        "samples": entries,
    }
    manifest_path = samples_dir / "manifest.json"
    save_json_atomic(manifest, manifest_path)
    return manifest_path


def _sample_filename(selected: SelectedDatasetSample) -> str:
    primary = _slug(selected.candidate.primary_group)
    scene_id = _slug(selected.candidate.scene_id)
    return (
        f"primary-{primary}__duration-{selected.duration_band}__"
        f"rank-{selected.rank_in_stratum:02d}__{scene_id}__"
        f"camera-{selected.camera_index}.gif"
    )


def _slug(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.casefold()).strip("-")
    if not slug:
        raise ValueError(f"Cannot build a sample filename from {value!r}.")
    return slug


def load_scene_visibility_summaries(
    dataset_root: Path,
    *,
    visibility_key: str,
) -> Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """Load exact per-camera object/court visibility from root metadata."""
    raw = load_json(dataset_root / "meta.json")
    root = as_config_mapping(raw, path=str(dataset_root / "meta.json"))
    scenes = require_config_value(
        root, "scenes", list, path=str(dataset_root / "meta.json")
    )
    result: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for index, raw_scene in enumerate(cast("Sequence[object]", scenes)):
        scene_path = f"{dataset_root / 'meta.json'}.scenes[{index}]"
        scene = as_config_mapping(raw_scene, path=scene_path)
        scene_id = cast(
            "str", require_config_value(scene, "scene_id", str, path=scene_path)
        )
        raw_cameras = require_config_value(scene, "cameras", list, path=scene_path)
        camera_visibilities: list[float] = []
        court_visibilities: list[float] = []
        for camera_index, raw_camera in enumerate(
            cast("Sequence[object]", raw_cameras)
        ):
            camera_path = f"{scene_path}.cameras[{camera_index}]"
            camera = as_config_mapping(raw_camera, path=camera_path)
            object_visibility = require_config_value(
                camera, visibility_key, (float, int), path=camera_path
            )
            court_visibility = require_config_value(
                camera, "court_visibility_count", (float, int), path=camera_path
            )
            camera_visibilities.append(float(cast("float | int", object_visibility)))
            court_visibilities.append(float(cast("float | int", court_visibility)))
        if not camera_visibilities:
            raise ValueError(f"{scene_path} contains no cameras.")
        if scene_id in result:
            raise ValueError(f"Duplicate root metadata scene_id: {scene_id}.")
        result[scene_id] = (tuple(camera_visibilities), tuple(court_visibilities))
    return result


__all__ = [
    "DatasetMode",
    "DatasetSampleCandidate",
    "DatasetSampleSelection",
    "DatasetSampleSpec",
    "DatasetSamplesConfig",
    "RenderedDatasetSample",
    "SelectedDatasetSample",
    "assign_tercile",
    "bounded_playback_fps",
    "choose_camera_index",
    "evenly_spaced_frame_indices",
    "load_scene_visibility_summaries",
    "materialize_dataset_samples",
    "remap_sample_track_instances",
    "save_animation_gif",
    "select_stratified_samples",
    "take_temporal_sample",
    "tercile_boundaries",
    "track_lifecycle_metrics",
    "validate_sample_frame_indices",
]
