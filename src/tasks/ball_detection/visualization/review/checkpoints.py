"""Read-only checkpoint discovery for the ball-detection review UI.

The UI suggests the checkpoints found under ``outputs/ball_detection`` and
``ckpt/ball_detection`` instead of asking for a free-text path, and describes
each one from the Hydra config stored in its own body (read with a
memory-mapped ``torch.load(..., mmap=True)`` so listing stays cheap for
hundreds of megabytes of weights).  Nothing is inferred from file names.

The settings that actually constrain inference are reported as-is:
``model.name`` (which architecture factory runs), ``model.num_frames`` (the
window the checkpoint was trained with), the minimum number of frames the
architecture accepts, and the checkpoint's saved ``metrics`` block (peak
threshold, NMS, distance threshold, subpixel refinement).  The training
``data.source`` is deliberately *not* used to decide compatibility: a
checkpoint trained on TrackNet still consumes any RGB source, and static web
frames are consumed through the dataset's own canonical repetition mode.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from omegaconf import DictConfig

from src.tasks.ball_detection.evaluation.configuration import read_checkpoint_config

#: Minimum temporal extent each architecture accepts.  ``model_io/factory.py``
#: binds ``minimum_frames`` per model; the values must stay in sync and
#: ``tests/unit/tasks/ball_detection/visualization/inference`` asserts it by
#: constructing the factories for the cheap architectures.
TASK_NAME: Final = "ball_detection"

MINIMUM_FRAMES_BY_MODEL: Final[dict[str, int]] = {
    "stunet": 8,
    "conv_next_unet": 1,
    "dinov3_rope": 1,
}

#: Fallbacks mirroring ``configs/metrics/default.yaml`` for checkpoints that
#: predate the saved metrics block.  They are evaluation settings with a
#: documented repository default, not model-architecture values.
DEFAULT_PEAK_THRESHOLD: Final = 0.5
DEFAULT_BALL_DISTANCE_THRESHOLD: Final = 4.0
DEFAULT_NMS_KERNEL: Final = 9
DEFAULT_MAX_PREDICTIONS_PER_FRAME: Final = 8


@dataclass(frozen=True, slots=True)
class BallMetricsDefaults:
    """Peak-decoding and evaluation settings saved with a checkpoint."""

    peak_threshold: float
    ball_distance_threshold: float
    nms_kernel: int
    max_predictions_per_frame: int
    subpixel_refine: bool


@dataclass(frozen=True, slots=True)
class MetricsSettings:
    """Resolved metrics settings plus what had to be assumed to get them."""

    defaults: BallMetricsDefaults
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BallCheckpointInfo:
    """Resolved, read-only description of one ball-detection checkpoint."""

    id: str
    path: Path
    label: str
    model: str
    num_frames: int
    minimum_frames: int
    image_size_hw: tuple[int, int] | None
    input_mode: str
    metrics: BallMetricsDefaults
    error: str | None
    warnings: tuple[str, ...] = ()
    size_bytes: int = 0
    modified_ns: int = 0

    @property
    def state(self) -> tuple[int, int]:
        """Return the ``(size, mtime_ns)`` pair this description was read from."""
        return self.size_bytes, self.modified_ns

    @property
    def usable(self) -> bool:
        """Return whether the checkpoint can drive inference."""
        return self.error is None

    @property
    def minimum_window(self) -> int:
        """Return the smallest temporal window this checkpoint can evaluate.

        MDD models consume a frame-difference channel, so a one-frame window
        would feed an all-zero motion pair that no checkpoint was trained on.
        A checkpoint whose own configured window is a single frame is the
        exception: that *is* its trained contract.
        """
        if self.input_mode == "mdd" and self.num_frames > 1:
            return max(self.minimum_frames, 2)
        return self.minimum_frames

    @property
    def maximum_window(self) -> int:
        """Return the trained temporal window length."""
        return self.num_frames

    def settings(self) -> dict[str, Any]:
        """Return the inference settings the UI offers for this checkpoint."""
        return {
            "count": self.num_frames,
            "threshold": self.metrics.peak_threshold,
        }

    def to_dict(self, *, compatible_datasets: Sequence[str]) -> dict[str, Any]:
        """Return the JSON payload for the catalog endpoint."""
        payload: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "path": str(self.path),
            "model": self.model,
            "compatible_datasets": list(compatible_datasets),
            "settings": self.settings(),
            "window": {
                "min": self.minimum_window,
                "max": self.maximum_window,
            },
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload


def _finite_number(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def read_metrics_settings(config: Mapping[str, Any]) -> MetricsSettings:
    """Read the saved ``metrics`` block strictly.

    A *missing* key falls back to the documented repository default
    (``configs/metrics/default.yaml``) and is reported as a warning.  A key that
    is present but not a legal value -- including ``nan``/``inf`` -- is an error
    that makes the checkpoint unusable, because silently substituting a
    different threshold would change what the UI reports.
    """
    raw = config.get("metrics")
    warnings: list[str] = []
    errors: list[str] = []
    if raw is None:
        warnings.append(
            "checkpoint has no saved metrics block; repository defaults apply."
        )
        block: Mapping[str, Any] = {}
    elif not isinstance(raw, Mapping):
        errors.append("checkpoint metrics block must be a mapping.")
        block = {}
    else:
        block = raw

    peak = block.get("peak_threshold")
    if "peak_threshold" not in block:
        warnings.append("metrics.peak_threshold is missing; using the default.")
        peak_threshold = DEFAULT_PEAK_THRESHOLD
    else:
        number = _finite_number(peak)
        if number is None or not 0.0 <= number <= 1.0:
            errors.append(
                f"metrics.peak_threshold must be a finite number in [0, 1], "
                f"got {peak!r}."
            )
            peak_threshold = DEFAULT_PEAK_THRESHOLD
        else:
            peak_threshold = number

    distance = block.get("ball_distance_threshold")
    if "ball_distance_threshold" not in block:
        warnings.append(
            "metrics.ball_distance_threshold is missing; using the default."
        )
        ball_distance_threshold = DEFAULT_BALL_DISTANCE_THRESHOLD
    else:
        number = _finite_number(distance)
        if number is None or number <= 0.0:
            errors.append(
                "metrics.ball_distance_threshold must be a finite positive "
                f"number, got {distance!r}."
            )
            ball_distance_threshold = DEFAULT_BALL_DISTANCE_THRESHOLD
        else:
            ball_distance_threshold = number

    kernel = block.get("nms_kernel")
    if "nms_kernel" not in block:
        warnings.append("metrics.nms_kernel is missing; using the default.")
        nms_kernel = DEFAULT_NMS_KERNEL
    else:
        value = _positive_int(kernel)
        if value is None or value % 2 == 0:
            errors.append(
                f"metrics.nms_kernel must be a positive odd integer, got {kernel!r}."
            )
            nms_kernel = DEFAULT_NMS_KERNEL
        else:
            nms_kernel = value

    peaks = block.get("max_predictions_per_frame")
    if "max_predictions_per_frame" not in block:
        warnings.append(
            "metrics.max_predictions_per_frame is missing; using the default."
        )
        max_predictions_per_frame = DEFAULT_MAX_PREDICTIONS_PER_FRAME
    else:
        value = _positive_int(peaks)
        if value is None:
            errors.append(
                "metrics.max_predictions_per_frame must be a positive integer, "
                f"got {peaks!r}."
            )
            max_predictions_per_frame = DEFAULT_MAX_PREDICTIONS_PER_FRAME
        else:
            max_predictions_per_frame = value

    subpixel = block.get("subpixel_refine")
    if "subpixel_refine" not in block:
        warnings.append("metrics.subpixel_refine is missing; using the default.")
        subpixel_refine = True
    elif isinstance(subpixel, bool):
        subpixel_refine = subpixel
    else:
        errors.append(
            f"metrics.subpixel_refine must be a bool, got {subpixel!r}."
        )
        subpixel_refine = True

    return MetricsSettings(
        defaults=BallMetricsDefaults(
            peak_threshold=peak_threshold,
            ball_distance_threshold=ball_distance_threshold,
            nms_kernel=nms_kernel,
            max_predictions_per_frame=max_predictions_per_frame,
            subpixel_refine=subpixel_refine,
        ),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def image_size_of(config: Mapping[str, Any], *, model: str) -> tuple[int, int] | None:
    """Return the spatial input size this checkpoint requires, if it fixes one."""
    model_block = config.get("model")
    if not isinstance(model_block, Mapping):
        return None
    source: Any
    if model == "dinov3_rope":
        source = model_block.get("image_size")
    else:
        data_block = config.get("data")
        source = data_block.get("image_size") if isinstance(data_block, Mapping) else None
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        return None
    if len(source) != 2:
        return None
    height, width = source[0], source[1]
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height <= 0
        or width <= 0
    ):
        return None
    return height, width


def file_state(path: Path) -> tuple[int, int]:
    """Return the ``(size_bytes, mtime_ns)`` pair identifying one file revision."""
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def rejected_checkpoint(
    *,
    checkpoint_id: str,
    path: Path,
    label: str,
    error: str,
) -> BallCheckpointInfo:
    """Build a description for a checkpoint that must not be used.

    The architecture fields stay empty on purpose: a rejected checkpoint has no
    verified model, window, or thresholds, so nothing downstream may read one.
    """
    try:
        state = file_state(path)
    except OSError:
        state = (0, 0)
    return BallCheckpointInfo(
        id=checkpoint_id,
        path=path,
        label=label,
        model="",
        num_frames=0,
        minimum_frames=0,
        image_size_hw=None,
        input_mode="",
        metrics=read_metrics_settings({}).defaults,
        error=error,
        size_bytes=state[0],
        modified_ns=state[1],
    )


def describe_checkpoint(
    *,
    checkpoint_id: str,
    path: Path,
    label: str,
) -> BallCheckpointInfo:
    """Read one checkpoint body into a description, capturing failures.

    The file's ``(size, mtime_ns)`` is recorded so callers can detect a replaced
    body and re-read the metadata instead of inferring with a stale description.
    """
    state = file_state(path)
    try:
        config = read_checkpoint_config(path)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim in the catalog
        return rejected_checkpoint(
            checkpoint_id=checkpoint_id,
            path=path,
            label=label,
            error=f"{type(error).__name__}: {error}",
        )
    return describe_config(
        checkpoint_id=checkpoint_id,
        path=path,
        label=label,
        config=config,
        state=state,
    )


def describe_config(
    *,
    checkpoint_id: str,
    path: Path,
    label: str,
    config: DictConfig,
    state: tuple[int, int] | None = None,
) -> BallCheckpointInfo:
    """Validate the saved config fields the review UI depends on.

    Everything the review UI must read before it can run a window is required
    here: the architecture name, ``model.num_frames``, ``model.input_mode``, and
    the spatial ``image_size`` the input is resized to.  A checkpoint missing any
    of them is reported unusable instead of failing later mid-request.
    """
    container: Mapping[str, Any] = cast(Mapping[str, Any], config)
    model_block = container.get("model")
    if not isinstance(model_block, Mapping):
        model_block = {}
    model = model_block.get("name")
    model_name = str(model) if isinstance(model, str) else ""
    input_mode = model_block.get("input_mode")
    num_frames = _positive_int(model_block.get("num_frames")) or 0
    minimum_frames = MINIMUM_FRAMES_BY_MODEL.get(model_name, 0)
    settings = read_metrics_settings(container)
    image_size = image_size_of(container, model=model_name)
    resolved_state = state if state is not None else (0, 0)

    errors: list[str] = list(settings.errors)
    if not model_name:
        errors.append("checkpoint config does not declare model.name.")
    elif model_name not in MINIMUM_FRAMES_BY_MODEL:
        supported = ", ".join(sorted(MINIMUM_FRAMES_BY_MODEL))
        errors.append(
            f"unsupported model.name={model_name!r}; expected one of [{supported}]."
        )
    elif num_frames <= 0:
        errors.append("checkpoint config does not declare a positive model.num_frames.")
    elif num_frames < minimum_frames:
        errors.append(
            f"{model_name} requires at least {minimum_frames} frames, but the "
            f"checkpoint was configured with model.num_frames={num_frames}."
        )
    if not isinstance(input_mode, str) or input_mode not in {"rgb", "mdd"}:
        errors.append(
            "checkpoint config must declare model.input_mode as 'rgb' or 'mdd', "
            f"got {input_mode!r}."
        )
    if image_size is None:
        errors.append(
            "checkpoint config does not declare a 2-element positive image_size "
            "for the configured model."
        )

    return BallCheckpointInfo(
        id=checkpoint_id,
        path=path,
        label=label,
        model=model_name,
        num_frames=num_frames,
        minimum_frames=minimum_frames,
        image_size_hw=image_size,
        input_mode=input_mode if isinstance(input_mode, str) else "",
        metrics=settings.defaults,
        error="; ".join(errors) if errors else None,
        warnings=settings.warnings,
        size_bytes=resolved_state[0],
        modified_ns=resolved_state[1],
    )


def checkpoint_roots(
    output_root: Path, checkpoint_root: Path
) -> tuple[tuple[Path, str], ...]:
    """Return ``(root, id prefix)`` pairs, primary outputs first.

    The CLI passes task-specific roots (``outputs/ball_detection``,
    ``ckpt/ball_detection``), but callers may also pass their parent.  Accepting
    the nested form keeps both spellings working without renaming anything.
    """
    primary = Path(output_root).expanduser().resolve(strict=False)
    extra = Path(checkpoint_root).expanduser().resolve(strict=False)
    roots: list[tuple[Path, str]] = [(primary, "")]
    nested = primary / TASK_NAME
    if nested != primary and nested.is_dir():
        roots.append((nested, f"{primary.name}/{TASK_NAME}/"))
    if extra not in {root for root, _ in roots}:
        prefix = f"{extra.parent.name}/{extra.name}/"
        roots.append((extra, prefix))
    return tuple(roots)


def scan_checkpoints(
    roots: Iterable[tuple[Path, str]],
) -> list[BallCheckpointInfo]:
    """Discover ``*.ckpt`` files under ``roots`` and describe each one.

    Only regular files that stay inside their configured root are offered.  A
    ``*.ckpt`` that resolves outside the root (a symlink pointing elsewhere) is
    listed with an explicit error instead of being read, because reading it
    would unpickle code from outside the directory the operator configured.
    """
    discovered: list[tuple[str, Path, Path]] = []
    rejected: list[BallCheckpointInfo] = []
    seen: set[Path] = set()
    for root, prefix in roots:
        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            continue
        for candidate in sorted(resolved_root.rglob("*.ckpt")):
            if candidate.is_symlink() and not candidate.exists():
                continue
            resolved = candidate.resolve()
            checkpoint_id = (
                f"{prefix}{candidate.relative_to(resolved_root).as_posix()}"
            )
            if not resolved.is_relative_to(resolved_root):
                rejected.append(
                    rejected_checkpoint(
                        checkpoint_id=checkpoint_id,
                        path=candidate,
                        label=candidate.stem,
                        error=(
                            f"checkpoint resolves outside the configured root "
                            f"{resolved_root}; refusing to load it."
                        ),
                    )
                )
                continue
            if not resolved.is_file():
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append((checkpoint_id, candidate, resolved))

    infos = [
        describe_checkpoint(
            checkpoint_id=checkpoint_id,
            path=candidate,
            label=resolved.stem,
        )
        for checkpoint_id, candidate, resolved in discovered
    ]
    return _deduplicate(infos + rejected)


def _deduplicate(infos: Sequence[BallCheckpointInfo]) -> list[BallCheckpointInfo]:
    """Keep ids unique by tagging duplicated relative paths with their root."""
    counts: dict[str, int] = {}
    for info in infos:
        counts[info.id] = counts.get(info.id, 0) + 1
    unique: list[BallCheckpointInfo] = []
    for info in infos:
        if counts[info.id] == 1:
            unique.append(info)
            continue
        parent = info.path.parent.name
        unique.append(
            BallCheckpointInfo(
                id=f"{parent}/{info.id}",
                path=info.path,
                label=info.label,
                model=info.model,
                num_frames=info.num_frames,
                minimum_frames=info.minimum_frames,
                image_size_hw=info.image_size_hw,
                input_mode=info.input_mode,
                metrics=info.metrics,
                error=info.error,
                warnings=info.warnings,
                size_bytes=info.size_bytes,
                modified_ns=info.modified_ns,
            )
        )
    return sorted(unique, key=lambda item: item.id)


__all__ = [
    "DEFAULT_BALL_DISTANCE_THRESHOLD",
    "DEFAULT_MAX_PREDICTIONS_PER_FRAME",
    "DEFAULT_NMS_KERNEL",
    "DEFAULT_PEAK_THRESHOLD",
    "MINIMUM_FRAMES_BY_MODEL",
    "TASK_NAME",
    "BallCheckpointInfo",
    "BallMetricsDefaults",
    "checkpoint_roots",
    "describe_checkpoint",
    "describe_config",
    "file_state",
    "image_size_of",
    "read_metrics_settings",
    "rejected_checkpoint",
    "scan_checkpoints",
]
