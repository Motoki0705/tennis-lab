"""Read-only checkpoint discovery for the PLCS inference UI.

The UI suggests checkpoints found under an outputs root instead of asking for a
free-text path first, and it narrows the offered scene families to the ones a
checkpoint can actually consume.  Listing a directory of multi-hundred-MB
checkpoints keeps the checkpoint body as the source of truth: its saved Hydra
config is read with a memory-mapped ``torch.load(..., mmap=True)`` so only the
metadata pages are touched.  A sibling ``hparams.yaml``/``config.yaml`` is used
only when the body cannot be read, and when both exist the critical fields must
agree; a stale sidecar that contradicts the body is rejected rather than
trusted.  The mapping from a checkpoint to its usable scene families is derived
from that config; nothing is inferred from file names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    PHYSICAL_V1_SELECTOR,
)

OBJECTNESS_SINGLE: Final = "single_object"
OBJECTNESS_MULTI: Final = "multi_object"

TRACK_QUERY_MODEL_NAMES: Final = frozenset(
    {"plcs_track_query", "plcs_track_query_reference"}
)
REFERENCE_MODEL_NAMES: Final = frozenset(
    {"plcs_multiview_axial_reference", "plcs_track_query_reference"}
)
SINGLE_OBJECT_MODEL_NAMES: Final = frozenset(
    {
        "plcs",
        "plcs_multiview_axial",
        "plcs_multiview_axial_split",
        "plcs_multiview_axial_camtoken",
        "plcs_multiview_axial_reference",
    }
)
SUPPORTED_INPUT_PROFILES: Final = frozenset({"frame", "sequence", "multiview"})
TRACK_QUERY_INPUT_PROFILE: Final = "track_query"

# A checkpoint consumes exactly one court selector and one object count, which
# together select the scene datasets it can run on.
FAMILIES_BY_TARGET: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    (OBJECTNESS_SINGLE, PHYSICAL_V1_SELECTOR): (
        "single_object",
        "single_object_broadcast",
    ),
    (OBJECTNESS_SINGLE, CAMERA_VIEW_V2_SELECTOR): ("single_object_camera_view_v2",),
    (OBJECTNESS_MULTI, PHYSICAL_V1_SELECTOR): (
        "multi_object",
        "multi_object_broadcast",
    ),
    (OBJECTNESS_MULTI, CAMERA_VIEW_V2_SELECTOR): ("multi_object_camera_view_v2",),
}

_SIDECAR_NAMES: Final = ("hparams.yaml", "config.yaml")


class CheckpointMetadataError(ValueError):
    """Raised when a checkpoint has no readable PLCS configuration."""


def objectness_for_model(model_name: str) -> str | None:
    """Return the object-count family a model name consumes, if known."""
    if model_name in TRACK_QUERY_MODEL_NAMES:
        return OBJECTNESS_MULTI
    if model_name in SINGLE_OBJECT_MODEL_NAMES:
        return OBJECTNESS_SINGLE
    return None


def is_reference_model(model_name: str) -> bool:
    """Return whether a model name requires an explicit reference camera."""
    return model_name in REFERENCE_MODEL_NAMES


def scene_family_of(scene_dir: str | None) -> str | None:
    """Return the ``data/plcs`` family name for a saved ``data.scene_dir``."""
    if not scene_dir:
        return None
    return Path(scene_dir).name or None


def allowed_scene_families(model_name: str, selector: str) -> tuple[str, ...]:
    """Return the ordered scene families a checkpoint can consume."""
    objects = objectness_for_model(model_name)
    if objects is None:
        return ()
    return FAMILIES_BY_TARGET.get((objects, selector), ())


def family_name(family: str) -> str:
    """Validate one scene family path component and return it unchanged."""
    stripped = family.strip()
    if not stripped or Path(stripped).name != stripped or stripped in {".", ".."}:
        raise ValueError(f"family must be a single path component, got {family!r}.")
    return stripped


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """Resolved, read-only description of one PLCS checkpoint."""

    path: Path
    relative: str
    label: str
    run_name: str
    checkpoint_name: str
    model_name: str
    selector: str
    input_profile: str | None
    objects: str | None
    reference: bool
    trained_scene_dir: str | None
    max_views: int | None
    max_seq_len: int | None
    num_queries: int | None
    seq_len_range: tuple[int, int] | None
    num_views_range: tuple[int, int] | None
    camera_candidates: tuple[int, ...] | None
    metadata_source: str
    supported: bool
    unsupported_reason: str | None
    families: tuple[str, ...]
    size_bytes: int
    modified_ns: int

    @property
    def id(self) -> str:
        """Return the stable id used by the HTTP API."""
        return self.relative

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload for the catalog endpoint."""
        return {
            "id": self.id,
            "path": str(self.path),
            "label": self.label,
            "run_name": self.run_name,
            "checkpoint_name": self.checkpoint_name,
            "model_name": self.model_name,
            "selector": self.selector,
            "input_profile": self.input_profile,
            "objects": self.objects,
            "reference": self.reference,
            "trained_scene_dir": self.trained_scene_dir,
            "max_views": self.max_views,
            "max_seq_len": self.max_seq_len,
            "num_queries": self.num_queries,
            "seq_len_range": (list(self.seq_len_range) if self.seq_len_range else None),
            "num_views_range": (
                list(self.num_views_range) if self.num_views_range else None
            ),
            "camera_candidates": (
                list(self.camera_candidates)
                if self.camera_candidates is not None
                else None
            ),
            "metadata_source": self.metadata_source,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "families": list(self.families),
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
        }


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    """Return a nested mapping value, or ``None`` when any key is missing."""
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _as_optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _as_optional_range(value: Any) -> tuple[int, int] | None:
    """Return a validated inclusive positive ``(min, max)`` int pair."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        return None
    low, high = value
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, int)
        or not isinstance(high, int)
        or low <= 0
        or high < low
    ):
        return None
    return (int(low), int(high))


def _as_optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    """Return a validated tuple of distinct non-negative camera indices."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        indices.append(int(item))
    if len(set(indices)) != len(indices):
        return None
    return tuple(indices)


def _config_payload(document: object) -> Mapping[str, Any] | None:
    """Return the composed Hydra config from a loaded YAML/mapping document."""
    if not isinstance(document, Mapping):
        return None
    nested = document.get("config")
    if isinstance(nested, Mapping):
        return cast("Mapping[str, Any]", nested)
    if "model" in document or "court_keypoints" in document:
        return cast("Mapping[str, Any]", document)
    return None


def _candidate_sidecars(path: Path) -> list[Path]:
    """Return sibling/ancestor config files, cheapest and closest first."""
    candidates: list[Path] = []
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    candidates.append(path.with_name(f"{stem}.config.yaml"))
    for ancestor in (path.parent, *path.parents[1:5]):
        candidates.extend(ancestor / name for name in _SIDECAR_NAMES)
    return candidates


def _document_from_sidecar(path: Path) -> tuple[Mapping[str, Any], str] | None:
    for candidate in _candidate_sidecars(path):
        if not candidate.is_file():
            continue
        try:
            with candidate.open(encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except yaml.YAMLError:
            continue
        payload = _config_payload(document)
        if payload is not None and _nested(payload, "model", "name") is not None:
            source = (
                "hparams_yaml" if candidate.name == "hparams.yaml" else "config_yaml"
            )
            return payload, source
    return None


def _document_from_archive(path: Path) -> Mapping[str, Any] | None:
    """Memory-map the checkpoint and read only its saved Hydra config."""
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        return None
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        return None
    config = hyper_parameters.get("config")
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=False)
        if isinstance(container, Mapping):
            return cast("Mapping[str, Any]", container)
        return None
    if isinstance(config, Mapping):
        return cast("Mapping[str, Any]", config)
    return None


def _unsupported(
    *,
    path: Path,
    relative: str,
    source: str,
    size_bytes: int,
    modified_ns: int,
    reason: str,
) -> CheckpointInfo:
    return CheckpointInfo(
        path=path,
        relative=relative,
        label=relative.removesuffix(path.suffix),
        run_name=path.parent.parent.name,
        checkpoint_name=path.name,
        model_name="",
        selector="",
        input_profile=None,
        objects=None,
        reference=False,
        trained_scene_dir=None,
        max_views=None,
        max_seq_len=None,
        num_queries=None,
        seq_len_range=None,
        num_views_range=None,
        camera_candidates=None,
        metadata_source=source,
        supported=False,
        unsupported_reason=reason,
        families=(),
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


def _build_info(
    *,
    path: Path,
    relative: str,
    config: Mapping[str, Any],
    source: str,
    size_bytes: int,
    modified_ns: int,
) -> CheckpointInfo:
    model_name = _nested(config, "model", "name")
    selector = _nested(config, "court_keypoints", "selector")
    if not isinstance(model_name, str) or not isinstance(selector, str):
        return _unsupported(
            path=path,
            relative=relative,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason="checkpoint の config に model.name / court_keypoints.selector がありません。",
        )

    objects = objectness_for_model(model_name)
    raw_profile = _nested(config, "model", "io", "input_profile")
    if isinstance(raw_profile, str):
        input_profile: str | None = raw_profile
    elif objects == OBJECTNESS_MULTI:
        # Track-query checkpoints carry no ``model.io`` block; the adapter
        # profile is fixed by the model architecture itself.
        input_profile = TRACK_QUERY_INPUT_PROFILE
    else:
        input_profile = None
    raw_scene_dir = _nested(config, "data", "scene_dir")
    trained_scene_dir = raw_scene_dir if isinstance(raw_scene_dir, str) else None
    max_views = _as_optional_positive_int(_nested(config, "model", "max_views"))
    max_seq_len = _as_optional_positive_int(_nested(config, "model", "max_seq_len"))
    num_queries = _as_optional_positive_int(_nested(config, "model", "num_queries"))
    seq_len_range = _as_optional_range(_nested(config, "data", "seq_len_range"))
    num_views_range = _as_optional_range(_nested(config, "data", "num_views_range"))
    camera_candidates = _as_optional_int_tuple(
        _nested(config, "data", "camera_candidates")
    )
    reference = is_reference_model(model_name)

    families = allowed_scene_families(model_name, selector)
    trained_family = scene_family_of(trained_scene_dir)
    if trained_family in families:
        families = (trained_family, *(f for f in families if f != trained_family))

    reason: str | None = None
    allowed_profiles = SUPPORTED_INPUT_PROFILES | (
        frozenset({TRACK_QUERY_INPUT_PROFILE})
        if objects == OBJECTNESS_MULTI
        else frozenset()
    )
    if reference and selector != CAMERA_VIEW_V2_SELECTOR:
        reason = "reference モデルは camera_view_v2 selector を要求します。"
    elif not families:
        reason = f"未対応の model/selector の組み合わせです: {model_name} / {selector}."
    elif input_profile not in allowed_profiles:
        reason = (
            "未対応の input profile です: "
            f"{input_profile!r} (対応: {sorted(allowed_profiles)})。"
        )

    return CheckpointInfo(
        path=path,
        relative=relative,
        label=relative.removesuffix(path.suffix),
        run_name=path.parent.parent.name,
        checkpoint_name=path.name,
        model_name=model_name,
        selector=selector,
        input_profile=input_profile,
        objects=objects,
        reference=reference,
        trained_scene_dir=trained_scene_dir,
        max_views=max_views,
        max_seq_len=max_seq_len,
        num_queries=num_queries,
        seq_len_range=seq_len_range,
        num_views_range=num_views_range,
        camera_candidates=camera_candidates,
        metadata_source=source,
        supported=reason is None,
        unsupported_reason=reason,
        families=families,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


# Fields whose disagreement between a checkpoint body and its sibling config
# means the sidecar is stale; the checkpoint is then rejected instead of
# described from untrusted metadata.
_CRITICAL_KEYS: Final = (
    ("model", "name"),
    ("model", "io", "input_profile"),
    ("model", "max_views"),
    ("model", "max_seq_len"),
    ("model", "num_queries"),
    ("court_keypoints", "selector"),
    ("data", "scene_dir"),
)


def _critical_fields(config: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(_nested(config, *keys) for keys in _CRITICAL_KEYS)


def _root_prefix(root: Path) -> str:
    """Return the id prefix that disambiguates a non-primary checkpoint root."""
    return f"{root.parent.name}/{root.name}/"


def _checkpoint_id(path: Path, root: Path, *, prefix: str) -> str:
    relative = (
        path.relative_to(root).as_posix()
        if path.is_relative_to(root)
        else path.as_posix()
    )
    return f"{prefix}{relative}"


def _checkpoint_roots(
    primary: Path, extra_roots: Sequence[str | Path] | None
) -> tuple[tuple[Path, str], ...]:
    """Return ``(root, id_prefix)`` pairs, primary first and deduplicated."""
    roots: list[tuple[Path, str]] = [(primary, "")]
    for raw in extra_roots or ():
        resolved = Path(raw).expanduser().resolve()
        if resolved == primary or not resolved.is_dir():
            continue
        roots.append((resolved, _root_prefix(resolved)))
    return tuple(roots)


def _resolve_config(path: Path) -> tuple[Mapping[str, Any] | None, str, str | None]:
    """Return ``(config, source, error)`` with the checkpoint body canonical."""
    sidecar = _document_from_sidecar(path)
    archive: Mapping[str, Any] | None = None
    archive_error: Exception | None = None
    try:
        archive = _document_from_archive(path)
    except Exception as error:  # pragma: no cover - torch raises many types
        archive_error = error
    if archive is not None:
        if sidecar is not None and _critical_fields(sidecar[0]) != _critical_fields(
            archive
        ):
            return (
                None,
                "checkpoint",
                "隣接する config ファイルが checkpoint 本体の config と一致しません。"
                "checkpoint 本体を正本として採用できないため、このチェックポイントを"
                "拒否しました。",
            )
        return archive, "checkpoint", None
    if sidecar is not None:
        config, source = sidecar
        return config, source, None
    if archive_error is not None:
        return None, "none", f"checkpoint を読み込めませんでした: {archive_error}"
    return None, "none", "checkpoint に PLCS config が見つかりません。"


def _describe(
    id_str: str, path_str: str, size_bytes: int, modified_ns: int
) -> CheckpointInfo:
    path = Path(path_str)
    config, source, error = _resolve_config(path)
    if config is None:
        return _unsupported(
            path=path,
            relative=id_str,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason=error or "checkpoint の config を解決できませんでした。",
        )
    return _build_info(
        path=path,
        relative=id_str,
        config=config,
        source=source,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


@lru_cache(maxsize=512)
def _describe_cached(
    id_str: str, path_str: str, size_bytes: int, modified_ns: int
) -> CheckpointInfo:
    return _describe(id_str, path_str, size_bytes, modified_ns)


def resolve_checkpoint_path(root: Path, value: str | Path) -> Path:
    """Resolve a checkpoint id (relative to ``root``) or an absolute path."""
    raw = Path(value)
    path = raw if raw.is_absolute() else root / raw
    path = path.expanduser().resolve()
    if path.suffix != ".ckpt":
        raise ValueError(f"checkpoint must end with '.ckpt': {path}")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def resolve_checkpoint(
    root: Path,
    value: str | Path,
    *,
    extra_roots: Sequence[str | Path] | None = None,
) -> tuple[Path, Path]:
    """Resolve a checkpoint id or path to its owning ``(root, path)``."""
    primary = Path(root).expanduser().resolve()
    if Path(value).is_absolute():
        return primary, resolve_checkpoint_path(primary, value)
    text = str(value).strip()
    for candidate_root, prefix in _checkpoint_roots(primary, extra_roots):
        relative = text[len(prefix) :] if prefix and text.startswith(prefix) else text
        try:
            return candidate_root, resolve_checkpoint_path(candidate_root, relative)
        except (FileNotFoundError, ValueError):
            continue
    # Nothing matched: raise the primary root's informative error instead of a
    # bare "not found" so the caller learns whether the value was malformed.
    return primary, resolve_checkpoint_path(primary, text)


def describe_checkpoint(
    root: Path,
    value: str | Path,
    *,
    extra_roots: Sequence[str | Path] | None = None,
) -> CheckpointInfo:
    """Resolve and describe one checkpoint id or path."""
    primary = Path(root).expanduser().resolve()
    owning_root, path = resolve_checkpoint(primary, value, extra_roots=extra_roots)
    prefix = "" if owning_root == primary else _root_prefix(owning_root)
    stat = path.stat()
    return _describe_cached(
        _checkpoint_id(path, owning_root, prefix=prefix),
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )


def load_checkpoint_config(
    root: Path,
    value: str | Path,
    *,
    extra_roots: Sequence[str | Path] | None = None,
) -> Mapping[str, Any]:
    """Return the canonical saved config that drives real inference or data."""
    _, path = resolve_checkpoint(root, value, extra_roots=extra_roots)
    config, _, error = _resolve_config(path)
    if config is None:
        raise CheckpointMetadataError(
            f"{path}: {error or 'checkpoint の config を解決できませんでした。'}"
        )
    return config


def scan_checkpoints(
    root: Path, *, extra_roots: Sequence[str | Path] | None = None
) -> list[CheckpointInfo]:
    """Describe every ``*.ckpt`` under each root in a stable id order."""
    primary = Path(root).expanduser().resolve()
    if not primary.is_dir():
        raise NotADirectoryError(f"checkpoint root is not a directory: {primary}")
    items: list[CheckpointInfo] = []
    seen: set[str] = set()
    for candidate_root, prefix in _checkpoint_roots(primary, extra_roots):
        for path in sorted(candidate_root.rglob("*.ckpt")):
            try:
                stat = path.stat()
            except OSError:
                continue
            id_str = _checkpoint_id(path, candidate_root, prefix=prefix)
            if id_str in seen:
                continue
            seen.add(id_str)
            items.append(
                _describe_cached(id_str, str(path), stat.st_size, stat.st_mtime_ns)
            )
    items.sort(key=lambda item: item.relative)
    return items


__all__ = [
    "CAMERA_VIEW_V2_SELECTOR",
    "FAMILIES_BY_TARGET",
    "OBJECTNESS_MULTI",
    "OBJECTNESS_SINGLE",
    "PHYSICAL_V1_SELECTOR",
    "REFERENCE_MODEL_NAMES",
    "SUPPORTED_INPUT_PROFILES",
    "TRACK_QUERY_INPUT_PROFILE",
    "TRACK_QUERY_MODEL_NAMES",
    "CheckpointInfo",
    "CheckpointMetadataError",
    "allowed_scene_families",
    "describe_checkpoint",
    "family_name",
    "is_reference_model",
    "load_checkpoint_config",
    "objectness_for_model",
    "resolve_checkpoint",
    "resolve_checkpoint_path",
    "scan_checkpoints",
    "scene_family_of",
]
