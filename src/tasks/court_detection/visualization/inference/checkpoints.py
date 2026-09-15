"""Read-only checkpoint discovery for the Court detection inference UI.

Only the modes the current contract can actually execute are offered.  The
checkpoint body is the sole authority: its saved Hydra config and
``target_bundle_state`` are read with a memory-mapped ``torch.load`` and
revalidated through the strict training-config and bundle contracts.  A sibling
``hparams.yaml`` is used only as a cross-check and never as a substitute, an
unreadable body is always unusable, and legacy single-head checkpoints without a
bundle snapshot are reported unsupported with the exact reason instead of being
migrated by guesswork.  Only ids below the configured roots can execute: an
absolute path, a ``..`` traversal, or a symlink that escapes the root is
rejected.
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

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.bundle_state import deserialize_target_bundle

COURT_TASK = "court_detection"
_SIDECAR_NAMES: Final = ("hparams.yaml", "config.yaml")
_CRITICAL_KEYS: Final = (
    ("target_bundle_state",),
    ("config", "data", "source", "kind"),
    ("config", "data", "source", "schema"),
    ("config", "model", "name"),
)


class CheckpointMetadataError(ValueError):
    """Raised when a checkpoint exposes no readable Court configuration."""


@dataclass(frozen=True, slots=True)
class CourtHeadSpec:
    """One trained head recovered from the checkpoint's serialized bundle."""

    kind: str
    schema: str
    output_channels: int
    channel_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CourtCheckpointInfo:
    """Resolved, read-only description of one Court checkpoint."""

    id: str
    path: Path
    label: str
    model: str
    metadata_source: str
    supported: bool
    reason: str | None
    heads: tuple[CourtHeadSpec, ...]
    size_bytes: int
    modified_ns: int

    def head(self, kind: str) -> CourtHeadSpec | None:
        for spec in self.heads:
            if spec.kind == kind:
                return spec
        return None

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(spec.kind for spec in self.heads)

    def dense_schemas(self) -> dict[str, str]:
        return {spec.kind: spec.schema for spec in self.heads if spec.kind != "kp"}

    def keypoint_channel_names(self) -> tuple[str, ...]:
        spec = self.head("kp")
        return spec.channel_names if spec is not None else ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "path": str(self.path),
            "label": self.label,
            "model": self.model,
            "metadata_source": self.metadata_source,
            "supported": self.supported,
            "heads": [
                {
                    "kind": spec.kind,
                    "schema": spec.schema,
                    "output_channels": spec.output_channels,
                }
                for spec in self.heads
            ],
            "settings": {"count": 1, "threshold": 0.5},
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
        }
        if self.reason is not None:
            payload["error"] = self.reason
        return payload


def checkpoint_roots(
    output_root: Path, checkpoint_root: Path
) -> tuple[tuple[Path, str], ...]:
    """Return ``(root, id prefix)`` pairs, primary outputs first.

    Both roots are already task-scoped by the shared CLI
    (``<project>/outputs/<task>`` and ``<project>/ckpt/<task>``), so the ids of
    the curated ``ckpt`` tree stay distinguishable without re-deriving the task
    directory here.
    """
    primary = Path(output_root).expanduser().resolve(strict=False)
    roots: list[tuple[Path, str]] = [(primary, "")]
    extra = Path(checkpoint_root).expanduser().resolve(strict=False)
    if extra != primary:
        roots.append((extra, f"{extra.parent.name}/"))
    return tuple(roots)


def _nested(document: object, *keys: str) -> object:
    value = document
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _critical_fields(document: object) -> tuple[object, ...]:
    return tuple(_nested(document, *keys) for keys in _CRITICAL_KEYS)


def _candidate_sidecars(path: Path) -> list[Path]:
    candidates: list[Path] = [path.with_name(f"{path.stem}.config.yaml")]
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
        if (
            isinstance(document, Mapping)
            and "config" in document
            and isinstance(document["config"], Mapping)
        ):
            source = (
                "hparams_yaml" if candidate.name == "hparams.yaml" else "config_yaml"
            )
            return cast("Mapping[str, Any]", document), source
    return None


def _document_from_archive(path: Path) -> Mapping[str, Any]:
    """Read the canonical checkpoint body; raise when it carries no config."""
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint 本体が mapping ではありません。")
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError("checkpoint 本体に hyper_parameters がありません。")
    config = hyper_parameters.get("config")
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint 本体に保存済み Court config がありません。")
    payload: dict[str, Any] = {"config": cast("Mapping[str, Any]", config)}
    bundle_state = hyper_parameters.get("target_bundle_state")
    if bundle_state is not None:
        payload["target_bundle_state"] = bundle_state
    return payload


def _resolve_document(
    path: Path,
) -> tuple[Mapping[str, Any] | None, str, str | None]:
    """Return ``(document, metadata_source, error)`` with the body canonical.

    The body is the only source that can grant support: a checkpoint whose bytes
    cannot be read, or whose body carries no config, is unusable even when a
    sibling ``hparams.yaml`` looks complete.
    """
    sidecar = _document_from_sidecar(path)
    try:
        archive = _document_from_archive(path)
    except Exception as error:  # torch raises several unrelated types
        return (
            None,
            "none",
            f"checkpoint 本体を読み込めず使用できません: "
            f"{type(error).__name__}: {error}",
        )
    if sidecar is not None and _critical_fields(sidecar[0]) != _critical_fields(
        archive
    ):
        return (
            None,
            "checkpoint",
            "隣接する hparams.yaml が checkpoint 本体と一致しません。stale な metadata を"
            "信用せず checkpoint を拒否しました。",
        )
    return archive, "checkpoint", None


def _bundle_heads(value: object) -> tuple[CourtHeadSpec, ...]:
    bundle = deserialize_target_bundle(value)
    return tuple(
        CourtHeadSpec(
            kind=spec.kind,
            schema=spec.schema,
            output_channels=spec.output_channels,
            channel_names=tuple(spec.channel_names),
        )
        for spec in bundle.targets.values()
    )


def _unsupported(
    *,
    id_str: str,
    path: Path,
    model: str,
    source: str,
    size_bytes: int,
    modified_ns: int,
    reason: str,
) -> CourtCheckpointInfo:
    return CourtCheckpointInfo(
        id=id_str,
        path=path,
        label=id_str.removesuffix(path.suffix),
        model=model,
        metadata_source=source,
        supported=False,
        reason=reason,
        heads=(),
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


def _describe(
    id_str: str, path_str: str, size_bytes: int, modified_ns: int
) -> CourtCheckpointInfo:
    path = Path(path_str)
    document, source, error = _resolve_document(path)
    if document is None:
        return _unsupported(
            id_str=id_str,
            path=path,
            model=COURT_TASK,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason=error or "checkpoint config を解決できませんでした。",
        )
    config = document.get("config")
    model_name = _nested(config, "model", "name")
    model = model_name if isinstance(model_name, str) else COURT_TASK
    bundle_state = document.get("target_bundle_state")
    if bundle_state is None:
        return _unsupported(
            id_str=id_str,
            path=path,
            model=model,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason=(
                "checkpoint に target_bundle_state が保存されていません。legacy な "
                "single-head run であり現行 bundle へは再構成できません。"
            ),
        )
    try:
        heads = _bundle_heads(bundle_state)
    except ValueError as error:
        return _unsupported(
            id_str=id_str,
            path=path,
            model=model,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason=f"checkpoint の target_bundle_state が不正です: {error}",
        )
    try:
        CourtTrainingConfig.from_config(config)
    except Exception as error:  # configuration contracts raise many types
        return _unsupported(
            id_str=id_str,
            path=path,
            model=model,
            source=source,
            size_bytes=size_bytes,
            modified_ns=modified_ns,
            reason=(
                "checkpoint config が現行契約より前のもので strict に読み込めません: "
                f"{type(error).__name__}: {error}"
            ),
        )
    return CourtCheckpointInfo(
        id=id_str,
        path=path,
        label=id_str.removesuffix(path.suffix),
        model=model,
        metadata_source=source,
        supported=True,
        reason=None,
        heads=heads,
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


@lru_cache(maxsize=512)
def _describe_cached(
    id_str: str, path_str: str, size_bytes: int, modified_ns: int
) -> CourtCheckpointInfo:
    return _describe(id_str, path_str, size_bytes, modified_ns)


def _checkpoint_id(path: Path, root: Path, *, prefix: str) -> str:
    relative = (
        path.relative_to(root).as_posix()
        if path.is_relative_to(root)
        else path.as_posix()
    )
    return f"{prefix}{relative}"


def resolve_checkpoint_path(root: Path, value: str | Path) -> Path:
    """Resolve a catalog-relative checkpoint id strictly inside ``root``.

    Only ids that stay below the configured root are accepted: absolute paths,
    ``..`` traversal, and symlinks whose target escapes the root are refused so
    an HTTP client can never select a checkpoint outside the reviewed trees.
    """
    resolved_root = Path(root).expanduser().resolve(strict=False)
    raw = Path(value)
    if raw.is_absolute():
        raise ValueError(
            "Court checkpoint は catalog ID で選択してください（絶対 path は受け付けません）。"
        )
    if not raw.parts or any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError("Court checkpoint ID は安全な相対 path である必要があります。")
    path = resolved_root / raw
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("Court checkpoint ID が許可 root の外を指すため拒否しました。")
    if path.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"Court checkpoint が見つかりません: {path}")
    if path.suffix != ".ckpt":
        raise ValueError(f"Court checkpoint は '.ckpt' で終わる必要があります: {path}")
    return resolved


def resolve_checkpoint(
    output_root: Path, checkpoint_root: Path, value: str | Path
) -> tuple[Path, Path]:
    """Resolve a checkpoint id or path to its owning ``(root, path)``."""
    text = str(value).strip()
    for root, prefix in checkpoint_roots(output_root, checkpoint_root):
        relative = text[len(prefix) :] if prefix and text.startswith(prefix) else text
        if any(part in {"", ".", ".."} for part in Path(relative).parts):
            raise ValueError(
                "Court checkpoint ID は安全な相対 path である必要があります。"
            )
        try:
            return root, resolve_checkpoint_path(root, relative)
        except FileNotFoundError:
            continue
    primary = checkpoint_roots(output_root, checkpoint_root)[0][0]
    return primary, resolve_checkpoint_path(primary, text)


def describe_checkpoint(
    output_root: Path, checkpoint_root: Path, value: str | Path
) -> CourtCheckpointInfo:
    """Resolve and describe one checkpoint id or path."""
    roots = checkpoint_roots(output_root, checkpoint_root)
    primary = roots[0][0]
    owning_root, path = resolve_checkpoint(output_root, checkpoint_root, value)
    prefix = "" if owning_root == primary else f"{owning_root.parent.name}/"
    stat = path.stat()
    return _describe_cached(
        _checkpoint_id(path, owning_root, prefix=prefix),
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )


def scan_checkpoints(
    output_root: Path, checkpoint_root: Path
) -> list[CourtCheckpointInfo]:
    """Describe every ordinary ``*.ckpt`` under both roots in stable id order."""
    items: list[CourtCheckpointInfo] = []
    seen: set[str] = set()
    for root, prefix in checkpoint_roots(output_root, checkpoint_root):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.ckpt")):
            if path.is_symlink():
                continue
            if not path.resolve(strict=False).is_relative_to(
                root.resolve(strict=False)
            ):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            id_str = _checkpoint_id(path, root, prefix=prefix)
            if id_str in seen:
                continue
            seen.add(id_str)
            items.append(
                _describe_cached(id_str, str(path), stat.st_size, stat.st_mtime_ns)
            )
    items.sort(key=lambda item: item.id)
    return items


def head_kinds(info: CourtCheckpointInfo) -> Sequence[str]:
    """Return the checkpoint's head kinds in serialized order."""
    return info.kinds


__all__ = [
    "COURT_TASK",
    "CheckpointMetadataError",
    "CourtCheckpointInfo",
    "CourtHeadSpec",
    "checkpoint_roots",
    "describe_checkpoint",
    "head_kinds",
    "resolve_checkpoint",
    "resolve_checkpoint_path",
    "scan_checkpoints",
]
