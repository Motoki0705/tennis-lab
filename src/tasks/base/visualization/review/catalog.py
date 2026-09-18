"""Read-only catalog over ``<data_root>/<task>/<form>/scenes``.

Only cheap directory and stat metadata is touched here: form discovery, scene
listing, per-form counts, and a revision derived from scene file stats.
Heavy scene arrays are never
loaded by this module, so catalogs over 10,000-scene datasets stay responsive.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    resolve_court_keypoint_contract,
)

SCENES_DIR_NAME = "scenes"
SAMPLES_DIR_NAME = "samples"
FORM_META_NAME = "meta.json"
SCENE_META_NAME = "meta.json"
SCENE_SCALARS_NAME = "scalars.json"
SINGLE_MODE = "single"
MULTI_MODE = "multi"


class DatasetCatalogError(ValueError):
    """Raised when the data root or a requested form/scene is invalid."""


@dataclass(frozen=True, slots=True)
class DatasetForm:
    """One detected dataset form (an immediate child of the task directory)."""

    name: str
    relative_path: str
    mode: str
    scene_count: int
    has_samples: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.relative_path,
            "mode": self.mode,
            "scene_count": self.scene_count,
            "has_samples": self.has_samples,
        }


def _single_component(value: object, *, label: str) -> str:
    """Return ``value`` as a safe single path component or raise."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise DatasetCatalogError(f"{label} must be a single path component.")
    if "/" in value or "\\" in value or Path(value).name != value:
        raise DatasetCatalogError(f"{label} must not contain a path separator.")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetCatalogError(f"{path}: invalid JSON ({error}).") from error
    if not isinstance(document, dict):
        raise DatasetCatalogError(f"{path}: expected a JSON object.")
    return document


def _form_mode(meta_path: Path) -> str:
    """Derive ``single``/``multi`` from the saved generation config."""
    document = _load_json_object(meta_path)
    config = document.get("config")
    generation = config.get("generation") if isinstance(config, dict) else None
    mode = generation.get("mode") if isinstance(generation, dict) else None
    if not isinstance(mode, str) or not mode:
        raise DatasetCatalogError(
            f"{meta_path}: config.generation.mode is required to classify a form."
        )
    return MULTI_MODE if mode.startswith("multi") else SINGLE_MODE


class DatasetCatalog:
    """Discover the forms and scenes of one BLCS/PLCS dataset root."""

    def __init__(
        self,
        data_root: str | Path,
        task: str,
        *,
        forms: Sequence[str] | None = None,
    ) -> None:
        try:
            root = Path(data_root).resolve(strict=True)
        except FileNotFoundError as error:
            raise DatasetCatalogError(
                f"Data root does not exist: {Path(data_root)}."
            ) from error
        if not root.is_dir():
            raise DatasetCatalogError(f"Data root is not a directory: {root}.")
        self.root = root
        self.task = _single_component(task, label="task")
        task_root = root / self.task
        if not task_root.is_dir():
            raise DatasetCatalogError(f"Task directory does not exist: {task_root}.")
        self.task_root = task_root

        discovered = self._discover_forms()
        if forms is not None:
            requested = tuple(_single_component(name, label="form") for name in forms)
            missing = [name for name in requested if name not in discovered]
            if missing:
                raise DatasetCatalogError(
                    f"Unknown form(s) {missing!r}; available: "
                    f"{sorted(discovered)!r}."
                )
            self._forms = tuple(discovered[name] for name in requested)
        else:
            self._forms = tuple(discovered[name] for name in sorted(discovered))

    def _discover_forms(self) -> dict[str, DatasetForm]:
        found: dict[str, DatasetForm] = {}
        for child in sorted(self.task_root.iterdir()):
            scenes_dir = child / SCENES_DIR_NAME
            if not child.is_dir() or not scenes_dir.is_dir():
                continue
            scene_count = sum(1 for entry in scenes_dir.iterdir() if entry.is_dir())
            found[child.name] = DatasetForm(
                name=child.name,
                relative_path=child.name,
                mode=_form_mode(child / FORM_META_NAME),
                scene_count=scene_count,
                has_samples=(child / SAMPLES_DIR_NAME).is_dir(),
            )
        if not found:
            raise DatasetCatalogError(
                f"No dataset forms with a {SCENES_DIR_NAME!r} directory under "
                f"{self.task_root}."
            )
        return found

    def forms(self) -> tuple[DatasetForm, ...]:
        return self._forms

    def form(self, name: str) -> DatasetForm:
        key = _single_component(name, label="form")
        for candidate in self._forms:
            if candidate.name == key:
                return candidate
        raise DatasetCatalogError(f"Unknown form {key!r}.")

    def form_dir(self, name: str) -> Path:
        form = self.form(name)
        path = (self.task_root / form.relative_path).resolve()
        if not path.is_relative_to(self.task_root):
            raise DatasetCatalogError(f"Form directory escapes the task root: {path}.")
        return path

    def scenes(self, name: str) -> tuple[str, ...]:
        scenes_dir = self.form_dir(name) / SCENES_DIR_NAME
        return tuple(
            sorted(entry.name for entry in scenes_dir.iterdir() if entry.is_dir())
        )

    def scene_path(self, form_name: str, scene_id: str) -> Path:
        key = _single_component(scene_id, label="scene")
        scenes_dir = self.form_dir(form_name) / SCENES_DIR_NAME
        path = (scenes_dir / key).resolve()
        if not path.is_relative_to(scenes_dir) or not path.is_dir():
            raise DatasetCatalogError(f"Unknown scene {key!r} in form {form_name!r}.")
        return path

    def revision(self, form_name: str, scene_id: str) -> str:
        scene_path = self.scene_path(form_name, scene_id)
        digest = hashlib.sha256()
        paths = [scene_path / SCENE_META_NAME, scene_path / SCENE_SCALARS_NAME,
                 self.form_dir(form_name) / FORM_META_NAME]
        paths.extend(sorted(scene_path.glob("*.npy")))
        for path in paths:
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()[:20]

    def court_contract(self, form_name: str) -> CourtKeypointContract:
        """Resolve the CourtKP20 contract recorded in the form's root metadata."""
        meta_path = self.form_dir(form_name) / FORM_META_NAME
        document = _load_json_object(meta_path)
        config = document.get("config")
        section = config.get("court_keypoints") if isinstance(config, dict) else None
        selector = section.get("selector") if isinstance(section, dict) else None
        if not isinstance(selector, str) or not selector:
            raise DatasetCatalogError(
                f"{meta_path}: config.court_keypoints.selector is required."
            )
        return resolve_court_keypoint_contract(selector)


__all__ = [
    "MULTI_MODE",
    "SAMPLES_DIR_NAME",
    "SCENES_DIR_NAME",
    "SINGLE_MODE",
    "DatasetCatalog",
    "DatasetCatalogError",
    "DatasetForm",
]
