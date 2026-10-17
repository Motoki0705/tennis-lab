"""Preserve whole legacy preparation runs and explicitly selected label evidence."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.annotation.migration.legacy_format.contracts import (
    BallAnnotation,
    ClipManifest,
    PlayerAnnotation,
    SourceInfo,
    SupportedAnnotation,
    annotation_clip_id,
    loads_json,
    parse_annotation,
)
from src.tasks.annotation.migration.legacy_format.layout import published_video_path
from src.tasks.annotation.migration.legacy_format.prepared import verify_prepared_run
from src.tasks.annotation.migration.legacy_format.raw_zip import (
    MAX_ZIP_BYTES,
    inspect_zip,
)
from src.tasks.annotation.migration.legacy_format.validation import validate_annotation
from src.tasks.base.annotation.contracts import Document, content_digest, relative_path
from src.tasks.base.annotation.storage.contracts import ArtifactKind
from src.tasks.base.annotation.storage.contracts import (
    after_validator as _after_validator,
)
from src.tasks.base.annotation.storage.migration import (
    Bundle,
    BundleFile,
    describe_file,
)
from src.tasks.base.annotation.storage.migration.files import checked_root, read_file

_Model = TypeVar("_Model")


class _AfterValidator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


after_validator = cast(_AfterValidator, _after_validator)


class LegacySource(Document):
    record: str
    video: str

    @after_validator
    def source_paths(self) -> Self:
        for path in (self.record, self.video):
            relative_path(path)
            if not path.startswith("sources/"):
                raise ValueError("legacy source must be selected below sources/")
        if self.record == self.video:
            raise ValueError("source record and video must be different files")
        return self


class LegacySelection(Document):
    prepared_runs: tuple[str, ...] = Field(min_length=1)
    sources: tuple[LegacySource, ...] = Field(min_length=1)
    annotations: tuple[str, ...] = ()
    raw_zips: tuple[str, ...] = ()
    max_clips: int = Field(ge=1)
    max_json_bytes: int = Field(ge=1)

    @after_validator
    def explicit_paths(self) -> Self:
        for paths in (self.prepared_runs, self.annotations, self.raw_zips):
            if len(set(paths)) != len(paths):
                raise ValueError("legacy selection paths must be unique")
            for path in paths:
                relative_path(path)
        for run in self.prepared_runs:
            if len(Path(run).parts) != 3 or Path(run).parts[0] != "_preparation":
                raise ValueError("select complete _preparation/source/run directories")
        for path in self.annotations:
            parts = Path(path).parts
            if (
                len(parts) != 4
                or parts[:2] != ("annotated", "processed")
                or parts[2] not in {"ball", "player"}
                or not path.endswith(".json")
            ):
                raise ValueError("select processed ball/player JSON paths explicitly")
        if any(not path.endswith(".zip") for path in self.raw_zips):
            raise ValueError("raw artifacts must be selected as whole ZIP files")
        return self


class LegacyReader:
    def __init__(self, selection: LegacySelection) -> None:
        self.selection = selection

    def _json(self, root: Path, path: str) -> dict[str, Any]:
        with read_file(root, path) as source:
            payload = source.read(self.selection.max_json_bytes + 1)
        if len(payload) > self.selection.max_json_bytes:
            raise ValueError("legacy JSON exceeds the explicit size limit")
        result = loads_json(payload)
        if not isinstance(result, dict):
            raise ValueError("legacy JSON must be an object")
        return result

    def inspect(self, root: Path) -> Bundle:
        checked_root(root)
        files: dict[str, BundleFile] = {}
        sources: dict[str, tuple[SourceInfo, LegacySource]] = {}
        manifests: dict[str, tuple[ClipManifest, str, str]] = {}
        semantics: list[object] = []

        def add(path: str, kind: ArtifactKind, *dependencies: str) -> BundleFile:
            item = describe_file(
                root, path, kind=kind, dependencies=tuple(sorted(set(dependencies)))
            )
            previous = files.setdefault(path, item)
            if previous != item:
                raise ValueError("legacy path has conflicting ownership or references")
            return item

        for selected in self.selection.sources:
            source = SourceInfo.model_validate(self._json(root, selected.record))
            media = add(selected.video, "source")
            if (
                source.source_id in sources
                or Path(selected.video).name != source.filename
                or media.digest != source.sha256
                or media.size_bytes != source.bytes
            ):
                raise ValueError(
                    "legacy original source identity/bytes are inconsistent"
                )
            add(selected.record, "configuration", selected.video)
            sources[source.source_id] = (source, selected)
            semantics.append(source.model_dump(mode="json"))

        used_sources: set[str] = set()
        for run in self.selection.prepared_runs:
            summary_path = run + "/prepared.json"
            summary = self._json(root, summary_path)
            names = summary.get("clips")
            if (
                not isinstance(names, list)
                or not names
                or any(
                    not isinstance(name, str) or Path(name).name != name
                    for name in names
                )
                or len(names) != len(set(names))
            ):
                raise ValueError("legacy preparation must enumerate unique clip names")
            if len(manifests) + len(names) > self.selection.max_clips:
                raise ValueError("legacy selection exceeds the explicit clip limit")
            actual = {
                path.name
                for path in (root / run / "clips").iterdir()
                if not path.name.startswith(".building-")
            }
            if actual != set(names):
                raise ValueError(
                    "legacy preparation contains unlisted or missing clips"
                )
            run_manifests: list[ClipManifest] = []
            dependencies: list[str] = []
            for name in names:
                manifest_path = f"{run}/clips/{name}/clip_manifest.json"
                manifest = ClipManifest.model_validate(self._json(root, manifest_path))
                clip_id = annotation_clip_id(manifest)
                source_pair = sources.get(manifest.source.source_id)
                if (
                    clip_id in manifests
                    or manifest.clip_id != name
                    or Path(run).parts[1] != manifest.source.source_id
                    or source_pair is None
                    or source_pair[0] != manifest.source
                ):
                    raise ValueError(
                        "legacy clip/source identity is duplicated or inconsistent"
                    )
                selected = source_pair[1]
                used_sources.add(manifest.source.source_id)
                video = (
                    published_video_path(root, manifest).relative_to(root).as_posix()
                )
                clip = add(video, "clip", selected.video)
                if clip.digest != manifest.sha256 or clip.size_bytes != manifest.bytes:
                    raise ValueError(
                        "legacy clip media hash/size differs from its manifest"
                    )
                add(manifest_path, "configuration", video, selected.record)
                ready_path = f"{run}/ready/{name}.json"
                self._json(root, ready_path)
                add(ready_path, "adoption_evidence", manifest_path, video)
                dependencies.extend((manifest_path, ready_path))
                manifests[clip_id] = (manifest, manifest_path, video)
                run_manifests.append(manifest)
                semantics.append(manifest.model_dump(mode="json"))
            # Reuse the preparation writer's complete-run checks, including receipts,
            # exact target coverage, sample slots, source bounds and byte ceilings.
            verify_prepared_run(
                root / run,
                run_manifests[0].source,
                run_manifests[0].kit_id,
                summary["settings"],
            )
            add(summary_path, "configuration", *dependencies)
            semantics.append(summary)
        if used_sources != set(sources):
            raise ValueError("legacy selection includes an unrelated original source")

        for path in self.selection.annotations:
            annotation = parse_annotation(self._json(root, path))
            expected_type = (
                BallAnnotation if Path(path).parts[2] == "ball" else PlayerAnnotation
            )
            if (
                not isinstance(annotation, expected_type)
                or annotation.clip_id != Path(path).stem
            ):
                raise ValueError(
                    "legacy processed annotation identity/schema differs from its path"
                )
            manifest, manifest_path, video = self._annotation(manifests, annotation)
            add(path, "adoption_evidence", manifest_path, video)
            semantics.append(annotation.model_dump(mode="json"))

        for path in self.selection.raw_zips:
            with read_file(root, path) as source:
                payload = source.read(MAX_ZIP_BYTES + 1)
            members = inspect_zip(payload)
            dependencies = []
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in members:
                    annotation = parse_annotation(loads_json(archive.read(member)))
                    _, manifest_path, video = self._annotation(manifests, annotation)
                    dependencies.extend((manifest_path, video))
                    semantics.append((member, annotation.model_dump(mode="json")))
            add(path, "adoption_evidence", *dependencies)
        return Bundle(
            reader="legacy-chat-preparation-v2",
            selection_digest=self.selection.sha256,
            semantic_digest=content_digest(semantics),
            files=tuple(files[path] for path in sorted(files)),
        )

    @staticmethod
    def _annotation(
        manifests: dict[str, tuple[ClipManifest, str, str]],
        annotation: SupportedAnnotation,
    ) -> tuple[ClipManifest, str, str]:
        selected = manifests.get(annotation.clip_id)
        if selected is None:
            raise ValueError(
                "annotation/ZIP references a clip outside the selected runs"
            )
        report = validate_annotation(annotation, selected[0])
        if report.errors:
            raise ValueError(
                "legacy annotation is invalid: " + "; ".join(report.errors)
            )
        return selected
