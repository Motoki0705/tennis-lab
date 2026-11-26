"""Stage immutable outputs and switch or restore the adopted dataset pointers."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.multiview_scene.inputs import INPUT_NODE, RECIPE
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    scene_array_manifest,
)
from src.tennis_scene.pipeline.storage.clip_store import clip_store_lock
from src.tennis_scene.pipeline.storage.scene_index import (
    indexed_scene_path,
    resolve_indexed_scene,
)
from src.tennis_scene.pipeline.storage.scene_pointers import (
    read_pending_scene_update,
    replace_scene_pointers,
    stable_scene_pointers,
)
from src.tennis_scene.pipeline.video_artifact import validate_digest
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256
from src.utils.io import utc_now_iso

if TYPE_CHECKING:
    from src.tennis_scene.generate_dataset.manifest import DatasetClipRecord

POINTERS = ("scene.json", "annotation.json")
PUBLICATION = "multiview_scene_publication"
LEGACY_GENERATIONS = ("active", "before-multiview")
_RECEIPT_DIRECTORY = Path("updates") / "multiview_scene"
_RECOVERY = (
    "re-run '.venv/bin/python -m src.tennis_scene.multiview_scene publish' "
    "with the original arguments, or run "
    "'.venv/bin/python -m src.tennis_scene.multiview_scene restore' "
    "with the same --dataset, --progress, --output, --players-per-side and clip selection."
)


def _receipt_path(root: Path, revision: str) -> Path:
    return root / _RECEIPT_DIRECTORY / f"{validate_digest(revision)}.json"


def read_pointers(root: Path) -> dict[str, str]:
    return {name: (root / name).read_bytes().decode("utf-8") for name in POINTERS}


def open_legacy_store(root: Path, generation: str) -> StoreReader:
    """Read the chosen legacy index in place, including its immutable generations."""
    active = StoreReader(root / "scene.json")
    if generation == "active":
        if PUBLICATION in active.document or INPUT_NODE in active.references:
            raise ValueError(
                f"{root}: choose --legacy-generation before-multiview for the adopted multiview scene"
            )
        return active
    if generation != "before-multiview":
        raise ValueError(f"Unknown legacy generation: {generation}")

    revision = active.document.get(PUBLICATION)
    if (
        not isinstance(revision, str)
        or len(revision) != 64
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError(
            f"{root}: no adopted multiview publication with a legacy generation"
        )
    journal = _receipt_path(root, revision)
    if not journal.is_file():
        raise FileNotFoundError(f"Missing multiview publication receipt: {journal}")
    receipt = json.loads(journal.read_text(encoding="utf-8"))
    with stable_scene_pointers(root):
        adopted = read_pointers(root)
    if receipt["revision"] != revision or receipt["after"] != adopted:
        raise ValueError(
            f"Multiview publication receipt disagrees with the adopted pointers: {journal}"
        )

    before = receipt["before"]["scene.json"]
    marker = json.loads(receipt["before"]["annotation.json"])
    frozen = active.descriptor(INPUT_NODE)
    pins = frozen["identity"]["pins"]
    legacy = StoreReader(root / "scene.json", document=json.loads(before))
    if (
        pins["recipe"] != RECIPE
        or pins["legacy_index_sha256"] != sha256(before.encode("utf-8")).hexdigest()
        or legacy.source_sha256 != active.source_sha256
        or INPUT_NODE in legacy.references
        or PUBLICATION in legacy.document
        or marker.get("scene_index") != "scene.json"
        or marker.get("scene_result")
        != legacy.document["exports"]["scene"]["files"]["scene"]["path"]
    ):
        raise ValueError(
            f"Invalid legacy generation in multiview publication receipt: {journal}"
        )
    active.assert_unchanged()
    return legacy


def require_known_pointers(
    current: dict[str, str],
    before: dict[str, str],
    after: dict[str, str],
) -> None:
    if any(current[name] not in (before[name], after[name]) for name in POINTERS):
        raise ValueError("Dataset pointers changed independently of this publication")


def _validate_snapshot(root: Path, pointers: dict[str, str]) -> None:
    index = json.loads(pointers["scene.json"])
    marker = json.loads(pointers["annotation.json"])
    exported = resolve_indexed_scene(index, root)
    if (
        marker.get("scene_index") != "scene.json"
        or marker.get("scene_result") != exported.relative_to(root).as_posix()
    ):
        raise ValueError(
            f"Publication snapshot has an inconsistent scene and marker: {root}"
        )


def _copy_generation(source: Path, destination: Path) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent, prefix=".stage-") as temporary:
            staged = Path(temporary) / "copy"
            shutil.copytree(source, staged)
            staged.replace(destination)
    source_files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
    destination_files = {
        p.relative_to(destination) for p in destination.rglob("*") if p.is_file()
    }
    if source_files != destination_files or any(
        dual_sha256(source / path) != dual_sha256(destination / path)
        for path in source_files
    ):
        raise ValueError(f"Immutable staged generation differs: {destination}")


def _stage_revision(
    root: Path, candidate: StoreReader, before: dict[str, str], revision: str
) -> dict[str, str]:
    original = json.loads(before["scene.json"])
    marker = json.loads(before["annotation.json"])
    exported = indexed_scene_path(candidate.index_path)
    scene = load_scene_result(exported)
    if (
        scene.metadata["player_reconstruction_method"] != RECIPE
        or scene.metadata["status"] != "ok"
    ):
        raise ValueError("Dataset publication requires a completed multiview scene")
    if original["source_sha256"] != candidate.source_sha256:
        raise ValueError("Multiview scene belongs to a different source")
    pins = candidate.descriptor("multiview_scene_inputs")["identity"]["pins"]
    if (
        sha256(before["scene.json"].encode("utf-8")).hexdigest()
        != pins["legacy_index_sha256"]
    ):
        raise ValueError("The original camera and ball publication has changed")
    if original["artifacts"]["ball_triangulation"] != scene.metadata["reused_ball"]:
        raise ValueError("The candidate does not preserve the active ball reference")
    # The copy is prepared while the old scene is still adopted.
    for ref in candidate.references.values():
        path = Path(ref["path"]).parent
        _copy_generation(candidate.root / path, root / path)
    destination = Path("exports") / revision
    _copy_generation(exported.parent, root / destination)
    original["artifacts"].update(candidate.references)
    original["exports"]["scene"] = candidate.document["exports"]["scene"]
    original["revision"] += 1
    original[PUBLICATION] = revision
    config = Path("configs") / f"{document_digest(pins)}.yaml"
    write_json_atomic(root / config, pins)
    marker.update(
        scene_result=str(destination / "scene.npz"),
        pipeline_config=str(config),
        generator="src.tennis_scene.multiview_scene",
        generated_at=utc_now_iso(),
        scene_schema_version=scene.schema_version,
        result_status=scene.metadata["status"],
        arrays=scene_array_manifest(scene),
        validity_statistics=scene.metadata["validity_statistics"],
        publication_identity_sha256=document_digest(pins),
        **{PUBLICATION: revision},
    )
    return {
        name: json.dumps(doc, ensure_ascii=False, sort_keys=True)
        for name, doc in zip(POINTERS, (original, marker), strict=True)
    }


def change_publication(
    dataset: Path,
    output: Path,
    records: Sequence[DatasetClipRecord],
    *,
    restore: bool,
) -> None:
    selected: list[tuple[Path, StoreReader | None]] = []
    for record in records:
        root = (dataset / record.path / ANNOTATION_RELATIVE_DIR).resolve(strict=True)
        candidate = None
        if not restore:
            candidate = StoreReader(output / record.clip_id / "scene.json")
            indexed_scene_path(candidate.index_path)
            metadata = candidate.descriptor("scene_assembly")["payload"]["metadata"]
            if (
                metadata["player_reconstruction_method"] != RECIPE
                or metadata["status"] != "ok"
            ):
                raise ValueError(f"Multiview result is incomplete: {record.clip_id}")
        selected.append((root, candidate))

    for root, candidate in selected:
        with clip_store_lock(root):
            pending = read_pending_scene_update(root)
            if pending is not None and (
                pending.receipt.parent != root / _RECEIPT_DIRECTORY
                or pending.receipt.suffix != ".json"
            ):
                raise pending.error()
            current = read_pointers(root)
            if candidate is not None:
                revision = candidate.references["scene_assembly"]["artifact_id"]
            elif pending is not None:
                revision = pending.receipt.stem
            else:
                revisions = {
                    value
                    for text in current.values()
                    if (value := json.loads(text).get(PUBLICATION)) is not None
                }
                if len(revisions) > 1:
                    raise ValueError(
                        "Dataset pointers name different multiview revisions"
                    )
                if not revisions:
                    continue
                revision = revisions.pop()

            journal = _receipt_path(root, revision)
            if pending is not None and pending.receipt != journal:
                raise pending.error()
            if not journal.is_file():
                if pending is not None or candidate is None:
                    raise FileNotFoundError(
                        f"Missing immutable publication receipt: {journal}"
                    )
                _validate_snapshot(root, current)
                before = current
                after = _stage_revision(root, candidate, before, revision)
                _validate_snapshot(root, after)
                write_json_atomic(
                    journal, {"before": before, "after": after, "revision": revision}
                )
            else:
                receipt: dict[str, Any] = json.loads(
                    journal.read_text(encoding="utf-8")
                )
                if receipt["revision"] != revision:
                    raise ValueError(
                        "Multiview publication receipt names another revision"
                    )
                before, after = receipt["before"], receipt["after"]
                require_known_pointers(current, before, after)
                _validate_snapshot(root, before if restore else after)
            desired = before if restore else after
            replace_scene_pointers(
                root,
                scene=desired["scene.json"],
                annotation=desired["annotation.json"],
                receipt=journal,
                recovery=_RECOVERY,
            )
