"""Retire only the owned source cache after durable, fully verified JPEG preparation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef, ClipManifest, Document, SourceProvenance
from ..media import check_clip
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, load_plan
from ..storage import StorageCatalog
from ..storage.contracts import ManagedArtifact, Protection
from .contracts import AcquisitionReceipt, PreparationConfig, PreparationReceipt
from .ledger import CollectionLedger
from .source_handoff import handoff_screening_inputs


class FileIdentity(Document):
    device: int
    inode: int
    size: int = Field(ge=0)
    modified_ns: int
    changed_ns: int

    @classmethod
    def read(cls, path: Path) -> FileIdentity:
        value = path.lstat()
        if not path.is_file() or path.is_symlink() or value.st_nlink != 1:
            raise ValueError(
                "source retirement requires one owned regular file with no hardlinks"
            )
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


class RetirementIntent(Document):
    schema_version: Literal["annotation.source-retirement.v1"] = (
        "annotation.source-retirement.v1"
    )
    acquisition: ArtifactRef
    preparation: ArtifactRef
    basis: Literal["prepared", "excluded"] = "prepared"
    source: SourceProvenance
    cache_path: str
    file: FileIdentity


class RetirementReceipt(Document):
    intent: ArtifactRef
    status: Literal["deleted"] = "deleted"


def verify_prepared(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    source_id: str,
) -> tuple[ArtifactRef, ArtifactRef, AcquisitionReceipt]:
    """The registry proves source/PTS validation happened before deletion was allowed."""
    ledger = CollectionLedger(layout)
    row = ledger.source(plan, source_id)
    if row["state"] != "prepared" or row["preparation"] is None:
        raise ValueError(
            "source retirement requires committed preparation of all selected clips"
        )
    acquisition_ref = ArtifactRef.model_validate_json(row["acquisition"])
    prepared_ref = ArtifactRef.model_validate_json(row["preparation"])
    acquired = read_document(layout, acquisition_ref, AcquisitionReceipt)
    prepared = read_document(layout, prepared_ref, PreparationReceipt)
    if (
        prepared.source_id,
        prepared.source_digest,
        prepared.configuration,
        prepared.selection,
    ) != (
        source_id,
        acquired.source.media.digest,
        plan.preparation,
        ArtifactRef.model_validate_json(row["selection"]),
    ):
        raise ValueError("prepared source differs from the immutable registered inputs")
    with ledger.registry.transaction() as database:
        registered = {
            r["id"]: r["manifest"]
            for r in database.execute(
                "SELECT id,manifest FROM clips WHERE source_id=? AND campaign_id=?",
                (source_id, plan.campaign_id),
            )
        }
    if len(registered) != len(prepared.clips) or not registered:
        raise ValueError("source retirement requires the complete registered clip set")
    for ref in prepared.clips:
        manifest = read_document(layout, ref, ClipManifest)
        if (
            manifest.jpeg is None
            or manifest.source != SourceProvenance.from_source(acquired.source)
            or registered.get(manifest.clip_id) != ref.canonical_bytes().decode()
        ):
            raise ValueError(
                "source retirement requires the exact registered JPEG manifests"
            )
        check_clip(verify_reference(layout, manifest.media), manifest)
    return acquisition_ref, prepared_ref, acquired


def _paths(
    layout: AnnotationLayout, plan: CampaignPlan, source_id: str
) -> tuple[Path, Path]:
    directory = layout.path("sources", source_id, plan.preparation.digest)
    return directory / "retirement-intent.json", directory / "retired.json"


def verify_retirement_intent(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    source_id: str,
    acquisition: ArtifactRef,
    preparation: ArtifactRef,
    acquired: AcquisitionReceipt,
    *,
    excluded: bool = False,
) -> RetirementIntent:
    path, _ = _paths(layout, plan, source_id)
    if not path.is_file():
        raise ValueError("source missing without a verified retirement intent")
    intent: RetirementIntent = read_document(layout, annotation_reference(layout, path), RetirementIntent)
    if intent.basis != ("excluded" if excluded else "prepared"):
        raise ValueError("source retirement basis differs from the registered state")
    expected = layout.path("sources", "blobs", acquired.source.media.digest)
    if (intent.acquisition, intent.preparation, intent.source, intent.cache_path) != (
        acquisition,
        preparation,
        SourceProvenance.from_source(acquired.source),
        expected.relative_to(layout.root).as_posix(),
    ):
        raise ValueError(
            "source retirement intent differs from the registered preparation"
        )
    return intent


def verify_acquisition_media(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    source_id: str,
    reference: ArtifactRef,
    acquired: AcquisitionReceipt,
) -> None:
    """Verify live source bytes, or their explicitly authorized retirement evidence."""
    if layout.artifact_path(acquired.source.media).exists():
        verify_reference(layout, acquired.source.media)
        return
    config = read_document(layout, plan.preparation, PreparationConfig)
    ledger = CollectionLedger(layout)
    row = ledger.source(plan, source_id)
    with ledger.registry.transaction() as database:
        owner = database.execute(
            "SELECT s.*, c.digest AS plan_digest FROM source_content sc "
            "JOIN sources s ON s.id=sc.source_id "
            "LEFT JOIN campaigns c ON c.id=s.campaign_id WHERE sc.digest=?",
            (acquired.source.media.digest,),
        ).fetchone()
    if owner is not None and owner["id"] != source_id:
        _require_content_duplicate(row, owner["id"])
        if row["acquisition"] != reference.canonical_bytes().decode():
            raise ValueError("duplicate source differs from its registered acquisition")
        owner_ref = ArtifactRef.model_validate_json(owner["acquisition"])
        original = read_document(layout, owner_ref, AcquisitionReceipt)
        if original.source.media != acquired.source.media:
            raise ValueError("duplicate source does not reference the same owned blob")
        owner_plan = load_plan(
            layout.path("campaigns", owner["campaign_id"], "plan.json"),
            expected_digest=owner["plan_digest"],
        )
        verify_acquisition_media(layout, owner_plan, owner["id"], owner_ref, original)
        return
    if (
        row["state"] == "prepared"
        and config.retained_media == "jpeg_q90"
        and config.source_retention == "delete_after_verified_frames"
    ):
        acquisition, basis, verified = verify_prepared(layout, plan, source_id)
        excluded = False
    elif (
        row["state"] == "excluded"
        and config.excluded_source_retention == "delete_after_verified_exclusion"
    ):
        from .exclusion import verify_excluded

        acquisition, basis, verified, _kept = verify_excluded(layout, plan, source_id)
        excluded = True
    else:
        raise ValueError("missing source media has no authorized retirement")
    if acquisition != reference or verified != acquired:
        raise ValueError("retired source differs from the acquisition receipt")
    verify_retirement_intent(
        layout, plan, source_id, acquisition, basis, verified, excluded=excluded
    )


def _require_content_duplicate(row: dict[str, Any], owner_id: str) -> None:
    if (
        row["state"] != "excluded"
        or row["reason"] != f"duplicate content of {owner_id}"
        or row["selection"] is not None
        or row["preparation"] is not None
    ):
        raise ValueError("shared source acquisition is not a verified content duplicate")


def _verify_shared_acquisitions(
    layout: AnnotationLayout,
    database: sqlite3.Connection,
    source_id: str,
    media: ArtifactRef,
) -> None:
    owner = database.execute(
        "SELECT source_id FROM source_content WHERE digest=?", (media.digest,)
    ).fetchone()
    if owner is None or owner["source_id"] != source_id:
        raise ValueError("source retirement requires the registered content owner")
    for row in database.execute(
        "SELECT * FROM sources WHERE acquisition IS NOT NULL AND id!=?", (source_id,)
    ):
        reference = ArtifactRef.model_validate_json(row["acquisition"])
        acquired = read_document(layout, reference, AcquisitionReceipt)
        if layout.artifact_path(acquired.source.media) != layout.artifact_path(media):
            continue
        if acquired.source.media != media or acquired.source.canonical_id != row["id"]:
            raise ValueError("shared source acquisition differs from the owned blob")
        _require_content_duplicate(dict(row), source_id)


def retire_source(
    layout: AnnotationLayout, plan: CampaignPlan, source_id: str, *, excluded: bool = False
) -> ArtifactRef:
    """Caller owns the source lock. Never deletes an input outside our content cache."""
    evidence: tuple[ArtifactRef, ...] = ()
    if excluded:
        from .exclusion import verify_excluded

        acquisition, prepared, acquired, evidence = verify_excluded(layout, plan, source_id)
    else:
        acquisition, prepared, acquired = verify_prepared(layout, plan, source_id)
    media = acquired.source.media
    path = layout.artifact_path(media)
    expected = layout.path("sources", "blobs", media.digest)
    if path != expected:
        raise ValueError(
            "source retirement is restricted to the owned content-addressed cache"
        )
    intent_path, complete_path = _paths(layout, plan, source_id)
    catalog = StorageCatalog(layout)
    with artifact_lock(layout, "source-content-" + media.digest):
        if intent_path.exists():
            intent = verify_retirement_intent(
                layout, plan, source_id, acquisition, prepared, acquired, excluded=excluded
            )
        else:
            verify_reference(layout, media)
            intent = RetirementIntent(
                acquisition=acquisition,
                preparation=prepared,
                basis="excluded" if excluded else "prepared",
                source=SourceProvenance.from_source(acquired.source),
                cache_path=media.path,
                file=FileIdentity.read(path),
            )
            write_document(layout, intent_path, intent)
        intent_ref = annotation_reference(layout, intent_path)
        if complete_path.exists():
            receipt = read_document(
                layout, annotation_reference(layout, complete_path), RetirementReceipt
            )
            if receipt.intent != intent_ref:
                raise ValueError("retirement completion belongs to another intent")
            if path.exists():
                # A separate explicit restore may have recreated the original.
                verify_reference(layout, media)
            return annotation_reference(layout, complete_path)
        # Commit dependency transfer before unlink. If completion writing fails,
        # a database rollback must never resurrect pins on a deleted source.
        handoff_screening_inputs(layout, plan, source_id, media, prepared, excluded=excluded, evidence=evidence)
        if excluded:
            catalog.protect_once(Protection(
                protection_id="excluded-source-evidence-v1-" + intent_ref.digest,
                kind="active",
                reason="Preserve the completed exclusion, URL and all available screening images after source retirement",
                references=evidence,
            ))
        # Hold the registry writer lock through unlink, so a new explicit reader
        # protection cannot race the last dependency check.
        with catalog.registry.transaction() as database:
            _verify_shared_acquisitions(layout, database, source_id, media)
            for row in database.execute(
                "SELECT document FROM storage_protections WHERE released_reason IS NULL"
            ):
                record = Protection.model_validate_json(row[0])
                if any(layout.artifact_path(ref) == path for ref in record.references):
                    raise ValueError(
                        "source still has a live storage protection: "
                        + record.protection_id
                    )
            for row in database.execute("SELECT document FROM storage_artifacts"):
                record = ManagedArtifact.model_validate_json(row[0])
                if any(
                    layout.artifact_path(ref) == path
                    for ref in (record.reference, *record.dependencies)
                ):
                    raise ValueError("source is still retained by the storage catalog")
            if path.exists():
                verify_reference(layout, media)
                if FileIdentity.read(path) != intent.file:
                    raise ValueError("source changed after retirement was prepared")
                path.unlink()
                sync_directory(path.parent)
            # If unlink succeeded but the process died before this write, the
            # durable intent and revalidated JPEGs make completion recoverable.
            return write_document(
                layout, complete_path, RetirementReceipt(intent=intent_ref)
            )


def resume_prepared(
    layout: AnnotationLayout, plan: CampaignPlan, source_id: str
) -> ArtifactRef:
    acquisition, prepared, acquired = verify_prepared(layout, plan, source_id)
    if layout.artifact_path(acquired.source.media).exists():
        verify_reference(layout, acquired.source.media)
    else:
        verify_retirement_intent(
            layout, plan, source_id, acquisition, prepared, acquired
        )
    return prepared
