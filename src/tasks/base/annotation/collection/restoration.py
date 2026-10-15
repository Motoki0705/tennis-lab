"""Explicit one-attempt restoration; JPEG processing never requires a download."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef, Document, Identifier, MediaFingerprint
from ..paths import AnnotationLayout
from ..plan import CampaignPlan, file_digest
from .acquisition import stage_source
from .contracts import PreparationConfig, SourceCandidate
from .ledger import CollectionLedger
from .retirement import verify_prepared, verify_retirement_intent


class RestorationReceipt(Document):
    acquisition: ArtifactRef
    attempt: Identifier
    status: Literal["restored", "unavailable", "different_content"]
    downloaded: MediaFingerprint | None
    reason: str


def restore_source(
    layout: AnnotationLayout, plan: CampaignPlan, source_id: str, attempt: str
) -> ArtifactRef:
    """Caller owns the source lock; failed attempts remain terminal until explicitly retried."""
    acquisition, preparation, acquired = verify_prepared(layout, plan, source_id)
    verify_retirement_intent(
        layout, plan, source_id, acquisition, preparation, acquired
    )
    path = layout.path(
        "sources", source_id, plan.preparation.digest, "restorations", attempt + ".json"
    )
    if path.exists():
        ref = annotation_reference(layout, path)
        record = read_document(layout, ref, RestorationReceipt)
        if record.acquisition != acquisition or record.attempt != attempt:
            raise ValueError("restoration receipt belongs to another source or attempt")
        if record.status == "restored":
            verify_reference(layout, acquired.source.media)
        return ref
    row = CollectionLedger(layout).source(plan, source_id)
    candidate = SourceCandidate.model_validate_json(row["candidate"])
    config = read_document(layout, plan.preparation, PreparationConfig)
    blob = layout.artifact_path(acquired.source.media)
    downloaded = None
    status: Literal["restored", "unavailable", "different_content"] = "restored"
    reason = "Exact original bytes restored; retained JPEGs and annotation identities unchanged."
    with artifact_lock(layout, "source-content-" + acquired.source.media.digest):
        if blob.exists():
            verify_reference(layout, acquired.source.media)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="restoring-", dir=path.parent
            ) as temporary:
                try:
                    media = stage_source(candidate, config, Path(temporary), layout)
                except (subprocess.CalledProcessError, FileNotFoundError) as error:
                    # A failed download is not proof of removal by YouTube. Record
                    # exactly what is known; do not loop or block the JPEG cycle.
                    status = "unavailable"
                    reason = f"Download did not complete ({type(error).__name__}); provider deletion is unconfirmed. No automatic retry."
                else:
                    downloaded = MediaFingerprint(
                        digest=file_digest(media), size_bytes=media.stat().st_size
                    )
                    if (downloaded.digest, downloaded.size_bytes) != (
                        acquired.source.media.digest,
                        acquired.source.media.size_bytes,
                    ):
                        status = "different_content"
                        reason = "URL returned different bytes; immutable source/JPEG provenance was not replaced."
                    else:
                        with media.open("rb") as handle:
                            os.fsync(handle.fileno())
                        os.rename(media, blob)
                        sync_directory(blob.parent)
    return write_document(
        layout,
        path,
        RestorationReceipt(
            acquisition=acquisition,
            attempt=attempt,
            status=status,
            downloaded=downloaded,
            reason=reason,
        ),
    )
