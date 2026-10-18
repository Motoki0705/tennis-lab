"""Acquire into one content-addressed source area; retain provider identity."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    sync_directory,
    verify_reference,
    write_document,
)
from ..contracts import ArtifactRef, SourceIdentity
from ..paths import AnnotationLayout
from ..plan import file_digest
from .contracts import AcquisitionReceipt, PreparationConfig, SourceCandidate


def stage_source(
    candidate: SourceCandidate,
    config: PreparationConfig,
    staging: Path,
    layout: AnnotationLayout,
) -> Path:
    """Acquire into caller-owned temporary storage and verify provider identity."""
    media: Path
    if candidate.local_media is not None:
        original = verify_reference(layout, candidate.local_media)
        media = staging / "media"
        shutil.copyfile(original, media)
        return media
    from src.utils.video.youtube import download_youtube_video

    media = download_youtube_video(
        url=candidate.origin_url,
        video_id=candidate.video_id,
        output_dir=staging,
        format_selector=config.format_selector,
        js_runtimes=config.js_runtimes,
        remote_components=config.remote_components,
    )
    info = json.loads((staging / f"{candidate.video_id}.info.json").read_bytes())
    if (
        info.get("id") != candidate.video_id
        or info.get("channel_id") != candidate.channel_id
    ):
        raise ValueError(
            "download metadata differs from canonical video/channel identity"
        )
    return media


def acquire_source(
    layout: AnnotationLayout,
    candidate: SourceCandidate,
    configuration: ArtifactRef,
    config: PreparationConfig,
) -> ArtifactRef:
    directory = layout.path("sources", candidate.source_id, configuration.digest)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="building-", dir=directory) as temporary:
        staging = Path(temporary)
        media = stage_source(candidate, config, staging, layout)
        digest = file_digest(media)
        if candidate.local_media is not None and digest != candidate.local_media.digest:
            raise ValueError("local source changed while copying the pinned input")
        blob = layout.path("sources", "blobs", digest)
        with artifact_lock(layout, "source-content-" + digest):
            blob.parent.mkdir(parents=True, exist_ok=True)
            if blob.exists():
                if (
                    blob.stat().st_size != media.stat().st_size
                    or file_digest(blob) != digest
                ):
                    raise ValueError("stored content-addressed source has changed")
            else:
                with media.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.rename(media, blob)
                sync_directory(blob.parent)
        identity = SourceIdentity(
            provider=candidate.provider,
            channel_id=candidate.channel_id,
            video_id=candidate.video_id,
            media=annotation_reference(layout, blob),
            origin_url=candidate.origin_url,
        )
        receipt = AcquisitionReceipt(
            candidate_digest=candidate.sha256,
            configuration=configuration,
            source=identity,
        )
        return write_document(layout, directory / "acquisition.json", receipt)
