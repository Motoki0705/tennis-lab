"""Read proposal caches and Codex annotations without running inference."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.tasks.annotation.migration.legacy_format.contracts import (
    BallAnnotation,
    BallFrameAnnotation,
    ClipManifest,
    Digest,
    Point,
    annotation_clip_id,
    read_json,
    sha256_file,
)
from src.tasks.annotation.migration.legacy_format.layout import published_video_path
from src.tasks.annotation.migration.legacy_format.validation import validate_annotation


class Correction(StrEnum):
    UNREVIEWED = "unreviewed"
    AGREE = "agree"
    RERANK = "rerank"
    MOVED = "moved"
    ADDED = "added"
    REMOVED = "removed"
    BOTH_NONE = "both_none"


CHANGES = frozenset(
    (Correction.RERANK, Correction.MOVED, Correction.ADDED, Correction.REMOVED)
)
_PRIORITY = {
    kind: i
    for i, kind in enumerate(
        (
            Correction.BOTH_NONE,
            Correction.AGREE,
            Correction.RERANK,
            Correction.ADDED,
            Correction.MOVED,
            Correction.REMOVED,
        )
    )
}


class _CacheModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Candidate(_CacheModel):
    center_px: Point
    score: float = Field(ge=0, le=1)
    # Retained caches optionally include blob diagnostics; drawing uses peaks.
    blob: dict[str, Any] | None = None


class CandidateIdentity(_CacheModel):
    video_sha256: Digest
    checkpoint_sha256: Digest
    threshold: float = Field(ge=0, le=1)
    max_peaks: int = Field(gt=0)


class CandidateFrame(_CacheModel):
    candidates: list[Candidate]


class CandidateCache(_CacheModel):
    schema_name: str = Field(alias="schema")
    proposals_only: bool
    identity: CandidateIdentity
    frames: dict[str, CandidateFrame]
    updated_at: str | None = None


def classify_frame(
    candidates: Sequence[Candidate],
    frame_annotation: BallFrameAnnotation,
    radius_px: float,
) -> Correction:
    """Classify each track against score-ordered peaks, then take precedence.

    ``radius_px`` is already in source pixels; the reader scales the configured
    1920-pixel reference radius once. Null-position tracks participate as
    removed/both_none, including when another track has a position.
    """
    if isinstance(radius_px, bool) or not math.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("radius_px must be finite and positive")
    if not frame_annotation.reviewed:
        return Correction.UNREVIEWED
    if not frame_annotation.balls:
        return Correction.REMOVED if candidates else Correction.BOTH_NONE
    results: list[Correction] = []
    for ball in frame_annotation.balls:
        if ball.center_px is None:
            kind = Correction.REMOVED if candidates else Correction.BOTH_NONE
        elif not candidates:
            kind = Correction.ADDED
        elif math.dist(candidates[0].center_px, ball.center_px) <= radius_px:
            kind = Correction.AGREE
        elif any(
            math.dist(peak.center_px, ball.center_px) <= radius_px
            for peak in candidates[1:]
        ):
            kind = Correction.RERANK
        else:
            kind = Correction.MOVED
        results.append(kind)
    return max(results, key=_PRIORITY.__getitem__)


def read_candidate_cache(path: Path, manifest: ClipManifest) -> CandidateCache:
    """The replaceable boundary for raw proposals, with no sorting or filling."""
    cache: CandidateCache = CandidateCache.model_validate(read_json(path))
    if cache.schema_name != "campaign_ball_candidates.v1":
        raise ValueError(f"unsupported candidate schema: {cache.schema_name}")
    if cache.proposals_only is not True:
        raise ValueError("candidate cache requires proposals_only=true")
    if cache.identity.video_sha256 != manifest.sha256:
        raise ValueError("candidate video identity does not match manifest SHA-256")
    if set(cache.frames) != {str(i) for i in range(len(manifest.frames))}:
        raise ValueError("candidate frames must cover every manifest frame exactly")
    for index, frame in cache.frames.items():
        peaks = frame.candidates
        if len(peaks) > cache.identity.max_peaks:
            raise ValueError(f"frame {index}: candidates exceed max_peaks")
        if any(a.score < b.score for a, b in zip(peaks, peaks[1:], strict=False)):
            raise ValueError(
                f"frame {index}: candidates must be in descending score order"
            )
        if any(
            not (
                0 <= peak.center_px[0] < manifest.width
                and 0 <= peak.center_px[1] < manifest.height
            )
            for peak in peaks
        ):
            raise ValueError(f"frame {index}: candidate position outside image")
    return cache


@dataclass(frozen=True)
class PreparedClip:
    path: Path
    manifest: ClipManifest


def prepared_clips(annotation_root: Path) -> dict[str, PreparedClip]:
    """Index retained manifests by the global filename stem, rejecting duplicates."""
    result: dict[str, PreparedClip] = {}
    for path in sorted(
        (annotation_root / "_preparation").glob("*/*/clips/*/clip_manifest.json")
    ):
        if path.parent.name.startswith(".building-"):
            continue
        manifest = ClipManifest.model_validate(read_json(path))
        clip_id = annotation_clip_id(manifest)
        if clip_id in result:
            raise ValueError(f"duplicate prepared clip: {clip_id}")
        result[clip_id] = PreparedClip(path, manifest)
    if not result:
        raise ValueError(f"no prepared clip manifests under {annotation_root}")
    return result


@dataclass(frozen=True)
class AnnotationComparison:
    prepared: PreparedClip
    video: Path
    cache_path: Path
    cache: CandidateCache
    annotation_path: Path
    annotation: BallAnnotation
    radius_px: float
    corrections: tuple[Correction, ...]

    def input_manifest(self) -> dict[str, Any]:
        return {
            "candidate_cache": {
                "path": str(self.cache_path),
                "sha256": sha256_file(self.cache_path),
            },
            "annotation": {
                "path": str(self.annotation_path),
                "sha256": sha256_file(self.annotation_path),
            },
            "clip_manifest": {
                "path": str(self.prepared.path),
                "sha256": sha256_file(self.prepared.path),
            },
            "video": {
                "path": str(self.video),
                "sha256": self.prepared.manifest.sha256,
                "role": "original_media_identity",
            },
            "checkpoint_sha256": self.cache.identity.checkpoint_sha256,
            "candidate_identity": self.cache.identity.model_dump(),
        }


def load_comparison(
    annotation_root: Path,
    campaign_dir: Path,
    clip_id: str,
    *,
    radius_px: float,
    prepared: PreparedClip,
) -> AnnotationComparison:
    """Validate annotations and proposals; media decoding is a render-only step."""
    if clip_id != annotation_clip_id(prepared.manifest):
        raise ValueError("requested clip_id does not match prepared manifest")
    cache_path = campaign_dir / "cache/cands_ball" / f"{clip_id}.json"
    cache = read_candidate_cache(cache_path, prepared.manifest)
    annotation_path = annotation_root / "annotated/processed/ball" / f"{clip_id}.json"
    annotation = BallAnnotation.model_validate(read_json(annotation_path))
    report = validate_annotation(annotation, prepared.manifest)
    if report.errors:
        raise ValueError(f"{annotation_path}: {'; '.join(report.errors)}")
    effective_radius = radius_px * prepared.manifest.width / 1920
    corrections = tuple(
        classify_frame(cache.frames[str(i)].candidates, frame, effective_radius)
        for i, frame in enumerate(annotation.frames)
    )
    return AnnotationComparison(
        prepared,
        published_video_path(annotation_root, prepared.manifest),
        cache_path,
        cache,
        annotation_path,
        annotation,
        effective_radius,
        corrections,
    )


def class_counts(corrections: Sequence[Correction]) -> dict[str, int]:
    counts = Counter(corrections)
    return {kind.value: counts[kind] for kind in Correction}


def rank_clips(
    annotation_root: Path, campaign_dir: Path, *, radius_px: float
) -> list[dict[str, Any]]:
    """Rank all cache/processed pairs; validate JSON, without decoding videos."""
    index = prepared_clips(annotation_root)
    paired = sorted(
        path.stem
        for path in (campaign_dir / "cache/cands_ball").glob("*.json")
        if (annotation_root / "annotated/processed/ball" / path.name).is_file()
    )
    if not paired:
        raise ValueError("no clips with both candidate cache and processed annotation")
    rows: list[dict[str, Any]] = []
    for clip_id in paired:
        if clip_id not in index:
            raise ValueError(f"no manifest for paired clip: {clip_id}")
        clip = load_comparison(
            annotation_root,
            campaign_dir,
            clip_id,
            radius_px=radius_px,
            prepared=index[clip_id],
        )
        counts = class_counts(clip.corrections)
        changed = sum(counts[kind] for kind in CHANGES)
        rows.append(
            {
                "clip_id": clip_id,
                "frame_count": len(clip.corrections),
                "class_counts": counts,
                "has_all_changes": all(counts[kind] > 0 for kind in CHANGES),
                "changed_ratio": changed / len(clip.corrections),
                "radius_source_px": clip.radius_px,
                "inputs": clip.input_manifest(),
            }
        )
    rows.sort(
        key=lambda row: (
            not row["has_all_changes"],
            -row["changed_ratio"],
            row["clip_id"],
        )
    )
    return rows
