"""Verify retained preparation summaries and receipts without running preparation."""

from pathlib import Path
from typing import Any

from .contracts import ClipManifest, FrameRange, SourceInfo, read_json, sha256_file
from .layout import published_video_path


def _ready_path(directory: Path) -> Path:
    return directory.parent.parent / "ready" / f"{directory.name}.json"


def _output_root(directory: Path) -> Path:
    return directory.parents[4]


def _verify_published(directory: Path) -> ClipManifest:
    ready = read_json(_ready_path(directory))
    manifest: ClipManifest = ClipManifest.model_validate(
        read_json(directory / "clip_manifest.json")
    )
    expected = {manifest.filename, "clip_manifest.json"}
    if set(ready["files"]) != expected or {p.name for p in directory.iterdir()} != {
        "clip_manifest.json"
    }:
        raise ValueError("prepared metadata must cover only video and manifest")
    for name, digest in ready["files"].items():
        path = (
            directory / name
            if name == "clip_manifest.json"
            else published_video_path(_output_root(directory), manifest)
        )
        if (
            Path(name).name != name
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != digest
        ):
            raise ValueError(f"prepared clip is incomplete or changed: {directory}")
    if ready["files"][manifest.filename] != manifest.sha256:
        raise ValueError("ready marker does not identify the clip hash")
    return manifest


def _covers_source(ranges: list[FrameRange], count: int) -> bool:
    cursor = 0
    for interval in ranges:
        if interval.start != cursor:
            return False
        cursor = interval.stop
    return cursor == count


def verify_prepared_run(
    root: Path, source: SourceInfo, kit_id: str, settings: dict[str, Any]
) -> None:
    """Validate a complete preparation run, including each published clip receipt."""
    summary = read_json(root / "prepared.json")
    if (
        summary["source_sha256"] != source.sha256
        or summary["kit_id"] != kit_id
        or summary["settings"] != settings
    ):
        raise ValueError("prepared run identity mismatch")
    if summary["schema_version"] != "tennis_chat_preparation.v2":
        raise ValueError("unsupported preparation summary")
    requested = [
        FrameRange.model_validate(value) for value in summary["requested_target_ranges"]
    ]
    maximum = settings["sampling"]["max_clips_per_video"]
    expected_slots = (
        summary["candidate_clip_count"]
        if maximum is None
        else min(maximum, summary["candidate_clip_count"])
    )
    if len(requested) != expected_slots or len(set(summary["clips"])) != len(
        summary["clips"]
    ):
        raise ValueError("preparation summary has missing/duplicate sampling slots")
    if maximum is not None and len(summary["clips"]) != expected_slots:
        raise ValueError(
            "final clip count must equal the requested sampling slot count"
        )
    realized: list[FrameRange] = []
    for index, name in enumerate(summary["clips"]):
        if Path(name).name != name:
            raise ValueError("invalid clip name in preparation summary")
        manifest = _verify_published(root / "clips" / name)
        if manifest.source.sha256 != source.sha256 or manifest.kit_id != kit_id:
            raise ValueError("prepared clip source/kit identity mismatch")
        target = manifest.target_range
        if target.stop > summary["source_frame_count"] or (
            realized and target.start < realized[-1].stop
        ):
            raise ValueError("prepared target frames overlap or exceed the source")
        if maximum is not None and not (
            requested[index].start
            <= target.start
            < target.stop
            <= requested[index].stop
        ):
            raise ValueError("prepared target escaped its requested sampling slot")
        if manifest.bytes > settings["max_bytes"]:
            raise ValueError("prepared clip exceeds the byte limit")
        realized.append(target)
    full = _covers_source(realized, summary["source_frame_count"])
    if maximum is None and not full:
        raise ValueError("full extraction did not cover all source frames")
    if summary["coverage_mode"] != ("full" if full else "sampled"):
        raise ValueError("preparation coverage claim differs from actual target frames")
    if summary["selected_target_ranges"] != [
        target.model_dump() for target in realized
    ]:
        raise ValueError("selected target ranges differ from the prepared clips")
    if summary["selected_frame_count"] != sum(
        target.stop - target.start for target in realized
    ):
        raise ValueError("selected frame count differs from actual target ranges")
