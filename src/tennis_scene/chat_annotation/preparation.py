"""Download once, encode bounded overlapping clips, and publish verified packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from bisect import bisect_left, bisect_right
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from src.utils.video.youtube import download_youtube_video

from .configuration import PrepareConfig, youtube_id
from .kit import build_kit
from .runtime.contracts import (
    KIT_VERSION,
    ClipManifest,
    FrameMap,
    FrameRange,
    SourceInfo,
    read_json,
    sha256_file,
    write_json,
)
from .runtime.media import Timeline, check_clip, decode_range, encode_video, probe_video
from .sampling import centered_subrange, select_target_ranges


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _acquire(config: PrepareConfig) -> tuple[Path, SourceInfo]:
    if config.local_video is not None:
        digest = sha256_file(config.local_video)
        source_id = "local_" + digest[:16]
        directory = config.output / "sources" / source_id
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / (source_id + config.local_video.suffix)
        if not source.exists():
            temporary = source.with_name(source.name + ".partial")
            shutil.copyfile(config.local_video, temporary)
            temporary.replace(source)
        title, video_id = config.local_video.name, None
    else:
        assert config.url is not None
        video_id = youtube_id(config.url)
        source_id = video_id
        selection = _digest(
            [config.format_selector, config.js_runtimes, config.remote_components]
        )[:16]
        directory = config.output / "sources" / video_id / selection
        source = download_youtube_video(
            url=config.url,
            video_id=video_id,
            output_dir=directory / "original",
            format_selector=config.format_selector,
            write_info_json=True,
            no_playlist=True,
            js_runtimes=config.js_runtimes,
            remote_components=config.remote_components,
        )
        info = read_json(source.parent / f"{video_id}.info.json")
        if info.get("id") != video_id or not isinstance(info.get("title"), str):
            raise ValueError("yt-dlp metadata does not match requested video")
        title = info["title"]
        digest = sha256_file(source)
    record = directory / "source_record.json"
    if sha256_file(source) != digest:
        raise ValueError("stored original differs from source hash")
    if record.exists():
        metadata = SourceInfo.model_validate(read_json(record))
        if metadata.sha256 != digest or sha256_file(source) != digest:
            raise ValueError("cached source has changed; use a new output directory")
        return source, metadata
    metadata = SourceInfo(
        source_id=source_id,
        youtube_id=video_id,
        url=config.url,
        title=title,
        filename=source.name,
        sha256=digest,
        bytes=source.stat().st_size,
        acquired_at=datetime.now(UTC).isoformat(),
    )
    write_json(record, metadata.model_dump(mode="json"))
    return source, metadata


def media_range(timeline: Timeline, target: FrameRange, context: float) -> FrameRange:
    boundaries = [timeline.boundary(i) for i in range(len(timeline.pts) + 1)]
    margin = Fraction(str(context))
    return FrameRange(
        start=bisect_left(
            boundaries, max(Fraction(0), boundaries[target.start] - margin)
        ),
        stop=bisect_right(
            boundaries, min(boundaries[-1], boundaries[target.stop] + margin)
        )
        - 1,
    )


def _request(manifest: ClipManifest) -> str:
    template = (Path(__file__).parent / "resources" / "REQUEST.txt").read_text(
        encoding="utf-8"
    )
    return (
        template.replace("{{CLIP_ID}}", manifest.clip_id)
        .replace("{{VIDEO}}", manifest.filename)
        .replace("{{KIT_ID}}", manifest.kit_id)
        .replace("{{KIT_VERSION}}", manifest.kit_version)
    )


def _verify_published(directory: Path) -> ClipManifest:
    ready = read_json(directory / "ready.json")
    for name, digest in ready["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError(f"prepared clip is incomplete or changed: {directory}")
    manifest = cast(
        ClipManifest,
        ClipManifest.model_validate(read_json(directory / "clip_manifest.json")),
    )
    if set(ready["files"]) != {manifest.filename, "clip_manifest.json", "REQUEST.txt"}:
        raise ValueError("ready marker must cover video, manifest and request")
    if ready["files"].get(manifest.filename) != manifest.sha256:
        raise ValueError("ready marker does not identify the clip hash")
    return manifest


def _make_clip(
    config: PrepareConfig,
    source: Path,
    source_info: SourceInfo,
    timeline: Timeline,
    kit_id: str,
    clips_root: Path,
    target: FrameRange,
) -> list[Path]:
    clip_id = f"f{target.start:09d}-{target.stop:09d}"
    destination = clips_root / clip_id
    if destination.exists():
        manifest = _verify_published(destination)
        if (
            manifest.source.sha256 != source_info.sha256
            or manifest.kit_id != kit_id
            or manifest.target_range != target
        ):
            raise ValueError("existing clip identity differs from this preparation run")
        return [destination]
    media = media_range(timeline, target, config.context_seconds)
    duration = timeline.boundary(media.stop) - timeline.boundary(media.start)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=clips_root) as temporary:
        staging = Path(temporary)
        video = staging / f"{source_info.source_id}__{clip_id}.mp4"
        if duration <= Fraction(str(config.duration_seconds)):
            frames = (
                (
                    frame,
                    timeline.pts[index] - timeline.pts[media.start],
                    timeline.durations[index],
                )
                for index, frame in enumerate(
                    decode_range(source, timeline, media.start, media.stop),
                    start=media.start,
                )
            )
            encode_video(
                video,
                width=timeline.width,
                height=timeline.height,
                time_base=timeline.time_base,
                rate=timeline.rate,
                frames=frames,
                crf=config.crf,
                preset=config.preset,
            )
        if not video.exists() or video.stat().st_size > config.max_bytes:
            if target.stop - target.start < 2:
                raise ValueError(
                    "one target frame with its context exceeds the duration/byte limit; cannot split without dropping frames or context"
                )
            if config.max_clips_per_video is not None:
                # A sampling slot is one output file. Keep its temporal center
                # and reduce duration rather than exceeding the video's cap.
                smaller = centered_subrange(timeline, target)
                print(
                    f"  sampling slot [{target.start}, {target.stop}) shortened to [{smaller.start}, {smaller.stop}) to meet the duration/byte limit"
                )
                return _make_clip(
                    config,
                    source,
                    source_info,
                    timeline,
                    kit_id,
                    clips_root,
                    smaller,
                )
            middle = (target.start + target.stop) // 2
            return _make_clip(
                config,
                source,
                source_info,
                timeline,
                kit_id,
                clips_root,
                FrameRange(start=target.start, stop=middle),
            ) + _make_clip(
                config,
                source,
                source_info,
                timeline,
                kit_id,
                clips_root,
                FrameRange(start=middle, stop=target.stop),
            )
        manifest = ClipManifest(
            schema_version="tennis_chat_clip.v1",
            kit_version=KIT_VERSION,
            kit_id=kit_id,
            clip_id=clip_id,
            source=source_info,
            filename=video.name,
            sha256=sha256_file(video),
            bytes=video.stat().st_size,
            width=timeline.width,
            height=timeline.height,
            time_base=str(timeline.time_base),
            nominal_fps=str(timeline.rate),
            source_start_pts=timeline.pts[0],
            media_range=media,
            target_range=target,
            policies=config.policies,
            frames=[
                FrameMap(
                    frame_index=index - media.start,
                    source_frame_index=index,
                    source_pts=timeline.pts[index],
                    clip_pts=timeline.pts[index] - timeline.pts[media.start],
                    duration_pts=timeline.durations[index],
                    is_target=target.start <= index < target.stop,
                )
                for index in range(media.start, media.stop)
            ],
        )
        check_clip(video, manifest)
        write_json(staging / "clip_manifest.json", manifest.model_dump(mode="json"))
        (staging / "REQUEST.txt").write_text(_request(manifest), encoding="utf-8")
        write_json(
            staging / "ready.json",
            {
                "files": {
                    name: sha256_file(staging / name)
                    for name in (video.name, "clip_manifest.json", "REQUEST.txt")
                }
            },
        )
        staging.rename(destination)
    print(f"  {clip_id}: {manifest.bytes} bytes, {len(manifest.frames)} frames")
    return [destination]


def _covers_source(ranges: list[FrameRange], count: int) -> bool:
    cursor = 0
    for interval in ranges:
        if interval.start != cursor:
            return False
        cursor = interval.stop
    return cursor == count


def _verify_run(
    root: Path, source: SourceInfo, kit_id: str, settings: dict[str, Any]
) -> None:
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


def prepare(config: PrepareConfig) -> Path:
    if config.urls:
        raise ValueError("use prepare_batch for multiple source URLs")
    config.output.mkdir(parents=True, exist_ok=True)
    kit_directory, kit_id = build_kit(config.output / "project_kits")
    source, source_info = _acquire(config)
    return _prepare_acquired(config, source, source_info, kit_directory, kit_id)


def _prepare_acquired(
    config: PrepareConfig,
    source: Path,
    source_info: SourceInfo,
    kit_directory: Path,
    kit_id: str,
) -> Path:
    settings = {
        "duration_seconds": config.duration_seconds,
        "context_seconds": config.context_seconds,
        "max_bytes": config.max_bytes,
        "crf": config.crf,
        "preset": config.preset,
        "policies": config.policies.model_dump(mode="json"),
        "sampling": {
            "max_clips_per_video": config.max_clips_per_video,
            "strategy": config.sampling_strategy,
        },
    }
    run_id = _digest([source_info.sha256, kit_id, settings])[:16]
    root = Path(config.output) / "videos" / str(source_info.source_id) / run_id
    if (root / "prepared.json").exists():
        _verify_run(root, source_info, kit_id, settings)
        return root
    clips_root = root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)
    timeline = probe_video(source)
    outputs: list[Path] = []
    requested, candidate_count = select_target_ranges(
        timeline,
        config.duration_seconds,
        config.context_seconds,
        config.max_clips_per_video,
        config.sampling_strategy,
    )
    for target in requested:
        outputs.extend(
            _make_clip(
                config, source, source_info, timeline, kit_id, clips_root, target
            )
        )
    realized = [_verify_published(path).target_range for path in outputs]
    write_json(
        root / "prepared.json",
        {
            "schema_version": "tennis_chat_preparation.v2",
            "source_sha256": source_info.sha256,
            "kit_id": kit_id,
            "project_kit_directory": str(kit_directory.resolve()),
            "source_frame_count": len(timeline.pts),
            "candidate_clip_count": candidate_count,
            "requested_target_ranges": [target.model_dump() for target in requested],
            "selected_target_ranges": [target.model_dump() for target in realized],
            "selected_frame_count": sum(
                target.stop - target.start for target in realized
            ),
            "coverage_mode": "full"
            if _covers_source(realized, len(timeline.pts))
            else "sampled",
            "settings": settings,
            "clips": [path.name for path in outputs],
        },
    )
    _verify_run(root, source_info, kit_id, settings)
    return root
