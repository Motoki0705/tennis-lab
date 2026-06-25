"""Build a tennis-scene image corpus for DINOv3 SSL from YouTube videos.

Usage:
    python -m src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images
    python -m src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images workflow.discovery.max_results_per_query=2
    python -m src.tasks.ball_detection.scripts.youtube.prepare_dinov3_ssl_images workflow.gate.backend=mock workflow.processing.max_new_videos=1

Notes:
    - Hydra loads configuration from `src/tasks/ball_detection/configs/prepare_dinov3_ssl_images.yaml`.
    - The intended production gate is Qwen/Qwen3.5-0.8B served by local vLLM.
    - `mock` gate is kept for pipeline dry-runs and tests when the local VLM runtime is unavailable.
"""

from __future__ import annotations

import base64
import json
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import av
import hydra
import requests
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from src.utils.io import (
    JSONDict,
    ensure_dirs,
    load_json_if_exists,
    read_jsonl,
    relative_path,
    save_json_atomic,
    utc_now_iso,
    write_jsonl,
)
from src.utils.video import (
    FramePacket,
    VideoInfo,
    sample_uniform_frame_indices,
)
from src.utils.video.youtube import download_youtube_video, transcode_h264_video

F = TypeVar("F", bound=Callable[..., Any])
TENNIS_TERMS = ("tennis", "atp", "wta", "grand slam", "rally", "court")
VIDEO_FILE_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@dataclass(frozen=True)
class GateDecision:
    """Video-level gate decision for sampled frames."""

    accepted: bool
    label: str
    confidence: float | None
    reason: str
    backend: str
    raw_response: str | None = None
    reasoning: str | None = None
    raw_payload: JSONDict | None = None


class FrameGate(Protocol):
    """Protocol for a video/frame-domain classifier."""

    def classify(
        self,
        *,
        source: Mapping[str, Any],
        sampled_frames: Sequence[Mapping[str, Any]],
        contact_sheet: Path,
    ) -> GateDecision:
        """Classify whether a sampled video belongs to the tennis domain."""


class MockTennisGate:
    """Deterministic gate for tests and dry-runs without a local VLM."""

    def __init__(self, *, accept_all: bool = False) -> None:
        self.accept_all = accept_all

    def classify(
        self,
        *,
        source: Mapping[str, Any],
        sampled_frames: Sequence[Mapping[str, Any]],
        contact_sheet: Path,
    ) -> GateDecision:
        text = " ".join(
            str(source.get(key, ""))
            for key in ("title", "query", "url", "source_url", "video_id")
        ).lower()
        accepted = self.accept_all or any(term in text for term in TENNIS_TERMS)
        return GateDecision(
            accepted=accepted,
            label="tennis" if accepted else "non_tennis",
            confidence=1.0 if accepted else 0.0,
            reason="mock gate matched tennis metadata"
            if accepted
            else "mock gate rejected metadata",
            backend="mock",
        )


class VllmGate:
    """Call a local vLLM OpenAI-compatible VLM endpoint."""

    def __init__(self, cfg: DictConfig) -> None:
        self.base_url = str(cfg.base_url).rstrip("/")
        self.model = str(cfg.model)
        self.prompt = str(cfg.prompt)
        self.timeout_sec = float(cfg.timeout_sec)
        self.max_tokens = int(cfg.get("max_tokens", 128))
        self.accept_labels = {str(value).lower() for value in cfg.accept_labels}
        self.extra_body = cast(
            JSONDict,
            OmegaConf.to_container(cfg.get("extra_body", {}), resolve=True),
        )

    def classify(
        self,
        *,
        source: Mapping[str, Any],
        sampled_frames: Sequence[Mapping[str, Any]],
        contact_sheet: Path,
    ) -> GateDecision:
        image_data = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
        request_body: JSONDict = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        request_body.update(self.extra_body)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=request_body,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        raw = str(message.get("content") or "").strip()
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        parsed = _parse_gate_output(raw)
        label = parsed["label"]
        accepted = label in self.accept_labels
        return GateDecision(
            accepted=accepted,
            label=label,
            confidence=cast(float | None, parsed["confidence"]),
            reason=str(parsed["reason"]),
            backend="vllm",
            raw_response=raw,
            reasoning=None if reasoning is None else str(reasoning),
            raw_payload=cast(JSONDict, payload),
        )


@hydra_main(
    config_path="../../configs",
    config_name="prepare_dinov3_ssl_images",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""
    run_pipeline(cfg)
    return 0


def run_pipeline(cfg: DictConfig) -> dict[str, int]:
    """Run the YouTube-to-DINOv3-SSL image pipeline."""
    workflow = cfg.workflow
    root = Path(to_absolute_path(str(workflow.root))).resolve()
    paths = workflow.paths
    videos_dir = root / str(paths.videos_dir)
    source_dir = videos_dir / str(paths.source_dir)
    h264_dir = videos_dir / str(paths.h264_dir)
    sampled_dir = root / str(paths.sampled_dir)
    images_dir = root / str(paths.images_dir)
    manifests_dir = root / str(paths.manifests_dir)
    ensure_dirs([source_dir, h264_dir, sampled_dir, images_dir, manifests_dir])

    sources = _collect_sources(workflow)
    existing_video_ids = _existing_video_ids(manifests_dir)
    gate = _build_gate(workflow.gate)

    source_records: list[JSONDict] = []
    sample_records: list[JSONDict] = []
    image_records: list[JSONDict] = []
    gate_records: list[JSONDict] = []
    processed_new = 0
    accepted_videos = 0
    failed_video_count = 0
    cleanup_video_file_count = 0

    for source in sources:
        video_id = str(source["video_id"])
        if video_id in existing_video_ids and not bool(
            workflow.processing.reprocess_existing
        ):
            print(f"[prepare_dinov3_ssl_images] skip existing video_id={video_id}")
            continue
        if processed_new >= int(workflow.processing.max_new_videos):
            break

        print(f"[prepare_dinov3_ssl_images] source={video_id}")
        processed_new += 1
        source_records.append(dict(source))
        local_video_supplied = bool(source.get("local_video"))
        try:
            source_video = _resolve_source_video(source, source_dir, workflow.download)
        except Exception as exc:
            failed_video_count += 1
            print(
                "  failed to resolve/download source video: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        sample_video = _transcode_source_video(
            source_video, video_id, h264_dir, workflow.transcode
        )
        info = _read_info_json(source_dir / f"{video_id}.info.json")
        source_with_info = {**source, "title": info.get("title") or source.get("title")}

        try:
            sampled_frames = _sample_video_frames(
                source=source_with_info,
                video_path=sample_video,
                output_dir=sampled_dir / video_id,
                root=root,
                cfg=workflow.frames,
            )
        except RuntimeError:
            if bool(workflow.transcode.enabled) or not bool(
                workflow.transcode.fallback_on_decode_error
            ):
                raise
            print("  source decode failed; falling back to H.264 transcode")
            sample_video = _force_transcode_source_video(
                source_video, video_id, h264_dir, workflow.transcode
            )
            sampled_frames = _sample_video_frames(
                source=source_with_info,
                video_path=sample_video,
                output_dir=sampled_dir / video_id,
                root=root,
                cfg=workflow.frames,
            )
        sample_records.extend(sampled_frames)
        contact_sheet = _write_contact_sheet(
            sampled_frames=sampled_frames,
            output_path=sampled_dir / video_id / "contact_sheet.jpg",
            root=root,
            cfg=workflow.gate.contact_sheet,
        )
        try:
            decision = gate.classify(
                source=source_with_info,
                sampled_frames=sampled_frames,
                contact_sheet=contact_sheet,
            )
        except Exception as exc:
            failed_video_count += 1
            decision = _failed_gate_decision(str(workflow.gate.backend), exc)
        gate_record = _gate_record(
            video_id, source_with_info, decision, root, contact_sheet
        )
        gate_records.append(gate_record)
        print(
            f"  gate: {decision.label} accepted={decision.accepted} reason={decision.reason}"
        )
        if decision.accepted:
            accepted_videos += 1
            image_records.extend(
                _promote_sampled_frames(
                    sampled_frames=sampled_frames,
                    images_dir=images_dir,
                    root=root,
                    decision=decision,
                    overwrite=bool(workflow.frames.overwrite),
                )
            )
        elif decision.label == "gate_error":
            print("  skip promotion because gate failed")
        cleanup_video_file_count += _cleanup_video_files(
            video_id=video_id,
            videos_dir=videos_dir,
            source_video=source_video,
            sample_video=sample_video,
            enabled=bool(workflow.processing.cleanup_videos_after_processing),
            keep_info_json=bool(workflow.processing.cleanup_keep_info_json),
            skip=local_video_supplied,
        )

    write_jsonl(manifests_dir / "sources.jsonl", source_records)
    write_jsonl(manifests_dir / "sampled_frames.jsonl", sample_records)
    write_jsonl(manifests_dir / "gate_decisions.jsonl", gate_records)
    write_jsonl(manifests_dir / "images.jsonl", image_records)
    summary = {
        "schema_name": "dinov3_ssl_youtube_images_summary_v1",
        "source_count": len(source_records),
        "sampled_frame_count": len(sample_records),
        "accepted_video_count": accepted_videos,
        "failed_video_count": failed_video_count,
        "image_count": len(image_records),
        "cleanup_video_file_count": cleanup_video_file_count,
        "images_dir": relative_path(images_dir, root),
        "written_at": utc_now_iso(),
    }
    save_json_atomic(summary, manifests_dir / "summary.json")
    return {
        "source_count": len(source_records),
        "sampled_frame_count": len(sample_records),
        "accepted_video_count": accepted_videos,
        "failed_video_count": failed_video_count,
        "image_count": len(image_records),
        "cleanup_video_file_count": cleanup_video_file_count,
    }


def _collect_sources(workflow: DictConfig) -> list[JSONDict]:
    manual_sources = _manual_sources(workflow.get("sources", []))
    discovered_sources = _discover_sources(workflow.discovery)
    deduped: dict[str, JSONDict] = {}
    for source in [*manual_sources, *discovered_sources]:
        video_id = str(source["video_id"])
        if video_id not in deduped:
            deduped[video_id] = source
    return list(deduped.values())


def _manual_sources(raw_sources: Iterable[Any]) -> list[JSONDict]:
    sources: list[JSONDict] = []
    for raw_source in raw_sources:
        source = cast(JSONDict, OmegaConf.to_container(raw_source, resolve=True))
        url = str(source.get("url") or "")
        video_id = str(source.get("video_id") or source.get("source_id") or "")
        if not video_id:
            video_id = _video_id_from_url(url)
        if not video_id:
            raise ValueError(
                f"Manual source must define video_id/source_id or url: {source}"
            )
        sources.append({**source, "video_id": video_id, "url": url})
    return sources


def _discover_sources(cfg: DictConfig) -> list[JSONDict]:
    if not bool(cfg.enabled):
        return []
    import yt_dlp  # type: ignore[import-untyped]

    sources: list[JSONDict] = []
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "ignoreerrors": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in cfg.queries:
            result = ydl.extract_info(
                f"ytsearch{int(cfg.max_results_per_query)}:{query}",
                download=False,
            )
            entries = [] if result is None else result.get("entries", [])
            for entry in entries:
                if not entry:
                    continue
                video_id = str(entry.get("id") or "")
                if not video_id:
                    continue
                duration_sec = _duration_seconds(entry.get("duration"))
                if not _duration_allowed(duration_sec, cfg):
                    print(
                        "  skip discovery result by duration: "
                        f"video_id={video_id} duration_sec={duration_sec}"
                    )
                    continue
                url = str(
                    entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
                )
                if not url.startswith("http"):
                    url = f"https://www.youtube.com/watch?v={video_id}"
                sources.append(
                    {
                        "video_id": video_id,
                        "url": url,
                        "title": entry.get("title"),
                        "duration_sec": duration_sec,
                        "query": str(query),
                        "source": "ytsearch",
                    }
                )
    return sources


def _existing_video_ids(manifests_dir: Path) -> set[str]:
    video_ids: set[str] = set()
    for manifest_name in (
        "images.jsonl",
        "gate_decisions.jsonl",
        "sampled_frames.jsonl",
    ):
        for record in read_jsonl(manifests_dir / manifest_name):
            if record.get("video_id") is not None:
                video_ids.add(str(record["video_id"]))
    return video_ids


def _resolve_source_video(
    source: Mapping[str, Any], source_dir: Path, cfg: DictConfig
) -> Path:
    local_video = source.get("local_video")
    if local_video:
        return Path(to_absolute_path(str(local_video))).resolve()
    archive = None
    if cfg.get("download_archive") is not None:
        archive = Path(to_absolute_path(str(cfg.download_archive))).resolve()
    return cast(
        Path,
        download_youtube_video(
            url=str(source["url"]),
            video_id=str(source["video_id"]),
            output_dir=source_dir,
            format_selector=str(cfg.format),
            merge_output_format=str(cfg.merge_output_format),
            enabled=bool(cfg.enabled),
            overwrite=bool(cfg.overwrite),
            js_runtimes=None
            if cfg.get("js_runtimes") is None
            else str(cfg.js_runtimes),
            remote_components=(
                None
                if cfg.get("remote_components") is None
                else str(cfg.remote_components)
            ),
            download_archive=archive,
            extra_args=[str(value) for value in cfg.get("extra_args", [])],
        ),
    )


def _transcode_source_video(
    source_video: Path,
    video_id: str,
    h264_dir: Path,
    cfg: DictConfig,
) -> Path:
    if not bool(cfg.enabled):
        print(f"  transcode disabled; sampling source video directly: {source_video}")
        return source_video
    return cast(
        Path,
        transcode_h264_video(
            source_video=source_video,
            output_path=h264_dir / f"{video_id}.mp4",
            enabled=bool(cfg.enabled),
            overwrite=bool(cfg.overwrite),
            ffmpeg_binary=str(cfg.ffmpeg_binary),
            encoder=str(cfg.encoder),
            hwaccel=None if cfg.get("hwaccel") is None else str(cfg.hwaccel),
            hwaccel_output_format=(
                None
                if cfg.get("hwaccel_output_format") is None
                else str(cfg.hwaccel_output_format)
            ),
            preset=str(cfg.preset),
            tune=None if cfg.get("tune") is None else str(cfg.tune),
            rate_control=None
            if cfg.get("rate_control") is None
            else str(cfg.rate_control),
            cq=None if cfg.get("cq") is None else cfg.cq,
            bitrate=None if cfg.get("bitrate") is None else str(cfg.bitrate),
            maxrate=None if cfg.get("maxrate") is None else str(cfg.maxrate),
            bufsize=None if cfg.get("bufsize") is None else str(cfg.bufsize),
            profile=None if cfg.get("profile") is None else str(cfg.profile),
            pix_fmt=str(cfg.pix_fmt),
            crf=cfg.crf,
        ),
    )


def _force_transcode_source_video(
    source_video: Path,
    video_id: str,
    h264_dir: Path,
    cfg: DictConfig,
) -> Path:
    return cast(
        Path,
        transcode_h264_video(
            source_video=source_video,
            output_path=h264_dir / f"{video_id}.mp4",
            enabled=True,
            overwrite=bool(cfg.overwrite),
            ffmpeg_binary=str(cfg.ffmpeg_binary),
            encoder=str(cfg.encoder),
            hwaccel=None if cfg.get("hwaccel") is None else str(cfg.hwaccel),
            hwaccel_output_format=(
                None
                if cfg.get("hwaccel_output_format") is None
                else str(cfg.hwaccel_output_format)
            ),
            preset=str(cfg.preset),
            tune=None if cfg.get("tune") is None else str(cfg.tune),
            rate_control=None
            if cfg.get("rate_control") is None
            else str(cfg.rate_control),
            cq=None if cfg.get("cq") is None else cfg.cq,
            bitrate=None if cfg.get("bitrate") is None else str(cfg.bitrate),
            maxrate=None if cfg.get("maxrate") is None else str(cfg.maxrate),
            bufsize=None if cfg.get("bufsize") is None else str(cfg.bufsize),
            profile=None if cfg.get("profile") is None else str(cfg.profile),
            pix_fmt=str(cfg.pix_fmt),
            crf=cfg.crf,
        ),
    )


def _probe_video_info(video_path: Path) -> VideoInfo:
    container = av.open(str(video_path))
    try:
        stream = _video_stream(container)
        fps = float(stream.average_rate or stream.base_rate or 0.0)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and stream.duration is not None and stream.time_base:
            frame_count = int(float(stream.duration * stream.time_base) * fps)
        if frame_count <= 0 and container.duration is not None:
            frame_count = int((float(container.duration) / 1_000_000.0) * fps)
        return VideoInfo(
            fps=fps,
            width=int(stream.width),
            height=int(stream.height),
            frame_count=frame_count,
        )
    finally:
        container.close()


def _iter_selected_video_frames(
    video_path: Path, frame_indices: Sequence[int]
) -> Iterator[FramePacket[Any]]:
    targets = sorted(set(frame_indices))
    if not targets:
        return
    container = av.open(str(video_path))
    try:
        stream = _video_stream(container)
        target_offset = 0
        for frame_index, frame in enumerate(container.decode(stream)):
            while target_offset < len(targets) and targets[target_offset] < frame_index:
                target_offset += 1
            if target_offset >= len(targets):
                break
            if frame_index != targets[target_offset]:
                continue
            frame_rgb = frame.to_ndarray(format="rgb24")
            yield FramePacket(
                index=frame_index,
                frame=frame_rgb,
                original_size=(int(frame.width), int(frame.height)),
            )
            target_offset += 1
            if target_offset >= len(targets):
                break
    finally:
        container.close()


def _video_stream(container: Any) -> Any:
    for stream in container.streams:
        if stream.type == "video":
            return stream
    raise RuntimeError("No video stream found.")


def _sample_video_frames(
    *,
    source: Mapping[str, Any],
    video_path: Path,
    output_dir: Path,
    root: Path,
    cfg: DictConfig,
) -> list[JSONDict]:
    manifest_path = output_dir / "frames.jsonl"
    if manifest_path.exists() and not bool(cfg.overwrite):
        existing_records = read_jsonl(manifest_path)
        if existing_records:
            print(f"  sampled frames exist: {len(existing_records)} -> {output_dir}")
            return cast(list[JSONDict], existing_records)

    ensure_dirs([output_dir])
    info = _probe_video_info(video_path)
    if info.frame_count <= 0:
        raise RuntimeError(f"PyAV reported no frames for {video_path}")
    frame_indices = sample_uniform_frame_indices(
        info.frame_count, int(cfg.frames_per_video)
    )
    records: list[JSONDict] = []
    for packet in _iter_selected_video_frames(video_path, frame_indices):
        frame_index = packet.index
        image_id = f"{source['video_id']}_f{frame_index:08d}"
        output_path = output_dir / f"{image_id}.{cfg.output_ext}"
        if not output_path.exists() or bool(cfg.overwrite):
            Image.fromarray(packet.frame).save(
                output_path,
                quality=int(cfg.jpeg_quality),
            )
        records.append(
            {
                "image_id": image_id,
                "image_path": relative_path(output_path, root),
                "video_id": source["video_id"],
                "source_url": source.get("url") or source.get("source_url"),
                "source_title": source.get("title"),
                "source_frame_index": frame_index,
                "timestamp_sec": frame_index / info.fps if info.fps > 0 else None,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "sampled_at": utc_now_iso(),
            }
        )
    if len(records) != len(frame_indices):
        raise RuntimeError(
            f"Decoded {len(records)}/{len(frame_indices)} sampled frames from {video_path}"
        )
    write_jsonl(manifest_path, records)
    print(f"  sampled frames: {len(records)} -> {output_dir}")
    return records


def _write_contact_sheet(
    *,
    sampled_frames: Sequence[Mapping[str, Any]],
    output_path: Path,
    root: Path,
    cfg: DictConfig,
) -> Path:
    selected = list(sampled_frames)[
        :: max(1, len(sampled_frames) // int(cfg.max_images))
    ]
    selected = selected[: int(cfg.max_images)]
    thumbs = []
    for record in selected:
        image_path = root / str(record["image_path"])
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            thumbs.append(
                image.convert("RGB").resize(
                    (int(cfg.thumb_width), int(cfg.thumb_height))
                )
            )
    if not thumbs:
        raise RuntimeError("No sampled frames available for contact sheet.")
    cols = int(cfg.columns)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new(
        "RGB",
        (cols * int(cfg.thumb_width), rows * int(cfg.thumb_height)),
    )
    for index, thumb in enumerate(thumbs):
        col = index % cols
        row = index // cols
        sheet.paste(thumb, (col * int(cfg.thumb_width), row * int(cfg.thumb_height)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
    return output_path


def _promote_sampled_frames(
    *,
    sampled_frames: Sequence[Mapping[str, Any]],
    images_dir: Path,
    root: Path,
    decision: GateDecision,
    overwrite: bool,
) -> list[JSONDict]:
    records: list[JSONDict] = []
    for frame in sampled_frames:
        source_path = root / str(frame["image_path"])
        destination = images_dir / source_path.name
        if overwrite or not destination.exists():
            shutil.copy2(source_path, destination)
        records.append(
            {
                **dict(frame),
                "image_path": relative_path(destination, root),
                "gate_label": decision.label,
                "gate_backend": decision.backend,
                "accepted_at": utc_now_iso(),
            }
        )
    return records


def _cleanup_video_files(
    *,
    video_id: str,
    videos_dir: Path,
    source_video: Path,
    sample_video: Path,
    enabled: bool,
    keep_info_json: bool,
    skip: bool,
) -> int:
    """Delete processed downloaded video files while preserving manifests."""
    if not enabled or skip:
        return 0

    videos_root = videos_dir.resolve()
    candidates = {source_video, sample_video}
    candidates.update(videos_dir.glob(f"**/{video_id}*"))

    deleted = 0
    for path in sorted(candidates, key=str):
        if not path.is_file():
            continue
        if keep_info_json and path.name == f"{video_id}.info.json":
            continue
        if path.suffix.lower() not in VIDEO_FILE_SUFFIXES:
            continue
        if not _is_relative_to(path.resolve(), videos_root):
            continue
        path.unlink()
        deleted += 1
        print(f"  cleanup video: {path}")
    return deleted


def _build_gate(cfg: DictConfig) -> FrameGate:
    backend = str(cfg.backend)
    if backend == "mock":
        return MockTennisGate(accept_all=bool(cfg.mock.accept_all))
    if backend == "vllm":
        return VllmGate(cfg.vllm)
    raise ValueError(f"Unsupported gate.backend={backend!r}.")


def _failed_gate_decision(backend: str, exc: Exception) -> GateDecision:
    return GateDecision(
        accepted=False,
        label="gate_error",
        confidence=None,
        reason=f"{type(exc).__name__}: {exc}",
        backend=backend,
    )


def _gate_record(
    video_id: str,
    source: Mapping[str, Any],
    decision: GateDecision,
    root: Path,
    contact_sheet: Path,
) -> JSONDict:
    return {
        "video_id": video_id,
        "source_url": source.get("url") or source.get("source_url"),
        "source_title": source.get("title"),
        "accepted": decision.accepted,
        "label": decision.label,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "backend": decision.backend,
        "raw_response": decision.raw_response,
        "reasoning": decision.reasoning,
        "raw_payload": decision.raw_payload,
        "contact_sheet": relative_path(contact_sheet, root),
        "processed_at": utc_now_iso(),
    }


def _read_info_json(path: Path) -> JSONDict:
    return cast(JSONDict, load_json_if_exists(path, {}))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _parse_gate_label(text: str) -> str:
    normalized = text.strip().lower()
    if "non_tennis" in normalized or "not tennis" in normalized:
        return "non_tennis"
    if "tennis" in normalized:
        return "tennis"
    if "unknown" in normalized or "uncertain" in normalized:
        return "unknown"
    return normalized.split()[0] if normalized.split() else "unknown"


def _parse_gate_output(text: str) -> JSONDict:
    payload = _parse_json_object(text)
    if payload is not None:
        label = _parse_gate_label(str(payload.get("label", "")))
        confidence = payload.get("confidence")
        reason = payload.get("reason")
        return {
            "label": label,
            "confidence": float(confidence)
            if isinstance(confidence, int | float)
            else None,
            "reason": str(reason)
            if reason is not None
            else f"vLLM gate returned label={label!r}",
        }
    label = _parse_gate_label(text)
    return {
        "label": label,
        "confidence": None,
        "reason": f"vLLM gate returned label={label!r}",
    }


def _parse_json_object(text: str) -> JSONDict | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "```" in stripped:
        fenced = stripped.split("```")
        candidates.extend(part.strip() for part in fenced if part.strip())
    if "{" in stripped and "}" in stripped:
        candidates.append(stripped[stripped.find("{") : stripped.rfind("}") + 1])
    for candidate in candidates:
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(JSONDict, parsed)
    return None


def _duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value)
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
        return seconds
    except ValueError:
        return None


def _duration_allowed(duration_sec: float | None, cfg: DictConfig) -> bool:
    if duration_sec is None:
        return bool(cfg.allow_unknown_duration)
    min_duration = cfg.get("min_duration_sec")
    max_duration = cfg.get("max_duration_sec")
    if min_duration is not None and duration_sec < float(min_duration):
        return False
    return not (max_duration is not None and duration_sec > float(max_duration))


def _video_id_from_url(url: str) -> str:
    if "watch?v=" in url:
        return url.split("watch?v=", 1)[1].split("&", 1)[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/", 1)[1].split("?", 1)[0]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())  # type: ignore[call-arg]
