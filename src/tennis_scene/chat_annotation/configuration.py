"""Strict preparation configuration; defaults live exclusively in the YAML."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

from .runtime.contracts import Policies

PREPARE_SCHEMA = StrictConfigSchema(
    name="chat_annotation.prepare",
    fields={
        "paths": ConfigField.mapping(
            StrictConfigSchema(
                name="paths",
                fields={f"{role.value}_root": ConfigField.of(str) for role in PathRole},
            )
        ),
        "source": ConfigField.mapping(
            StrictConfigSchema(
                name="source",
                fields={
                    "url": ConfigField.of(str, type(None)),
                    "urls": ConfigField.sequence(ConfigField.of(str)),
                    "local_video": ConfigField.of(str, type(None)),
                    "format_selector": ConfigField.of(str),
                    "js_runtimes": ConfigField.of(str, type(None)),
                    "remote_components": ConfigField.of(str, type(None)),
                },
            )
        ),
        "batch": ConfigField.mapping(
            StrictConfigSchema(
                name="batch",
                fields={"download_workers": ConfigField.of(int)},
            )
        ),
        "sampling": ConfigField.mapping(
            StrictConfigSchema(
                name="sampling",
                fields={
                    "max_clips_per_video": ConfigField.of(int, type(None)),
                    "strategy": ConfigField.of(str),
                },
            )
        ),
        "output_directory": ConfigField.of(str),
        "clip": ConfigField.mapping(
            StrictConfigSchema(
                name="clip",
                fields={
                    "duration_seconds": ConfigField.of(int, float),
                    "context_seconds": ConfigField.of(int, float),
                    "max_bytes": ConfigField.of(int),
                },
            )
        ),
        "encoding": ConfigField.mapping(
            StrictConfigSchema(
                name="encoding",
                fields={"crf": ConfigField.of(int), "preset": ConfigField.of(str)},
            )
        ),
        "annotation": ConfigField.mapping(
            StrictConfigSchema(
                name="annotation",
                fields={
                    "static_tolerance_px_at_1080p": ConfigField.of(int, float),
                    "homography_max_error_px_at_1080p": ConfigField.of(int, float),
                    "ball_max_gap_seconds": ConfigField.of(int, float),
                },
            )
        ),
    },
)


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme != "https" or host not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }:
        raise ValueError("source.url must be an HTTPS YouTube video URL")
    parts = parsed.path.strip("/").split("/")
    if host in {"youtu.be", "www.youtu.be"} and len(parts) == 1:
        value = parts[0]
    elif parsed.path == "/watch":
        values = parse_qs(parsed.query).get("v", [])
        value = values[0] if len(values) == 1 else ""
    elif len(parts) == 2 and parts[0] in {"shorts", "live", "embed"}:
        value = parts[1]
    else:
        value = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError(
            "source.url must identify one video, not a channel or playlist"
        )
    return value


@dataclass(frozen=True)
class PrepareConfig:
    url: str | None
    urls: tuple[str, ...]
    local_video: Path | None
    format_selector: str
    js_runtimes: str | None
    remote_components: str | None
    output: Path
    duration_seconds: float
    context_seconds: float
    max_bytes: int
    crf: int
    preset: str
    policies: Policies
    download_workers: int
    max_clips_per_video: int | None
    sampling_strategy: str

    @classmethod
    def from_config(cls, cfg: DictConfig) -> PrepareConfig:
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError("preparation config must be a mapping")
        validated = PREPARE_SCHEMA.validate(cast(dict[str, object], raw))
        roots = RuntimePathRoots.from_mapping(
            cast(dict[str, object], validated["paths"]), repository_root=PROJECT_ROOT
        )
        resolver = PathResolver(roots)
        source = cast(dict[str, object], validated["source"])
        url = cast(str | None, source["url"])
        urls = cast(tuple[str, ...], source["urls"])
        local = cast(str | None, source["local_video"])
        if sum((url is not None, local is not None, bool(urls))) != 1:
            raise ValueError(
                "specify exactly one of source.url, source.urls and source.local_video"
            )
        if url is not None:
            youtube_id(url)
        video_ids = [youtube_id(item) for item in urls]
        if len(video_ids) != len(set(video_ids)):
            raise ValueError("source.urls contains duplicate YouTube video IDs")
        clip = cast(dict[str, object], validated["clip"])
        duration = float(cast(float, clip["duration_seconds"]))
        context = float(cast(float, clip["context_seconds"]))
        maximum = cast(int, clip["max_bytes"])
        if (
            not all(math.isfinite(value) for value in (duration, context))
            or not 0 <= context * 2 < duration
        ):
            raise ValueError(
                "require 0 <= 2 * context_seconds < duration_seconds, all finite"
            )
        if not 0 < maximum <= 500_000_000:
            raise ValueError("clip.max_bytes must be between 1 and 500000000")
        encoding = cast(dict[str, object], validated["encoding"])
        crf, preset = cast(int, encoding["crf"]), cast(str, encoding["preset"])
        if not 0 <= crf <= 51 or preset not in {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        }:
            raise ValueError("invalid libx264 CRF/preset")
        selector = cast(str, source["format_selector"])
        if not selector.strip():
            raise ValueError("source.format_selector cannot be empty")
        batch = cast(dict[str, object], validated["batch"])
        download_workers = cast(int, batch["download_workers"])
        if download_workers < 1:
            raise ValueError("batch.download_workers must be at least 1")
        sampling = cast(dict[str, object], validated["sampling"])
        max_clips = cast(int | None, sampling["max_clips_per_video"])
        if max_clips is not None and max_clips < 1:
            raise ValueError("sampling.max_clips_per_video must be positive or null")
        sampling_strategy = cast(str, sampling["strategy"])
        if sampling_strategy != "uniform_midpoints":
            raise ValueError("sampling.strategy must be uniform_midpoints")
        return cls(
            url=url,
            urls=urls,
            local_video=resolver.resolve(PathRole.DATA, local)
            if local is not None
            else None,
            format_selector=selector,
            js_runtimes=cast(str | None, source["js_runtimes"]),
            remote_components=cast(str | None, source["remote_components"]),
            output=resolver.resolve(
                PathRole.OUTPUT, cast(str, validated["output_directory"])
            ),
            duration_seconds=duration,
            context_seconds=context,
            max_bytes=maximum,
            crf=crf,
            preset=preset,
            policies=Policies.model_validate(
                dict(cast(dict[str, object], validated["annotation"]))
            ),
            download_workers=download_workers,
            max_clips_per_video=max_clips,
            sampling_strategy=sampling_strategy,
        )


def validate_prepare_config(cfg: DictConfig) -> None:
    PrepareConfig.from_config(cfg)
