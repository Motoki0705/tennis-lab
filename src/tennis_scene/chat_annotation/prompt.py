"""Self-contained request text with lossless, compact per-video frame metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .layout import video_path
from .runtime.contracts import ClipManifest, FrameMap, read_json, sha256_file


def compact_manifest(manifest: ClipManifest) -> dict[str, Any]:
    value: dict[str, Any] = manifest.model_dump(mode="json", exclude={"frames"})
    runs: list[list[int]] = []
    for frame in manifest.frames:
        if runs and runs[-1][3] == frame.duration_pts:
            runs[-1][1] += 1
        else:
            runs.append([frame.frame_index, 1, frame.source_pts, frame.duration_pts])
    value["frame_runs"] = runs
    return value


def expand_manifest(value: dict[str, Any]) -> ClipManifest:
    """Decode the documented frame-run representation without losing VFR timing."""
    expanded = dict(value)
    runs = expanded.pop("frame_runs")
    first_pts = runs[0][2]
    frames: list[dict[str, Any]] = []
    for start, count, source_pts, duration in runs:
        if count <= 0 or start != len(frames):
            raise ValueError("frame runs must be positive and contiguous from zero")
        for offset in range(count):
            index = start + offset
            source_index = expanded["media_range"]["start"] + index
            stamp = source_pts + offset * duration
            frames.append(
                FrameMap(
                    frame_index=index,
                    source_frame_index=source_index,
                    source_pts=stamp,
                    clip_pts=stamp - first_pts,
                    duration_pts=duration,
                    is_target=expanded["target_range"]["start"]
                    <= source_index
                    < expanded["target_range"]["stop"],
                ).model_dump()
            )
    expanded["frames"] = frames
    result: ClipManifest = ClipManifest.model_validate(expanded)
    return result


def render_request(contents: dict[str, bytes], manifests: list[ClipManifest]) -> str:
    resources = Path(__file__).parent / "resources"
    sections = [
        (resources / "REQUEST.txt").read_text(encoding="utf-8"),
        contents["PROTOCOL.md"].decode("utf-8"),
    ]
    for name in (
        "annotation.schema.json",
        "court_definition.json",
        "kit_manifest.json",
    ):
        sections.append(
            f"## {name}\n\n```json\n{contents[name].decode('utf-8').rstrip()}\n```\n"
        )
    catalog = [compact_manifest(manifest) for manifest in manifests]
    sections.append(
        "## 動画入力定義\n\n```json\n"
        + json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n```\n"
    )
    return "\n".join(sections)


def write_request(
    root: Path, contents: dict[str, bytes], *, project_directory: Path | None = None
) -> None:
    """Include every published video, so preparing another source keeps prior inputs."""
    manifests = []
    names: set[str] = set()
    expected_paths: set[Path] = set()
    current_id = json.loads(contents["kit_manifest.json"])["kit_id"]
    for path in sorted((root / "_preparation").glob("*/*/clips/*/clip_manifest.json")):
        if path.parent.name.startswith(".building-"):
            continue
        ready_path = path.parents[2] / "ready" / f"{path.parent.name}.json"
        if not ready_path.exists():
            raise ValueError(f"prepared video metadata is incomplete: {path}")
        ready = read_json(ready_path)
        manifest = ClipManifest.model_validate(read_json(path))
        video = video_path(root, manifest)
        if manifest.kit_id != current_id:
            raise ValueError(
                "existing videos use a different request version; use a new output directory"
            )
        if manifest.filename in names:
            raise ValueError("duplicate video filename in request catalog")
        if (
            ready["files"]
            != {
                "clip_manifest.json": sha256_file(path),
                manifest.filename: sha256_file(video),
            }
            or manifest.sha256 != ready["files"][manifest.filename]
        ):
            raise ValueError("published video metadata changed")
        names.add(manifest.filename)
        expected_paths.add(video)
        manifests.append(manifest)
    folders = list((root / "videos").iterdir()) if (root / "videos").exists() else []
    if any(not folder.is_dir() or folder.is_symlink() for folder in folders):
        raise ValueError("videos must contain source video directories only")
    videos = [path for folder in folders for path in folder.iterdir()]
    if any(
        not path.is_file() or path.is_symlink() or path.suffix != ".mp4"
        for path in videos
    ):
        raise ValueError(
            "source video directories must contain only MP4 files directly"
        )
    if set(videos) != expected_paths:
        raise ValueError("video catalog is incomplete or contains an unpublished video")
    value = render_request(contents, manifests)
    path = (
        project_directory if project_directory is not None else root / "project_kits"
    ) / "REQUEST.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != value:
        temporary = path.with_suffix(".txt.partial")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
