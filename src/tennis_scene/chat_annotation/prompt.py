"""Target-specific requests whose video information comes from the attachment."""

from __future__ import annotations

import json
from pathlib import Path

from .layout import video_path
from .runtime.contracts import ClipManifest, read_json, sha256_file


def render_request(contents: dict[str, bytes], target: str = "ball_detection") -> str:
    if target not in {"ball_detection", "player_detection"}:
        raise ValueError(f"unsupported annotation target: {target}")
    request = contents[f"{target}_REQUEST.txt"].decode("utf-8")
    schema = contents[f"{target}_annotation.schema.json"].decode("utf-8")
    return f"{request}\n\n## JSON Schema\n\n```json\n{schema.rstrip()}\n```\n"


def write_request(
    root: Path, contents: dict[str, bytes], *, project_directory: Path | None = None
) -> None:
    """Verify local provenance before publishing the shared request."""
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
            raise ValueError("duplicate published video filename")
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
    directory = (
        project_directory if project_directory is not None else root / "project_kits"
    )
    for target in ("ball_detection", "player_detection"):
        value = render_request(contents, target)
        path = directory / target / "REQUEST.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding="utf-8") != value:
            temporary = path.with_suffix(".txt.partial")
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
    for legacy_name in ("PROJECT_INSTRUCTIONS.txt", "REQUEST.txt"):
        (directory / legacy_name).unlink(missing_ok=True)
