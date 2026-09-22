"""A concise shared request with only essential per-clip input information."""

from __future__ import annotations

import json
from pathlib import Path

from .layout import video_path
from .runtime.contracts import ClipManifest, read_json, sha256_file


def render_request(contents: dict[str, bytes], manifests: list[ClipManifest]) -> str:
    resources = Path(__file__).parent / "resources"
    inputs = [
        {
            "filename": manifest.filename,
            "width": manifest.width,
            "height": manifest.height,
            "frame_count": len(manifest.frames),
            "ball_max_gap_seconds": manifest.policies.ball_max_gap_seconds,
        }
        for manifest in manifests
    ]
    catalog = "\n".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for value in inputs
    )
    return "\n".join(
        (
            (resources / "REQUEST.txt").read_text(encoding="utf-8"),
            contents["PROTOCOL.md"].decode("utf-8"),
            "## 入力一覧\n\n各行は1本の動画の入力情報です。添付ファイル名と一致する行だけを使います。\n\n"
            + "```jsonl\n"
            + catalog
            + "\n```\n",
        )
    )


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
