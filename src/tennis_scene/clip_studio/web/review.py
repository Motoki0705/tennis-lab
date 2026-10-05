"""Read-only raw/project/export correspondence; never repairs dataset records."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.tennis_scene.clip_studio.export import (
    ExportSettings,
    build_clip_manifest,
    plan_clip_export,
)
from src.tennis_scene.clip_studio.project import Clip, ClipStudioProject
from src.tennis_scene.clip_studio.timeline import source_frame_index
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifest,
    load_dataset_manifest,
)
from src.utils.io import load_json
from src.utils.video import VideoInfo


def frame_correspondence(
    project: ClipStudioProject, infos: Sequence[VideoInfo], time: float
) -> dict[str, Any]:
    """Use the same nearest-frame mapping as the paused image API (0 based)."""
    if not math.isfinite(time):
        raise ValueError("共通時刻は有限値で指定してください。")
    cameras = []
    for source, info in zip(project.sources, infos, strict=True):
        index = source_frame_index(
            time,
            offset_sec=source.offset_sec,
            fps=info.fps,
            frame_count=info.frame_count,
        )
        cameras.append(
            {
                "camera_id": source.camera_id,
                "offset_sec": source.offset_sec,
                "local_time_sec": time + source.offset_sec,
                "frame_index": index,
                "frame_time_sec": None if index is None else index / info.fps,
                "coverage_sec": (
                    -source.offset_sec,
                    info.frame_count / info.fps - source.offset_sec,
                ),
                "available": index is not None,
            }
        )
    return {
        "global_time_sec": time,
        "cameras": cameras,
        "containing_clips": [
            clip.name for clip in project.clips if clip.contains(time)
        ],
    }


def _inspect_clip(
    project: ClipStudioProject,
    infos: Sequence[VideoInfo],
    clip: Clip,
    output: Path,
    dataset: DatasetManifest | None,
    index_error: str | None,
) -> dict[str, Any]:
    directory = output / "videos" / project.video_id / "clips" / clip.name
    result: dict[str, Any] = {
        **clip.to_dict(),
        "clip_id": f"{project.video_id}/{clip.name}",
        "manifest_path": str(directory / "clip.json"),
        "state": "unexported",
        "reason": "clip.jsonがありません。保存した区間と切出済み動画は別です。",
        "export": None,
    }
    if not (directory / "clip.json").is_file():
        if directory.exists() and any(directory.iterdir()):
            result.update(
                state="incomplete",
                reason="出力先にファイルがありますがclip.jsonがありません。",
            )
        return result
    try:
        manifest = ClipManifest.load(directory)
        saved = load_json(manifest.manifest_path)
        settings = ExportSettings(
            output, manifest.fps, manifest.width, manifest.height, 17, False
        )
        plan = plan_clip_export(project, infos, clip, settings)
        expected = build_clip_manifest(plan)
        differences = [
            key
            for key, value in expected.items()
            if key != "exported_at" and saved[key] != value
        ]
        result["export"] = {
            "fps": manifest.fps,
            "num_frames": manifest.num_frames,
            "width": manifest.width,
            "height": manifest.height,
            "cameras": list(manifest.cameras),
            "global_start_sec": saved["global_start_sec"],
            "global_end_sec": saved["global_end_sec"],
        }
        if differences:
            result.update(
                state="conflict",
                reason="保存projectとの不一致: " + ", ".join(differences),
            )
            return result
        missing = [
            camera
            for camera in manifest.camera_ids
            if not manifest.media_path(camera, must_exist=False).is_file()
        ]
        if missing:
            result.update(
                state="incomplete", reason="動画ファイル不足: " + ", ".join(missing)
            )
            return result
        if dataset is None:
            result.update(
                state="index_unknown",
                reason="dataset登録を確認できません: " + str(index_error),
            )
            return result
        record = dataset.clips.get(manifest.clip_id)
        if record is None:
            result.update(
                state="unregistered",
                reason="clipと動画は存在しますがdataset.jsonに登録がありません。",
            )
            return result
        record_fields = {
            "dataset_id": (dataset.dataset_id, manifest.dataset_id),
            "num_cameras": (record.num_cameras, len(manifest.camera_ids)),
            "num_frames": (record.num_frames, manifest.num_frames),
            "fps": (record.fps, manifest.fps),
            "width": (record.width, manifest.width),
            "height": (record.height, manifest.height),
        }
        mismatches = [
            key for key, (actual, target) in record_fields.items() if actual != target
        ]
        if mismatches:
            result.update(
                state="conflict",
                reason="dataset登録との不一致: " + ", ".join(mismatches),
            )
        else:
            result.update(
                state="indexed",
                reason="dataset登録・保存区間・同期値・動画ファイル存在が一致。動画内容の全frame検証や教師の品質確認は未実施。",
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(state="invalid", reason=f"出力を確認できません: {error}")
    return result


def review_catalog(
    project: ClipStudioProject,
    infos: Sequence[VideoInfo],
    projects_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Inspect only existing exports; registration does not imply QA or labels."""
    dataset = None
    index_error = None
    try:
        dataset = load_dataset_manifest(output)
        if dataset.dataset_id != project.dataset_id:
            raise ValueError("dataset.jsonのdataset_idが保存projectと異なります。")
    except (OSError, ValueError, KeyError, TypeError) as error:
        dataset = None
        index_error = str(error)
    clips = [
        _inspect_clip(project, infos, clip, output, dataset, index_error)
        for clip in project.clips
    ]
    return {
        "projects_sha256": hashlib.sha256(projects_path.read_bytes()).hexdigest()
        if projects_path.is_file()
        else None,
        "dataset_path": str(output / "dataset.json"),
        "index_error": index_error,
        "dataset_registered_total": None if dataset is None else len(dataset.clips),
        "video_registered_total": None
        if dataset is None
        else sum(r.video_id == project.video_id for r in dataset.clips.values()),
        "counts": dict(Counter(clip["state"] for clip in clips)),
        "clips": clips,
    }
