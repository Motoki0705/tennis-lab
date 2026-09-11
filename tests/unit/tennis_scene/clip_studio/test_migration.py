"""Legacy recording layout migration tests."""

from pathlib import Path

from src.tennis_scene.clip_studio.migration import (
    apply_video_clip_migration,
    plan_video_clip_migration,
)
from src.tennis_scene.clip_studio.project import ClipStudioProjects
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.io import save_json_atomic


def _legacy_clip_manifest(recording_id: str) -> dict[str, object]:
    return {
        "version": 1,
        "clip_id": f"{recording_id}/clip_000",
        "recording_id": recording_id,
        "clip_name": "clip_000",
        "fps": 30.0,
        "num_frames": 3,
        "width": 64,
        "height": 36,
        "global_start_sec": 0.0,
        "global_end_sec": 0.1,
        "camera_ids": ["cam0"],
        "video_paths": ["media/cam0.mp4"],
        "cameras": [
            {
                "camera_id": "cam0",
                "video": "media/cam0.mp4",
                "source_path": f"/old/raw/{recording_id}/cam0.mp4",
                "offset_sec": 0.0,
                "source_fps": 30.0,
                "source_frame_start": 0,
                "source_frame_end": 2,
                "letterbox": None,
            }
        ],
        "sync_source": "clip_studio",
        "exported_at": "2026-09-11T00:00:00+00:00",
    }


def test_migrates_raw_projects_and_exported_clips(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    multiview = data_root / "tennis_multivew"
    for recording_id in ("meiji_3cam", "meiji_3cam_2"):
        raw = multiview / "raw" / recording_id
        raw.mkdir(parents=True)
        (raw / "cam0.mp4").write_bytes(b"raw")

    processed = multiview / "processed/meiji_3cam"
    records = []
    for index, recording_id in enumerate(("meiji_3cam", "meiji_3cam_2")):
        save_json_atomic(
            {
                "version": 2,
                "recording_id": recording_id,
                "sources": [
                    {
                        "path": f"tennis_multivew/raw/{recording_id}/cam0.mp4",
                        "camera_id": "cam0",
                        "offset_sec": 1.25 + index,
                    }
                ],
                "clips": [
                    {"name": "clip_000", "start_sec": 0.0, "end_sec": 0.1}
                ],
            },
            processed / f"projects/{recording_id}/project.json",
        )
        clip_dir = processed / f"dataset/clips/{recording_id}/clip_000"
        save_json_atomic(_legacy_clip_manifest(recording_id), clip_dir / "clip.json")
        (clip_dir / "media").mkdir()
        (clip_dir / "media/cam0.mp4").write_bytes(b"processed")
        records.append(
            {
                "clip_id": f"{recording_id}/clip_000",
                "recording_id": recording_id,
                "clip_name": "clip_000",
                "path": f"clips/{recording_id}/clip_000",
                "num_cameras": 1,
                "num_frames": 3,
                "fps": 30.0,
                "width": 64,
                "height": 36,
            }
        )
    save_json_atomic(
        {
            "version": 1,
            "created_at": "created",
            "updated_at": "updated",
            "clips": records,
        },
        processed / "dataset/dataset.json",
    )

    plan = plan_video_clip_migration(data_root, "meiji_3cam")
    assert [video.video_id for video in plan.videos] == ["video_000", "video_001"]
    apply_video_clip_migration(plan)

    assert (multiview / "raw/meiji_3cam/video_000/cam0.mp4").is_file()
    assert (multiview / "raw/meiji_3cam/video_001/cam0.mp4").is_file()
    assert not (multiview / "raw/meiji_3cam_2").exists()

    roots = RuntimePathRoots(
        project_root=tmp_path.resolve(),
        data_root=data_root.resolve(),
        checkpoint_root=(tmp_path / "ckpt").resolve(),
        artifact_root=(tmp_path / "artifacts").resolve(),
        output_root=(tmp_path / "outputs").resolve(),
        cache_root=(tmp_path / "cache").resolve(),
        external_asset_root=(tmp_path / "third_party").resolve(),
    )
    projects = ClipStudioProjects.load(
        (processed / "projects.json").resolve(), PathResolver(roots)
    )
    assert list(projects.projects) == ["video_000", "video_001"]
    assert projects.projects["video_000"].sources[0].offset_sec == 1.25
    assert projects.projects["video_001"].sources[0].offset_sec == 2.25
    assert (
        projects.projects["video_000"].sources[0].path
        == (multiview / "raw/meiji_3cam/video_000/cam0.mp4").resolve()
    )

    dataset = load_dataset_manifest(processed / "dataset")
    assert dataset.dataset_id == "meiji_3cam"
    assert dataset.clips["video_000/clip_000"].path == (
        "videos/video_000/clips/clip_000"
    )
    assert dataset.clips["video_001/clip_000"].path == (
        "videos/video_001/clips/clip_000"
    )
    manifest = ClipManifest.load(processed / "dataset/videos/video_000/clips/clip_000")
    assert manifest.dataset_id == "meiji_3cam"
    assert manifest.video_id == "video_000"
    assert manifest.cameras[0]["source_path"].endswith(
        "raw/meiji_3cam/video_000/cam0.mp4"
    )
    second_manifest = ClipManifest.load(
        processed / "dataset/videos/video_001/clips/clip_000"
    )
    assert second_manifest.video_id == "video_001"
    assert not (processed / "projects").exists()
    assert not (processed / "dataset/clips").exists()
