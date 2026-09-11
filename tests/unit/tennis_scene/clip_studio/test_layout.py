"""Canonical raw dataset/video discovery tests."""

from pathlib import Path

import pytest

from src.tennis_scene.clip_studio.layout import discover_clip_studio_layout
from src.utils.configuration import PathResolver


def _source(
    path_resolver: PathResolver, relative: str, names: tuple[str, ...]
) -> Path:
    directory: Path = path_resolver.roots.data_root / relative
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).touch()
    return directory


def test_discovers_video_and_derives_processed_layout(path_resolver) -> None:
    source = _source(
        path_resolver,
        "tennis_multivew/raw/meiji_3cam/video_002",
        ("cam2.mp4", "cam0.mp4", "cam1.mp4"),
    )

    layout = discover_clip_studio_layout(
        path_resolver, "tennis_multivew/raw/meiji_3cam/video_002"
    )

    assert layout.dataset_id == "meiji_3cam"
    assert layout.video_id == "video_002"
    assert layout.source_directory == source.resolve()
    assert layout.camera_ids == ("cam0", "cam1", "cam2")
    assert [path.name for path in layout.video_paths] == [
        "cam0.mp4",
        "cam1.mp4",
        "cam2.mp4",
    ]
    processed = path_resolver.roots.data_root / "tennis_multivew/processed/meiji_3cam"
    assert layout.projects_path == (processed / "projects.json").resolve()
    assert layout.dataset_directory == (processed / "dataset").resolve()


@pytest.mark.parametrize(
    ("relative", "files", "message"),
    [
        ("other/raw/set/video_000", ("cam0.mp4",), "source_directory must be"),
        (
            "tennis_multivew/raw/set/session_000",
            ("cam0.mp4",),
            "video_id must match",
        ),
        (
            "tennis_multivew/raw/set/video_000",
            ("cam0.mp4", "cam2.mp4"),
            "contiguous",
        ),
        (
            "tennis_multivew/raw/set/video_000",
            ("cam0.mp4", "preview.mp4"),
            "unsupported MP4",
        ),
    ],
)
def test_rejects_noncanonical_source_layout(
    path_resolver, relative: str, files: tuple[str, ...], message: str
) -> None:
    _source(path_resolver, relative, files)
    with pytest.raises(ValueError, match=message):
        discover_clip_studio_layout(path_resolver, relative)


def test_rejects_missing_source_directory(path_resolver) -> None:
    with pytest.raises(FileNotFoundError, match="source directory not found"):
        discover_clip_studio_layout(path_resolver, "tennis_multivew/raw/set/video_000")
