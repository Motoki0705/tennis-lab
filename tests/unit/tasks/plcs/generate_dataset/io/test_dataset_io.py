"""PLCS dataset publication-preservation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize("precreate_empty_root", [False, True])
def test_writer_accepts_missing_or_empty_destination(
    tmp_path: Path,
    precreate_empty_root: bool,
) -> None:
    output_dir = tmp_path / "plcs_broadcast_norm_v2"
    if precreate_empty_root:
        output_dir.mkdir()

    writer = PLCSDatasetWriter(
        output_dir,
        court_coordinate_normalization=resolve_court_coordinate_normalization("v2"),
    )

    assert writer.output_dir == output_dir
    assert writer.scenes_dir == output_dir / "scenes"
    assert writer.scenes_dir.is_dir()


@pytest.mark.parametrize(
    "sentinel_relative",
    [
        "sentinel.bin",
        "config.yaml",
        "meta.json",
        "scenes/scene_000001/sentinel.bin",
    ],
)
def test_writer_rejects_non_empty_destination_without_changing_any_bytes(
    tmp_path: Path,
    sentinel_relative: str,
) -> None:
    output_dir = tmp_path / "plcs_broadcast_norm_v2"
    sentinel = output_dir / sentinel_relative
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"legacy-plcs-bytes\x00\xff")
    before = _tree_bytes(output_dir)

    with pytest.raises(FileExistsError, match="non-empty or non-directory"):
        PLCSDatasetWriter(
            output_dir,
            court_coordinate_normalization=resolve_court_coordinate_normalization("v2"),
        )

    assert _tree_bytes(output_dir) == before
    assert sentinel.read_bytes() == b"legacy-plcs-bytes\x00\xff"


def test_writer_rejects_non_directory_destination_without_changing_bytes(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "plcs_broadcast_norm_v2"
    output_path.write_bytes(b"occupied-file-destination")

    with pytest.raises(FileExistsError, match="non-empty or non-directory"):
        PLCSDatasetWriter(
            output_path,
            court_coordinate_normalization=resolve_court_coordinate_normalization("v2"),
        )

    assert output_path.read_bytes() == b"occupied-file-destination"


def test_writer_rejects_scene_collision_before_writing_scene_files(
    tmp_path: Path,
) -> None:
    writer = PLCSDatasetWriter(
        tmp_path / "plcs_broadcast_norm_v2",
        court_coordinate_normalization=resolve_court_coordinate_normalization("v2"),
    )
    scene_dir = writer.scenes_dir / "scene_existing"
    scene_dir.mkdir()
    sentinel = scene_dir / "position.npy"
    sentinel.write_bytes(b"existing-scene-position")
    before = _tree_bytes(writer.output_dir)
    colliding_scene = SimpleNamespace(meta={"scene_id": "scene_existing"})

    with pytest.raises(FileExistsError):
        writer.save_scene(colliding_scene)  # type: ignore[arg-type]

    assert _tree_bytes(writer.output_dir) == before
    assert sentinel.read_bytes() == b"existing-scene-position"
    assert writer.scene_records == []
    assert writer.scene_counter == 0
