"""Tests for the DINOv3 token annotation contract."""

from __future__ import annotations

from pathlib import Path
from struct import unpack_from
from zipfile import BadZipFile, ZipFile

import numpy as np
import pytest

from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.dino_tokens import (
    DinoTokenSpec,
    dino_dir,
    has_dino_tokens,
    load_dino_tokens,
    sample_frame_indices,
    write_dino_tokens,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifestError,
)
from tests.support.tasks.slcs.dataset import (
    DEFAULT_FIXTURE_DINO_SPEC,
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def test_spec_grid_and_validation() -> None:
    spec = DinoTokenSpec(
        backbone="b",
        patch_size=16,
        image_height=48,
        image_width=64,
        embed_dim=8,
        frame_stride=10,
    )
    assert (spec.grid_h, spec.grid_w, spec.num_tokens) == (3, 4, 12)
    with pytest.raises(DatasetManifestError, match="divisible"):
        DinoTokenSpec(
            backbone="b",
            patch_size=16,
            image_height=50,
            image_width=64,
            embed_dim=8,
            frame_stride=10,
        )


def test_sample_frame_indices_includes_last_frame() -> None:
    assert sample_frame_indices(37, 10).tolist() == [0, 10, 20, 30, 36]
    assert sample_frame_indices(31, 10).tolist() == [0, 10, 20, 30]
    assert sample_frame_indices(1, 10).tolist() == [0]


def test_roundtrip_and_overwrite_protection(synthetic_dataset: SLCSDataIndex) -> None:
    clip_dir = synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    manifest = ClipManifest.load(clip_dir)
    assert has_dino_tokens(clip_dir)
    tokens, frame_idx, spec = load_dino_tokens(
        manifest, "cam0", expected_spec=DEFAULT_FIXTURE_DINO_SPEC
    )
    assert tokens.shape == (len(frame_idx), spec.num_tokens, spec.embed_dim)
    assert tokens.dtype == np.float32
    with pytest.raises(DatasetManifestError, match="overwrite"):
        write_dino_tokens(
            manifest, {"cam0": (tokens.astype(np.float16), frame_idx)}, spec
        )


def test_spec_mismatch_is_error(synthetic_dataset: SLCSDataIndex) -> None:
    manifest = ClipManifest.load(synthetic_dataset.clip_dir(synthetic_dataset.clips[0]))
    wrong = DinoTokenSpec(
        backbone="other",
        patch_size=16,
        image_height=48,
        image_width=64,
        embed_dim=8,
        frame_stride=10,
    )
    with pytest.raises(DatasetManifestError, match="expected spec"):
        load_dino_tokens(manifest, "cam0", expected_spec=wrong)


def test_non_monotonic_frame_idx_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_slcs_dataset_fixture(
        root, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    spec = DEFAULT_FIXTURE_DINO_SPEC
    bad_frames = np.array([0, 5, 5], dtype=np.int64)
    tokens: np.ndarray = np.zeros(
        (3, spec.num_tokens, spec.embed_dim), dtype=np.float16
    )
    with pytest.raises(DatasetManifestError, match="strictly increasing"):
        write_dino_tokens(
            manifest, {"cam0": (tokens, bad_frames)}, spec, overwrite=True
        )


def test_frame_idx_out_of_range_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_slcs_dataset_fixture(
        root, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    spec = DEFAULT_FIXTURE_DINO_SPEC
    frames = np.array([0, 10_000], dtype=np.int64)
    tokens: np.ndarray = np.zeros(
        (2, spec.num_tokens, spec.embed_dim), dtype=np.float16
    )
    with pytest.raises(DatasetManifestError, match="outside clip frames"):
        write_dino_tokens(manifest, {"cam0": (tokens, frames)}, spec, overwrite=True)


def test_missing_archive_is_error(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_slcs_dataset_fixture(
        root, SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    (dino_dir(clip_dir) / "cam0.npz").unlink()
    with pytest.raises(DatasetManifestError, match="archive missing"):
        load_dino_tokens(ClipManifest.load(clip_dir), "cam0")


@pytest.mark.parametrize("member", ["tokens.npy", "frame_idx.npy"])
def test_corrupt_member_crc_reports_archive_context(
    tmp_path: Path, member: str
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "ds", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    path = dino_dir(manifest.clip_dir) / "cam0.npz"
    tokens, frames, _ = load_dino_tokens(manifest, "cam0")
    # Stored ZIP members allow a deterministic payload-bit flip without touching
    # the central-directory CRC or relying on compressor-specific bit layouts.
    np.savez(path, tokens=tokens.astype(np.float16), frame_idx=frames)
    with ZipFile(path) as archive:
        entry = archive.getinfo(member)
    payload = bytearray(path.read_bytes())
    name_length, extra_length = unpack_from("<HH", payload, entry.header_offset + 26)
    data_start = entry.header_offset + 30 + name_length + extra_length
    payload[data_start + entry.file_size - 1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(DatasetManifestError) as caught:
        load_dino_tokens(manifest, "cam0")
    assert isinstance(caught.value.__cause__, BadZipFile)
    assert "CRC-32" in str(caught.value.__cause__)
    assert member in str(caught.value.__cause__)
    for value in (manifest.clip_id, "cam0", str(path)):
        assert value in str(caught.value)


def test_archive_open_io_error_keeps_cause_and_context(
    synthetic_dataset: SLCSDataIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ClipManifest.load(synthetic_dataset.clip_dir(synthetic_dataset.clips[0]))
    path = dino_dir(manifest.clip_dir) / "cam0.npz"
    original = OSError("simulated storage read failure")

    def unreadable(*args: object, **kwargs: object) -> None:
        raise original

    monkeypatch.setattr("src.tasks.slcs.data.dino_tokens.np.load", unreadable)
    with pytest.raises(DatasetManifestError) as caught:
        load_dino_tokens(manifest, "cam0")
    assert caught.value.__cause__ is original
    for value in (manifest.clip_id, "cam0", str(path)):
        assert value in str(caught.value)


@pytest.mark.parametrize("failure", ["missing_key", "shape"])
def test_readable_archive_schema_error_is_not_wrapped(
    tmp_path: Path, failure: str
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "ds", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    path = dino_dir(manifest.clip_dir) / "cam0.npz"
    if failure == "missing_key":
        np.savez(path, tokens=np.zeros((1, 12, 8), dtype=np.float16))
        message = "must contain 'tokens' and 'frame_idx'"
    else:
        np.savez(
            path, tokens=np.zeros((1, 12), dtype=np.float16), frame_idx=np.array([0])
        )
        message = "tokens must be"
    with pytest.raises(DatasetManifestError, match=message) as caught:
        load_dino_tokens(manifest, "cam0")
    assert caught.value.__cause__ is None
    assert "Failed to read" not in str(caught.value)
