import json
from pathlib import Path

import pytest

from src.tasks.slcs.data.annotation import SLCSDataIndex, load_slcs_annotation
from src.tasks.slcs.data.dino_tokens import load_dino_spec, load_dino_tokens
from src.tasks.slcs.data.splits import load_split_assignments
from src.tennis_scene.dataset_pipeline.assemble import assemble_dataset
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def test_assembly_rebinds_markers_preserves_sources_and_rejects_changed_inputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    index = build_slcs_dataset_fixture(source, SLCSFixtureDatasetConfig())
    for record in index.clips:
        marker = index.clip_dir(record) / "annotations/dino_v3/annotation.json"
        data = json.loads(marker.read_text())
        data["generator"]["checkpoint_sha256"] = "0" * 64
        marker.write_text(json.dumps(data))
    first = ClipManifest.load(index.clip_dir(index.clips[0]))
    original = first.digest()
    splits = {
        f"video_{i:03d}": split for i, split in enumerate(("train", "val", "test"))
    }
    destination = tmp_path / "assembled"
    report = assemble_dataset(
        [source], destination, dataset_id="combined_v1", video_splits=splits, seed=42
    )
    assert report["num_clips"] == 3 and first.digest() == original
    combined = SLCSDataIndex.load(destination)
    assert load_split_assignments(destination / "splits.json", combined) == splits
    for record in combined.clips:
        clip = ClipManifest.load(combined.clip_dir(record))
        assert clip.dataset_id == "combined_v1"
        load_slcs_annotation(clip)
        load_dino_tokens(
            clip, clip.camera_ids[0], expected_spec=load_dino_spec(clip.clip_dir)
        )
    assert (
        assemble_dataset(
            [source],
            destination,
            dataset_id="combined_v1",
            video_splits=splits,
            seed=42,
        )
        == report
    )
    with pytest.raises(ValueError, match="inputs changed"):
        assemble_dataset(
            [source],
            destination,
            dataset_id="combined_v2",
            video_splits=splits,
            seed=42,
        )
    split_path = destination / "splits.json"
    original_splits = split_path.read_bytes()
    payload = json.loads(original_splits)
    payload["assignments"]["video_002"] = "train"
    split_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split assignments changed"):
        assemble_dataset(
            [source], destination, dataset_id="combined_v1", video_splits=splits, seed=42
        )
    split_path.write_bytes(original_splits)
    marker = combined.clip_dir(combined.clips[0]) / "annotations/dino_v3/annotation.json"
    feature_receipt = json.loads(marker.read_text())
    feature_receipt["generator"]["checkpoint_sha256"] = "1" * 64
    marker.write_text(json.dumps(feature_receipt))
    with pytest.raises(ValueError, match="feature identity changed"):
        assemble_dataset(
            [source], destination, dataset_id="combined_v1", video_splits=splits, seed=42
        )
