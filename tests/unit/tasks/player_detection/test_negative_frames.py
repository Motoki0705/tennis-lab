from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.tasks.player_detection.configuration import (
    AugmentationConfig,
    FrameSelectionConfig,
    InputSizeConfig,
    PlayerDataConfig,
)
from src.tasks.player_detection.data.datamodule import PlayerDetectionDataModule
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    collate_detection,
    select_detection_frames,
)
from src.tasks.player_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
    PlayerFrameStore,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.review.service import StoreReview
from tests.support.tasks.player_detection.chat_root import SyntheticRoot
from tests.support.tasks.player_detection.store import legacy_store

SELECTION = FrameSelectionConfig(require_reviewed=True, require_located_players=True, min_visible_box_px=4.)
AUGMENTATION = AugmentationConfig(hflip_prob=1., short_side_choices=(64,), brightness=0., contrast=0., saturation=0.)


def data_config(directory) -> PlayerDataConfig:
    return PlayerDataConfig(dataset_dir=directory, batch_size=2, eval_batch_size=2, num_workers=0, pin_memory=False,
        train_samples_per_epoch=4, val_frame_stride=1, test_frame_stride=1, selection=SELECTION,
        input_size=InputSizeConfig(short_side=64, max_long_side=96), augmentation=AUGMENTATION)


def test_mixed_samples_preserve_empty_shapes_through_augmentation_and_collate(synthetic_root: SyntheticRoot) -> None:
    store = PlayerFrameStore(build_dataset(synthetic_root.config))
    selected = select_detection_frames(store, "train", SELECTION, frame_stride=1)
    data = PlayerDetectionDataset(store, selected, input_size=data_config(store.directory).input_size, augmentation=AUGMENTATION)
    negative, positive = data[0], data[1]
    batch = collate_detection([negative, positive])
    assert batch.boxes_cxcywh[0].shape == batch.boxes_xyxy_px[0].shape == (0, 4)
    assert batch.labels[0].shape == (0,) and batch.labels[0].dtype == torch.int64
    assert batch.boxes_cxcywh[1].shape == (2, 4) and batch.labels[1].shape == (2,)
    assert all(torch.isfinite(image).all() for image in batch.images)
    assert batch.frames == [0, 1]
    review = StoreReview(store.directory, SELECTION)
    assert review.frame(store.clips[0].clip_id, 0)["confirmed_empty"] is True
    assert review.frame(store.clips[0].clip_id, 0)["annotations"] == []
    assert review.summary()["counts"]["confirmed_empty"] == 9
    assert "confirmed_empty" in review.timeline(store.clips[0].clip_id)["frames"][0]["flags"]


def test_negative_only_clips_reach_train_loader_but_cannot_supply_ap_validation(synthetic_root: SyntheticRoot) -> None:
    for path in (synthetic_root.root / "annotated/processed/player").glob("*.json"):
        raw = json.loads(path.read_text())
        for frame in raw["frames"]:
            frame["players"] = []
        path.write_text(json.dumps(raw))
    directory = build_dataset(synthetic_root.config)
    module = PlayerDetectionDataModule(data_config(directory))
    batches = list(module.train_dataloader())
    assert len(batches) == 2 and module.selection_stats["train"]["negative_frames"] == 12
    assert module.store.instances["bbox_xyxy"].shape == (0, 4)
    assert all(boxes.shape == (0, 4) for batch in batches for boxes in batch.boxes_cxcywh)
    assert all(labels.shape == (0,) for batch in batches for labels in batch.labels)
    path = directory / METADATA_FILE
    raw = json.loads(path.read_text())
    raw["clips"][0]["split"] = "val"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="positive target instances"):
        PlayerDetectionDataModule(data_config(directory)).val_dataloader()


def test_unknown_context_and_removed_boxes_never_turn_into_negative_teachers(synthetic_root: SyntheticRoot) -> None:
    directory = build_dataset(synthetic_root.config)
    with np.load(directory / INDEX_FILE) as saved:
        columns = {name: saved[name] for name in saved.files}
    columns["reviewed"][0] = False
    columns["is_target"][3] = False
    columns["confirmed_empty"][[0, 3, 5]] = False
    np.savez(directory / INDEX_FILE, **columns)
    store = PlayerFrameStore(directory)
    selection = FrameSelectionConfig(require_reviewed=False, require_located_players=True, min_visible_box_px=20.)
    selected = select_detection_frames(store, "train", selection, frame_stride=1)
    assert selected.frames.tolist() == [1, 6, 7, 8, 9, 10, 11]
    assert selected.stats["negative_frames"] == 6
    assert selected.stats["dropped_unconfirmed_empty"] == 2
    assert selected.stats["dropped_context"] == selected.stats["dropped_no_visible_box"] == 1
    assert selected.stats["dropped_unresolved_player"] == 1
    review = StoreReview(directory, selection)
    assert [i for i, reason in enumerate(review.reasons) if reason == "eligible"] == selected.frames.tolist()


@pytest.mark.parametrize("fault", ["positive", "unreviewed", "context", "missing"])
def test_invalid_confirmed_empty_flags_are_rejected(synthetic_root: SyntheticRoot, fault: str) -> None:
    directory = build_dataset(synthetic_root.config)
    with np.load(directory / INDEX_FILE) as saved:
        columns = {name: saved[name] for name in saved.files}
    if fault == "positive":
        columns["confirmed_empty"][1] = True
    elif fault == "missing":
        columns.pop("confirmed_empty")
    else:
        columns["reviewed" if fault == "unreviewed" else "is_target"][0] = False
    np.savez(directory / INDEX_FILE, **columns)
    with pytest.raises(ValueError, match="Confirmed empty|missing column"):
        PlayerFrameStore(directory)


def test_regular_reader_rejects_legacy_store(synthetic_root: SyntheticRoot) -> None:
    current = build_dataset(synthetic_root.config)
    directory = legacy_store(current, current.parent / "legacy-v1")
    with pytest.raises(ValueError, match="Unsupported player store schema"):
        PlayerFrameStore(directory)
