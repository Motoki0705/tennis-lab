"""Store labels, split isolation, windows and batch supervision are one contract."""

import json
from collections import Counter
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.store_datamodule import BallStoreDataModule
from src.tasks.ball_detection.data.store_dataset import select_store_windows
from src.tasks.ball_detection.data.supervision import (
    OBSERVED_ONLY,
    FrameSupervisionPolicy,
    resolve_frame_supervision,
)
from tests.support.tasks.ball_detection.store import (
    ball,
    frame,
    store_config,
    write_store_clip,
)


def test_supervision_does_not_turn_unknown_or_estimated_balls_into_negatives(
    tmp_path: Path,
) -> None:
    directory = write_store_clip(
        tmp_path / "store",
        "chat/clip",
        [
            frame(0, ball()),
            frame(1),
            frame(2, ball("out_of_frame", None)),
            frame(3, ball("interpolated")),
            frame(4, ball("occlusion_estimated")),
            frame(5, ball("unresolved", None)),
            frame(6, annotated=False),
            frame(7, ball(), ball("unresolved", None, track="b002")),
        ],
        source="chat_annotation",
    )
    store = BallFrameStore(directory)
    labels = resolve_frame_supervision(store, OBSERVED_ONLY)
    assert labels.supervised.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    selection = select_store_windows(store, labels, store.clips, length=2, stride=1)
    assert [window.start for window in selection.windows] == [0, 1, 2]
    assert selection.stats["chat_annotation"]["windows_without_supervision"] == 4
    short = select_store_windows(store, labels, store.clips, length=9, stride=1)
    assert not short.windows
    assert short.stats["chat_annotation"]["clips_shorter_than_window"] == 1


def test_mixed_loader_split_isolation_and_masked_targets(tmp_path: Path) -> None:
    directory = tmp_path / "ball_detection/test-v1"
    for split in ("train", "val", "test"):
        for source in ("tracknet", "meiji"):
            write_store_clip(
                directory,
                f"{source}/{split}/clip",
                [
                    frame(0, ball()),
                    frame(1, ball("unresolved", None)),
                    frame(2),
                    frame(3, ball()),
                ],
                source=source,
                split=split,
            )
    cfg = store_config(tmp_path)
    module = BallStoreDataModule(cfg)
    module.setup()
    for split, dataset in [
        ("train", module.train_dataset),
        ("val", module.val_dataset),
        ("test", module.test_dataset),
    ]:
        assert dataset is not None
        assert {
            dataset.store.clips[window.clip].split for window in dataset.windows
        } == {split}
    batches = list(module.train_dataloader())
    assert len(batches[-1]["source"]) == 1  # Keep the epoch quota in the partial batch.
    assert Counter(source for batch in batches for source in batch["source"]) == {
        "tracknet": 4,
        "meiji": 5,
    }
    assert module.val_dataset is not None
    sample = module.val_dataset[0]
    assert sample["supervised"].tolist() == [True, False]
    assert sample["visibility"][0].sum() == 1
    assert sample["visibility"][1].sum() == 0
    assert sample["heatmaps"][0].max() > 0.5
    assert sample["heatmaps"][1].sum() == 0
    torch.testing.assert_close(sample["coords"][0, 0], torch.tensor([10.0, 20.0]))
    assert sample["images"].shape == (2, 3, 48, 64)
    prefix = module.val_dataset[0, 1]
    assert prefix["images"].shape[0] == 1
    assert prefix["supervised"].tolist() == [True]
    # Dataset pickling must reopen shards in workers; a loader with workers has identical eval batches.
    cfg.data.num_workers = 2
    worker_module = BallStoreDataModule(cfg)
    worker_module.setup("validate")
    for actual, expected in zip(
        worker_module.val_dataloader(), module.val_dataloader(), strict=True
    ):
        torch.testing.assert_close(actual["images"], expected["images"])
        assert actual["window_id"] == expected["window_id"]


def test_absent_source_and_empty_split_fail_explicitly(tmp_path: Path) -> None:
    write_store_clip(
        tmp_path / "ball_detection/test-v1", "tracknet/train/clip", [frame(0), frame(1)]
    )
    cfg = store_config(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        BallStoreDataModule(cfg)
    cfg.data.sources = ["tracknet"]
    cfg.data.train_sampling.source_weights = {"tracknet": 1.0}
    with pytest.raises(RuntimeError, match="No supervised val windows"):
        BallStoreDataModule(cfg).setup("validate")


def test_validation_covers_tail_and_unknown_windows_without_padding(tmp_path: Path) -> None:
    directory = write_store_clip(tmp_path / "store", "meiji/val/cam0",
                                 [frame(i, ball("unresolved", None)) for i in range(9)], split="val")
    store = BallFrameStore(directory)
    supervision = resolve_frame_supervision(store, OBSERVED_ONLY)
    selected = select_store_windows(store, supervision, store.clips, length=4, stride=3, validation=True)
    assert [window.start for window in selected.windows] == [0, 3, 5]
    assert selected.stats["tracknet"]["windows_without_supervision"] == 3
    with pytest.raises(ValueError, match="gaps"):
        select_store_windows(store, supervision, store.clips, length=4, stride=5, validation=True)
    with pytest.raises(ValueError, match="shorter"):
        select_store_windows(store, supervision, store.clips, length=10, stride=3, validation=True)


def test_candidate_reference_uses_raw_single_observed_and_source_scale(tmp_path: Path) -> None:
    from src.tasks.ball_detection.data.store_dataset import (
        BallStoreDataset,
        StoreWindow,
    )

    directory = write_store_clip(
        tmp_path / "ball_detection/test-v1", "meiji/val/cam0",
        [frame(0, ball()), frame(1, ball("interpolated")),
         frame(2, ball(), ball("out_of_frame", None, track="b002")), frame(3, ball(), annotated=False)],
        source="meiji", split="val",
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["clips"][0].update(source_width=128, source_height=96, camera_id="cam0")
    metadata_path.write_text(json.dumps(metadata))
    store = BallFrameStore(directory)
    # Even a policy that trains on estimated positions cannot admit them to
    # candidate recall, or collapse one observed + one other instance.
    policy = FrameSupervisionPolicy(frozenset({"observed", "interpolated"}),
                                    frozenset({"out_of_frame"}), frozenset({"unresolved", "occlusion_estimated"}))
    data = BallStoreDataset(store=store, supervision=resolve_frame_supervision(store, policy),
                           windows=[StoreWindow(0, 0)], config=store_config(tmp_path, frames=4))
    reference = data[0]["candidate_reference"]
    assert reference["observed"].tolist() == [True, False, False, False]
    assert reference["source_scale"].item() == 0.5
    assert reference["camera"] == "cam0"
    assert reference["namespace"] == "store"
    assert reference["frame_id"].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(reference["xy"][0], torch.tensor([10.0, 20.0]))


@pytest.mark.parametrize(
    "policy",
    [
        {
            "positive": ["observed"],
            "absent": ["unresolved", "out_of_frame"],
            "ignore": ["interpolated", "occlusion_estimated"],
        },
        {
            "positive": ["observed"],
            "absent": ["interpolated", "out_of_frame"],
            "ignore": ["unresolved", "occlusion_estimated"],
        },
        {
            "positive": ["observed"],
            "absent": ["out_of_frame"],
            "ignore": ["unresolved"],
        },
        {
            "positive": ["observed"],
            "absent": ["out_of_frame"],
            "ignore": ["observed", "unresolved", "interpolated", "occlusion_estimated"],
        },
    ],
)
def test_invalid_label_policy_fails_at_configuration_boundary(
    tmp_path: Path, policy: dict[str, list[str]]
) -> None:
    from src.tasks.ball_detection.configuration import validate_data

    cfg = store_config(tmp_path)
    cfg.data.supervision = OmegaConf.create(policy)
    with pytest.raises(ValueError):
        validate_data(cfg)
    with pytest.raises(ValueError):
        FrameSupervisionPolicy.from_mapping(policy)
