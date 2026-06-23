from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.data.web_store import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    STORE_FILE,
    STORE_SHARD,
    WebFrameStore,
)
from src.tasks.ball_detection.scripts.convert_web_dataset import (
    IndexBuilder,
    SampleRecord,
    ShardWriter,
    make_group_split_map,
    roboflow_source_group,
)


def _encode(width: int, height: int, value: int) -> bytes:
    image: np.ndarray = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _build_store(output_dir: Path) -> None:
    """Create a tiny two-backend store: a packed video frame + a referenced file."""
    refs = output_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    ref_path = refs / "still.jpg"
    ref_path.write_bytes(_encode(80, 60, 200))

    writer = ShardWriter(output_dir / "shards", shard_size_bytes=1 << 30)
    index = IndexBuilder()
    index.add(
        SampleRecord(
            instances=[(40.0, 30.0, 1)],
            orig_w=160,
            orig_h=120,
            temporal=1,
            source="racketvision",
            sequence="racketvision:match1_000",
            frame_index=7,
            split="train",
            jpeg=_encode(160, 120, 100),
        ),
        writer,
    )
    index.add(
        SampleRecord(
            instances=[],
            orig_w=160,
            orig_h=120,
            temporal=1,
            source="racketvision",
            sequence="racketvision:match1_000",
            frame_index=9,
            split="train",
            jpeg=_encode(160, 120, 110),
        ),
        writer,
    )
    index.add(
        SampleRecord(
            instances=[(20.0, 15.0, 1), (60.0, 45.0, 1)],
            orig_w=80,
            orig_h=60,
            temporal=0,
            source="roboflow",
            sequence="roboflow:still1",
            frame_index=-1,
            split="train",
            file_path=ref_path,
        ),
        writer,
    )
    index.add(
        SampleRecord(
            instances=[(40.0, 30.0, 1)],
            orig_w=160,
            orig_h=120,
            temporal=1,
            source="racketvision",
            sequence="racketvision:match2_000",
            frame_index=11,
            split="val",
            jpeg=_encode(160, 120, 120),
        ),
        writer,
    )
    index.add(
        SampleRecord(
            instances=[],
            orig_w=160,
            orig_h=120,
            temporal=1,
            source="racketvision",
            sequence="racketvision:match2_000",
            frame_index=13,
            split="val",
            jpeg=_encode(160, 120, 130),
        ),
        writer,
    )
    writer.close()
    index.save(output_dir)


def _config(data_dir: Path, **data_overrides: object) -> dict:
    data = {
        "source": "web",
        "data_dir": str(data_dir),
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "image_size": [64, 96],
        "heatmap_size": [64, 96],
        "sigma_ratio": 0.012,
        "max_instances": 8,
        "sampling": {
            "mode": "static",
            "seed": 1234,
            "train_negative_fraction": None,
            "temporal": {
                "frame_step": 1,
                "sample_stride": 1,
                "max_frame_gap": None,
            },
        },
        "augmentation": {"normalize_imagenet": {"enabled": False}},
    }
    data.update(data_overrides)
    return {"model": {"num_frames": 4}, "data": data}


def test_group_split_is_complete_balanced_and_deterministic() -> None:
    weights = {f"sequence-{index}": index + 1 for index in range(16)}
    first = make_group_split_map(weights, val_ratio=0.1, test_ratio=0.1, seed=1234)
    second = make_group_split_map(weights, val_ratio=0.1, test_ratio=0.1, seed=1234)
    assert first == second
    assert set(first) == set(weights)
    assert list(first.values()).count("train") == 12
    assert list(first.values()).count("val") == 2
    assert list(first.values()).count("test") == 2
    assert roboflow_source_group("frame_10_jpg.rf.012345.jpg") == "frame_10_jpg"


def test_web_store_backends_roundtrip(tmp_path: Path) -> None:
    _build_store(tmp_path)
    store = WebFrameStore(tmp_path)

    assert len(store) == 5
    backends = {int(store._columns["store"][i]) for i in range(len(store))}
    assert backends == {STORE_SHARD, STORE_FILE}

    assert store.temporal(0) is True
    assert store.original_size(0) == (160, 120)
    assert store.decode_bgr(0).shape == (120, 160, 3)
    assert store.sequence_name(0) == "racketvision:match1_000"
    assert store.frame_index(0) == 7
    assert store.label_state(0) == LABEL_POSITIVE
    assert store.is_positive(0) is True

    # Explicit negative from the same temporal sequence.
    assert store.label_state(1) == LABEL_NEGATIVE
    assert store.labels(1) == ()
    assert store.is_positive(1) is False

    # File-backed (referenced in place) sample with two ball instances.
    assert store.temporal(2) is False
    assert store.decode_bgr(2).shape == (60, 80, 3)
    assert len(store.labels(2)) == 2

    assert store.split_indices("train").tolist() == [0, 1, 2]
    assert store.split_indices("val").tolist() == [3, 4]
    assert store.split_indices("train", temporal_only=True).tolist() == [0, 1]
    assert store.split_indices("train", sources=["roboflow"]).tolist() == [2]
    assert store.temporal_windows("train", num_frames=2) == [(0, 1)]


def test_web_datamodule_sample_contract(tmp_path: Path) -> None:
    _build_store(tmp_path)
    datamodule = build_ball_detection_datamodule(_config(tmp_path))
    datamodule.setup("fit")

    assert len(datamodule.train_dataset) == 3
    assert len(datamodule.val_dataset) == 2

    sample = datamodule.train_dataset[0]
    assert set(sample) == {
        "images",
        "heatmaps",
        "coords",
        "visibility",
        "original_size",
        "heatmap_size",
    }
    assert tuple(sample["images"].shape) == (4, 3, 64, 96)
    assert sample["images"].dtype == torch.float32
    assert tuple(sample["heatmaps"].shape) == (4, 64, 96)
    assert tuple(sample["coords"].shape) == (4, 8, 2)
    assert tuple(sample["visibility"].shape) == (4, 8)
    # Static clip: every replicated frame shares the annotation, peak > 0.
    assert sample["heatmaps"].amax() > 0
    assert torch.equal(sample["images"][0], sample["images"][1])

    batch = next(iter(datamodule.train_dataloader()))
    assert tuple(batch["images"].shape) == (2, 4, 3, 64, 96)


def test_web_datamodule_temporal_windows(tmp_path: Path) -> None:
    _build_store(tmp_path)
    config = _config(
        tmp_path,
        sampling={
            "mode": "temporal",
            "seed": 1234,
            "train_negative_fraction": None,
            "temporal": {
                "frame_step": 1,
                "sample_stride": 1,
                "max_frame_gap": 2,
            },
        },
    )
    config["model"]["num_frames"] = 2
    datamodule = build_ball_detection_datamodule(config)
    datamodule.setup("fit")
    # The window remains inside one split-safe video sequence.
    assert len(datamodule.train_dataset) == 1
    sample = datamodule.train_dataset[0]
    assert tuple(sample["images"].shape) == (2, 3, 64, 96)
    assert sample["heatmaps"][0].amax() > 0
    assert sample["heatmaps"][1].amax() == 0
    assert not torch.equal(sample["images"][0], sample["images"][1])


def test_web_datamodule_can_cap_static_negative_fraction(tmp_path: Path) -> None:
    _build_store(tmp_path)
    config = _config(
        tmp_path,
        sampling={
            "mode": "static",
            "seed": 1234,
            "train_negative_fraction": 0.0,
            "temporal": {},
        },
    )
    datamodule = build_ball_detection_datamodule(config)
    datamodule.setup("fit")
    assert len(datamodule.train_dataset) == 2
    assert all(
        window.frame_names[0] != "1" for window in datamodule.train_dataset.windows
    )
