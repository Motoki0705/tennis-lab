from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.io import write_jpeg

from src.datasets.collate_tracking import SceneBatch
from src.training.scene_model.datamodule import DancetrackDataModule


def _make_fake_sequence(root: Path, split: str, name: str) -> None:
    seq_root = root / split / name
    img_dir = seq_root / "img1"
    gt_dir = seq_root / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    (seq_root / "seqinfo.ini").write_text(
        f"""[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength=3\nimWidth=32\nimHeight=32\nimExt=.jpg\n""",
        encoding="utf-8",
    )
    for frame_id in range(1, 4):
        img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        write_jpeg(img, str(img_dir / f"{frame_id:06d}.jpg"))
    (gt_dir / "gt.txt").write_text(
        "1,1,5,5,10,10,1,1,1\n2,1,6,6,10,10,1,1,1\n3,1,7,7,10,10,1,1,1\n",
        encoding="utf-8",
    )


def _dataset_cfg(root: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "root": str(root),
            "split": {"train": "train", "val": "val"},
            "window": {"size": 2, "stride": 1},
            "image": {"resize": 32, "horizontal_flip_prob": 0.0},
            "loader": {
                "train": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "val": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            },
            "cache": {"enabled": False},
        }
    )


def test_datamodule_produces_scene_batch(tmp_path: Path) -> None:
    for split in ("train", "val"):
        _make_fake_sequence(tmp_path, split, f"demo_{split}")
    cfg = _dataset_cfg(tmp_path)
    dm = DancetrackDataModule(cfg, {"minimal": True})
    dm.setup("fit")
    loader = dm.train_dataloader()
    batches = list(loader)
    assert len(batches) == 2
    batch = batches[0]
    assert isinstance(batch, SceneBatch)
    assert batch.frames.shape[-1] == 32
    assert batch.padding_mask.shape[1] == 2
    val_batch = next(iter(dm.val_dataloader()))
    assert val_batch.frames.shape[0] == 1
