"""The task inference entry point accepts only exported player weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.submodules.models.dino.person_detector import DinoPersonDetector
from src.tasks.player_detection.inference import (
    DinoPlayerDetector,
    inspect_player_checkpoint,
)
from src.tasks.player_detection.model_export import EXPORT_FORMAT


def _write_checkpoint(path: Path, **overrides: object) -> Path:
    metadata: dict[str, object] = {
        "format": EXPORT_FORMAT,
        "task": "player_detection",
        "class_id": 1,
        "source_checkpoint_sha256": "a" * 64,
        "epoch": 3,
        "global_step": 8000,
    }
    metadata.update(overrides)
    torch.save({"model": {}, "args": object(), "tennis_lab": metadata}, path)
    return path


def test_player_checkpoint_inspection_reports_training_identity(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "player.pth")

    info = inspect_player_checkpoint(checkpoint)

    assert info.path == checkpoint
    assert info.source_checkpoint_sha256 == "a" * 64
    assert (info.epoch, info.global_step) == (3, 8000)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"format": "unknown"}, "not a supported tennis-player"),
        ({"task": "ball_detection"}, "not a supported tennis-player"),
        ({"class_id": 0}, "not a supported tennis-player"),
        ({"class_id": True}, "not a supported tennis-player"),
        ({"source_checkpoint_sha256": "invalid"}, "source_checkpoint_sha256"),
        ({"epoch": -1}, "invalid epoch"),
        ({"global_step": True}, "invalid global_step"),
    ],
)
def test_player_checkpoint_rejects_wrong_provenance(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "wrong.pth", **override)

    with pytest.raises(ValueError, match=message):
        inspect_player_checkpoint(checkpoint)


def test_player_detector_rejects_coco_checkpoint_before_model_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "coco.pth"
    torch.save({"model": {}, "args": object()}, checkpoint)
    detector = DinoPlayerDetector(
        checkpoint, tmp_path, device="cpu", confidence=0.3,
        short_side=800, max_long_side=1333,
    )
    built = False

    def unexpected_build(self: DinoPersonDetector) -> None:
        nonlocal built
        built = True

    monkeypatch.setattr(DinoPersonDetector, "_load_impl", unexpected_build)
    with pytest.raises(ValueError, match="exported player DINO checkpoint"):
        detector.load()
    assert not built
    assert not detector.is_loaded


def test_player_detector_loads_valid_export_and_tracks_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "player.pth")
    detector = DinoPlayerDetector(
        checkpoint, tmp_path, device="cpu", confidence=0.3,
        short_side=800, max_long_side=1333,
    )
    calls: list[str] = []

    def fake_load(self: DinoPersonDetector) -> None:
        calls.append("load")

    def fake_unload(self: DinoPersonDetector) -> None:
        calls.append("unload")

    monkeypatch.setattr(DinoPersonDetector, "_load_impl", fake_load)
    monkeypatch.setattr(DinoPersonDetector, "_unload_impl", fake_unload)
    detector.load()
    detector.load()
    assert calls == ["load"]
    assert detector.is_loaded
    assert detector.checkpoint_info is not None
    assert detector.checkpoint_info.epoch == 3
    detector.unload()
    assert calls == ["load", "unload"]
    assert not detector.is_loaded
    assert detector.checkpoint_info is None


def test_player_checkpoint_requires_absolute_existing_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        inspect_player_checkpoint(Path("player.pth"))
    with pytest.raises(FileNotFoundError, match="not found"):
        inspect_player_checkpoint(tmp_path / "missing.pth")
