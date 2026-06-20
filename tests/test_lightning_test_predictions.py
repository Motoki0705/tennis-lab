"""Test-split prediction saving on BaseLightningModule (issue #533).

Exercises the buffer/scene-id/save machinery without a real Trainer: a tiny
subclass declares a payload, two batches with different sequence lengths are
collected, and the resulting npz is checked for scene-id mapping and time
padding.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.base.training.lightning_module import BaseLightningModule


class _PayloadModule(BaseLightningModule):
    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {
            "pred_position": result["pred"],
            "target_position": result["target"],
        }


def _attach_fake_trainer(
    module: BaseLightningModule, scenes: list[Path], log_dir: str
) -> None:
    module._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(test_dataset=SimpleNamespace(scenes=scenes)),
        log_dir=log_dir,
    )


def test_predictions_saved_with_scene_ids_and_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _PayloadModule()
    scenes = [Path(f"data/plcs/scenes/scene_{i:03d}") for i in range(3)]
    _attach_fake_trainer(module, scenes, str(tmp_path))
    module._reset_test_prediction_buffer()

    # Batch 1: 2 samples, T=4. Batch 2: 1 sample, T=6.
    module.collect_test_predictions(
        None, {"pred": torch.zeros(2, 4, 3), "target": torch.ones(2, 4, 3)}
    )
    module.collect_test_predictions(
        None, {"pred": torch.zeros(1, 6, 3), "target": torch.ones(1, 6, 3)}
    )

    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    npz_path = module.save_test_predictions(metrics={"position_error_m": 0.5})

    assert npz_path is not None and npz_path.exists()
    assert npz_path == tmp_path / "predictions" / "pred_test.npz"
    data = np.load(npz_path, allow_pickle=False)
    assert data["scene_ids"].tolist() == ["scene_000", "scene_001", "scene_002"]
    # Padded to global max T=6, stacked across both batches.
    assert data["pred_position"].shape == (3, 6, 3)
    assert data["target_position"].shape == (3, 6, 3)

    metrics = json.loads((tmp_path / "predictions" / "metrics.json").read_text())
    assert metrics["position_error_m"] == 0.5


def test_bfloat16_predictions_are_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under bf16-mixed precision the model emits bfloat16 tensors, which numpy
    # cannot convert directly; _to_numpy must upcast to float32 (issue #533).
    module = _PayloadModule()
    _attach_fake_trainer(module, [Path("d/scene_000")], str(tmp_path))
    module._reset_test_prediction_buffer()
    module.collect_test_predictions(
        None,
        {
            "pred": torch.zeros(1, 4, 3, dtype=torch.bfloat16),
            "target": torch.ones(1, 4, 3, dtype=torch.bfloat16),
        },
    )
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    npz_path = module.save_test_predictions()
    assert npz_path is not None and npz_path.exists()
    data = np.load(npz_path, allow_pickle=False)
    assert data["pred_position"].dtype == np.float32
    assert data["pred_position"].shape == (1, 4, 3)


def test_empty_payload_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = BaseLightningModule()  # default payload -> {}
    _attach_fake_trainer(module, [], str(tmp_path))
    module._reset_test_prediction_buffer()
    module.collect_test_predictions(None, {"pred": torch.zeros(1, 2, 3)})
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(tmp_path))
    assert module.save_test_predictions() is None
    assert not (tmp_path / "predictions").exists()
