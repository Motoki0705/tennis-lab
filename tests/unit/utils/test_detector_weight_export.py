"""The one-off release script preserves inference state and reproducible bytes."""

from __future__ import annotations

import hashlib
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def exporter() -> Callable[..., dict[str, Any]]:
    script = runpy.run_path(str(PROJECT_ROOT / "scripts/export_detector_weights.py"))
    return cast(Callable[..., dict[str, Any]], script["export_checkpoint"])


def training_checkpoint(path: Path) -> str:
    torch.save(
        {
            "state_dict": {
                "backbone.weight": torch.arange(8.0),
                "head.bias": torch.ones(2),
            },
            "hyper_parameters": {
                "config": {"model": {"depth": 3}, "num_frames": 8},
                "target_bundle_state": {"schema": "court_target_bundle_v1"},
            },
            "pytorch-lightning_version": "2.6.5",
            "hparams_name": None,
            "optimizer_states": [{"momentum": torch.ones(10)}],
            "lr_schedulers": [{}],
            "loops": {"fit_loop": {}},
            "callbacks": {},
            "epoch": 9,
            "global_step": 10,
        },
        path,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_preserves_all_inference_keys_and_is_path_independent(
    tmp_path: Path,
    exporter: Callable[..., dict[str, Any]],
) -> None:
    source = tmp_path / "original.ckpt"
    digest = training_checkpoint(source)
    first, second = tmp_path / "one.ckpt", tmp_path / "nested/two.ckpt"
    a, b = exporter(source, first, digest), exporter(source, second, digest)
    assert a == b
    assert a["all_tensors_equal"] is True
    assert a["removed_keys"] == [
        "callbacks",
        "epoch",
        "global_step",
        "loops",
        "lr_schedulers",
        "optimizer_states",
    ]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    saved = torch.load(first, map_location="cpu", weights_only=False)
    assert set(saved) == {
        "state_dict",
        "hyper_parameters",
        "hparams_name",
        "pytorch-lightning_version",
    }
    assert (
        saved["hyper_parameters"]["target_bundle_state"]["schema"]
        == "court_target_bundle_v1"
    )
    assert first.read_bytes() == second.read_bytes()


def test_untrusted_source_is_rejected_before_deserialization(
    tmp_path: Path,
    exporter: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "untrusted.ckpt"
    source.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: pytest.fail("must not deserialize")
    )
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        exporter(source, tmp_path / "out.ckpt", "0" * 64)
    assert not (tmp_path / "out.ckpt").exists()


def test_export_refuses_to_replace_an_existing_file(
    tmp_path: Path,
    exporter: Callable[..., dict[str, Any]],
) -> None:
    source = tmp_path / "original.ckpt"
    digest = training_checkpoint(source)
    destination = tmp_path / "existing.ckpt"
    destination.write_bytes(b"existing asset")
    with pytest.raises(FileExistsError):
        exporter(source, destination, digest)
    assert destination.read_bytes() == b"existing asset"
