"""Tiny CPU-only checks; no dataset/checkpoint loading."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def test_gradient_geometry_and_unused_entries() -> None:
    spec = importlib.util.spec_from_file_location(
        "gradient_probe", Path(__file__).with_name("probe.py")
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    unused = torch.tensor([3.0], requires_grad=True)
    objectives = {
        "a": (first * torch.tensor([3.0, 4.0])).sum(),
        "b": (first * torch.tensor([-4.0, 3.0])).sum(),
        "zero": (first * 0).sum(),
    }
    result = probe.gradient_summary(objectives, [first, unused])
    assert result["norms"] == {"a": 5.0, "b": 5.0, "zero": 0.0}
    assert result["pairs"]["a__b"] == {"dot": 0.0, "cosine": 0.0}
    assert result["pairs"]["a__zero"] == {"dot": 0.0, "cosine": None}
    assert result["unused_as_zero"]["a"] == {"unused_tensors": 1, "unused_elements": 1}
    assert first.grad is None and unused.grad is None
    assert result["element_count"] == 3


def test_failure_publication_preserves_progress_and_original_error(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "gradient_probe", Path(__file__).with_name("probe.py")
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    original = ValueError("synthetic failure")
    with pytest.raises(ValueError) as raised, probe.publish_run(tmp_path) as report:
        assert (
            json.loads((tmp_path / "results.json").read_text())["status"] == "running"
        )
        report["domains"]["meiji"] = {"modes": {"eval": {"value": 1}}}
        report["stage"] = "meiji/train_dropout"
        probe.publish_results(tmp_path, report)
        raise original
    assert raised.value is original
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "meiji/train_dropout"
    assert saved["error"] == {"type": "ValueError", "message": "synthetic failure"}
    assert saved["domains"]["meiji"]["modes"]["eval"]["value"] == 1
    assert not (tmp_path / "results.json.tmp").exists()
