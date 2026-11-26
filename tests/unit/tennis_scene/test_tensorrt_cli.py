"""Check the documented model export CLI before model loading or ONNX work."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.scripts import tensorrt


@pytest.mark.parametrize("options", [
    [], ["--override=~event_detection", "--override=~execution.event_detection"],
], ids=["default", "absent"])
def test_export_cli_reaches_model_dispatch_without_an_event_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str],
) -> None:
    bundle = tmp_path / "bundle"
    seen = []

    def export_model(args: argparse.Namespace, config: PipelineRuntimeConfig) -> None:
        assert config.people.vitpose_checkpoint.name == "vitpose-h-multi-coco.pth"
        assert config.event_detection is None
        seen.append((args.model, args.directory))

    monkeypatch.setattr(sys, "argv", ["tensorrt", "export", "vitpose", str(bundle), *options])
    monkeypatch.setattr(tensorrt.torch.backends.mha, "set_fastpath_enabled", lambda _: None)
    monkeypatch.setattr(tensorrt, "_export_model", export_model)
    tensorrt.main()
    assert seen == [("vitpose", bundle)]
