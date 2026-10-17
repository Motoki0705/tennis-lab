import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.tasks.ball_detection.checkpoints.hybrid_training import (
    HYBRID_CHECKPOINT_SCHEMA,
    HYBRID_MODEL_KIND,
)
from src.tasks.ball_detection.evaluation import hybrid_cli
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from src.utils.checksum import dual_sha256
from tests.integration.tasks.ball_detection.test_rgb_mdd_contract import config


@pytest.mark.parametrize("mode,limit", [
    ("default", 3), ("reduce-overhead", 13), ("max-autotune", 11), ("off", 5),
])
def test_cli_applies_saved_compilation_after_device_placement_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, limit: int,
) -> None:
    model = DeepMDDHybridDetector(MDDHybridConfig(**asdict(config())))
    manifest = tmp_path / "windows.json"
    manifest.write_text("{}")
    checkpoint, output = tmp_path / "model.pt", tmp_path / "evaluation.json"
    contract = model.mdd.input_contract()
    decoder = jpeg_decoder_contract("opencv")
    recipe = dict(
        model=asdict(model.config), input_contract=contract, image_decode=decoder,
        parent=dict(input_contract=contract, image_decode=decoder),
        hybrid_manifest_sha256=dual_sha256(manifest),
        runtime=asdict(CoordinateRuntime(compile_mode=mode, compile_recompile_limit=limit)),
        batch_size=1, presence_threshold=.6,
        loss=dict(heatmap_weight=1., presence_weight=1., sigma_ratio=.012, focal_gamma=2.),
    )
    torch.save(dict(schema=HYBRID_CHECKPOINT_SCHEMA, model_kind=HYBRID_MODEL_KIND,
                    stage="hybrid", recipe=recipe, model_config=asdict(model.config),
                    state_dict=model.state_dict()), checkpoint)
    monkeypatch.setattr(sys, "argv", ["hybrid", "--checkpoint", str(checkpoint),
        "--manifest", str(manifest), "--output", str(output), "--split", "val", "--device", "cuda"])
    events: list[str] = []

    def place(restored: DeepMDDHybridDetector, device: torch.device) -> DeepMDDHybridDetector:
        assert device.type == "cuda"
        events.append("to")
        return restored

    def compile_model(restored: DeepMDDHybridDetector, **settings: Any) -> None:
        assert events == ["to"]
        assert settings == dict(mode=mode, recompile_limit=limit)
        events.append("compile")

    def evaluate(restored: DeepMDDHybridDetector, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert events == ["to", "compile"]
        assert kwargs["presence_threshold"] == .6
        events.append("evaluate")
        return dict(scopes={})

    # Restore a real checkpoint; replace only the CUDA and data execution edges.
    with patch.object(CoordinateRuntime, "configure"), \
            patch.object(DeepMDDHybridDetector, "to", place), \
            patch.object(hybrid_cli, "compile_coordinate_model", side_effect=compile_model) as compile_call, \
            patch.object(hybrid_cli, "FrozenBallWindowDataset", return_value=SimpleNamespace(frame_steps=(1,))), \
            patch.object(CoordinateRuntime, "loader", return_value=[]), \
            patch.object(hybrid_cli, "evaluate_hybrid", side_effect=evaluate) as evaluate_call:
        hybrid_cli.main()
    compile_call.assert_called_once()
    assert compile_call.call_args.args[0] is evaluate_call.call_args.args[0]
    assert events == ["to", "compile", "evaluate"]
    assert json.loads(output.read_text())["checkpoint_sha256"] == dual_sha256(checkpoint)
