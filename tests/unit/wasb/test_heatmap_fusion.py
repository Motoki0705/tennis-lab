from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch


def _load_predictor_module():
    root = Path(__file__).resolve().parents[3]
    module_path = root / "src" / "wasb" / "inference" / "ball_detection" / "heatmap_ensemble_predictor.py"
    spec = importlib.util.spec_from_file_location("heatmap_ensemble_predictor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load heatmap_ensemble_predictor module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _DummyRunner:
    def __init__(self, logits_list: list[torch.Tensor]) -> None:
        self._logits_list = logits_list

    def reset(self) -> None:
        return None

    def predict_batch_heatmaps_tta(self, frames_rgb, *, device, tta_transforms, target_hw):
        return self._logits_list


def test_poe_fusion_matches_expected() -> None:
    logits1 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
    logits2 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    module = _load_predictor_module()
    predictor_cls = module.HeatmapEnsemblePredictor
    runner1 = _DummyRunner([logits1])
    runner2 = _DummyRunner([logits2])
    predictor = predictor_cls(
        runners=[runner1, runner2],
        device=torch.device("cpu"),
        output_heatmap_hw=(3, 3),
        tta_transforms=[{"type": "identity"}],
        calibration_t=[1.0, 1.0],
        calibration_b=[0.0, 0.0],
        fusion_cfg={"weight_mode": "fixed", "model_weights": [1.0, 1.0], "eps": 1.0e-6},
        smoothing_cfg={"enabled": False},
        decode_cfg={"mode": "map", "return_heatmap": True},
    )

    frames = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    result = predictor.predict(frames)
    fused = torch.from_numpy(result["heatmap"])

    eps = 1.0e-6
    p1 = torch.sigmoid(logits1) + eps
    p2 = torch.sigmoid(logits2) + eps
    p1 = p1 / p1.sum(dim=(-2, -1), keepdim=True)
    p2 = p2 / p2.sum(dim=(-2, -1), keepdim=True)
    logp = (p1 + eps).log() + (p2 + eps).log()
    expected = torch.softmax(logp.view(1, -1), dim=-1).view(1, 3, 3)

    assert torch.allclose(fused, expected, atol=1e-5)
