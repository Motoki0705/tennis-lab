from pathlib import Path
import importlib.util
import sys

import torch


def _load_module():
    root = Path(__file__).resolve().parents[3]
    module_path = root / "src" / "wasb" / "inference" / "ball_detection" / "heatmap_ensemble_predictor.py"
    spec = importlib.util.spec_from_file_location("heatmap_ensemble_predictor", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load heatmap_ensemble_predictor module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_forward_backward_warmup_mixes_uniform() -> None:
    emissions = torch.zeros((3, 5, 5), dtype=torch.float32)
    emissions[0, 2, 2] = 1.0
    emissions[1, 2, 3] = 1.0
    emissions[2, 2, 4] = 1.0
    emissions = emissions / emissions.sum(dim=(-2, -1), keepdim=True)

    module = _load_module()
    kernel = module._build_kernel({"type": "disk", "radius": 1}, device=emissions.device, dtype=emissions.dtype)
    smoothed, state = module._forward_backward_smooth(
        emissions,
        kernel=kernel,
        accel_weight=0.0,
        warmup_tau=1.0,
        forward_state=None,
        eps=1.0e-6,
    )

    uniform = torch.full((5, 5), 1.0 / 25.0)
    assert torch.allclose(smoothed[0], uniform, atol=1e-6)
    assert torch.allclose(smoothed.sum(dim=(-2, -1)), torch.ones(3), atol=1e-5)
    assert state.shape == (5, 5)
