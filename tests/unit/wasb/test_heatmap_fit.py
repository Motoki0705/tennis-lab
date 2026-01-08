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


def _gaussian_heatmap(center_x: float, center_y: float, size: int, sigma: float) -> torch.Tensor:
    ys = torch.linspace(0, size - 1, size)
    xs = torch.linspace(0, size - 1, size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    heat = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
    heat = heat / heat.sum()
    return heat.unsqueeze(0)


def test_fit_peak_quadratic_and_gaussian() -> None:
    heatmap = _gaussian_heatmap(center_x=3.2, center_y=2.7, size=7, sigma=1.0)
    module = _load_module()
    offsets_q = module._fit_peak(heatmap, fit_mode="quadratic", fit_window=5, eps=1.0e-6)
    offsets_g = module._fit_peak(heatmap, fit_mode="gaussian", fit_window=5, eps=1.0e-6)

    flat = heatmap.view(1, -1)
    idx = torch.argmax(flat, dim=-1)
    y = (idx // 7).to(dtype=heatmap.dtype)
    x = (idx % 7).to(dtype=heatmap.dtype)
    coord_q = torch.stack([x, y], dim=-1) + offsets_q
    coord_g = torch.stack([x, y], dim=-1) + offsets_g

    assert torch.allclose(coord_q[0, 0], torch.tensor(3.2), atol=0.4)
    assert torch.allclose(coord_q[0, 1], torch.tensor(2.7), atol=0.4)
    assert torch.allclose(coord_g[0, 0], torch.tensor(3.2), atol=0.4)
    assert torch.allclose(coord_g[0, 1], torch.tensor(2.7), atol=0.4)
