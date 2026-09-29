"""SOLIDER loading is exact and semantic conditioning follows the CPU input."""
import numpy as np
import pytest
import torch
from torch import nn

from src.tasks.player_association.appearance.solider import load_backbone
from src.tasks.player_association.appearance.solider_vendor.swin_transformer import (
    SwinTransformer,
)


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {**{f'base.{k}': v for k, v in model.state_dict().items()},
        'classifier.weight': torch.zeros(1041, 1024),
        **{f'bottleneck.{k}': torch.zeros(1024) for k in ('weight', 'bias', 'running_mean', 'running_var')},
        'bottleneck.num_batches_tracked': torch.tensor(0)}


def test_exact_loading_rejects_missing_backbone_and_unknown_tensor() -> None:
    model = nn.Linear(2, 3)
    state = _state(model)
    load_backbone(model, state)
    del state['base.weight']
    with pytest.raises(RuntimeError, match='Missing key'):
        load_backbone(model, state)
    state = _state(model)
    state['other_model.weight'] = torch.zeros(1)
    with pytest.raises(ValueError, match='explicit MSMT17'):
        load_backbone(model, state)


def test_inference_semantic_conditioning_uses_input_device_and_weights() -> None:
    model = SwinTransformer(pretrain_img_size=(32, 16), embed_dims=8,
        depths=(1, 1, 1, 1), num_heads=(1, 2, 4, 8), semantic_weight=.2, drop_path_rate=0.)
    model.eval()
    image = torch.from_numpy(np.linspace(-1., 1., 3 * 32 * 16, dtype=np.float32).reshape(1, 3, 32, 16))
    with torch.inference_mode():
        implicit, _ = model(image)
        explicit, _ = model(image, semantic_weight=torch.tensor([[.2, .8]]))
    assert implicit.device.type == 'cpu' and implicit.shape == (1, 64)
    torch.testing.assert_close(implicit, explicit, rtol=0, atol=0)
