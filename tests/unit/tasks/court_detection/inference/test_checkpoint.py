"""Historical training state must not alter strict inference weight loading."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.bundle_state import serialize_target_bundle
from src.tasks.court_detection.data.contracts import (
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.inference.checkpoint import load_court_pair
from src.tasks.court_detection.models.encoders import CourtDINOv3Encoder
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel

_CONFIG = Path(__file__).resolve().parents[5] / 'src/tasks/court_detection/configs'


def test_inference_loads_exact_weights_without_training_schema(tmp_path, monkeypatch):
    bundle = CourtTargetBundleSpec({'kp': CourtTargetSpec('kp', 'test', 1, ('point',), torch.float32, False)})
    with initialize_config_dir(config_dir=str(_CONFIG), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[])
    resolver = CourtTrainingConfig.from_config(cfg).shared.resolver
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw.pop('run')
    raw.pop('training')
    raw['data']['augmentation']['train_scales'] = [256, 384, 512]

    def make_model(config, target_bundle):
        assert target_bundle == bundle
        model = object.__new__(CourtHierarchicalModel)
        nn.Module.__init__(model)
        model.in_channels = 3
        model.target_bundle_spec = target_bundle
        encoder = object.__new__(CourtDINOv3Encoder)
        nn.Module.__init__(encoder)
        model.encoder = encoder
        model.register_parameter('weight', nn.Parameter(torch.zeros(2)))
        return model

    monkeypatch.setattr(CourtHierarchicalModel, 'from_config', staticmethod(make_model))
    checkpoint = {'hyper_parameters': {'config': raw, 'target_bundle_state': serialize_target_bundle(bundle)},
                  'state_dict': {'model.weight': torch.tensor([1., 3.]), 'criterion.training_only': torch.ones(1)}}
    path = tmp_path / 'court.ckpt'
    torch.save(checkpoint, path)
    pair = load_court_pair(path, resolver=resolver)
    torch.testing.assert_close(pair.model.weight, torch.tensor([1., 3.]))
    with pytest.raises(ValueError, match='strict'):
        load_court_pair(path, resolver=resolver, strict=False)
    checkpoint['state_dict']['model.unexpected'] = torch.ones(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        load_court_pair(path, resolver=resolver)
